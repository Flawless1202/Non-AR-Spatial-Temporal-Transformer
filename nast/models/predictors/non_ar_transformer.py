from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.utils.config import ConfigDict

from ...datasets.pipelines.box_transforms import multi_dim_boxes_convert, multi_dim_boxes_denormalize
from ..builder import PREDICTORS, build_encoder, build_decoder, build_embedding, build_loss


@PREDICTORS.register_module()
class NonAutoRegressionTransformer(nn.Module):

    def __init__(self,
                 encoder: ConfigDict,
                 box_encoder: ConfigDict,
                 loss: ConfigDict,
                 decoder: Optional[ConfigDict] = None,
                 time_encoder: Optional[ConfigDict] = None,
                 vision_encoder: Optional[ConfigDict] = None,
                 pose_encoder: Optional[ConfigDict] = None,
                 spatial_pos_embed: Optional[ConfigDict] = None,
                 temporal_pos_embed: Optional[ConfigDict] = None,
                 index_pos_embed: Optional[ConfigDict] = None):
        super().__init__()

        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder) if decoder else None
        self.box_encoder = build_embedding(box_encoder)

        self.output_box_encoder = build_embedding(box_encoder)

        self.loss_func = build_loss(loss)
        self.time_encoder = build_embedding(time_encoder) if time_encoder else None
        self.vision_encoder = build_encoder(vision_encoder) if vision_encoder else None
        self.spatial_pos_embed = build_embedding(spatial_pos_embed) if spatial_pos_embed else None
        self.temporal_pos_embed = build_embedding(temporal_pos_embed) if temporal_pos_embed else None
        self.index_pos_embed = build_embedding(index_pos_embed) if index_pos_embed else None
        self.pose_encoder = build_embedding(pose_encoder) if pose_encoder else None
        self.mlp = nn.Linear(encoder.d_model, 4)
        self.tgt_embed = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model),
            nn.ReLU(inplace=True)
        )

        self.enc_weights_embed = nn.Linear(encoder.d_model, 1)
        self.futr_weights_embed = nn.Linear(encoder.d_model, 1)

    def forward(self,
                data: Dict[str, Dict[str, Tensor]],
                mode: str = 'one_step') -> Tensor:
        hist, futr = data['hist'], data['futr']

        if 'features' not in hist:
            hist['features'] = None

        hist_embed = self._input_embed(hist['boxes'], hist['timestamps'], hist['features'])
        hist_embed *= hist_embed.shape[-1] ** 0.5
        enc_pos_embed = self._pos_embed(hist['boxes'])
        enc_output, enc_attn_list = self.encoder(hist_embed, key_padding_mask=hist['masks'], pos_embed=enc_pos_embed)

        query_pos_embed = self._pos_embed(futr['masks'])
        tgt_seq = self._tgt_generate(hist_embed, enc_output, query_pos_embed)

        if self.decoder is not None:
            dec_output, dec_attn_list, dec_enc_attn_list = self.decoder(dec=tgt_seq, enc=enc_output,
                                                                        dec_key_padding_mask=futr['masks'],
                                                                        enc_key_padding_mask=hist['masks'],
                                                                        pos_embed=enc_pos_embed,
                                                                        dec_mask=None,
                                                                        query_pos_embed=query_pos_embed)
        else:
            dec_output = tgt_seq
        pred = self.mlp(dec_output)

        return pred

    def loss(self,
             pred_boxes: Tensor,
             current_boxes: Tensor,
             gold_boxes: Tensor,
             gold_masks: Tensor,
             box_range: Tuple[int, int] = (512, 320)) -> Tensor:
        masks = gold_masks.reshape(-1, 1).float()
        pred = multi_dim_boxes_denormalize(pred_boxes + current_boxes, box_range).reshape(-1, 4)
        gold = multi_dim_boxes_denormalize(gold_boxes, box_range).reshape(-1, 4)
        loss = self.loss_func(pred * masks, gold * masks)

        return loss

    @staticmethod
    def get_target_boxes(gold: Dict[str, Tensor], pred_fmt: str = 'cxcywh', box_range: Tuple[int, int] = (512, 320)):
        gold_boxes, gold_masks = gold['boxes'], gold['masks']
        gold_boxes = multi_dim_boxes_convert(gold_boxes, in_fmt=pred_fmt, out_fmt='xyxy')
        gold_boxes = multi_dim_boxes_denormalize(gold_boxes, box_range=box_range)

        return gold_boxes, gold_masks

    def predict(self, data: Dict[str, Dict[str, Tensor]], pred_fmt: str = 'cxcywh',
                box_range: Tuple[int, int] = (512, 320)):
        with torch.no_grad():
            current_boxes = data['hist']['boxes'][:, -1:]
            pred = self(data)
            pred_boxes = pred + current_boxes
            pred_boxes = multi_dim_boxes_convert(pred_boxes, in_fmt=pred_fmt, out_fmt='xyxy')
            pred_boxes = multi_dim_boxes_denormalize(pred_boxes, box_range=box_range)

        return pred_boxes

    def _input_embed(self, boxes: Tensor, timestamps: Optional[Tensor] = None, features: Optional[Tensor] = None):
        embed = self.box_encoder(boxes)

        if self.time_encoder is not None:
            time_embed = self.time_embed(timestamps).repeat(1, 1, boxes.shape[-2], 1)
            embed = torch.cat((embed, time_embed), dim=-1)

        if self.vision_encoder is not None:
            batch_size, sample_len, object_len = features.size()[:3]
            features = features.reshape(-1, *features.size()[3:])
            vision_embed = self.vision_encoder(features)
            vision_embed = vision_embed.reshape(batch_size, sample_len, object_len, -1)
            embed = torch.cat((embed, vision_embed), dim=-1)

        return embed

    def _pos_embed(self, boxes: Tensor, masks: Optional[Tensor] = None, timestamps: Optional[Tensor] = None):
        pos_embed = None
        bsize, seq_len, obj_len = boxes.shape[:-1]

        if self.index_pos_embed:
            idx = torch.arange(seq_len, device=boxes.device).reshape(1, seq_len, 1, 1).repeat(bsize, 1, obj_len, 1)
            pos_embed = pos_embed + self.index_pos_embed(idx) if pos_embed else self.index_pos_embed(idx)

        if self.spatial_pos_embed:
            sp_embed = self.spatial_pos_embed(boxes, masks)
            pos_embed = pos_embed + sp_embed if pos_embed else sp_embed

        if self.temporal_pos_embed:
            tp_embed = self.temporal_pos_embed(timestamps).unsqueeze(-2).repeat(1, 1, obj_len, 1)
            pos_embed = pos_embed + tp_embed if pos_embed else tp_embed

        return pos_embed

    def _tgt_generate(self, hist_embed, enc_output, query_pos_embed):
        _, hist_len, obj_len, d_model = hist_embed.shape
        futr_len = query_pos_embed.shape[1]

        futr_weights_embed = self.futr_weights_embed(query_pos_embed).permute(0, 2, 1, 3).reshape(-1, futr_len, 1)
        enc_weights_embed = self.enc_weights_embed(enc_output).permute(0, 2, 1, 3).reshape(-1, hist_len, 1)
        futr_weights = torch.bmm(futr_weights_embed, enc_weights_embed.transpose(-1, -2))
        tgt_seq = torch.bmm(futr_weights, self.tgt_embed(hist_embed).permute(0, 2, 1, 3).reshape(-1, hist_len, d_model))
        tgt_seq = tgt_seq.reshape(-1, obj_len, futr_len, d_model).permute(0, 2, 1, 3)
        
        return tgt_seq

    @staticmethod
    def _get_subsequent_mask(seq: Tensor):
        len_seq = seq.shape[1]
        subsequent_mask = torch.tril(torch.ones((len_seq, len_seq), device=seq.device), diagonal=0).bool()
        return subsequent_mask

from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .masked_tensor_attention import MaskedTensorAttention
from .spatial_temporal_tensor_attention import SpatialTemporalTensorAttention


class CausalConvResidual(nn.Module):

    def __init__(self, d_model: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.1,
                 norm_befor: bool = False):
        super().__init__()

        padding = kernel_size // 2

        self.conv_layers = nn.ModuleList()
        self.dilated_conv_layers = nn.ModuleList()
        for _ in range(2):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=1, padding=padding, dilation=1),
                nn.ReLU(inplace=True)))
        for _ in range(2):
            self.dilated_conv_layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=1, padding=dilation, dilation=dilation),
                nn.ReLU(inplace=True)))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm_befor = norm_befor

    def forward(self, x: Tensor, masks: Optional[Tensor] = None):
        batch_size, seq_len, obj_len, d_model = x.shape
        
        no_zero_mask = masks.permute(0, 2, 3, 1).reshape(batch_size * obj_len, 1, seq_len).float() \
            if masks is not None else x.new_ones(batch_size * obj_len, 1, seq_len).float()

        x2 = self.norm(x) if self.norm_befor else x

        x2 = x2.permute(0, 2, 3, 1).reshape(batch_size * obj_len, d_model, seq_len)

        for idx in range(len(self.conv_layers)):
            x2 = self.conv_layers[idx](x2) * no_zero_mask
            x2 = self.dilated_conv_layers[idx](x2) * no_zero_mask
        x2 = x2.reshape(-1, obj_len, d_model, seq_len).permute(0, 3, 1, 2)

        x = x + self.dropout(x2)

        return self.norm(x) if not self.norm_befor else x


class TensorAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 num_head: int,
                 mode: str = 'temporal',
                 dropout: float = 0.1,
                 norm_before: bool = False):
        super().__init__()

        self.mode = mode
        self.mta = MaskedTensorAttention(d_model, num_head, mode, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_before = norm_before

    def forward(self,
                x: Union[Tensor, Tuple[Tensor, Tensor]],
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                pos_embed: Union[Optional[Tensor], Tuple[Optional[Tensor], Optional[Tensor]]] = None) \
            -> Tuple[Tensor, Tensor]:

        if isinstance(x, Tensor):
            x2 = self.norm(x) if self.norm_before else x
            q = k = self._with_pos_embed(x2, pos_embed)
            x2, attn = self.mta(q, k, x2, key_padding_mask, attn_mask)

            x = x + self.dropout(x2)

            x = self.norm(x) if not self.norm_before else x

            return x, attn

        elif isinstance(x, Tuple) and len(x) == 2 and isinstance(pos_embed, Tuple) and len(pos_embed) == 2:
            dec, enc = x
            query_pos_embed, enc_pos_embed = pos_embed
            dec2 = self.norm(dec) if self.norm_before else dec
            dec2, dec_enc_attn = self.mta(self._with_pos_embed(dec2, query_pos_embed),
                                          self._with_pos_embed(enc, enc_pos_embed),
                                          enc,
                                          key_padding_mask,
                                          attn_mask)
            dec = dec + self.dropout(dec2)
            dec = self.norm(dec) if not self.norm_before else dec
            return dec, dec_enc_attn
        
        else:
            raise NotImplementedError

    @staticmethod
    def _with_pos_embed(tensor: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        return tensor if pos is None else tensor + pos


class SpatialTemporalTensorAttentionBlock(TensorAttentionBlock):

    def __init__(self,
                 d_model: int,
                 num_head: int,
                 dropout: float = 0.1,
                 norm_before: bool = False):
        super().__init__(d_model, num_head=num_head, dropout=dropout, norm_before=norm_before)
        self.mta = SpatialTemporalTensorAttention(d_model, num_head, dropout)


class PoswiseFeedForward(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_hid: int,
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 norm_before: bool = False):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_hid)
        self.act = getattr(F, activation)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hid, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.norm_before = norm_before

    def forward(self, x: Tensor) -> Tensor:

        x2 = self.norm(x) if self.norm_before else x
        x2 = self.fc2(self.dropout1(self.act(self.fc1(x2))))
        x = x + self.dropout2(x2)

        return self.norm(x) if not self.norm_before else x


class EncoderLayer(nn.Module):
    def __init__(self,
                 num_head: int,
                 d_model: int,
                 d_hid: int,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 norm_before: bool = False,
                 with_global_conv: bool = True):

        super().__init__()

        self.tab = SpatialTemporalTensorAttentionBlock(d_model, num_head, dropout=dropout, norm_before=norm_before)

        self.causal_conv = CausalConvResidual(d_model, kernel_size=3, dilation=3,
                                              dropout=dropout, norm_befor=norm_before) if with_global_conv else None

        self.ff = PoswiseFeedForward(d_model, d_hid, activation, dropout, norm_before)

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor]:

        x, attn = self.tab(x, key_padding_mask=key_padding_mask, pos_embed=pos_embed)
        x = self.causal_conv(x, key_padding_mask) if self.causal_conv is not None else x

        return self.ff(x), attn


class DecoderLayer(nn.Module):

    def __init__(self,
                 num_head: int,
                 d_model: int,
                 d_hid: int,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 norm_before: bool = False,
                 with_global_conv: bool = True):
        super().__init__()

        self.dec_slf_tab = SpatialTemporalTensorAttentionBlock(d_model, num_head, dropout=dropout,
                                                norm_before=norm_before)
        self.causal_conv1 = CausalConvResidual(d_model, kernel_size=3, dilation=3,
                                               dropout=dropout, norm_befor=norm_before) if with_global_conv else None
        self.dec_enc_tab = TensorAttentionBlock(d_model, num_head, mode='temporal', dropout=dropout,
                                                norm_before=norm_before)
        self.causal_conv2 = CausalConvResidual(d_model, kernel_size=3, dilation=3,
                                               dropout=dropout, norm_befor=norm_before) if with_global_conv else None
        self.ff = PoswiseFeedForward(d_model, d_hid, activation, dropout, norm_before)

    def forward(self,
                dec: Tensor,
                enc: Tensor,
                dec_key_padding_mask: Optional[Tensor] = None,
                enc_key_padding_mask: Optional[Tensor] = None,
                dec_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None,
                query_pos_embed: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor, Tensor]:

        dec_output = dec

        dec_output, dec_slf_attn = self.dec_slf_tab(dec_output,
                                                    key_padding_mask=dec_key_padding_mask,
                                                    attn_mask=dec_mask,
                                                    pos_embed=query_pos_embed)
        dec_output = self.causal_conv1(dec_output, dec_key_padding_mask) \
            if self.causal_conv1 is not None else dec_output

        dec_output, dec_enc_attn = self.dec_enc_tab((dec_output, enc),
                                                    key_padding_mask=enc_key_padding_mask,
                                                    pos_embed=(query_pos_embed, pos_embed))
        dec_output = self.causal_conv2(dec_output, None) \
            if self.causal_conv2 is not None else dec_output

        return self.ff(dec_output), dec_slf_attn, dec_enc_attn

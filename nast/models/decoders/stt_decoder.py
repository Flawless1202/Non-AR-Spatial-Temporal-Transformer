from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..builder import DECODERS
from ..utils.transformer_layers import DecoderLayer


@DECODERS.register_module()
class SpatialTemporalDecoder(nn.Module):

    def __init__(self,
                 num_head: int,
                 d_model: int,
                 d_hid: int,
                 num_stack: int = 4,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 norm_before: bool = False,
                 embed_only_first: bool = False,
                 with_global_conv: bool = True):
        super().__init__()

        self.embed_only_first = embed_only_first

        self.decoder_stack = nn.ModuleList([
            DecoderLayer(num_head, d_model, d_hid, activation=activation,
                         dropout=dropout, norm_before=norm_before,
                         with_global_conv=with_global_conv) for _ in range(num_stack)
        ])

        self._reset_parameters()

    def forward(self,
                dec: Tensor,
                enc: Tensor,
                dec_key_padding_mask: Optional[Tensor] = None,
                enc_key_padding_mask: Optional[Tensor] = None,
                dec_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None,
                query_pos_embed: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tuple[Tensor, ...], Tuple[Tensor, ...]]:

        dec_output = dec
        dec_slf_attn_list = []
        dec_enc_attn_list = []

        for idx, layer in enumerate(self.decoder_stack):
            query_embed = None if self.embed_only_first and idx > 0 else query_pos_embed
            embed = None if self.embed_only_first and idx > 0 else pos_embed
            dec_output, dec_slf_attn, dec_enc_attn = layer(
                dec_output, enc, dec_key_padding_mask, enc_key_padding_mask, dec_mask, embed, query_embed)

            dec_slf_attn_list.append(dec_slf_attn)
            dec_enc_attn_list.append(dec_enc_attn)

        return dec_output, tuple(dec_slf_attn_list), tuple(dec_enc_attn_list)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

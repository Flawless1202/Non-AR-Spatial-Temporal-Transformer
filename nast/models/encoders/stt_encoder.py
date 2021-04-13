from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.transformer_layers import EncoderLayer
from ..builder import ENCODERS


@ENCODERS.register_module()
class SpatialTemporalEncoder(nn.Module):

    def __init__(self,
                 num_head: int,
                 d_model: int,
                 d_hid: int,
                 num_stack: int = 4,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 norm_before: bool = True,
                 embed_only_first: bool = False,
                 with_global_conv: bool = True):
        super().__init__()

        self.embed_only_first = embed_only_first

        self.encoder_stack = nn.ModuleList([
            EncoderLayer(num_head, d_model, d_hid, activation=activation,
                         dropout=dropout, norm_before=norm_before,
                         with_global_conv=with_global_conv) for _ in range(num_stack)
        ])

        self._reset_parameters()

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        enc_output = x
        enc_attn_list = []

        for idx, layer in enumerate(self.encoder_stack):
            embed = None if self.embed_only_first and idx > 0 else pos_embed
            enc_output, enc_attn = layer(enc_output, key_padding_mask, embed)
            enc_attn_list.append(enc_attn)

        return enc_output, tuple(enc_attn_list)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

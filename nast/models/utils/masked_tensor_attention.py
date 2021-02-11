from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .multi_head_attention import MultiheadAttention


class MaskedTensorAttention(nn.Module):
    """A custom class to implement the multi-dimension MultiheadAttention based on the
        `torch.nn.MultiheadAttention` class.
        
    Args:
        d_model: total dimension of the model.
        num_head: parallel attention heads.
        mode: choose from 'spatial' and 'temporal'
        dropout: a Dropout layer on attn_output_weights. Default: 0.1.
    """

    def __init__(self,
                 d_model: int,
                 num_head: int,
                 mode: str = 'spatial',
                 dropout: float = 0.1):
        super().__init__()

        assert mode in ['spatial', 'temporal']
        self.mode = mode
        self.d_model = d_model

        self.attn = MultiheadAttention(d_model, num_head, dropout)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor]:

        batch_size, seq_len, obj_len, d_model = query.shape
        attn_dim = 2 if self.mode == 'spatial' else 1
        target_len = query.shape[attn_dim]
        source_len = key.shape[attn_dim]
        reserve_size = [query.shape[idx] for idx in range(len(query.shape) - 1) if idx != attn_dim]

        if key_padding_mask is not None:
            kp_mask = ~key_padding_mask.transpose(1, 2) if attn_dim == 1 else ~key_padding_mask
            kp_mask = kp_mask.reshape(-1, source_len)
        else:
            kp_mask = None

        a_mask = ~attn_mask if attn_mask is not None else None

        if a_mask is not None and attn_mask.shape[1] > 2:
            no_zero_mask = ~(torch.bmm((~kp_mask).unsqueeze(2).float(), (~kp_mask).unsqueeze(1).float()).bool())
            no_zero_mask += a_mask.unsqueeze(0)
            no_zero_mask = (no_zero_mask.reshape(kp_mask.size(0), -1).sum(dim=1)) < target_len * source_len
        else:
            # print(kp_mask.any())
            no_zero_mask = kp_mask.sum(dim=1) < source_len

        q = query if attn_dim == 2 else query.permute(0, 2, 1, 3)
        k = key if attn_dim == 2 else key.permute(0, 2, 1, 3)
        v = value if attn_dim == 2 else value.permute(0, 2, 1, 3)

        q = q.reshape(-1, target_len, self.d_model)[no_zero_mask].permute(1, 0, 2)
        k = k.reshape(-1, source_len, self.d_model)[no_zero_mask].permute(1, 0, 2)
        v = v.reshape(-1, source_len, self.d_model)[no_zero_mask].permute(1, 0, 2)
        
        # print(no_zero_mask.any())
        no_zero_output, no_zero_attn = self.attn(q, k, v, key_padding_mask=kp_mask[no_zero_mask], attn_mask=a_mask)

        output = no_zero_output.new_zeros((kp_mask.size(0), target_len, d_model))
        attn = no_zero_output.new_zeros((kp_mask.size(0), target_len, source_len))
        output[no_zero_mask] += no_zero_output.permute(1, 0, 2)
        output = output.permute(1, 0, 2)
        attn[no_zero_mask] += no_zero_attn

        if attn_dim == 1:
            output = output.reshape(batch_size, obj_len, seq_len, d_model).transpose(1, 2)
        else:
            output = output.reshape(batch_size, seq_len, obj_len, d_model)

        return output, attn.reshape(reserve_size + list(attn.size()[-2:]))

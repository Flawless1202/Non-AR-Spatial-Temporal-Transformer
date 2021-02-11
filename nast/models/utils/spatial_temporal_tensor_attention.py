import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch._jit_internal import Optional, Tuple
from torch.nn import grad  # noqa: F401
from torch.nn.functional import linear
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.parameter import Parameter
from torch.overrides import has_torch_function, handle_torch_function


def multi_head_attention_weights(query: Tensor,
                                 key: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 dropout_p: float,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 attn_mask: Optional[Tensor] = None
                                 ) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    if not torch.jit.is_scripting():
        tens_ops = (query, key, in_proj_weight, in_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                multi_head_attention_weights, tens_ops, query, key,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                dropout_p, training=training, key_padding_mask=key_padding_mask,
                attn_mask=attn_mask)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    # assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if torch.equal(query, key):
        q, k = linear(query, in_proj_weight, in_proj_bias).chunk(2, dim=-1)

    else:
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = 0
        _end = embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = linear(query, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = linear(key, _w, _b)

    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
               attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    return attn_output_weights


class SpatialTemporalTensorAttention(nn.Module):
    """A custom class to implement the multi-dimension Spatial-Temporal MultiheadAttention based on the
        `torch.nn.MultiheadAttention` class.
        
    Args:
        d_model: total dimension of the model.
        num_head: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.1.
    """

    def __init__(self,
                 d_model: int,
                 num_head: int,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()

        # assert mode in ['spatial', 'temporal']
        # self.mode = mode
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.head_dim = d_model // num_head

        assert self.head_dim * num_head == self.d_model, "embed_dim must be divisible by num_heads"

        self.in_temporal_proj_weight = Parameter(torch.empty(3 * d_model, d_model))
        self.in_spatial_proj_weight = Parameter(torch.empty(2 * d_model, d_model))

        if bias:
            self.in_temporal_proj_bias = Parameter(torch.empty(3 * d_model))
            self.in_spatial_proj_bias = Parameter(torch.empty(2 * d_model))
        else:
            self.register_parameter('in_temporal_proj_bias', None)
            self.register_parameter('in_spatial_proj_bias', None)

        self.out_proj = _LinearWithBias(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.in_temporal_proj_weight)
        torch.nn.init.xavier_uniform_(self.in_spatial_proj_weight)

        if self.in_temporal_proj_bias is not None:
            torch.nn.init.constant_(self.in_temporal_proj_bias, 0.)

        if self.in_spatial_proj_bias is not None:
            torch.nn.init.constant_(self.in_spatial_proj_bias, 0.)

        # self.attn = MultiheadAttention(d_model, num_head, dropout)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(SpatialTemporalTensorAttention, self).__setstate__(state)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor]:

        batch_size, seq_len, obj_len, d_model = query.shape

        v = value.permute(0, 2, 1, 3)
        v = v.reshape(-1, seq_len, d_model)
        v = F.linear(v, self.in_temporal_proj_weight[self.d_model * 2:],
                     self.in_temporal_proj_bias[self.d_model * 2:])
        v = v.contiguous().view(-1, batch_size * obj_len * self.num_head, self.head_dim).transpose(0, 1)

        # for temporal attention
        temporal_attn, no_zero_mask = self._compute_attn(query, key, self.d_model, self.num_head,
                                                         self.in_temporal_proj_weight[:self.d_model * 2],
                                                         self.in_temporal_proj_bias[:self.d_model * 2],
                                                         self.dropout, self.training,
                                                         key_padding_mask, attn_mask, attn_dim=1)
        # for spatial attention
        spatial_attn, _ = self._compute_attn(query, key, self.d_model, self.num_head,
                                             self.in_spatial_proj_weight,
                                             self.in_spatial_proj_bias[:self.d_model * 2],
                                             self.dropout, self.training,
                                             key_padding_mask, attn_mask=None, attn_dim=2)

        temporal_attn = temporal_attn.reshape(batch_size, obj_len, self.num_head, seq_len, seq_len
                                              ).transpose(1, 2).reshape(-1, obj_len, seq_len, seq_len)
        spatial_attn = spatial_attn.reshape(batch_size, seq_len, self.num_head, obj_len, obj_len
                                            ).transpose(1, 2).reshape(-1, seq_len, obj_len, obj_len)
        # import pdb; pdb.set_trace()
        spatial_temporal_attn = torch.bmm(spatial_attn.reshape(-1, obj_len, obj_len),
                                          temporal_attn.transpose(1, 2).reshape(-1, obj_len, seq_len))
        spatial_temporal_attn = spatial_temporal_attn.reshape(-1, seq_len, obj_len, seq_len).transpose(1, 2).reshape(
            -1, self.num_head, obj_len, seq_len, seq_len).transpose(1, 2).reshape(-1, seq_len, seq_len)

        # attn_weights = torch.zeros_like(spatial_temporal_attn)
        # attn_weights[no_zero_mask] = F.softmax(spatial_temporal_attn[no_zero_mask], dim=-1)
        attn_weights = F.softmax(spatial_temporal_attn, dim=-1)
        # spatial_temporal_attn = F.tanh(spatial_temporal_attn)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.reshape(batch_size, obj_len, seq_len, d_model).transpose(1, 2)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        
        return attn_output, attn_weights

    @staticmethod
    def _compute_attn(query, key, d_model, num_heads, in_proj_weight, in_proj_bias, dropout_p, training,
                      key_padding_mask=None, attn_mask=None, attn_dim=1):
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
            no_zero_mask = kp_mask.sum(dim=1) < source_len

        q = query if attn_dim == 2 else query.permute(0, 2, 1, 3)
        k = key if attn_dim == 2 else key.permute(0, 2, 1, 3)

        q = q.reshape(-1, target_len, d_model)[no_zero_mask].permute(1, 0, 2)
        k = k.reshape(-1, source_len, d_model)[no_zero_mask].permute(1, 0, 2)
        
        # print(q.size(), k.size())

        # assert k.size(0) == v.size(0) and k.size(1) == v.size(1)

        no_zero_attn_weights = multi_head_attention_weights(q, k, d_model, num_heads,
                                                            in_proj_weight, in_proj_bias,
                                                            dropout_p, training,
                                                            kp_mask[no_zero_mask], a_mask)
        no_zero_attn_weights = F.softmax(no_zero_attn_weights, dim=-1)
        # no_zero_attn_weights = F.tanh(no_zero_attn_weights)

        attn_weights = no_zero_attn_weights.new_zeros((kp_mask.size(0) * num_heads, target_len, source_len))
        # print(no_zero_attn_weights.size(), attn_weights.size())
        # import pdb; pdb.set_trace()
        no_zero_mask = torch.cat([no_zero_mask.unsqueeze(1) for _ in range(num_heads)], dim=1).reshape(-1)  # TODO: uncertain
        attn_weights[no_zero_mask] += no_zero_attn_weights

        return attn_weights, no_zero_mask  #.reshape(reserve_size + list(attn_weights.size()[-2:]))


if __name__ == '__main__':
    stta = SpatialTemporalTensorAttention(64, 4, 0.1)
    inp = torch.rand(16, 15, 9, 64)
    key_padding_mask = inp.new_ones(16, 15, 9, 1).bool()
    out = stta(inp, inp, inp, key_padding_mask, attn_mask=None)
    print(out[0].size())

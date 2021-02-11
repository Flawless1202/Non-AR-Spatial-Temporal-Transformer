import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..builder import EMBEDDINGS


@EMBEDDINGS.register_module()
class MLP(nn.Module):

    def __init__(self,
                 d_in: int,
                 d_hid: int,
                 d_out: int,
                 num_layers: int,
                 activation: str = "relu",
                 norm: bool = True,
                 dropout: float = 0.1):
        super().__init__()

        self.num_layers = num_layers
        d_hids = [d_hid] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([d_in] + d_hids, d_hids + [d_out]))
        self.act = getattr(F, activation) if activation is not None else nn.Sequential()
        self.norm = nn.LayerNorm(d_out) if norm is True else nn.Sequential()
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = self.act(layer(x)) if idx < self.num_layers - 1 else layer(x)
        x = self.norm(self.dropout(x))
        return x

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


@EMBEDDINGS.register_module()
class TimeEmbeddingSine(nn.Module):

    def __init__(self,
                 d_model: int = 64,
                 temperature: int = 10000,
                 scale: Optional[float] = None,
                 requires_grad: bool = False):
        super().__init__()

        self.d_model = d_model
        self.temperature = temperature
        self.scale = 2 * math.pi if scale is None else scale
        self.requires_grad = requires_grad

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.clone()
        d_embed = self.d_model
        dim_t = torch.arange(d_embed, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / d_embed)
        x = x / dim_t
        x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(-2)
        return x if self.requires_grad else x.detach()


@EMBEDDINGS.register_module()
class Time2Vec(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()

        self.fc_periodic = nn.Linear(1, d_model - 1, bias=True)
        self.fc_no_periodic = nn.Linear(1, 1, bias=True)
        self.act = torch.sin
        self._reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        periodic_x = self.act(self.fc_periodic(x))
        no_periodic_x = self.fc_no_periodic(x)
        return torch.cat([periodic_x, no_periodic_x], dim=-1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

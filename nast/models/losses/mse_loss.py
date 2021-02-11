from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction: The method that reduces the loss to a scalar. Options are "none", "mean" and "sum".
        loss_weight: The weight of the loss.
    """

    def __init__(self, reduction: str = 'mean', loss_weight: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor, weight: Optional[Tensor] = None, avg_factor: Optional[int] = None) \
            -> Tensor:
        """Forward function of loss.

        Args:
            pred: The prediction.
            target: The learning target of the prediction.
            weight: Weight of the loss for each prediction.
            avg_factor: Average factor that is used to average the loss.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor)
        return loss

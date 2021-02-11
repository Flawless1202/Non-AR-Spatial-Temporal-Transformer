from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    """Smooth L1 loss.

    Args:
        pred: The prediction.
        target: The learning target of the prediction.
        beta: The threshold in the piecewise function.

    Returns:
        torch.Tensor: Calculated loss
    """

    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


@weighted_loss
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss.

    Args:
        pred: The prediction.
        target: The learning target of the prediction.

    Returns:
        loss: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super().__init__()

        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred: The prediction.
            target: The learning target of the prediction.
            weight: The weight of loss for each prediction.
            avg_factor: Average factor that is used to average the loss.
            reduction_override: The reduction method used to override the original reduction method of the loss.
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction: The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight: The weight of loss.
    """

    def __init__(self, reduction: str = 'mean', loss_weight: float = 1.0):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred: The prediction.
            target: The learning target of the prediction.
            weight: The weight of loss for each prediction.
            avg_factor: Average factor that is used to average the loss.
            reduction_override: The reduction method used to override the original reduction method of the loss.
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

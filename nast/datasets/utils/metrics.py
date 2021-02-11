from typing import List, Dict, Optional

import torch
from torch import Tensor
from torchvision.ops import box_iou, box_convert


def calculate_metrics(pred: Tensor, gold: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
    if mask is None:
        mask = pred.new_ones(pred.shape[:-1]).unsqueeze(-1)
    pred_center = box_convert(pred, in_fmt='xyxy', out_fmt='cxcywh')[..., :2]
    gold_center = box_convert(gold, in_fmt='xyxy', out_fmt='cxcywh')[..., :2]
    sum_ade = ((pred_center - gold_center) ** 2 * mask.float()).sum(dim=-1).sqrt().sum()
    sum_fde = ((pred_center[:, -1] - gold_center[:, -1]) ** 2 * mask.float()[:, -1]).sum(dim=-1).sqrt().sum()
    num_ade = mask.float().sum()
    num_fde = mask.float()[:, -1].sum()
    sum_fiou = box_iou(pred[:, -1].reshape(-1, 4).contiguous(), gold[:, -1].reshape(-1, 4).contiguous()).diag()
    sum_fiou = sum_fiou[~sum_fiou.isnan()].sum()

    return {"sum_ade": sum_ade, "sum_fde": sum_fde, "num_ade": num_ade, "num_fde": num_fde, "sum_fiou": sum_fiou}


def summarize_metrics(outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    seq_len = len(outputs)
    stats = {k: sum([o[k] for o in outputs]) for k in outputs[0].keys()}
    ade = stats['sum_ade'] / stats['num_ade']
    fde = stats['sum_fde'] / stats['num_fde']
    fiou = stats['sum_fiou'] / stats['num_fde']
    avg_loss = stats["val_loss"] / seq_len if 'val_loss' in stats.keys() else None

    return {'ade': ade, 'fde': fde, 'fiou': fiou, "avg_val_loss": avg_loss}

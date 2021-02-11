from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_convert, clip_boxes_to_image


def multi_dim_boxes_convert(boxes: Union[Tensor, np.ndarray], in_fmt: str, out_fmt: str) \
        -> Union[Tensor, np.ndarray]:
    """
    A multi-dim wrapper of `torchvision.ops.box_convert`.

    Args:
        boxes: boxes which will be converted. The original size could be any format matching :math:`(*shape, 4)`.
        in_fmt: Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
        out_fmt: Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].

    Returns:
        Boxes into converted format with original shape and type.
    """

    new_boxes = torch.from_numpy(boxes).clone() if isinstance(boxes, np.ndarray) else boxes.clone()
    ori_size = new_boxes.size()
    new_boxes = box_convert(new_boxes.reshape(-1, 4), in_fmt, out_fmt).reshape(ori_size)
    return new_boxes.numpy() if isinstance(boxes, np.ndarray) else new_boxes


def multi_dim_boxes_normalize(boxes: Union[Tensor, np.ndarray], box_range: Tuple[int, int]) \
        -> Union[Tensor, np.ndarray]:
    """
    Normalize the coordinates of boxes to :math:`[0, 1]` with the given box_range.

    Args:
        boxes: boxes which will be normalized. The original size could be any format matching :math:`(*shape, 4)`.
        box_range: The width and height used to normalize the boxes.

    Returns:
        new_boxes: boxes normalized to :math:`[0, 1]`.
    """

    x_scale, y_scale = box_range
    new_boxes = boxes.copy() if isinstance(boxes, np.ndarray) else boxes.clone()
    new_boxes[..., 0::2] /= x_scale
    new_boxes[..., 1::2] /= y_scale
    return new_boxes


def multi_dim_boxes_denormalize(boxes: Union[Tensor, np.ndarray], box_range: Tuple[int, int]) \
        -> Union[Tensor, np.ndarray]:
    """
    Denormalize the coordinates of boxes to match the `box_range`:math:`(width, height)`.

    Args:
        boxes: boxes which will be denormalized. The original size could be any format matching :math:`(*shape, 4)`.
        box_range: The width and height used to normalize the boxes.

    Returns:
        new_boxes: boxes denormalized to match the `box_range`.
    """

    x_scale, y_scale = box_range
    new_boxes = boxes.copy() if isinstance(boxes, np.ndarray) else boxes.clone()
    new_boxes[..., 0::2] *= x_scale
    new_boxes[..., 1::2] *= y_scale
    return new_boxes


def multi_dim_boxes_clip(boxes: Union[Tensor, np.ndarray], box_range: Tuple[int, int]) \
        -> Union[Tensor, np.ndarray]:
    """
    A multi-dim wrapper of `torchvision.ops.clip_boxes_to_image`.

    Args:
        boxes: boxes which will be clipped. The original size could be any format matching :math:`(*shape, 4)`.
        box_range: The width and height used to clip the boxes.

    Returns:
        new_boxes: boxes clipped to match the `box_range`.
    """

    new_boxes = torch.from_numpy(boxes).clone() if isinstance(boxes, np.ndarray) else boxes.clone()
    ori_size = new_boxes.size()
    new_boxes = clip_boxes_to_image(new_boxes.reshape(-1, 4), box_range[::-1]).reshape(ori_size)
    return new_boxes.numpy() if isinstance(boxes, np.ndarray) else new_boxes


def multi_dim_boxes_rescale(boxes: Union[Tensor, np.ndarray], src_range: Tuple[int, int], dst_range: Tuple[int, int]) \
        -> Union[Tensor, np.ndarray]:
    """
    Rescale the coordinates of boxes from `src_range` to `dst_range`.

    Args:
        boxes: boxes which will be normalized. The original size could be any format matching :math:`(*shape, 4)`.
        src_range: The width and height of source range.
        dst_range: The width and height of destination range.

    Returns:
        new_boxes: boxes rescaled to `dst_range`.
    """

    x_scale, y_scale = dst_range[0] / src_range[0], dst_range[1] / src_range[1]
    new_boxes = boxes.copy() if isinstance(boxes, np.ndarray) else boxes.clone()
    new_boxes[..., 0::2] *= x_scale
    new_boxes[..., 1::2] *= y_scale
    return new_boxes

from typing import Tuple, Sequence, Dict, Union, Callable

from ..builder import PIPELINES, build_pipelines
from .box_transforms import *


class Compose(object):

    def __init__(self, transforms: Sequence[Union[dict, Callable]]):
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_pipelines(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data: Dict[str, Dict[str, Union[Tensor, np.ndarray]]]) \
            -> Dict[str, Dict[str, Union[Tensor, np.ndarray]]]:

        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


@PIPELINES.register_module()
class ToTensor(object):

    def __init__(self, move_features_dim: bool = True):
        self.move_features_dim = move_features_dim

    def __call__(self, data: Dict[str, Dict[str, Union[Tensor, np.ndarray]]]) \
            -> Dict[str, Dict[str, Union[Tensor, np.ndarray]]]:

        for phase in data:
            for key in data[phase]:
                data[phase][key] = torch.from_numpy(data[phase][key]) if isinstance(data[phase][key], np.ndarray) \
                              else data[phase][key]
                if key == 'features' and self.move_features_dim:
                    data[phase][key] = torch.movedim(data[phase][key], -1, -3)

        return data

    def __repr__(self) -> str:
        return f"self.__class__.__name__(move_features_dim={self.move_features_dim})"


@PIPELINES.register_module()
class BoxesConvert(object):

    def __init__(self, in_fmt: str, out_fmt: str):
        self.in_fmt = in_fmt
        self.out_fmt = out_fmt

    def __call__(self, data: Dict[str, Dict[str, Union[Tensor, np.ndarray]]]) \
            -> Dict[str, Dict[str, Union[Tensor, np.ndarray]]]:
        for phase in data.keys():
            data[phase]['boxes'] = multi_dim_boxes_convert(data[phase]['boxes'], self.in_fmt, self.out_fmt)
        return data

    def __repr__(self) -> str:
        return f"self.__class__.__name__(in_fmt={self.in_fmt}, out_fmt={self.out_fmt})"


@PIPELINES.register_module()
class BoxesClip(object):

    def __init__(self, box_range: Tuple[int, int]):
        self.box_range = box_range

    def __call__(self, data: Dict[str, Dict[str, Union[Tensor, np.ndarray]]]) \
            -> Dict[str, Dict[str, Union[Tensor, np.ndarray]]]:
        for phase in data.keys():
            data[phase]['boxes'] = multi_dim_boxes_clip(data[phase]['boxes'], self.box_range)
        return data

    def __repr__(self) -> str:
        return f"self.__class__.__name__(box_range={self.box_range})"


@PIPELINES.register_module()
class BoxesNormalize(object):

    def __init__(self, box_range: Tuple[int, int]):
        self.box_range = box_range

    def __call__(self, data: Dict[str,  Dict[str, Union[Tensor, np.ndarray]]]) \
            -> Dict[str,  Dict[str, Union[Tensor, np.ndarray]]]:
        for phase in data.keys():
            data[phase]['boxes'] = multi_dim_boxes_normalize(data[phase]['boxes'], self.box_range)
        return data

    def __repr__(self) -> str:
        return f"self.__class__.__name__(size={self.box_range})"


@PIPELINES.register_module()
class BoxesDenormalize(object):

    def __init__(self, box_range: Tuple[int, int]):
        self.box_range = box_range

    def __call__(self, data: Dict[str,  Dict[str, Union[Tensor, np.ndarray]]]) \
            -> Dict[str,  Dict[str, Union[Tensor, np.ndarray]]]:
        for phase in data.keys():
            data[phase]['boxes'] = multi_dim_boxes_denormalize(data[phase]['boxes'], self.box_range)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(box_range={self.box_range})"


@PIPELINES.register_module()
class BoxesRescale(object):

    def __init__(self, src_range: Tuple[int, int], dst_range: Tuple[int, int]):
        self.src_range = src_range
        self.dst_range = dst_range

    def __call__(self, data: Dict[str, Dict[str, Union[Tensor, np.ndarray]]]) \
            -> Dict[str, Dict[str, Union[Tensor, np.ndarray]]]:
        for phase in data.keys():
            data[phase]['boxes'] = multi_dim_boxes_rescale(data[phase]['boxes'], self.src_range, self.dst_range)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(src_range={self.src_range}, dst_range={self.dst_range})"


@PIPELINES.register_module()
class AlignTime(object):

    def __init__(self, align_type: str = 'current'):
        assert align_type in ['current', 'start'], f"align_type '{align_type}' is not support!"
        self.align_type = align_type

    def __call__(self, data: Dict[str, Dict[str, Union[Tensor, np.ndarray]]])\
            -> Dict[str, Dict[str, Union[Tensor, np.ndarray]]]:
        now = data['hist']['timestamps'][-1:] if self.align_type == 'current' else data['hist']['timestamps'][:1]
        for phase in ('futr', 'hist'):
            data[phase]['timestamps'] = data[phase]['timestamps'] - now
            data[phase]['timestamps'] = data[phase]['timestamps'] / 1e6

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(align_type={self.align_type})"

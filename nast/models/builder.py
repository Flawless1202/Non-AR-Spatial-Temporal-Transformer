from typing import Union, Optional, Any, Dict, List

import torch.nn as nn
from mmcv.utils import Registry, Config, build_from_cfg


ENCODERS = Registry("encoders")
DECODERS = Registry("decoders")
PREDICTORS = Registry("predictors")
EMBEDDINGS = Registry("embeddings")
LOSSES = Registry("losses")


def build(cfg: Union[Dict, List[Dict]],
          registry: Registry,
          default_args: Optional[Dict] = None) -> Any:
    """Build a module.

    Args:
        cfg: The config of modules, is is either a dict or a list of configs.
        registry: A registry the module belongs to.
        default_args: Default arguments to build the module. Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_encoder(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build backbone."""
    return build(cfg, ENCODERS)


def build_decoder(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build neck."""
    return build(cfg, DECODERS)


def build_predictor(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build head."""
    return build(cfg, PREDICTORS)


def build_embedding(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build embeddings."""
    return build(cfg, EMBEDDINGS)


def build_loss(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build losses."""
    return build(cfg, LOSSES)

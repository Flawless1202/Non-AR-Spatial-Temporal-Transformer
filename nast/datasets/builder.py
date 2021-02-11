from typing import Optional

from mmcv.utils import Registry, build_from_cfg


DATASETS = Registry('datasets')
PIPELINES = Registry('pipelines')


def build_dataset(cfg: dict, default_args: Optional[dict] = None):
    return build_from_cfg(cfg, DATASETS, default_args)


def build_pipelines(cfg: dict, default_args: Optional[dict] = None):
    return build_from_cfg(cfg, PIPELINES, default_args)

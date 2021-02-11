import multiprocessing
import os
import pickle as pkl
from typing import Tuple
from functools import partial

import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

from .builder import DATASETS
from .pipelines import Compose
from .pipelines.box_transforms import multi_dim_boxes_rescale


@DATASETS.register_module()
class NuScenes(Dataset):

    def __init__(self,
                 root_dir: str,
                 step: int = 3,
                 hist_len: int = 3,
                 futr_len: int = 6,
                 roi_size: Tuple[int, int, int] = (224, 224, 3),
                 image_size: Tuple[int, int] = (1600, 900),
                 resize: Tuple[int, int] = (512, 320),
                 extend_ratio: float = 1.3,
                 force_regenerate: bool = False,
                 use_flow: bool = False,
                 scenes_list_file: str = None,
                 with_origin_image: bool = False,
                 pipelines: dict = None):
        super().__init__()

        self.root_dir = root_dir
        self.step = step
        self.hist_len = hist_len
        self.futr_len = futr_len
        self.roi_size = roi_size
        self.extend_ratio = extend_ratio
        self.image_size = image_size
        self.resize = resize
        self.use_flow = use_flow
        self.with_origin_image = with_origin_image
        self.pipelines = Compose(pipelines) if pipelines is not None else None

        cache_dir = f"__pycache__/{self.__class__.__name__}/step_{step}_sample_len_{hist_len}_{futr_len}_" \
                    f"roi_size_{roi_size[0]}_{roi_size[1]}_{roi_size[2]}_extend_ratio_{extend_ratio}_" \
                    f"{scenes_list_file.split('/')[-1].split('_')[0]}_{'flow' if use_flow else 'img'}_" \
                    f"{'with_origin_img' if with_origin_image else 'without_origin_img'}"

        if not os.path.exists(cache_dir) or force_regenerate:
            self._make_cache(cache_dir, scenes_list_file)

        self.__samples__ = [f"{cache_dir}/{file}" for file in sorted(os.listdir(cache_dir))]

    def __getitem__(self, idx: int) -> dict:
        with open(self.__samples__[idx], 'rb') as f:
            data = pkl.load(f)

        return self.pipelines(data) if self.pipelines is not None else data

    def __len__(self) -> int:
        return len(self.__samples__)

    def _make_cache(self, cache_dir: str, scenes_list_file: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        with open(scenes_list_file, "r") as f:
            scenes_list = f.readlines()
            scenes = [scene.rstrip('\n') for scene in scenes_list]

        with multiprocessing.Pool(20) as p:
            for _ in tqdm(p.imap_unordered(partial(self._process_single_scene, cache_dir=cache_dir), scenes),
                          total=len(scenes)):
                pass

    def _process_single_scene(self, scene_id: str, cache_dir: str) -> None:

        ann_file = f"{self.root_dir}/annotations/{scene_id}_instances_ann.csv"
        if not os.path.exists(ann_file):
            return

        ann = np.loadtxt(ann_file, delimiter=',')

        if len(ann) == 0 or len(ann.shape) != 2:
            return

        sample_len = self.hist_len + self.futr_len
        start_save_idx = 1 if self.use_flow else 0

        sample_idx = 0
        scene_timestamps = ann[:, 0].astype(np.int64)
        unique_timestamps = np.unique(scene_timestamps)
        sample_start_ids = np.arange(start_save_idx, len(unique_timestamps) - sample_len, self.step).tolist()

        for sample_start_id in sample_start_ids:
            sample_timestamps = unique_timestamps[sample_start_id:sample_start_id + sample_len]
            sample_mask = np.any(scene_timestamps.reshape(-1, 1) == sample_timestamps.reshape(1, -1), axis=1)
            sample_object_uids = np.unique(ann[sample_mask, 1].astype(np.int))

            sample_object_box = np.zeros((sample_len, len(sample_object_uids), 4), dtype=np.float32)
            sample_object_mask = np.zeros((sample_len, len(sample_object_uids)), dtype=np.bool)

            for time_idx, timestamp in enumerate(sample_timestamps):

                for object_idx, object_uid in enumerate(sample_object_uids):
                    matched_ann = ann[np.logical_and(ann[:, 0] == timestamp, ann[:, 1] == object_uid)]
                    if 0 == len(matched_ann):
                        continue
                    box = multi_dim_boxes_rescale(matched_ann[0, -4:], src_range=self.image_size, dst_range=self.resize)
                    sample_object_box[time_idx, object_idx] = box
                    sample_object_mask[time_idx, object_idx] = True

            sample_timestamps = np.expand_dims(sample_timestamps, axis=-1)
            sample_object_mask = np.expand_dims(sample_object_mask, axis=-1)

            data = dict(hist=dict(timestamps=sample_timestamps[:self.hist_len],
                                  boxes=sample_object_box[:self.hist_len],
                                  masks=sample_object_mask[:self.hist_len]),
                        futr=dict(timestamps=sample_timestamps[self.hist_len - 1:],
                                  boxes=sample_object_box[self.hist_len - 1:],
                                  masks=sample_object_mask[self.hist_len - 1:]))

            # refine the future masks
            data['futr']['masks'] = np.repeat(
                np.expand_dims(data['hist']['masks'].sum(axis=0) == data['hist']['masks'].shape[0], axis=0),
                data['futr']['masks'].shape[0], axis=0) & data['futr']['masks']

            with open(f"{cache_dir}/{scene_id}_{sample_idx:04d}.pkl", "wb") as f:
                pkl.dump(data, f)

            sample_idx += 1

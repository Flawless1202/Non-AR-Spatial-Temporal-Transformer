import os
import argparse
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    depth: float,
                    sample_data_token: str,
                    filename: str) -> OrderedDict:
    """
    Generate one 2D annotation record given various informations on top of the 2D bounding box coordinates.
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data token.
    :param filename:The corresponding image file where the annotation is present.
    :return: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2, depth]
    repro_rec['filename'] = filename

    return repro_rec


def get_2d_boxes(nusc, sample_data_token: str, visibilities: List[str]) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param visibilities: Visibility filter.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]
        depth = np.mean(corners_3d)

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, depth, sample_data_token, sd_rec['filename'])
        repro_recs.append(repro_rec)

    return repro_recs


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the NuScenes dataset.")
    parser.add_argument("dataroot", help="The root dir of NuScenes.")
    parser.add_argument("saveroot", help="The save root dir of NuScenes.")
    parser.add_argument("-v", "--version", type=str,
                        help="The version of dataset. Select from 'v1.0-mini, v1.0-trainval or v1.0-test'")
    parser.add_argument("--verbose", action="store_true", help="Whether use verbose mode.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=args.verbose)

    ann_dir = os.path.join(args.saveroot, "frames_ann")
    os.makedirs(ann_dir, exist_ok=True)

    ego_pose_dir = os.path.join(args.saveroot, "ego_pose")
    os.makedirs(ego_pose_dir, exist_ok=True)

    scene_count = 0

    for scene in nusc.scene:

        instance_ids = dict()

        first_sample_token = scene["first_sample_token"]
        sample = nusc.get("sample", first_sample_token)
        cam_sample = nusc.get("sample_data", sample["data"]["CAM_FRONT"])

        instances_ann = list()
        ego_poses = list()

        scene_dir = os.path.join(args.saveroot, "frames", scene["name"])
        os.makedirs(scene_dir, exist_ok=True)

        has_more_frames = True
        while has_more_frames:

            frame_path = nusc.get_sample_data_path(cam_sample["token"])
            # img = cv2.imread(frame_path)

            ego_pose = nusc.get("ego_pose", cam_sample["ego_pose_token"])
            ego_poses.append([ego_pose["timestamp"]] + ego_pose["rotation"] + ego_pose["translation"])

            if cam_sample["is_key_frame"]:
                frame_time_stamp = int(frame_path.split("__")[-1].split(".")[0])
                reproj_bboxes = get_2d_boxes(nusc, cam_sample["token"], visibilities=['3', '4'])

                for reproj_bbox in reproj_bboxes:
                    if "human" in reproj_bbox["category_name"] or "vehicle" in reproj_bbox["category_name"]:

                        instance_token = reproj_bbox["instance_token"]
                        if instance_token not in instance_ids:
                            instance_id = len(instance_ids.keys()) + 1
                            instance_ids[instance_token] = instance_id

                        x1, y1, x2, y2, depth = reproj_bbox["bbox_corners"]
                        instance_id = instance_ids[instance_token]
                        instances_ann.append([frame_time_stamp, instance_id, x1, y1, x2, y2, depth])

            # shutil.copy(frame_path, os.path.join(scene_dir, frame_path.split("/")[-1]))
            if not cam_sample["next"] == "":
                cam_sample = nusc.get("sample_data", cam_sample["next"])
            else:
                has_more_frames = False

        df_ins = pd.DataFrame(instances_ann)
        df_ins.to_csv(os.path.join(ann_dir, "{}_instances_ann.csv".format(scene["name"])),
                      header=False, index=False)

        df_ego_pose = pd.DataFrame(ego_poses)
        df_ego_pose.to_csv(os.path.join(ego_pose_dir, "{}_ego_pose.csv".format(scene["name"])),
                           header=False, index=False)

        scene_count += 1
        print("{} scenes have been successfully processed!".format(scene_count))


if __name__ == '__main__':
    main()

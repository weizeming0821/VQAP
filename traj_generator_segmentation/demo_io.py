# -*- coding: utf-8 -*-
"""子阶段 Demo 保存模块"""

import os
import pickle
import json
import numpy as np
from PIL import Image

from rlbench.backend.const import (
    LEFT_SHOULDER_RGB_FOLDER, LEFT_SHOULDER_DEPTH_FOLDER, LEFT_SHOULDER_MASK_FOLDER,
    RIGHT_SHOULDER_RGB_FOLDER, RIGHT_SHOULDER_DEPTH_FOLDER, RIGHT_SHOULDER_MASK_FOLDER,
    OVERHEAD_RGB_FOLDER, OVERHEAD_DEPTH_FOLDER, OVERHEAD_MASK_FOLDER,
    WRIST_RGB_FOLDER, WRIST_DEPTH_FOLDER, WRIST_MASK_FOLDER,
    FRONT_RGB_FOLDER, FRONT_DEPTH_FOLDER, FRONT_MASK_FOLDER,
    LOW_DIM_PICKLE, IMAGE_FORMAT, DEPTH_SCALE,
)
from rlbench.backend import utils
from .config import GRIPPER_OPEN_THR
from .keyframe import extract_keyframes


def check_and_make(d):
    if not os.path.exists(d):
        os.makedirs(d)


def _clear_obs_images(obs):
    """清空观测中的图像数据以减小pickle体积"""
    obs.left_shoulder_rgb = None
    obs.left_shoulder_depth = None
    obs.left_shoulder_point_cloud = None
    obs.left_shoulder_mask = None
    obs.right_shoulder_rgb = None
    obs.right_shoulder_depth = None
    obs.right_shoulder_point_cloud = None
    obs.right_shoulder_mask = None
    obs.overhead_rgb = None
    obs.overhead_depth = None
    obs.overhead_point_cloud = None
    obs.overhead_mask = None
    obs.wrist_rgb = None
    obs.wrist_depth = None
    obs.wrist_point_cloud = None
    obs.wrist_mask = None
    obs.front_rgb = None
    obs.front_depth = None
    obs.front_point_cloud = None
    obs.front_mask = None


def _save_camera_images(obs, out_path, frame_idx):
    """保存单个观测的所有相机图像"""
    if obs.left_shoulder_rgb is not None:
        Image.fromarray(obs.left_shoulder_rgb).save(
            os.path.join(out_path, LEFT_SHOULDER_RGB_FOLDER, IMAGE_FORMAT % frame_idx))
    if obs.left_shoulder_depth is not None:
        utils.float_array_to_rgb_image(obs.left_shoulder_depth, scale_factor=DEPTH_SCALE).save(
            os.path.join(out_path, LEFT_SHOULDER_DEPTH_FOLDER, IMAGE_FORMAT % frame_idx))
    if obs.left_shoulder_mask is not None:
        Image.fromarray((obs.left_shoulder_mask * 255).astype(np.uint8)).save(
            os.path.join(out_path, LEFT_SHOULDER_MASK_FOLDER, IMAGE_FORMAT % frame_idx))

    if obs.right_shoulder_rgb is not None:
        Image.fromarray(obs.right_shoulder_rgb).save(
            os.path.join(out_path, RIGHT_SHOULDER_RGB_FOLDER, IMAGE_FORMAT % frame_idx))
    if obs.right_shoulder_depth is not None:
        utils.float_array_to_rgb_image(obs.right_shoulder_depth, scale_factor=DEPTH_SCALE).save(
            os.path.join(out_path, RIGHT_SHOULDER_DEPTH_FOLDER, IMAGE_FORMAT % frame_idx))
    if obs.right_shoulder_mask is not None:
        Image.fromarray((obs.right_shoulder_mask * 255).astype(np.uint8)).save(
            os.path.join(out_path, RIGHT_SHOULDER_MASK_FOLDER, IMAGE_FORMAT % frame_idx))

    if obs.overhead_rgb is not None:
        Image.fromarray(obs.overhead_rgb).save(
            os.path.join(out_path, OVERHEAD_RGB_FOLDER, IMAGE_FORMAT % frame_idx))
    if obs.overhead_depth is not None:
        utils.float_array_to_rgb_image(obs.overhead_depth, scale_factor=DEPTH_SCALE).save(
            os.path.join(out_path, OVERHEAD_DEPTH_FOLDER, IMAGE_FORMAT % frame_idx))
    if obs.overhead_mask is not None:
        Image.fromarray((obs.overhead_mask * 255).astype(np.uint8)).save(
            os.path.join(out_path, OVERHEAD_MASK_FOLDER, IMAGE_FORMAT % frame_idx))

    if obs.wrist_rgb is not None:
        Image.fromarray(obs.wrist_rgb).save(
            os.path.join(out_path, WRIST_RGB_FOLDER, IMAGE_FORMAT % frame_idx))
    if obs.wrist_depth is not None:
        utils.float_array_to_rgb_image(obs.wrist_depth, scale_factor=DEPTH_SCALE).save(
            os.path.join(out_path, WRIST_DEPTH_FOLDER, IMAGE_FORMAT % frame_idx))
    if obs.wrist_mask is not None:
        Image.fromarray((obs.wrist_mask * 255).astype(np.uint8)).save(
            os.path.join(out_path, WRIST_MASK_FOLDER, IMAGE_FORMAT % frame_idx))

    if obs.front_rgb is not None:
        Image.fromarray(obs.front_rgb).save(
            os.path.join(out_path, FRONT_RGB_FOLDER, IMAGE_FORMAT % frame_idx))
    if obs.front_depth is not None:
        utils.float_array_to_rgb_image(obs.front_depth, scale_factor=DEPTH_SCALE).save(
            os.path.join(out_path, FRONT_DEPTH_FOLDER, IMAGE_FORMAT % frame_idx))
    if obs.front_mask is not None:
        Image.fromarray((obs.front_mask * 255).astype(np.uint8)).save(
            os.path.join(out_path, FRONT_MASK_FOLDER, IMAGE_FORMAT % frame_idx))


def save_subphase_demo_from_memory(phase_obs_all, out_path, image_obs=None,
                                    image_frame_indices=None):
    """
    将内存中的子阶段观测列表保存到 out_path

    Args:
        phase_obs_all: 该段的所有观测列表（用于保存低维数据）
        out_path: 输出路径
        image_obs: 用于保存图像的观测列表（None表示与phase_obs_all相同）
        image_frame_indices: 图像保存的帧索引（用于图像命名，None表示使用本地索引）
    """
    os.makedirs(out_path, exist_ok=True)

    folders = [
        (LEFT_SHOULDER_RGB_FOLDER, LEFT_SHOULDER_DEPTH_FOLDER, LEFT_SHOULDER_MASK_FOLDER),
        (RIGHT_SHOULDER_RGB_FOLDER, RIGHT_SHOULDER_DEPTH_FOLDER, RIGHT_SHOULDER_MASK_FOLDER),
        (OVERHEAD_RGB_FOLDER, OVERHEAD_DEPTH_FOLDER, OVERHEAD_MASK_FOLDER),
        (WRIST_RGB_FOLDER, WRIST_DEPTH_FOLDER, WRIST_MASK_FOLDER),
        (FRONT_RGB_FOLDER, FRONT_DEPTH_FOLDER, FRONT_MASK_FOLDER),
    ]
    for rgb_dir, depth_dir, mask_dir in folders:
        check_and_make(os.path.join(out_path, rgb_dir))
        check_and_make(os.path.join(out_path, depth_dir))
        check_and_make(os.path.join(out_path, mask_dir))

    # 确定用于保存图像的观测列表
    obs_for_images = image_obs if image_obs is not None else phase_obs_all

    # 保存图像
    for local_idx, obs in enumerate(obs_for_images):
        if obs is None:
            continue
        frame_idx = image_frame_indices[local_idx] if image_frame_indices else local_idx
        _save_camera_images(obs, out_path, frame_idx)

    # 清空低维数据中的图像（phase_obs_all 中的对象会被修改）
    for obs in phase_obs_all:
        if obs is not None:
            _clear_obs_images(obs)

    # 保存低维数据（整个段的所有帧）
    with open(os.path.join(out_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(phase_obs_all, f)


def save_phase_metadata(out_ep_path, demo, phase_info, keyframe_inds,
                        debug_info, save_mode='full'):
    """保存 episode 层级的元数据"""
    ep_meta = {
        'total_frames': len(demo),
        'save_mode': save_mode,
        'num_phases': len(phase_info),
        'keyframe_inds': [int(p['keyframe_index']) for p in phase_info],
        'auto_thresholds': {k: float(v) for k, v in debug_info.get('thresholds', {}).items()},
        'segmentation_trace': debug_info.get('trace', {}),
        'phases': phase_info,
    }
    with open(os.path.join(out_ep_path, 'phase_metadata.json'), 'w') as f:
        json.dump(ep_meta, f, ensure_ascii=False, indent=2)


def process_demo_in_memory(demo, out_ep_path, descriptions,
                           signals=None, min_phase_len=5,
                           save_mode='full', fixed_phase_num=None):
    """对内存中的 demo 执行关键帧分割并保存结果"""
    keyframe_inds, debug_info, num_phases = extract_keyframes(
        demo, signals=signals, min_phase_len=min_phase_len)

    valid = (fixed_phase_num is None) or (num_phases == fixed_phase_num)
    if not valid:
        return [], num_phases, False

    kept_segments = debug_info.get('trace', {}).get('stage3_kept_segments', None)
    if kept_segments:
        phase_ranges = [(int(seg['start']), int(seg['end']), int(seg['keyframe']))
                        for seg in kept_segments]
    else:
        n = len(demo)
        boundaries = [0]
        for kf in keyframe_inds:
            if kf + 1 < n:
                boundaries.append(kf + 1)
        if boundaries[-1] != n:
            boundaries.append(n)
        phase_ranges = [(boundaries[p], boundaries[p + 1], keyframe_inds[p])
                        for p in range(len(keyframe_inds))]

    os.makedirs(out_ep_path, exist_ok=True)
    phase_info = []

    for p, (start, end, kf_idx) in enumerate(phase_ranges):
        # 获取该段的所有观测（用于保存低维数据）
        all_phase_obs = demo[start:end]
        all_frame_indices = list(range(start, end))

        if save_mode == 'keyframe_only':
            # 图像只保存关键帧（节省磁盘空间）
            image_frame_indices = [start, kf_idx] if start != kf_idx else [kf_idx]
            image_obs = [demo[i] for i in image_frame_indices]
        else:
            # full 模式：图像也保存所有帧
            image_obs = all_phase_obs
            image_frame_indices = all_frame_indices

        phase_out = os.path.join(out_ep_path, f'phase_{p}')
        save_subphase_demo_from_memory(
            phase_obs_all=all_phase_obs,
            out_path=phase_out,
            image_obs=image_obs,
            image_frame_indices=image_frame_indices
        )

        gripper_states = [float(o.gripper_open) > GRIPPER_OPEN_THR
                          for o in all_phase_obs if o is not None]
        trigger = debug_info.get('frame_signals', {}).get(kf_idx, [])
        is_interacting = debug_info.get('seg_interacting', {}).get(kf_idx, None)
        phase_info.append({
            'phase_index': p,
            'keyframe_index': kf_idx,
            'start_frame': start,
            'end_frame': end,
            'length': end - start,
            'saved_image_frames': len(image_frame_indices),
            'saved_low_dim_frames': len(all_phase_obs),
            'gripper_open_ratio': float(np.mean(gripper_states)) if gripper_states else 0.0,
            'trigger_signals': trigger,
            'is_interacting': is_interacting,
        })

    save_phase_metadata(out_ep_path, demo, phase_info, keyframe_inds,
                        debug_info, save_mode)

    return phase_info, num_phases, valid

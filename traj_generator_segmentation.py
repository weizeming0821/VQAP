"""
一体化 Demo 生成与关键帧分割流水线

直接在内存中采集 RLBench 演示轨迹，立即进行关键帧分割，仅将分割后的子
阶段结果写入磁盘，避免先保存完整 demo 再分割带来的巨大磁盘开销。

核心流程（每条 episode）：
  1. task_env.get_demos() → 完整 demo（含所有相机图像，驻留内存）
  2. extract_keyframes(demo)  → 关键帧边界
  3. 按子阶段调用 save_phase_demo() → 仅写入对应帧的图像 + 低维 pickle
  4. 原始完整 demo 由 GC 回收，从不落盘

输出目录结构（与 demos_subphase 保持一致）：
  output_path/
    task_name/
      variationN/
        split_summary.json
        episodes/
          episodeN/
            phase_metadata.json
            phase_0/
              low_dim_obs.pkl
              front_rgb/0.png, 1.png, ...
              wrist_rgb/...
              ...
            phase_1/
              ...

分割算法（三阶段，与 test_traj_segmentation.py 完全一致）：
  阶段 1：逐信号候选帧提取（自动估计各信号阈值）
  阶段 2：信号优先级筛选
  阶段 3：距离合并

用法：
  python traj_generator_segmentation.py --tasks open_drawer --episodes_per_task 5
  python traj_generator_segmentation.py --signals gripper vel --min_phase_len 8
  python traj_generator_segmentation.py --save_mode full --score_threshold 15
  python traj_generator_segmentation.py --tasks open_drawer remove_cups \\
      --episodes_per_task 10 --variations 3 --signal_weights gripper=10 vel=8
"""

import argparse
import copy
import json
import os
import pickle
from datetime import datetime
from multiprocessing import Manager, Process

import numpy as np
from PIL import Image
from pyrep.const import RenderMode

import rlbench.backend.task as task
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils
from rlbench.backend.const import (
    DEPTH_SCALE, EPISODE_FOLDER, EPISODES_FOLDER,
    FRONT_DEPTH_FOLDER, FRONT_MASK_FOLDER, FRONT_RGB_FOLDER,
    IMAGE_FORMAT,
    LEFT_SHOULDER_DEPTH_FOLDER, LEFT_SHOULDER_MASK_FOLDER, LEFT_SHOULDER_RGB_FOLDER,
    LOW_DIM_PICKLE,
    OVERHEAD_DEPTH_FOLDER, OVERHEAD_MASK_FOLDER, OVERHEAD_RGB_FOLDER,
    RIGHT_SHOULDER_DEPTH_FOLDER, RIGHT_SHOULDER_MASK_FOLDER, RIGHT_SHOULDER_RGB_FOLDER,
    VARIATIONS_FOLDER,
    WRIST_DEPTH_FOLDER, WRIST_MASK_FOLDER, WRIST_RGB_FOLDER,
)
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment

# 从 test_traj_segmentation 导入分割逻辑（不修改该文件）
from test_traj_segmentation import (
    ALL_SIGNALS, DEFAULT_WEIGHTS, SIGNAL_DESC,
    auto_thresholds, extract_keyframes,
)


# ─────────────────────────────────────────────────────────────────────────────
# 相机图像配置
# ─────────────────────────────────────────────────────────────────────────────

# (obs 属性名, 保存子目录, 数据类型 in {rgb, depth, mask})
_CAMERAS = [
    ('left_shoulder_rgb',    LEFT_SHOULDER_RGB_FOLDER,    'rgb'),
    ('left_shoulder_depth',  LEFT_SHOULDER_DEPTH_FOLDER,  'depth'),
    ('left_shoulder_mask',   LEFT_SHOULDER_MASK_FOLDER,   'mask'),
    ('right_shoulder_rgb',   RIGHT_SHOULDER_RGB_FOLDER,   'rgb'),
    ('right_shoulder_depth', RIGHT_SHOULDER_DEPTH_FOLDER, 'depth'),
    ('right_shoulder_mask',  RIGHT_SHOULDER_MASK_FOLDER,  'mask'),
    ('overhead_rgb',         OVERHEAD_RGB_FOLDER,         'rgb'),
    ('overhead_depth',       OVERHEAD_DEPTH_FOLDER,       'depth'),
    ('overhead_mask',        OVERHEAD_MASK_FOLDER,        'mask'),
    ('wrist_rgb',            WRIST_RGB_FOLDER,            'rgb'),
    ('wrist_depth',          WRIST_DEPTH_FOLDER,          'depth'),
    ('wrist_mask',           WRIST_MASK_FOLDER,           'mask'),
    ('front_rgb',            FRONT_RGB_FOLDER,            'rgb'),
    ('front_depth',          FRONT_DEPTH_FOLDER,          'depth'),
    ('front_mask',           FRONT_MASK_FOLDER,           'mask'),
]

# pickle 时需置 None 的图像属性（含 point_cloud）
_IMAGE_ATTRS = [
    'left_shoulder_rgb', 'left_shoulder_depth',
    'left_shoulder_point_cloud', 'left_shoulder_mask',
    'right_shoulder_rgb', 'right_shoulder_depth',
    'right_shoulder_point_cloud', 'right_shoulder_mask',
    'overhead_rgb', 'overhead_depth',
    'overhead_point_cloud', 'overhead_mask',
    'wrist_rgb', 'wrist_depth',
    'wrist_point_cloud', 'wrist_mask',
    'front_rgb', 'front_depth',
    'front_point_cloud', 'front_mask',
]

# 快速查找 attr -> 数据类型
_ATTR_TYPE = {attr: dtype for attr, _, dtype in _CAMERAS}


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def check_and_make(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_obs_config_dict(obs_config):
    """将 ObservationConfig 转换为可序列化的字典（用于 dataset_metadata）。"""
    cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']
    result = {}
    for cam in cameras:
        cam_cfg = getattr(obs_config, f'{cam}_camera')
        result[cam] = {
            'rgb':        cam_cfg.rgb,
            'depth':      cam_cfg.depth,
            'mask':       cam_cfg.mask,
            'image_size': list(cam_cfg.image_size),
            'render_mode': str(cam_cfg.render_mode),
        }
    result['joint_velocities']       = obs_config.joint_velocities
    result['joint_positions']        = obs_config.joint_positions
    result['joint_forces']           = obs_config.joint_forces
    result['gripper_open']           = obs_config.gripper_open
    result['gripper_pose']           = obs_config.gripper_pose
    result['gripper_joint_positions'] = obs_config.gripper_joint_positions
    result['gripper_touch_forces']   = obs_config.gripper_touch_forces
    result['task_low_dim_state']     = obs_config.task_low_dim_state
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 子阶段 demo 保存（从内存中的完整 demo 保存指定帧范围）
# ─────────────────────────────────────────────────────────────────────────────

def _save_obs_image(obs, attr, img_dir, orig_idx):
    """保存单帧单相机图像，根据数据类型自动选择编码方式。"""
    arr = getattr(obs, attr, None)
    if arr is None:
        return
    dtype = _ATTR_TYPE.get(attr, 'rgb')
    if dtype == 'depth':
        # float_array_to_rgb_image 直接返回 PIL Image
        img = utils.float_array_to_rgb_image(arr, scale_factor=DEPTH_SCALE)
    elif dtype == 'mask':
        img = Image.fromarray((arr * 255).astype(np.uint8))
    else:
        img = Image.fromarray(arr)
    img.save(os.path.join(img_dir, IMAGE_FORMAT % orig_idx))


def save_phase_demo(demo, out_path, frame_indices):
    """
    将完整 demo（含图像数据）中指定帧保存到 out_path：
      1. 按相机/类型分目录保存图像（文件名保留原始帧编号）
      2. 保存低维观测 pickle（图像字段置 None 以节省空间）

    Args:
        demo:          完整 demo（List[Observation]），图像数据仍在内存中
        out_path:      子阶段输出目录
        frame_indices: 该子阶段覆盖的原始帧索引列表
    """
    os.makedirs(out_path, exist_ok=True)

    # ── 1. 创建图像目录并保存图像 ────────────────────────────────────────
    img_dirs = {}
    for attr, folder, _ in _CAMERAS:
        dir_path = os.path.join(out_path, folder)
        os.makedirs(dir_path, exist_ok=True)
        img_dirs[attr] = dir_path

    for orig_idx in frame_indices:
        obs = demo[orig_idx]
        if obs is None:
            continue
        for attr, _, _ in _CAMERAS:
            _save_obs_image(obs, attr, img_dirs[attr], orig_idx)

    # ── 2. 保存低维 pickle（图像属性置 None）───────────────────────────
    phase_obs = []
    for orig_idx in frame_indices:
        obs = demo[orig_idx]
        if obs is None:
            phase_obs.append(None)
            continue
        obs_copy = copy.copy(obs)       # 浅拷贝后覆盖图像属性，不影响原始 demo
        for attr in _IMAGE_ATTRS:
            try:
                setattr(obs_copy, attr, None)
            except (AttributeError, TypeError):
                pass
        phase_obs.append(obs_copy)

    with open(os.path.join(out_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(phase_obs, f)


# ─────────────────────────────────────────────────────────────────────────────
# 内存中 demo 的分割与保存
# ─────────────────────────────────────────────────────────────────────────────

def process_demo_in_memory(demo, out_ep_path, descriptions,
                            signals=None, weights=None,
                            score_threshold=None, min_phase_len=5,
                            merge_sim=0.95, save_mode='keyframe_only'):
    """
    对内存中的完整 demo（含图像）执行三阶段关键帧分割，直接保存各子阶段数据。

    Args:
        demo:         List[Observation]，图像数据仍在内存中
        out_ep_path:  该 episode 的输出目录
        descriptions: 变体语言描述列表
        signals:      启用的信号集合
        weights:      信号权重字典
        score_threshold: 候选帧筛选阈值
        min_phase_len:   相邻帧最小距离（距离合并）
        merge_sim:       语义合并余弦相似度阈值
        save_mode:    'full' | 'keyframe_only'

    Returns:
        phase_info: List[dict]
    """
    keyframe_inds, debug_info = extract_keyframes(
        demo,
        signals=signals,
        weights=weights,
        score_threshold=score_threshold,
        min_phase_len=min_phase_len,
        merge_sim=merge_sim,
    )
    num_phases = len(keyframe_inds)

    # 计算段边界 [start0, start1, ..., end_of_demo]
    boundaries = [0]
    for kf in keyframe_inds:
        next_start = kf + 1
        if next_start < len(demo):
            boundaries.append(next_start)
    if boundaries[-1] != len(demo):
        boundaries.append(len(demo))

    os.makedirs(out_ep_path, exist_ok=True)

    phase_info = []
    for p, kf_idx in enumerate(keyframe_inds):
        start = boundaries[p]
        if save_mode == 'keyframe_only':
            frame_indices = [start, kf_idx] if start != kf_idx else [kf_idx]
        else:
            end = boundaries[p + 1]
            frame_indices = list(range(start, end))

        phase_out = os.path.join(out_ep_path, f'phase_{p}')
        save_phase_demo(demo, phase_out, frame_indices)

        phase_obs = [demo[i] for i in frame_indices]
        gripper_states = [float(o.gripper_open) > 0.5
                          for o in phase_obs if o is not None]
        trigger = debug_info.get('frame_signals', {}).get(kf_idx, [])
        phase_info.append({
            'phase_index':        p,
            'keyframe_index':     kf_idx,
            'start_frame':        boundaries[p],
            'end_frame':          boundaries[p + 1],
            'length':             boundaries[p + 1] - boundaries[p],
            'saved_frames':       len(frame_indices),
            'gripper_open_ratio': float(np.mean(gripper_states)) if gripper_states else 0.0,
            'trigger_signals':    trigger,
        })

    ep_meta = {
        'total_frames':      len(demo),
        'save_mode':         save_mode,
        'num_phases':        num_phases,
        'keyframe_inds':     keyframe_inds,
        'boundaries':        boundaries,
        'task_descriptions': descriptions,
        'auto_thresholds':   {k: float(v) for k, v in
                              debug_info.get('thresholds', {}).items()},
        'score_threshold':   debug_info.get('score_threshold', None),
        'phases':            phase_info,
    }
    with open(os.path.join(out_ep_path, 'phase_metadata.json'), 'w') as f:
        json.dump(ep_meta, f, ensure_ascii=False, indent=2)

    return phase_info


# ─────────────────────────────────────────────────────────────────────────────
# 元数据保存
# ─────────────────────────────────────────────────────────────────────────────

def save_variation_metadata(variation_path, variation_index, descriptions,
                             episode_lengths, var_summary, seg_params):
    """保存 variation 层级元数据（含分割汇总）。"""
    num_episodes = len(episode_lengths)
    num_phases_list = [ep.get('num_phases', 0) for ep in var_summary]

    # variation_metadata.json
    metadata = {
        'variation_index':        variation_index,
        'num_episodes':           num_episodes,
        'episode_lengths':        [int(l) for l in episode_lengths],
        'avg_episode_length':     float(np.mean(episode_lengths)) if episode_lengths else 0,
        'min_episode_length':     int(min(episode_lengths))       if episode_lengths else 0,
        'max_episode_length':     int(max(episode_lengths))       if episode_lengths else 0,
        'avg_phases_per_episode': float(np.mean(num_phases_list)) if num_phases_list else 0,
    }
    with open(os.path.join(variation_path, 'variation_metadata.json'), 'w',
              encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # split_summary.json（与 test_traj_segmentation.py 格式一致）
    task_name = os.path.basename(os.path.dirname(variation_path))
    var_dir   = os.path.basename(variation_path)
    with open(os.path.join(variation_path, 'split_summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump({
            'task':         task_name,
            'variation':    var_dir,
            'descriptions': descriptions,
            'save_mode':    seg_params.get('save_mode', 'keyframe_only'),
            'signals':      seg_params.get('signals') or list(ALL_SIGNALS),
            'weights':      seg_params.get('weights') or DEFAULT_WEIGHTS,
            'episodes':     var_summary,
        }, f, ensure_ascii=False, indent=2)


def save_task_metadata(task_path, task_name):
    """保存 task 层级元数据（扫描 variation 子目录汇总）。"""
    variation_folders = sorted([
        d for d in os.listdir(task_path)
        if d.startswith('variation') and
        os.path.isdir(os.path.join(task_path, d))
    ])

    total_episodes = 0
    all_lengths    = []

    for var_folder in variation_folders:
        var_meta_path = os.path.join(task_path, var_folder, 'variation_metadata.json')
        if os.path.exists(var_meta_path):
            with open(var_meta_path, 'r') as f:
                var_meta = json.load(f)
            lengths = var_meta.get('episode_lengths')
            if lengths:
                all_lengths.extend(lengths)
            else:
                ep_count = var_meta.get('num_episodes', 0)
                avg_len  = var_meta.get('avg_episode_length', 0)
                if ep_count > 0:
                    all_lengths.extend([avg_len] * ep_count)
            total_episodes += var_meta.get('num_episodes', 0)

    task_class = ''.join(w.title() for w in task_name.split('_'))
    metadata = {
        'task_name':          task_name,
        'task_class':         task_class,
        'num_variations':     len(variation_folders),
        'total_episodes':     total_episodes,
        'total_steps':        int(sum(all_lengths)),
        'avg_episode_length': float(np.mean(all_lengths)) if all_lengths else 0,
    }
    with open(os.path.join(task_path, 'task_metadata.json'), 'w',
              encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_dataset_metadata(output_path, args, obs_config):
    """保存数据集层级元数据。"""
    task_folders = sorted([
        d for d in os.listdir(output_path)
        if os.path.isdir(os.path.join(output_path, d))
    ])
    metadata = {
        'created_at':            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pipeline':              'traj_generator_segmentation',
        'action_mode':           'MoveArmThenGripper',
        'arm_action_mode':       'JointVelocity',
        'gripper_action_mode':   'Discrete',
        'renderer':              args.renderer,
        'image_size':            list(args.image_size),
        'arm_max_velocity':      args.arm_max_velocity,
        'arm_max_acceleration':  args.arm_max_acceleration,
        'episodes_per_task':     args.episodes_per_task,
        'max_variations':        args.variations,
        'save_mode':             args.save_mode,
        'signals':               args.signals or list(ALL_SIGNALS),
        'score_threshold':       args.score_threshold,
        'min_phase_len':         args.min_phase_len,
        'merge_sim':             args.merge_sim,
        'observation_config':    get_obs_config_dict(obs_config),
        'total_tasks':           len(task_folders),
        'task_list':             task_folders,
    }
    path = os.path.join(output_path, 'dataset_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f'Dataset metadata saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 采集进程（多进程并行）
# ─────────────────────────────────────────────────────────────────────────────

def run(i, lock, task_index, variation_count, results, file_lock, tasks, args):
    """
    每个进程独立采集一个 variation 下的所有 episode，
    采集后立即在内存中分割并仅保存子阶段数据。
    """
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, args.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size  = img_size
    obs_config.overhead_camera.image_size        = img_size
    obs_config.wrist_camera.image_size           = img_size
    obs_config.front_camera.image_size           = img_size

    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters  = False
    obs_config.overhead_camera.depth_in_meters        = False
    obs_config.wrist_camera.depth_in_meters           = False
    obs_config.front_camera.depth_in_meters           = False

    obs_config.left_shoulder_camera.masks_as_one_channel  = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel        = False
    obs_config.wrist_camera.masks_as_one_channel           = False
    obs_config.front_camera.masks_as_one_channel           = False

    if args.renderer == 'opengl':
        for cam in [obs_config.right_shoulder_camera, obs_config.left_shoulder_camera,
                    obs_config.overhead_camera, obs_config.wrist_camera,
                    obs_config.front_camera]:
            cam.render_mode = RenderMode.OPENGL
    elif args.renderer == 'opengl3':
        for cam in [obs_config.right_shoulder_camera, obs_config.left_shoulder_camera,
                    obs_config.overhead_camera, obs_config.wrist_camera,
                    obs_config.front_camera]:
            cam.render_mode = RenderMode.OPENGL3

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        headless=True,
    )
    rlbench_env.launch()

    # 分割参数打包，方便传入 process_demo_in_memory
    seg_params = {
        'signals':         args.signals,
        'weights':         args.seg_weights,
        'score_threshold': args.score_threshold,
        'min_phase_len':   args.min_phase_len,
        'merge_sim':       args.merge_sim,
        'save_mode':       args.save_mode,
    }

    tasks_with_problems = results[i] = ''

    while True:
        with lock:
            if task_index.value >= num_tasks:
                print(f'Process {i} finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if args.variations >= 0:
                var_target = min(args.variations, var_target)
            if my_variation_count >= var_target:
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print(f'Process {i} finished')
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)
        descriptions, _ = task_env.reset()

        variation_path = os.path.join(
            args.output_path,
            task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count,
        )
        check_and_make(variation_path)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        episode_lengths = []
        var_summary     = []

        for ex_idx in range(args.episodes_per_task):
            print(f'Process {i} // Task: {task_env.get_name()} '
                  f'// Variation: {my_variation_count} // Demo: {ex_idx}')
            attempts = 10
            while attempts > 0:
                try:
                    demo, = task_env.get_demos(amount=1, live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        f'Process {i} failed collecting task {task_env.get_name()} '
                        f'(variation: {my_variation_count}, example: {ex_idx}). '
                        f'Skipping.\n{e}\n'
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break

                ep_out_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)

                # ── 核心：在内存中分割并仅写入子阶段数据 ──────────────────
                with file_lock:
                    phase_info = process_demo_in_memory(
                        demo, ep_out_path, descriptions,
                        **seg_params,
                    )

                episode_lengths.append(len(demo))

                if phase_info:
                    phases_str = ' | '.join(
                        f"phase_{p['phase_index']}"
                        f"(kf={p['keyframe_index']}, "
                        f"len={p['length']}f, "
                        f"sig={'+'.join(p['trigger_signals']) or '-'}, "
                        f"gripper={'open' if p['gripper_open_ratio'] > 0.5 else 'closed'})"
                        for p in phase_info
                    )
                    print(f'  {VARIATIONS_FOLDER % my_variation_count}/'
                          f'{EPISODE_FOLDER % ex_idx}: {phases_str}')

                var_summary.append({
                    'episode':    EPISODE_FOLDER % ex_idx,
                    'num_phases': len(phase_info),
                    'phases':     phase_info,
                })
                break

            if abort_variation:
                break

        # 保存 variation 层级元数据
        save_variation_metadata(
            variation_path, my_variation_count, descriptions,
            episode_lengths, var_summary, seg_params,
        )

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='RLBench 一体化 Demo 生成与关键帧分割流水线',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── 生成参数（与 action_traj_generator.py 一致）──────────────────────
    parser.add_argument('--output_path', type=str, default='./demos_subphase',
                        help='分割结果的保存路径（默认: ./demos_subphase）')
    parser.add_argument('--tasks', nargs='*', default=[],
                        help='要采集的任务列表，不填则采集所有任务')
    parser.add_argument('--image_size', nargs=2, type=int, default=[128, 128],
                        help='保存图像的分辨率')
    parser.add_argument('--renderer', type=str,
                        choices=['opengl', 'opengl3'], default='opengl3',
                        help='渲染器类型，opengl 无阴影但速度更快')
    parser.add_argument('--processes', type=int, default=4,
                        help='并行采集的进程数')
    parser.add_argument('--episodes_per_task', type=int, default=2,
                        help='每个任务变体采集的 episode 数量')
    parser.add_argument('--variations', type=int, default=5,
                        help='每个任务采集的变体数量上限，-1 表示全部 (默认: 5)')
    parser.add_argument('--arm_max_velocity', type=float, default=1.0,
                        help='运动规划使用的最大手臂速度')
    parser.add_argument('--arm_max_acceleration', type=float, default=4.0,
                        help='运动规划使用的最大手臂加速度')

    # ── 分割参数（与 test_traj_segmentation.py 一致）──────────────────────
    parser.add_argument('--save_mode', default='keyframe_only',
                        choices=['full', 'keyframe_only'],
                        help='子阶段保存模式 (默认: keyframe_only)\n'
                             '  full          保留每段完整轨迹\n'
                             '  keyframe_only 仅保留起始帧和关键帧')
    parser.add_argument(
        '--signals', nargs='+',
        choices=list(ALL_SIGNALS), default=None, metavar='SIG',
        help=('启用的信号（空格分隔），默认全部启用。\n可选: '
              + ' | '.join(f'{s}({SIGNAL_DESC[s]})' for s in ALL_SIGNALS)),
    )
    parser.add_argument(
        '--signal_weights', nargs='+', default=None, metavar='SIG=W',
        help=('自定义信号权重，格式：gripper=10 vel=8 ...\n'
              f'默认: {" ".join(f"{k}={v}" for k, v in DEFAULT_WEIGHTS.items())}'),
    )
    parser.add_argument('--score_threshold', type=float, default=None,
                        help=('候选帧保留的最低分数，None = 最高优先级信号的权重\n'
                              '提高此值 → 要求更多信号同时触发 → 段数减少'))
    parser.add_argument('--min_phase_len', type=int, default=5,
                        help='相邻关键帧最小帧距（距离合并阈值，默认: 5）')
    parser.add_argument('--merge_sim', type=float, default=0.95,
                        help=('语义余弦相似度合并阈值（默认: 0.95）\n'
                              '降低 → 合并更激进\n升高 → 保留更多细粒度分割'))

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 解析自定义信号权重
    seg_weights = None
    if args.signal_weights:
        seg_weights = dict(DEFAULT_WEIGHTS)
        for item in args.signal_weights:
            if '=' in item:
                k, v = item.split('=', 1)
                k = k.strip()
                if k in DEFAULT_WEIGHTS:
                    seg_weights[k] = float(v)
    # 将解析后的权重挂到 args 上，方便传入子进程
    args.seg_weights = seg_weights

    task_files = [
        t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
        if t != '__init__.py' and t.endswith('.py')
    ]
    if len(args.tasks) > 0:
        for t in args.tasks:
            if t not in task_files:
                raise ValueError(f'Task {t} not recognised!')
        task_files = args.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager   = Manager()
    result_dict   = manager.dict()
    file_lock     = manager.Lock()
    task_index    = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock          = manager.Lock()

    check_and_make(args.output_path)

    processes = [
        Process(target=run, args=(
            i, lock, task_index, variation_count,
            result_dict, file_lock, tasks, args,
        ))
        for i in range(args.processes)
    ]
    [p.start() for p in processes]
    [p.join()  for p in processes]

    print('Data collection and segmentation done!')
    for i in range(args.processes):
        print(result_dict[i])

    # 保存 task 层级元数据
    for task_name in task_files:
        task_path = os.path.join(args.output_path, task_name)
        if os.path.exists(task_path):
            save_task_metadata(task_path, task_name)

    # 保存数据集层级元数据
    img_size = list(map(int, args.image_size))
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    for cam in [obs_config.right_shoulder_camera, obs_config.left_shoulder_camera,
                obs_config.overhead_camera, obs_config.wrist_camera,
                obs_config.front_camera]:
        cam.image_size = img_size
    save_dataset_metadata(args.output_path, args, obs_config)


if __name__ == '__main__':
    main()

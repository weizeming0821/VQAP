import argparse
import json
import os
import pickle
import signal
import time
import traceback
from datetime import datetime
from multiprocessing import Manager, Process

import numpy as np
from PIL import Image
from pyrep.const import RenderMode
from tqdm import tqdm

import rlbench.backend.task as task
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment


class DemoTimeoutError(Exception):
    """单条演示采集超时时抛出。"""
    pass


def _demo_timeout_handler(signum, frame):
    raise DemoTimeoutError('Demo collection timed out')


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def append_log(log_path, log_lock, level, message):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] [{level}] {message}\n'
    with log_lock:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line)


def update_progress(progress, progress_lock, **deltas):
    with progress_lock:
        for k, v in deltas.items():
            progress[k] = int(progress.get(k, 0)) + int(v)


def estimate_total_episodes(task_files, args):
    """预估总 episode 数（不在主进程启动 RLBench，避免 Qt/DBus 死锁）。"""
    if args.variations >= 0:
        return int(len(task_files) * args.variations * args.episodes_per_task)
    # variations=-1 时未知精确总量，给一个保守下界。
    return int(len(task_files) * args.episodes_per_task)


def summarize_collection(task_files, args, progress, variation_stats):
    planned_tasks = len(task_files)
    stat_values = list(variation_stats.values())

    total_variations = len(stat_values)
    success_variations = [s for s in stat_values if s.get('status') == 'completed']
    failed_variations = [s for s in stat_values if s.get('status') != 'completed']

    planned_demos = sum(int(s.get('planned_demos', 0)) for s in stat_values)
    success_demos = sum(int(s.get('success_demos', 0)) for s in stat_values)
    failed_demos = max(0, planned_demos - success_demos)

    success_tasks = sorted({s.get('task_name') for s in success_variations if s.get('task_name')})
    failed_tasks = sorted({s.get('task_name') for s in failed_variations if s.get('task_name')})

    lines = [
        '===== Collection Summary =====',
        f'Planned tasks: {planned_tasks}',
        f'Success tasks: {len(success_tasks)}',
        f'Failed tasks: {len(failed_tasks)}',
        f'Success variations: {len(success_variations)} / {total_variations}',
        f'Failed variations: {len(failed_variations)} / {total_variations}',
        f'Success demos: {success_demos} / {planned_demos}',
        f'Failed demos: {failed_demos} / {planned_demos}',
    ]

    detail_lines = []
    if failed_variations:
        task_failed_demo = {}
        task_failed_var = {}
        for s in failed_variations:
            tn = s.get('task_name')
            if not tn:
                continue
            task_failed_var[tn] = task_failed_var.get(tn, 0) + 1
            task_failed_demo[tn] = task_failed_demo.get(tn, 0) + int(s.get('failed_demos', 0))

        detail_lines.append('Failed task breakdown:')
        for tn in sorted(task_failed_var.keys()):
            detail_lines.append(
                f'  - task={tn} failed_variations={task_failed_var[tn]} failed_demos={task_failed_demo[tn]}')

    return lines, detail_lines


def get_obs_config_dict(obs_config):
    """将 ObservationConfig 转换为可序列化的字典"""
    cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']
    result = {}
    for cam in cameras:
        cam_cfg = getattr(obs_config, f'{cam}_camera')
        result[cam] = {
            'rgb':   cam_cfg.rgb,
            'depth': cam_cfg.depth,
            'mask':  cam_cfg.mask,
            'image_size': list(cam_cfg.image_size),
            'render_mode': str(cam_cfg.render_mode),
        }
    result['joint_velocities'] = obs_config.joint_velocities
    result['joint_positions']  = obs_config.joint_positions
    result['joint_forces']     = obs_config.joint_forces
    result['gripper_open']     = obs_config.gripper_open
    result['gripper_pose']     = obs_config.gripper_pose
    result['gripper_joint_positions'] = obs_config.gripper_joint_positions
    result['gripper_touch_forces']    = obs_config.gripper_touch_forces
    result['task_low_dim_state']      = obs_config.task_low_dim_state
    return result


def save_variation_metadata(variation_path, variation_index, descriptions,
                            episode_lengths):
    """
    保存 variation 层级元数据

    Args:
        variation_path: variation 文件夹路径
        variation_index: 变体序号
        descriptions: 该变体的语言描述列表
        episode_lengths: 各 episode 的步数列表
    """
    num_episodes = len(episode_lengths)
    metadata = {
        'variation_index':        variation_index,
        'variation_descriptions': descriptions,
        'num_episodes':           num_episodes,
        # episode_lengths 记录每条 episode 的实际步数，供上层汇总时精确累加
        'episode_lengths':        [int(l) for l in episode_lengths],
        'avg_episode_length':     float(np.mean(episode_lengths)) if episode_lengths else 0,
        'min_episode_length':     int(min(episode_lengths))       if episode_lengths else 0,
        'max_episode_length':     int(max(episode_lengths))       if episode_lengths else 0,
    }
    path = os.path.join(variation_path, 'variation_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_task_metadata(task_path, task_name):
    """
    保存 task 层级元数据（扫描 variation 子文件夹汇总）

    Args:
        task_path: task 文件夹路径
        task_name: 任务名称（snake_case）
    """
    variation_folders = sorted([
        d for d in os.listdir(task_path)
        if d.startswith('variation') and os.path.isdir(os.path.join(task_path, d))
    ])

    total_episodes = 0
    all_lengths = []  # 所有 episode 的实际步数（从 variation_metadata 中读取）

    for var_folder in variation_folders:
        var_meta_path = os.path.join(task_path, var_folder, 'variation_metadata.json')
        if os.path.exists(var_meta_path):
            with open(var_meta_path, 'r') as f:
                var_meta = json.load(f)
            # 优先使用精确列表，回退到按 avg_len 近似
            lengths = var_meta.get('episode_lengths')
            if lengths:
                all_lengths.extend(lengths)
            else:
                ep_count = var_meta.get('num_episodes', 0)
                avg_len  = var_meta.get('avg_episode_length', 0)
                if ep_count > 0:
                    all_lengths.extend([avg_len] * ep_count)
            total_episodes += var_meta.get('num_episodes', 0)

    # 将 snake_case 转为 CamelCase
    task_class = ''.join(w.title() for w in task_name.split('_'))

    metadata = {
        'task_name':          task_name,
        'task_class':         task_class,
        'num_variations':     len(variation_folders),
        'total_episodes':     total_episodes,
        'total_steps':        int(sum(all_lengths)),
        'avg_episode_length': float(np.mean(all_lengths)) if all_lengths else 0,
    }
    path = os.path.join(task_path, 'task_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_dataset_metadata(save_path, args, obs_config):
    """
    保存 demos 层级（数据集全局）元数据

    Args:
        save_path: demos 根目录
        args: 命令行参数
        obs_config: 观测配置对象
    """
    task_folders = sorted([
        d for d in os.listdir(save_path)
        if os.path.isdir(os.path.join(save_path, d))
    ])

    # 各任务的汇总由 task_metadata.json 负责，此处只记录全局配置与任务列表
    metadata = {
        'created_at':          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'action_mode':         'MoveArmThenGripper',
        'arm_action_mode':     'JointVelocity',
        'gripper_action_mode': 'Discrete',
        'renderer':            args.renderer,
        'image_size':          list(args.image_size),
        'arm_max_velocity':    args.arm_max_velocity,
        'arm_max_acceleration':args.arm_max_acceleration,
        'episodes_per_task':   args.episodes_per_task,
        'max_variations':      args.variations,
        'observation_config':  get_obs_config_dict(obs_config),
        'total_tasks':         len(task_folders),
        'task_list':           task_folders,
    }
    path = os.path.join(save_path, 'dataset_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f'Dataset metadata saved: {path}')


def save_demo(demo, example_path):

    # 先保存图像数据，然后将图像字段置为 None，最后 pickle 低维数据
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(
        example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(
        example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    check_and_make(left_shoulder_rgb_path)
    check_and_make(left_shoulder_depth_path)
    check_and_make(left_shoulder_mask_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(right_shoulder_depth_path)
    check_and_make(right_shoulder_mask_path)
    check_and_make(overhead_rgb_path)
    check_and_make(overhead_depth_path)
    check_and_make(overhead_mask_path)
    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)
    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8))
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8))
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray(
            (obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(
            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        overhead_rgb.save(
            os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(
            os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(
            os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # 图像已单独保存，将其置为 None 以节省 pickle 体积
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

    # 保存低维状态数据
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def run(i, lock, task_index, variation_count, results, file_lock, tasks, args,
    log_path, log_lock, progress, progress_lock, variation_stats, worker_state):
    """每个进程独立选择一个任务和变体，采集该变体下所有 episode 的演示数据。"""

    # 为每个进程初始化随机种子
    np.random.seed(None)
    num_tasks = len(tasks)

    # 注册超时信号处理器（每个子进程独立设置）
    if args.demo_timeout > 0:
        signal.signal(signal.SIGALRM, _demo_timeout_handler)

    img_size = list(map(int, args.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # 深度值存储为 0~1 的归一化值（非米制单位）
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # 将分割掩码保存为 RGB 编码格式
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if args.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL
    elif args.renderer == 'opengl3':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL3
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL3
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL3
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL3
        obs_config.front_camera.render_mode = RenderMode.OPENGL3

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        headless=True)
    rlbench_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ''
    append_log(log_path, log_lock, 'INFO', f'process-{i} started')
    worker_state[i] = {
        'status': 'idle',
        'last_heartbeat': time.time(),
    }

    while True:
        # 通过锁确定当前进程分配的任务和变体
        with lock:

            if task_index.value >= num_tasks:
                append_log(log_path, log_lock, 'INFO', f'process-{i} finished')
                worker_state[i] = {
                    'status': 'finished',
                    'last_heartbeat': time.time(),
                }
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if args.variations >= 0:
                var_target = np.minimum(args.variations, var_target)
            if my_variation_count >= var_target:
                # 当前任务的所有变体已采集完毕，切换到下一个任务
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                append_log(log_path, log_lock, 'INFO', f'process-{i} finished')
                worker_state[i] = {
                    'status': 'finished',
                    'last_heartbeat': time.time(),
                }
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        worker_state[i] = {
            'status': 'running',
            'task_name': task_env.get_name(),
            'variation_index': int(my_variation_count),
            'demo_index': -1,
            'last_heartbeat': time.time(),
        }

        var_key = f'{task_env.get_name()}::{my_variation_count}'
        variation_stats[var_key] = {
            'task_name': task_env.get_name(),
            'variation_index': int(my_variation_count),
            'planned_demos': int(args.episodes_per_task),
            'success_demos': 0,
            'timeout_demos': 0,
            'failed_exception_demos': 0,
            'skipped_demos': 0,
            'failed_demos': 0,
            'status': 'in_progress',
        }

        update_progress(progress, progress_lock, planned_episodes=args.episodes_per_task)

        task_env.set_variation(my_variation_count)
        descriptions, _ = task_env.reset()

        variation_path = os.path.join(
            args.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count)

        check_and_make(variation_path)

        with open(os.path.join(
                variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        append_log(
            log_path,
            log_lock,
            'INFO',
            f'process-{i} start task={task_env.get_name()} variation={my_variation_count}')

        abort_variation = False
        episode_lengths = []  # 记录每个 episode 的步数
        timeout_count = 0
        failed_exc_count = 0
        skipped_count = 0
        for ex_idx in range(args.episodes_per_task):
            worker_state[i] = {
                'status': 'running',
                'task_name': task_env.get_name(),
                'variation_index': int(my_variation_count),
                'demo_index': int(ex_idx),
                'demo_started_at': time.time(),
                'last_heartbeat': time.time(),
            }

            if args.debug:
                print('Process', i, '// Task:', task_env.get_name(),
                      '// Variation:', my_variation_count, '// Demo:', ex_idx)

            attempts = 10
            while attempts > 0:
                try:
                    worker_state[i] = {
                        'status': 'running',
                        'task_name': task_env.get_name(),
                        'variation_index': int(my_variation_count),
                        'demo_index': int(ex_idx),
                        'demo_started_at': worker_state.get(i, {}).get('demo_started_at', time.time()),
                        'last_heartbeat': time.time(),
                    }
                    # 采集一条演示轨迹（设置超时）
                    if args.demo_timeout > 0:
                        signal.alarm(args.demo_timeout)
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                    if args.demo_timeout > 0:
                        signal.alarm(0)  # 采集成功，取消定时器
                except DemoTimeoutError:
                    signal.alarm(0)
                    problem = (
                        'Process %d TIMEOUT (>%ds) collecting task %s '
                        '(variation: %d, example: %d). Skipping this demo.\n' % (
                            i, args.demo_timeout, task_env.get_name(),
                            my_variation_count, ex_idx)
                    )
                    if args.debug:
                        print(problem)
                    tasks_with_problems += problem
                    append_log(log_path, log_lock, 'WARN', problem.strip())
                    update_progress(
                        progress,
                        progress_lock,
                        done_episodes=1,
                        timeout_episodes=1)
                    timeout_count += 1
                    cur = dict(variation_stats.get(var_key, {}))
                    cur['timeout_demos'] = int(cur.get('timeout_demos', 0)) + 1
                    variation_stats[var_key] = cur
                    break  # 仅跳过这一条 demo，继续采集下一条
                except Exception as e:
                    if args.demo_timeout > 0:
                        signal.alarm(0)
                    attempts -= 1
                    if attempts > 0:
                        if args.debug:
                            append_log(
                                log_path,
                                log_lock,
                                'DEBUG',
                                'retry process-%d task=%s variation=%d demo=%d left_attempts=%d error=%s' % (
                                    i, task_env.get_name(), my_variation_count,
                                    ex_idx, attempts, str(e)))
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), my_variation_count, ex_idx,
                            str(e))
                    )
                    if args.debug:
                        print(problem)
                    tasks_with_problems += problem
                    append_log(log_path, log_lock, 'ERROR', problem.strip())
                    if args.debug:
                        append_log(log_path, log_lock, 'DEBUG', traceback.format_exc())

                    remain = args.episodes_per_task - ex_idx - 1
                    update_progress(
                        progress,
                        progress_lock,
                        done_episodes=1 + max(0, remain),
                        failed_episodes=1,
                        skipped_episodes=max(0, remain))
                    failed_exc_count += 1
                    skipped_count += max(0, remain)
                    cur = dict(variation_stats.get(var_key, {}))
                    cur['failed_exception_demos'] = int(cur.get('failed_exception_demos', 0)) + 1
                    cur['skipped_demos'] = int(cur.get('skipped_demos', 0)) + max(0, remain)
                    variation_stats[var_key] = cur
                    abort_variation = True
                    break

                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                with file_lock:
                    save_demo(demo, episode_path)
                episode_lengths.append(len(demo))  # 记录步数
                append_log(
                    log_path,
                    log_lock,
                    'INFO',
                    'process-%d saved task=%s variation=%d demo=%d steps=%d' % (
                        i, task_env.get_name(), my_variation_count, ex_idx, len(demo)))
                update_progress(
                    progress,
                    progress_lock,
                    done_episodes=1,
                    success_episodes=1)
                cur = dict(variation_stats.get(var_key, {}))
                cur['success_demos'] = int(cur.get('success_demos', 0)) + 1
                variation_stats[var_key] = cur
                break
            if abort_variation:
                break

        # 保存 variation 层级元数据
        save_variation_metadata(
            variation_path, my_variation_count, descriptions, episode_lengths)

        append_log(
            log_path,
            log_lock,
            'INFO',
            'process-%d finish task=%s variation=%d success_demos=%d planned_demos=%d' % (
                i, task_env.get_name(), my_variation_count,
                len(episode_lengths), args.episodes_per_task))

        failed_demos = timeout_count + failed_exc_count + skipped_count
        status = 'completed' if failed_demos == 0 else 'partial_failed'
        variation_stats[var_key] = {
            'task_name': task_env.get_name(),
            'variation_index': int(my_variation_count),
            'planned_demos': int(args.episodes_per_task),
            'success_demos': int(len(episode_lengths)),
            'timeout_demos': int(timeout_count),
            'failed_exception_demos': int(failed_exc_count),
            'skipped_demos': int(skipped_count),
            'failed_demos': int(failed_demos),
            'status': status,
        }
        worker_state[i] = {
            'status': 'idle',
            'demo_index': -1,
            'last_heartbeat': time.time(),
        }

    results[i] = tasks_with_problems
    append_log(log_path, log_lock, 'INFO', f'process-{i} shutdown env')
    worker_state[i] = {
        'status': 'shutdown',
        'demo_index': -1,
        'last_heartbeat': time.time(),
    }
    rlbench_env.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description='RLBench 数据集生成器')
    parser.add_argument('--save_path', type=str, default='./demos_all_0323', help='演示数据的保存路径')
    parser.add_argument('--tasks', nargs='*', default=[], help='要采集的任务列表，不填则采集所有任务')
    parser.add_argument('--image_size', nargs=2, type=int, default=[128, 128], help='保存图像的分辨率')
    parser.add_argument('--renderer', type=str, choices=['opengl', 'opengl3'], default='opengl3', help='渲染器类型，opengl 无阴影但速度更快')
    parser.add_argument('--processes', type=int, default=8, help='并行采集的进程数')
    parser.add_argument('--episodes_per_task', type=int, default=2, help='每个任务变体采集的 episode 数量')
    parser.add_argument('--variations', type=int, default=5, help='每个任务采集的变体数量上限，-1 表示全部 (默认: 5)')
    parser.add_argument('--arm_max_velocity', type=float, default=1.0, help='运动规划使用的最大手臂速度')
    parser.add_argument('--arm_max_acceleration', type=float, default=4.0, help='运动规划使用的最大手臂加速度')
    parser.add_argument('--demo_timeout', type=int, default=90,
                        help='单条演示采集的超时秒数，超时则跳过该 variation（0 = 不限时，默认 120）')
    parser.add_argument('--worker_stuck_timeout', type=int, default=180,
                        help='worker 卡住判定秒数。长期无心跳会终止并重启该 worker (默认: 180)')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式：终端输出更详细，并在日志写入重试/堆栈信息')
    return parser.parse_args()


def main():
    args = parse_args()

    # demo_timeout 是软超时（signal）；watchdog 是硬超时（杀进程）。默认保证硬超时 > 软超时。
    if args.worker_stuck_timeout > 0 and args.demo_timeout > 0 and args.worker_stuck_timeout <= args.demo_timeout:
        args.worker_stuck_timeout = args.demo_timeout + 60

    check_and_make('./log')
    log_file = os.path.join(
        'log', f'collect_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [INFO] start collection\n')
        f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [INFO] args={vars(args)}\n')

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(args.tasks) > 0:
        for t in args.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = args.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()
    log_lock = manager.Lock()
    progress_lock = manager.Lock()
    progress = manager.dict({
        'planned_episodes': 0,
        'done_episodes': 0,
        'success_episodes': 0,
        'timeout_episodes': 0,
        'failed_episodes': 0,
        'skipped_episodes': 0,
    })
    variation_stats = manager.dict()
    worker_state = manager.dict()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(args.save_path)
    estimated_total_episodes = estimate_total_episodes(task_files, args)
    progress['planned_episodes'] = int(estimated_total_episodes)

    print(
        f'[Info] Start collecting. tasks={len(task_files)} '
        f'processes={args.processes} episodes_per_variation={args.episodes_per_task}')
    append_log(
        log_file,
        log_lock,
        'INFO',
        f'planned tasks={len(task_files)} processes={args.processes} '
        f'episodes_per_variation={args.episodes_per_task} estimated_total_episodes={estimated_total_episodes}')

    processes = [Process(
        target=run, args=(
            i, lock, task_index, variation_count, result_dict, file_lock,
            tasks, args, log_file, log_lock, progress, progress_lock,
            variation_stats, worker_state))
        for i in range(args.processes)]

    for t in processes:
        t.start()

    bar_total = estimated_total_episodes if estimated_total_episodes > 0 else None
    bar = tqdm(total=bar_total, desc='Collecting demos', dynamic_ncols=True)
    last_done = 0
    while any(t.is_alive() for t in processes):
        done = int(progress.get('done_episodes', 0))
        delta = done - last_done
        if delta > 0:
            bar.update(delta)
            last_done = done
        bar.set_postfix({
            'ok': int(progress.get('success_episodes', 0)),
            'timeout': int(progress.get('timeout_episodes', 0)),
            'fail': int(progress.get('failed_episodes', 0)),
            'skip': int(progress.get('skipped_episodes', 0)),
        })

        if args.worker_stuck_timeout > 0:
            now = time.time()
            for idx, proc in enumerate(processes):
                if not proc.is_alive():
                    continue
                st = worker_state.get(idx)
                if not st:
                    continue
                if st.get('status') != 'running':
                    continue
                last_hb = float(st.get('last_heartbeat', now))
                if now - last_hb < args.worker_stuck_timeout:
                    continue

                task_name = st.get('task_name', 'unknown')
                variation_index = st.get('variation_index', -1)
                demo_index = st.get('demo_index', -1)
                demo_started_at = float(st.get('demo_started_at', last_hb))
                var_key = f'{task_name}::{variation_index}'
                cur = dict(variation_stats.get(var_key, {}))
                planned = int(cur.get('planned_demos', args.episodes_per_task))
                succ = int(cur.get('success_demos', 0))
                tout = int(cur.get('timeout_demos', 0))
                fexc = int(cur.get('failed_exception_demos', 0))
                sk = int(cur.get('skipped_demos', 0))
                consumed = succ + tout + fexc + sk
                remain = max(0, planned - consumed)

                append_log(
                    log_file,
                    log_lock,
                    'ERROR',
                    f'watchdog kill worker-{idx} task={task_name} variation={variation_index} '
                    f'demo={demo_index} stalled_for={int(now - last_hb)}s '
                    f'demo_elapsed={int(now - demo_started_at)}s remain_demos={remain}')

                proc.terminate()
                proc.join(timeout=3)

                if remain > 0:
                    update_progress(
                        progress,
                        progress_lock,
                        done_episodes=remain,
                        skipped_episodes=remain)

                cur['task_name'] = task_name
                cur['variation_index'] = int(variation_index)
                cur['planned_demos'] = planned
                cur['success_demos'] = succ
                cur['timeout_demos'] = tout
                cur['failed_exception_demos'] = fexc
                cur['skipped_demos'] = sk + remain
                cur['failed_demos'] = planned - succ
                cur['status'] = 'stuck_killed'
                variation_stats[var_key] = cur

                worker_state[idx] = {
                    'status': 'restarting',
                    'demo_index': -1,
                    'last_heartbeat': time.time(),
                }

                new_proc = Process(
                    target=run,
                    args=(
                        idx, lock, task_index, variation_count, result_dict,
                        file_lock, tasks, args, log_file, log_lock,
                        progress, progress_lock, variation_stats, worker_state))
                new_proc.start()
                processes[idx] = new_proc
                append_log(log_file, log_lock, 'INFO', f'watchdog restarted worker-{idx}')

        time.sleep(0.2)

    for t in processes:
        t.join()

    final_done = int(progress.get('done_episodes', 0))
    if final_done > last_done:
        bar.update(final_done - last_done)
    # 若预估总量与实际不一致，结束前校正 total，避免显示异常百分比。
    if bar.total is not None and bar.total != final_done:
        bar.total = final_done
        bar.refresh()
    bar.close()

    print('Data collection done!')
    for i in range(args.processes):
        print(result_dict.get(i, ''))

    append_log(log_file, log_lock, 'INFO', f'worker_problems={dict(result_dict)}')

    summary_lines, detail_lines = summarize_collection(
        task_files=task_files,
        args=args,
        progress=progress,
        variation_stats=variation_stats,
    )
    print('\n' + '\n'.join(summary_lines))
    if detail_lines:
        print('\n'.join(detail_lines))
    append_log(log_file, log_lock, 'INFO', ' | '.join(summary_lines))
    for line in detail_lines:
        append_log(log_file, log_lock, 'WARN', line)

    # 保存 task 层级元数据
    task_names = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    if len(args.tasks) > 0:
        task_names = args.tasks
    for task_name in task_names:
        task_path = os.path.join(args.save_path, task_name)
        if os.path.exists(task_path):
            save_task_metadata(task_path, task_name)

    # 构造 obs_config 用于元数据（与 run() 中相同的配置）
    img_size = list(map(int, args.image_size))
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size  = img_size
    obs_config.overhead_camera.image_size        = img_size
    obs_config.wrist_camera.image_size           = img_size
    obs_config.front_camera.image_size           = img_size

    # 保存数据集层级元数据
    save_dataset_metadata(args.save_path, args, obs_config)
    append_log(log_file, log_lock, 'INFO', 'metadata saved and collection done')
    print(f'[Info] Log saved: {log_file}')


if __name__ == '__main__':
    main()

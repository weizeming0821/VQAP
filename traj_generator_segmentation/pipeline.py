# -*- coding: utf-8 -*-
"""
一体化流水线主模块

直接在内存中采集 RLBench 演示轨迹，立即进行关键帧分割，
仅将分割后的子阶段结果写入磁盘，避免先保存完整 demo 再分割带来的巨大磁盘开销。
"""

import argparse
import os
import signal
import time
import traceback
from datetime import datetime
from multiprocessing import Manager, Process

import numpy as np
from tqdm import tqdm

import rlbench.backend.task as task
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.backend.const import VARIATIONS_FOLDER, EPISODES_FOLDER, EPISODE_FOLDER
from rlbench.environment import Environment
from pyrep.const import RenderMode

from .config import (
    RUN_SIGNALS, RUN_MIN_PHASE_LEN, RUN_SHOW_SEG_TRACE, RUN_DUMP_SENSORS,
    DEFAULT_IMAGE_SIZE, DEFAULT_RENDERER, DEFAULT_PROCESSES,
    DEFAULT_EPISODES_PER_TASK, DEFAULT_VARIATIONS,
    DEFAULT_ARM_MAX_VELOCITY, DEFAULT_ARM_MAX_ACCELERATION,
    DEFAULT_DEMO_TIMEOUT, DEFAULT_WORKER_STUCK_TIMEOUT,
    MAX_DEMO_ATTEMPTS, MAX_PHASE_VALIDATION_RETRIES,
)
from .signals import ALL_SIGNALS
from .demo_io import process_demo_in_memory
from .metadata import (
    save_variation_metadata, save_descriptions, save_task_metadata, save_split_summary
)
from .validation import load_fixed_phase_config, validate_phase_count


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
    """预估总 episode 数。"""
    if args.variations >= 0:
        return int(len(task_files) * args.variations * args.episodes_per_task)
    return int(len(task_files) * args.episodes_per_task)


def get_obs_config_dict(obs_config):
    """将 ObservationConfig 转换为可序列化的字典。"""
    cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']
    result = {}
    for cam in cameras:
        cam_cfg = getattr(obs_config, f'{cam}_camera')
        result[cam] = {
            'rgb': cam_cfg.rgb,
            'depth': cam_cfg.depth,
            'mask': cam_cfg.mask,
            'image_size': list(cam_cfg.image_size),
            'render_mode': str(cam_cfg.render_mode),
        }
    result['joint_velocities'] = obs_config.joint_velocities
    result['joint_positions'] = obs_config.joint_positions
    result['joint_forces'] = obs_config.joint_forces
    result['gripper_open'] = obs_config.gripper_open
    result['gripper_pose'] = obs_config.gripper_pose
    result['gripper_joint_positions'] = obs_config.gripper_joint_positions
    result['gripper_touch_forces'] = obs_config.gripper_touch_forces
    result['task_low_dim_state'] = obs_config.task_low_dim_state
    return result


def create_obs_config(args):
    """创建观测配置。"""
    img_size = list(map(int, args.image_size))
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False
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

    return obs_config


def run_worker(i, lock, task_index, variation_count, results, file_lock, tasks, args,
               log_path, log_lock, progress, progress_lock, variation_stats, worker_state,
               fixed_phase_config):
    """
    每个进程独立选择一个任务和变体，采集演示数据，立即分割并保存。
    如果分割后的阶段数不符合 fixed_phase_num，则重新采集再分割。
    """
    np.random.seed(None)
    num_tasks = len(tasks)

    if args.demo_timeout > 0:
        signal.signal(signal.SIGALRM, _demo_timeout_handler)

    obs_config = create_obs_config(args)
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
    worker_state[i] = {'status': 'idle', 'last_heartbeat': time.time()}

    # 确定启用的信号
    signals = set(RUN_SIGNALS) if RUN_SIGNALS else set(ALL_SIGNALS)

    while True:
        with lock:
            if task_index.value >= num_tasks:
                append_log(log_path, log_lock, 'INFO', f'process-{i} finished')
                worker_state[i] = {'status': 'finished', 'last_heartbeat': time.time()}
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if args.variations >= 0:
                var_target = np.minimum(args.variations, var_target)
            if my_variation_count >= var_target:
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                append_log(log_path, log_lock, 'INFO', f'process-{i} finished')
                worker_state[i] = {'status': 'finished', 'last_heartbeat': time.time()}
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_name = task_env.get_name()
        worker_state[i] = {
            'status': 'running',
            'task_name': task_name,
            'variation_index': int(my_variation_count),
            'demo_index': -1,
            'last_heartbeat': time.time(),
        }

        # 获取该任务期望的阶段数
        expected_phase_num = fixed_phase_config.get(task_name, None)

        var_key = f'{task_name}::{my_variation_count}'
        variation_stats[var_key] = {
            'task_name': task_name,
            'variation_index': int(my_variation_count),
            'planned_demos': int(args.episodes_per_task),
            'success_demos': 0,
            'phase_valid_demos': 0,
            'timeout_demos': 0,
            'failed_exception_demos': 0,
            'phase_invalid_demos': 0,
            'status': 'in_progress',
        }

        update_progress(progress, progress_lock, planned_episodes=args.episodes_per_task)

        task_env.set_variation(my_variation_count)
        descriptions, _ = task_env.reset()

        # 创建输出目录
        variation_path = os.path.join(
            args.output_path, task_name,
            VARIATIONS_FOLDER % my_variation_count)
        check_and_make(variation_path)
        save_descriptions(variation_path, descriptions)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        append_log(log_path, log_lock, 'INFO',
                   f'process-{i} start task={task_name} variation={my_variation_count}')

        episode_stats = []
        timeout_count = 0
        failed_exc_count = 0
        phase_invalid_count = 0

        if args.debug:
            print(f'[DEBUG] process-{i} entering episode loop, episodes_per_task={args.episodes_per_task}', flush=True)

        for ex_idx in range(args.episodes_per_task):
            if args.debug:
                print(f'[DEBUG] process-{i} updating worker_state for episode {ex_idx}', flush=True)
            worker_state[i] = {
                'status': 'running',
                'task_name': task_name,
                'variation_index': int(my_variation_count),
                'demo_index': int(ex_idx),
                'demo_started_at': time.time(),
                'last_heartbeat': time.time(),
            }
            if args.debug:
                print(f'[DEBUG] process-{i} worker_state updated', flush=True)

            if args.debug:
                print(f'Process {i} // Task: {task_name} // Variation: {my_variation_count} // Demo: {ex_idx}')

            # 尝试采集并分割 demo，最多重试 MAX_PHASE_VALIDATION_RETRIES 次
            demo_collected = False
            phase_valid = False
            retry_count = 0
            max_retries = MAX_PHASE_VALIDATION_RETRIES if expected_phase_num is not None else 1

            while retry_count < max_retries and not (demo_collected and phase_valid):
                attempts = MAX_DEMO_ATTEMPTS
                demo = None

                while attempts > 0:
                    worker_state[i] = {
                        'status': 'running',
                        'task_name': task_name,
                        'variation_index': int(my_variation_count),
                        'demo_index': int(ex_idx),
                        'last_heartbeat': time.time(),
                    }

                    # 采集演示
                    if args.debug:
                        print(f'[DEBUG] process-{i} setting alarm, demo_timeout={args.demo_timeout}', flush=True)
                    if args.demo_timeout > 0:
                        signal.alarm(args.demo_timeout)

                    if args.debug:
                        print(f'[DEBUG] process-{i} calling get_demos()...', flush=True)
                    try:
                        demo, = task_env.get_demos(amount=1, live_demos=True)
                        if args.debug:
                            print(f'[DEBUG] process-{i} get_demos() returned, frames={len(demo)}', flush=True)
                        if args.demo_timeout > 0:
                            signal.alarm(0)
                        demo_collected = True
                        break  # 采集成功，跳出内层循环
                    except DemoTimeoutError:
                        signal.alarm(0)
                        problem = (f'Process {i} TIMEOUT collecting task {task_name} '
                                   f'(variation: {my_variation_count}, episode: {ex_idx})')
                        if args.debug:
                            print(problem)
                        tasks_with_problems += problem + '\n'
                        append_log(log_path, log_lock, 'WARN', problem)
                        timeout_count += 1
                        break
                    except Exception as e:
                        if args.demo_timeout > 0:
                            signal.alarm(0)
                        attempts -= 1
                        if attempts > 0:
                            continue
                        problem = (f'Process {i} failed collecting task {task_name} '
                                   f'(variation: {my_variation_count}, episode: {ex_idx}): {e}')
                        if args.debug:
                            print(problem)
                        tasks_with_problems += problem + '\n'
                        append_log(log_path, log_lock, 'ERROR', problem)
                        failed_exc_count += 1
                        break
                if demo is None:
                    print(f'[DEBUG] process-{i} failed to collect demo after {MAX_DEMO_ATTEMPTS} attempts, moving on...', flush=True)

                if demo is not None:
                    if args.debug:
                        print(f'[DEBUG] process-{i} demo collected, starting segmentation...', flush=True)
                    # 立即进行关键帧分割并保存
                    episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)

                    phase_info, num_phases, valid = process_demo_in_memory(
                        demo, episode_path, descriptions,
                        signals=signals,
                        min_phase_len=args.min_phase_len,
                        save_mode=args.save_mode,
                        fixed_phase_num=expected_phase_num,
                    )

                    if args.debug:
                        print(f'[DEBUG] process-{i} segmentation done: num_phases={num_phases}, expected={expected_phase_num}, valid={valid}', flush=True)

                    phase_valid = valid

                    if not valid and expected_phase_num is not None:
                        # 阶段数不匹配，需要重新采集
                        print(f'[INFO] process-{i} phase count mismatch, will retry: got {num_phases}, expected {expected_phase_num}, retry_count={retry_count + 1}/{max_retries}', flush=True)
                        phase_invalid_count += 1
                        retry_count += 1
                        append_log(log_path, log_lock, 'WARN',
                                   f'process-{i} task={task_name} variation={my_variation_count} '
                                   f'episode={ex_idx} phases={num_phases} expected={expected_phase_num} '
                                   f'retry={retry_count}/{max_retries}')
                        if args.debug:
                            print(f'  [WARN] Phase count mismatch: got {num_phases}, expected {expected_phase_num}, retry={retry_count}/{max_retries}', flush=True)
                        # 重置状态以便重试
                        demo_collected = False
                        continue

                    # 成功
                    episode_stats.append({
                        'episode': ex_idx,
                        'num_phases': num_phases,
                        'phase_valid': valid,
                        'phases': phase_info,
                    })

                    if RUN_SHOW_SEG_TRACE:
                        phases_str = ' | '.join(
                            f"phase_{p['phase_index']}(kf={p['keyframe_index']}, len={p['length']}f)"
                            for p in phase_info)
                        print(f'  variation{my_variation_count}/episode{ex_idx}: {phases_str}')

                    append_log(log_path, log_lock, 'INFO',
                               f'process-{i} saved task={task_name} variation={my_variation_count} '
                               f'episode={ex_idx} phases={num_phases}')
                    update_progress(progress, progress_lock,
                                    done_episodes=1, success_episodes=1)
                    cur = dict(variation_stats.get(var_key, {}))
                    cur['success_demos'] = int(cur.get('success_demos', 0)) + 1
                    if valid:
                        cur['phase_valid_demos'] = int(cur.get('phase_valid_demos', 0)) + 1
                    variation_stats[var_key] = cur
                    break

                # 采集失败，跳出重试循环
                update_progress(progress, progress_lock,
                                done_episodes=1, failed_episodes=1)
                break

        # 保存 variation 元数据
        save_variation_metadata(variation_path, my_variation_count, descriptions, episode_stats)

        # 保存 split_summary.json
        episode_summaries = [
            {
                'episode': f'episode{stat["episode"]}',
                'num_phases': stat['num_phases'],
                'phases': stat['phases'],
            }
            for stat in episode_stats
        ]
        save_split_summary(variation_path, task_name, f'variation{my_variation_count}',
                           descriptions, args.save_mode, signals, episode_summaries)

        failed_demos = timeout_count + failed_exc_count
        status = 'completed' if failed_demos == 0 and phase_invalid_count == 0 else 'partial_failed'
        variation_stats[var_key] = {
            'task_name': task_name,
            'variation_index': int(my_variation_count),
            'planned_demos': int(args.episodes_per_task),
            'success_demos': len(episode_stats),
            'phase_valid_demos': sum(1 for s in episode_stats if s.get('phase_valid', True)),
            'timeout_demos': int(timeout_count),
            'failed_exception_demos': int(failed_exc_count),
            'phase_invalid_demos': int(phase_invalid_count),
            'failed_demos': int(failed_demos),
            'status': status,
        }

        worker_state[i] = {'status': 'idle', 'demo_index': -1, 'last_heartbeat': time.time()}

    results[i] = tasks_with_problems
    append_log(log_path, log_lock, 'INFO', f'process-{i} shutdown env')
    worker_state[i] = {'status': 'shutdown', 'last_heartbeat': time.time()}
    rlbench_env.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(
        description='一体化 Demo 生成与关键帧分割流水线',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--output_path', type=str, default='./demos_subphase',
                        help='输出子阶段 demos 的根目录')
    parser.add_argument('--tasks', nargs='*', default=[],
                        help='要采集的任务列表，不填则采集所有任务')
    parser.add_argument('--image_size', nargs=2, type=int, default=DEFAULT_IMAGE_SIZE,
                        help='保存图像的分辨率')
    parser.add_argument('--renderer', type=str, choices=['opengl', 'opengl3'],
                        default=DEFAULT_RENDERER, help='渲染器类型')
    parser.add_argument('--processes', type=int, default=DEFAULT_PROCESSES,
                        help='并行采集的进程数')
    parser.add_argument('--episodes_per_task', type=int, default=DEFAULT_EPISODES_PER_TASK,
                        help='每个任务变体采集的 episode 数量')
    parser.add_argument('--variations', type=int, default=DEFAULT_VARIATIONS,
                        help='每个任务采集的变体数量上限，-1 表示全部')
    parser.add_argument('--arm_max_velocity', type=float, default=DEFAULT_ARM_MAX_VELOCITY,
                        help='运动规划使用的最大手臂速度')
    parser.add_argument('--arm_max_acceleration', type=float, default=DEFAULT_ARM_MAX_ACCELERATION,
                        help='运动规划使用的最大手臂加速度')
    parser.add_argument('--demo_timeout', type=int, default=DEFAULT_DEMO_TIMEOUT,
                        help='单条演示采集的超时秒数（0 = 不限时）')
    parser.add_argument('--worker_stuck_timeout', type=int, default=DEFAULT_WORKER_STUCK_TIMEOUT,
                        help='worker 卡住判定秒数')
    parser.add_argument('--min_phase_len', type=int, default=RUN_MIN_PHASE_LEN,
                        help='相邻关键帧最小帧距')
    parser.add_argument('--save_mode', type=str, default='keyframe_only',
                        choices=['full', 'keyframe_only'],
                        help='保存模式')
    parser.add_argument('--fixed_phase_csv', type=str, default='./TASK_FIXED_PHASE.csv',
                        help='固定阶段数配置文件路径')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式')
    return parser.parse_args()


def run_segmented_collection(args):
    """运行分割后的演示采集流程。"""
    if args.worker_stuck_timeout > 0 and args.demo_timeout > 0:
        if args.worker_stuck_timeout <= args.demo_timeout:
            args.worker_stuck_timeout = args.demo_timeout + 60

    check_and_make('./log')
    log_file = os.path.join('log',
                            f'segment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f'[{datetime.now()}] [INFO] start segmented collection\n')
        f.write(f'[{datetime.now()}] [INFO] args={vars(args)}\n')

    # 加载固定阶段数配置
    fixed_phase_config = load_fixed_phase_config(args.fixed_phase_csv)

    # 获取任务列表
    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    if len(args.tasks) > 0:
        for t in args.tasks:
            if t not in task_files:
                raise ValueError(f'Task {t} not recognised!')
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
    })
    variation_stats = manager.dict()
    worker_state = manager.dict()
    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(args.output_path)
    estimated_total = estimate_total_episodes(task_files, args)
    progress['planned_episodes'] = int(estimated_total)

    print(f'[Info] Start collecting. tasks={len(task_files)} '
          f'processes={args.processes} episodes_per_variation={args.episodes_per_task}')

    # 启动 worker 进程
    processes = [
        Process(target=run_worker, args=(
            i, lock, task_index, variation_count, result_dict, file_lock,
            tasks, args, log_file, log_lock, progress, progress_lock,
            variation_stats, worker_state, fixed_phase_config
        ))
        for i in range(args.processes)
    ]
    for p in processes:
        p.start()

    # 进度条
    bar_total = estimated_total if estimated_total > 0 else None
    bar = tqdm(total=bar_total, desc='Collecting & Segmenting', dynamic_ncols=True)
    last_done = 0

    while any(p.is_alive() for p in processes):
        done = int(progress.get('done_episodes', 0))
        delta = done - last_done
        if delta > 0:
            bar.update(delta)
            last_done = done
        bar.set_postfix({
            'ok': int(progress.get('success_episodes', 0)),
            'timeout': int(progress.get('timeout_episodes', 0)),
            'fail': int(progress.get('failed_episodes', 0)),
        })
        time.sleep(0.2)

    for p in processes:
        p.join()

    final_done = int(progress.get('done_episodes', 0))
    if final_done > last_done:
        bar.update(final_done - last_done)
    if bar.total is not None and bar.total != final_done:
        bar.total = final_done
        bar.refresh()
    bar.close()

    print('\nData collection & segmentation done!')
    for i in range(args.processes):
        if result_dict.get(i, ''):
            print(result_dict[i])

    # 保存 task 层级元数据
    task_names = args.tasks if len(args.tasks) > 0 else task_files
    for task_name in task_names:
        task_path = os.path.join(args.output_path, task_name)
        if os.path.exists(task_path):
            save_task_metadata(task_path, task_name,
                               fixed_phase_num=fixed_phase_config.get(task_name))

    append_log(log_file, log_lock, 'INFO', 'collection done')
    print(f'[Info] Log saved: {log_file}')


def main():
    args = parse_args()
    run_segmented_collection(args)


if __name__ == '__main__':
    main()

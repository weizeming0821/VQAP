# -*- coding: utf-8 -*-
"""
一体化流水线主模块

直接在内存中采集 RLBench 演示轨迹，立即进行关键帧分割，
仅将分割后的子阶段结果写入磁盘，避免先保存完整 demo 再分割带来的巨大磁盘开销。
"""

import argparse
import json
import os
import random
import signal
import shutil
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Manager, Process, get_context
from queue import Empty

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
    save_variation_metadata, save_task_metadata, save_dataset_metadata
)
from .validation import (
    load_fixed_phase_config,
    resolve_fixed_phase_csv_path,
    split_tasks_by_fixed_phase_config,
    validate_phase_count,
)


class DemoTimeoutError(Exception):
    """单条演示采集超时时抛出。"""
    pass


def _demo_timeout_handler(signum, frame):
    raise DemoTimeoutError('Demo collection timed out')


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def _remove_tree_if_exists(path):
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def _stream_is_tty(stream):
    isatty = getattr(stream, 'isatty', None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def _disable_tqdm_output():
    return not _stream_is_tty(sys.stderr)


def _build_failure_detail(task_name, variation_index, episode_index, failure_type, reason, **extra):
    detail = {
        'task_name': task_name,
        'variation_index': int(variation_index),
        'requested_episode': int(episode_index),
        'failure_type': failure_type,
        'reason': reason,
    }
    for key, value in extra.items():
        if value is not None:
            detail[key] = value
    return detail


def _write_json_atomic(path, payload):
    if not path:
        return
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    temp_path = f'{path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


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


def write_progress_snapshot(progress_file, started_at, args, progress, worker_state,
                            variation_stats=None, finished=False, log_file=None):
    if not progress_file:
        return

    progress_snapshot = {k: int(v) for k, v in dict(progress).items()}
    worker_snapshot = {str(k): v for k, v in dict(worker_state).items()}
    variation_stats_snapshot = dict(variation_stats) if variation_stats is not None else {}
    payload = {
        'started_at': started_at.isoformat(),
        'updated_at': datetime.now().isoformat(),
        'finished': bool(finished),
        'log_file': log_file,
        'progress': progress_snapshot,
        'worker_state': worker_snapshot,
        'variation_stats': variation_stats_snapshot,
        'config': {
            'processes': int(args.processes),
            'episodes_per_task': int(args.episodes_per_task),
            'variations': int(args.variations),
            'tasks': list(args.tasks),
            'base_seed': getattr(args, 'base_seed', None),
        },
    }
    _write_json_atomic(progress_file, payload)


def estimate_total_episodes(task_files, args):
    """基于真实 RLBench variation_count() 预估总 episode 数。"""
    task_variation_targets = resolve_task_variation_targets(task_files, args)
    total_variations = sum(task_variation_targets.values())
    return int(total_variations * args.episodes_per_task), task_variation_targets


def get_task_variation_target(task_env, args):
    """返回当前任务本次运行会处理的 variation 数量上限。"""
    var_target = int(task_env.variation_count())
    if args.variations >= 0:
        var_target = min(int(args.variations), var_target)
    return var_target


def _build_variation_probe_args(args):
    return {
        'image_size': list(args.image_size),
        'renderer': args.renderer,
        'arm_max_velocity': float(args.arm_max_velocity),
        'arm_max_acceleration': float(args.arm_max_acceleration),
        'variations': int(args.variations),
    }


def _fallback_variation_targets(task_files, args):
    if int(args.variations) >= 0:
        per_task_variations = int(args.variations)
    else:
        per_task_variations = 1
    return {task_name: per_task_variations for task_name in task_files}


def _resolve_task_variation_targets_in_process(task_files, args):
    """在当前进程中查询每个任务在当前参数下实际会处理的 variation 数量。"""
    if not task_files:
        return {}

    obs_config = create_obs_config(args)
    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        headless=True)

    variation_targets = {}
    rlbench_env.launch()
    try:
        for task_name in task_files:
            task_class = task_file_to_task_class(task_name)
            task_env = rlbench_env.get_task(task_class)
            variation_targets[task_name] = get_task_variation_target(task_env, args)
    finally:
        rlbench_env.shutdown()

    return variation_targets


def _variation_target_probe(task_files, probe_args, result_queue):
    try:
        args = argparse.Namespace(**probe_args)
        result_queue.put({
            'ok': True,
            'targets': _resolve_task_variation_targets_in_process(task_files, args),
        })
    except Exception as exc:
        result_queue.put({
            'ok': False,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })


def resolve_task_variation_targets(task_files, args):
    """在独立 spawn 子进程中查询 variation 数，避免污染后续 worker fork。"""
    if not task_files:
        return {}

    ctx = get_context('spawn')
    result_queue = ctx.Queue()
    probe = ctx.Process(
        target=_variation_target_probe,
        args=(list(task_files), _build_variation_probe_args(args), result_queue),
    )
    probe.start()

    payload = None
    try:
        payload = result_queue.get(timeout=max(30, 10 * len(task_files)))
    except Empty:
        payload = None

    probe.join(timeout=5)
    if probe.is_alive():
        probe.terminate()
        probe.join()

    if payload and payload.get('ok'):
        return payload.get('targets', {})

    warning_parts = ['[Warn] Failed to query real RLBench variation counts in probe process.']
    if payload and payload.get('error'):
        warning_parts.append(f'error={payload["error"]}')
    if payload and payload.get('traceback'):
        warning_parts.append(payload['traceback'].strip())
    elif probe.exitcode not in (0, None):
        warning_parts.append(f'probe_exitcode={probe.exitcode}')
    warning_parts.append('Falling back to rough planned_episodes estimate.')
    print(' '.join(warning_parts), flush=True)
    return _fallback_variation_targets(task_files, args)


def summarize_collection(task_files, progress, variation_stats, started_at, finished_at):
    """汇总本次生成任务的总体统计信息。"""
    stat_values = list(variation_stats.values())
    total_variations = len(stat_values)
    success_variations = [s for s in stat_values if s.get('status') == 'completed']
    failed_variations = [s for s in stat_values if s.get('status') != 'completed']

    planned_demos = sum(int(s.get('planned_demos', 0)) for s in stat_values)
    success_demos = sum(int(s.get('success_demos', 0)) for s in stat_values)
    failed_demos = sum(int(s.get('failed_demos', 0)) for s in stat_values)
    timeout_demos = sum(int(s.get('timeout_demos', 0)) for s in stat_values)
    exception_demos = sum(int(s.get('failed_exception_demos', 0)) for s in stat_values)
    phase_invalid_demos = sum(int(s.get('phase_invalid_demos', 0)) for s in stat_values)
    phase_invalid_attempts = sum(int(s.get('phase_invalid_attempts', 0)) for s in stat_values)
    phase_valid_demos = sum(int(s.get('phase_valid_demos', 0)) for s in stat_values)
    total_failed_demos = failed_demos + phase_invalid_demos
    duration_seconds = max(0.0, (finished_at - started_at).total_seconds())
    episodes_per_minute = (success_demos / duration_seconds * 60.0) if duration_seconds > 0 else 0.0

    failed_task_stats = {}
    failed_episode_details = []
    for stat in failed_variations:
        task_name = stat.get('task_name')
        if not task_name:
            continue
        item = failed_task_stats.setdefault(task_name, {
            'failed_variations': 0,
            'failed_demos': 0,
            'timeout_demos': 0,
            'failed_exception_demos': 0,
            'phase_invalid_demos': 0,
        })
        item['failed_variations'] += 1
        item['failed_demos'] += int(stat.get('failed_demos', 0))
        item['timeout_demos'] += int(stat.get('timeout_demos', 0))
        item['failed_exception_demos'] += int(stat.get('failed_exception_demos', 0))
        item['phase_invalid_demos'] += int(stat.get('phase_invalid_demos', 0))
        for detail in stat.get('failure_details', []):
            failed_episode_details.append({
                'task_name': detail.get('task_name', task_name),
                'variation_index': int(detail.get('variation_index', stat.get('variation_index', -1))),
                'requested_episode': int(detail.get('requested_episode', detail.get('episode', -1))),
                'failure_type': detail.get('failure_type', 'unknown'),
                'reason': detail.get('reason', ''),
                'observed_phases': detail.get('observed_phases'),
                'expected_phases': detail.get('expected_phases'),
                'retries': detail.get('retries'),
            })

    lines = [
        '===== Segmented Collection Summary =====',
        f'Started at: {started_at.strftime("%Y-%m-%d %H:%M:%S")}',
        f'Finished at: {finished_at.strftime("%Y-%m-%d %H:%M:%S")}',
        f'Total duration: {duration_seconds:.2f}s',
        f'Generation speed: {episodes_per_minute:.2f} successful episodes/min',
        f'Planned tasks: {len(task_files)}',
        f'Processed variations: {total_variations}',
        f'Success variations: {len(success_variations)} / {total_variations}',
        f'Failed variations: {len(failed_variations)} / {total_variations}',
        f'Planned demos: {planned_demos}',
        f'Done demos: {int(progress.get("done_episodes", 0))}',
        f'Success demos: {success_demos}',
        f'Total failed demos: {total_failed_demos}',
        f'Timeout/exception failed demos: {failed_demos}',
        f'Timeout demos: {timeout_demos}',
        f'Exception demos: {exception_demos}',
        f'Phase invalid demos: {phase_invalid_demos}',
        f'Phase invalid attempts: {phase_invalid_attempts}',
        f'Phase valid demos: {phase_valid_demos}',
        f'Phase valid rate: {phase_valid_demos} / {success_demos}',
    ]

    detail_lines = []
    if failed_task_stats:
        detail_lines.append('Failed task breakdown:')
        for task_name in sorted(failed_task_stats.keys()):
            item = failed_task_stats[task_name]
            detail_lines.append(
                f'  - task={task_name} failed_variations={item["failed_variations"]} '
                f'failed_demos={item["failed_demos"]} timeout_demos={item["timeout_demos"]} '
                f'exception_demos={item["failed_exception_demos"]} '
                f'phase_invalid_demos={item["phase_invalid_demos"]}')

    if failed_episode_details:
        detail_lines.append('Failed episode details:')
        for detail in sorted(
                failed_episode_details,
                key=lambda item: (item['task_name'], item['variation_index'], item['requested_episode'], item['failure_type'])):
            extra_parts = []
            if detail.get('observed_phases') is not None and detail.get('expected_phases') is not None:
                extra_parts.append(
                    f'phases={detail["observed_phases"]}/{detail["expected_phases"]}')
            if detail.get('retries') is not None:
                extra_parts.append(f'retries={detail["retries"]}')
            extra_suffix = (' ' + ' '.join(extra_parts)) if extra_parts else ''
            detail_lines.append(
                f'  - task={detail["task_name"]} variation={detail["variation_index"]} '
                f'requested_episode={detail["requested_episode"]} type={detail["failure_type"]} '
                f'reason={detail["reason"]}{extra_suffix}')

    return lines, detail_lines


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
    base_seed = getattr(args, 'base_seed', None)
    if base_seed is None or int(base_seed) < 0:
        worker_seed = None
        np.random.seed(None)
        random.seed()
    else:
        worker_seed = (int(base_seed) + int(i) * 100003 + int(os.getpid())) % (2 ** 32 - 1)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
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
    worker_state[i] = {
        'status': 'idle',
        'last_heartbeat': time.time(),
        'worker_seed': worker_seed,
    }

    # 确定启用的信号
    signals = set(RUN_SIGNALS) if RUN_SIGNALS else set(ALL_SIGNALS)

    while True:
        with lock:
            if task_index.value >= num_tasks:
                append_log(log_path, log_lock, 'INFO', f'process-{i} finished')
                worker_state[i] = {
                    'status': 'finished',
                    'last_heartbeat': time.time(),
                    'worker_seed': worker_seed,
                }
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = get_task_variation_target(task_env, args)
            if my_variation_count >= var_target:
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                append_log(log_path, log_lock, 'INFO', f'process-{i} finished')
                worker_state[i] = {
                    'status': 'finished',
                    'last_heartbeat': time.time(),
                    'worker_seed': worker_seed,
                }
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
            'worker_seed': worker_seed,
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
            'phase_invalid_attempts': 0,
            'phase_invalid_demos': 0,
            'failure_details': [],
            'status': 'in_progress',
        }

        task_env.set_variation(my_variation_count)
        descriptions, _ = task_env.reset()

        # 创建输出目录
        variation_path = os.path.join(
            args.output_path, task_name,
            VARIATIONS_FOLDER % my_variation_count)
        check_and_make(variation_path)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        append_log(log_path, log_lock, 'INFO',
                   f'process-{i} start task={task_name} variation={my_variation_count}')

        episode_stats = []
        timeout_count = 0
        failed_exc_count = 0
        phase_invalid_attempt_count = 0
        phase_invalid_demo_count = 0
        failure_details = []

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
                'worker_seed': worker_seed,
            }
            if args.debug:
                print(f'[DEBUG] process-{i} worker_state updated', flush=True)

            if args.debug:
                print(f'Process {i} // Task: {task_name} // Variation: {my_variation_count} // Demo: {ex_idx}')

            # 尝试采集并分割 demo，最多重试 MAX_PHASE_VALIDATION_RETRIES 次
            demo_collected = False
            phase_valid = False
            phase_invalid_terminal_failure = False
            retry_count = 0
            max_retries = MAX_PHASE_VALIDATION_RETRIES if expected_phase_num is not None else 1
            episode_accounted = False
            last_phase_count = None
            episode_failure_detail = None

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
                        'worker_seed': worker_seed,
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
                        episode_failure_detail = _build_failure_detail(
                            task_name,
                            my_variation_count,
                            ex_idx,
                            'timeout',
                            'demo collection timed out',
                        )
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
                        episode_failure_detail = _build_failure_detail(
                            task_name,
                            my_variation_count,
                            ex_idx,
                            'exception',
                            str(e),
                            attempts=MAX_DEMO_ATTEMPTS,
                        )
                        break
                if demo is None:
                    print(f'[DEBUG] process-{i} failed to collect demo after {MAX_DEMO_ATTEMPTS} attempts, moving on...', flush=True)

                if demo is not None:
                    if args.debug:
                        print(f'[DEBUG] process-{i} demo collected, starting segmentation...', flush=True)
                    # 立即进行关键帧分割并保存
                    saved_episode_index = len(episode_stats)
                    episode_path = os.path.join(episodes_path, EPISODE_FOLDER % saved_episode_index)

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
                    last_phase_count = num_phases

                    if not valid and expected_phase_num is not None:
                        # 阶段数不匹配，需要重新采集
                        print(f'[INFO] process-{i} phase count mismatch, will retry: got {num_phases}, expected {expected_phase_num}, retry_count={retry_count + 1}/{max_retries}', flush=True)
                        phase_invalid_attempt_count += 1
                        retry_count += 1
                        append_log(log_path, log_lock, 'WARN',
                                   f'process-{i} task={task_name} variation={my_variation_count} '
                                   f'episode={ex_idx} phases={num_phases} expected={expected_phase_num} '
                                   f'retry={retry_count}/{max_retries}')
                        if args.debug:
                            print(f'  [WARN] Phase count mismatch: got {num_phases}, expected {expected_phase_num}, retry={retry_count}/{max_retries}', flush=True)
                        if retry_count >= max_retries:
                            phase_invalid_terminal_failure = True
                            demo_collected = False
                            _remove_tree_if_exists(episode_path)
                            episode_failure_detail = _build_failure_detail(
                                task_name,
                                my_variation_count,
                                ex_idx,
                                'phase_invalid',
                                'phase validation retries exhausted',
                                observed_phases=int(num_phases),
                                expected_phases=int(expected_phase_num),
                                retries=int(retry_count),
                            )
                            break
                        # 重置状态以便重试
                        _remove_tree_if_exists(episode_path)
                        demo_collected = False
                        continue

                    # 成功
                    episode_stats.append({
                        'episode': saved_episode_index,
                        'requested_episode': ex_idx,
                        'num_phases': num_phases,
                        'phase_valid': valid,
                        'phases': phase_info,
                    })

                    if RUN_SHOW_SEG_TRACE:
                        phases_str = ' | '.join(
                            f"phase_{p['phase_index']}(kf={p['keyframe_index']}, len={p['length']}f)"
                            for p in phase_info)
                        print(
                            f'  variation{my_variation_count}/episode{saved_episode_index} '
                            f'requested_episode={ex_idx}: {phases_str}')

                    append_log(log_path, log_lock, 'INFO',
                               f'process-{i} saved task={task_name} variation={my_variation_count} '
                               f'episode={saved_episode_index} requested_episode={ex_idx} '
                               f'phases={num_phases}')
                    update_progress(progress, progress_lock,
                                    done_episodes=1, success_episodes=1)
                    cur = dict(variation_stats.get(var_key, {}))
                    cur['success_demos'] = int(cur.get('success_demos', 0)) + 1
                    if valid:
                        cur['phase_valid_demos'] = int(cur.get('phase_valid_demos', 0)) + 1
                    variation_stats[var_key] = cur
                    episode_accounted = True
                    break

                # 采集失败，跳出重试循环
                if phase_invalid_terminal_failure:
                    phase_invalid_demo_count += 1
                if episode_failure_detail is None:
                    episode_failure_detail = _build_failure_detail(
                        task_name,
                        my_variation_count,
                        ex_idx,
                        'collection_failed',
                        'demo collection failed without a captured exception',
                    )
                failure_details.append(dict(episode_failure_detail))
                update_progress(progress, progress_lock,
                                done_episodes=1, failed_episodes=1)
                episode_accounted = True
                break

            if phase_invalid_terminal_failure and not episode_accounted:
                if episode_failure_detail is None:
                    episode_failure_detail = _build_failure_detail(
                        task_name,
                        my_variation_count,
                        ex_idx,
                        'phase_invalid',
                        'phase validation retries exhausted',
                        observed_phases=last_phase_count,
                        expected_phases=int(expected_phase_num) if expected_phase_num is not None else None,
                        retries=int(retry_count),
                    )
                problem = (
                    f'Process {i} exhausted phase validation retries for task {task_name} '
                    f'(variation: {my_variation_count}, episode: {ex_idx}): '
                    f'got {last_phase_count}, expected {expected_phase_num}'
                )
                tasks_with_problems += problem + '\n'
                append_log(log_path, log_lock, 'ERROR', problem)
                failure_details.append(dict(episode_failure_detail))
                phase_invalid_demo_count += 1
                update_progress(progress, progress_lock,
                                done_episodes=1, failed_episodes=1)

        failed_demos = timeout_count + failed_exc_count
        status = 'completed' if failed_demos == 0 and phase_invalid_demo_count == 0 else 'partial_failed'
        variation_stats[var_key] = {
            'task_name': task_name,
            'variation_index': int(my_variation_count),
            'planned_demos': int(args.episodes_per_task),
            'success_demos': len(episode_stats),
            'phase_valid_demos': sum(1 for s in episode_stats if s.get('phase_valid', True)),
            'timeout_demos': int(timeout_count),
            'failed_exception_demos': int(failed_exc_count),
            'phase_invalid_attempts': int(phase_invalid_attempt_count),
            'phase_invalid_demos': int(phase_invalid_demo_count),
            'failed_demos': int(failed_demos),
            'failure_details': failure_details,
            'status': status,
        }

        save_variation_metadata(
            variation_path,
            my_variation_count,
            descriptions,
            episode_stats,
            args.save_mode,
            signals,
            generation_stats=variation_stats[var_key],
        )

        worker_state[i] = {
            'status': 'idle',
            'demo_index': -1,
            'last_heartbeat': time.time(),
            'worker_seed': worker_seed,
        }

    results[i] = tasks_with_problems
    append_log(log_path, log_lock, 'INFO', f'process-{i} shutdown env')
    worker_state[i] = {
        'status': 'shutdown',
        'last_heartbeat': time.time(),
        'worker_seed': worker_seed,
    }
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
    parser.add_argument('--fixed_phase_csv', type=str, default='./TASK_FIXED_PHASE_NUM.csv',
                        help='固定阶段数配置文件路径')
    parser.add_argument('--base_seed', type=int, default=-1,
                        help='可选，整个采集作业的基础随机种子；每个 worker 会在此基础上派生自己的种子')
    parser.add_argument('--progress_file', type=str, default='',
                        help='可选，若提供则持续写入 JSON 进度快照，供外部 launcher 聚合显示')
    parser.add_argument('--log_path', type=str, default='',
                        help='可选，若提供则将本次运行日志写到指定路径，而不是默认 log 目录')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式')
    return parser.parse_args()


def run_segmented_collection(args):
    """运行分割后的演示采集流程。"""
    started_at = datetime.now()
    if args.worker_stuck_timeout > 0 and args.demo_timeout > 0:
        if args.worker_stuck_timeout <= args.demo_timeout:
            args.worker_stuck_timeout = args.demo_timeout + 60

    if args.log_path:
        log_file = os.path.abspath(args.log_path)
        check_and_make(os.path.dirname(log_file))
    else:
        check_and_make('./log')
        log_file = os.path.join('log',
                                f'traj_gen_seg_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}_pid{os.getpid()}.log')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f'[{datetime.now()}] [INFO] start segmented collection\n')
        f.write(f'[{datetime.now()}] [INFO] args={vars(args)}\n')

    args.fixed_phase_csv = resolve_fixed_phase_csv_path(args.fixed_phase_csv)
    fixed_phase_csv_exists = os.path.exists(args.fixed_phase_csv)
    fixed_phase_config = load_fixed_phase_config(args.fixed_phase_csv)

    # 获取任务列表
    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    if len(args.tasks) > 0:
        for t in args.tasks:
            if t not in task_files:
                raise ValueError(f'Task {t} not recognised!')
        task_files = args.tasks

        fixed_phase_tasks, normal_tasks = split_tasks_by_fixed_phase_config(
            task_files, fixed_phase_config)
        startup_messages = []
        if fixed_phase_csv_exists:
            if fixed_phase_tasks:
                startup_messages.append(
                    '[Info] Explicit tasks using fixed phase config from '
                    f'{args.fixed_phase_csv}: ' + ', '.join(fixed_phase_tasks)
                )
            if normal_tasks:
                startup_messages.append(
                    '[Info] Explicit tasks not listed in TASK_FIXED_PHASE_NUM.csv '
                    'and will use normal segmentation: ' + ', '.join(normal_tasks)
                )
        else:
            startup_messages.append(
                f'[Info] Fixed phase CSV not found for explicit tasks: {args.fixed_phase_csv}'
            )
            startup_messages.append(
                '[Info] Explicit tasks will use normal segmentation: '
                + ', '.join(task_files)
            )

        if startup_messages:
            with open(log_file, 'a', encoding='utf-8') as f:
                for message in startup_messages:
                    print(message)
                    f.write(f'[{datetime.now()}] [INFO] {message}\n')

    tasks = [task_file_to_task_class(t) for t in task_files]
    args.signals = sorted(set(RUN_SIGNALS) if RUN_SIGNALS else set(ALL_SIGNALS))

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
    estimated_total, task_variation_targets = estimate_total_episodes(task_files, args)
    progress['planned_episodes'] = int(estimated_total)
    planned_variations = int(sum(task_variation_targets.values()))
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(
            f'[{datetime.now()}] [INFO] planned_variations={planned_variations} '
            f'planned_episodes={estimated_total} task_variation_targets={task_variation_targets}\n'
        )
    write_progress_snapshot(
        args.progress_file,
        started_at,
        args,
        progress,
        worker_state,
        variation_stats=variation_stats,
        finished=False,
        log_file=log_file,
    )

    print(f'[Info] Start collecting. tasks={len(task_files)} '
            f'planned_variations={planned_variations} planned_episodes={estimated_total} '
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
    bar = tqdm(
        total=bar_total,
        desc='Collecting & Segmenting',
        dynamic_ncols=True,
        disable=_disable_tqdm_output(),
    )
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
        write_progress_snapshot(
            args.progress_file,
            started_at,
            args,
            progress,
            worker_state,
            variation_stats=variation_stats,
            finished=False,
            log_file=log_file,
        )
        time.sleep(0.2)

    for p in processes:
        p.join()

    worker_exit_codes = [p.exitcode for p in processes]
    crashed_workers = [index for index, exit_code in enumerate(worker_exit_codes) if exit_code not in (0, None)]

    finished_at = datetime.now()

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

    if crashed_workers:
        message = (
            'Worker process crashed: ' +
            ', '.join(f'process-{index} exitcode={worker_exit_codes[index]}' for index in crashed_workers)
        )
        append_log(log_file, log_lock, 'ERROR', message)
        raise RuntimeError(message)

    # 保存 task 层级元数据
    task_names = args.tasks if len(args.tasks) > 0 else task_files
    for task_name in task_names:
        task_path = os.path.join(args.output_path, task_name)
        if os.path.exists(task_path):
            save_task_metadata(task_path, task_name,
                               fixed_phase_num=fixed_phase_config.get(task_name))

    progress_snapshot = dict(progress)
    variation_stats_snapshot = dict(variation_stats)
    save_dataset_metadata(
        args.output_path,
        started_at,
        finished_at,
        args,
        task_names,
        progress_snapshot,
        variation_stats_snapshot,
    )

    summary_lines, detail_lines = summarize_collection(
        task_files,
        progress_snapshot,
        variation_stats_snapshot,
        started_at,
        finished_at,
    )
    print('')
    for line in summary_lines + detail_lines:
        print(line)
        append_log(log_file, log_lock, 'INFO', line)

    append_log(log_file, log_lock, 'INFO', 'collection done')
    write_progress_snapshot(
        args.progress_file,
        started_at,
        args,
        progress_snapshot,
        worker_state,
        variation_stats=variation_stats_snapshot,
        finished=True,
        log_file=log_file,
    )
    print(f'[Info] Log saved: {log_file}')


def main():
    args = parse_args()
    run_segmented_collection(args)


if __name__ == '__main__':
    main()

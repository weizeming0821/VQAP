# -*- coding: utf-8 -*-
"""
元数据管理模块

保存 dataset、variation 和 task 层级的元数据。
"""

import os
import json
import numpy as np


def save_variation_metadata(variation_path, variation_index, descriptions,
                            episode_stats, save_mode, signals,
                            generation_stats=None):
    """
    保存 variation 层级元数据。

    Args:
        variation_path:  str，variation 文件夹路径
        variation_index: int，变体序号
        descriptions:    list，该变体的语言描述列表
        episode_stats:   list，各 episode 的统计信息
        save_mode:       str，保存模式
        signals:         Iterable[str]，启用的信号集合
        generation_stats: dict，variation 级生成统计信息
    """
    num_episodes = len(episode_stats)
    phase_counts = [s.get('num_phases', 0) for s in episode_stats]
    valid_episodes = sum(1 for s in episode_stats if s.get('phase_valid', True))
    episode_summaries = []
    for stat in episode_stats:
        episode_index = int(stat.get('episode', -1))
        episode_summaries.append({
            'episode': f'episode{episode_index}',
            'num_phases': int(stat.get('num_phases', 0)),
            'phase_valid': bool(stat.get('phase_valid', True)),
            'phase_metadata_path': os.path.join('episodes', f'episode{episode_index}', 'phase_metadata.json'),
        })

    generation_stats = dict(generation_stats or {})
    metadata = {
        'variation_index': int(variation_index),
        'descriptions': descriptions,
        'save_mode': save_mode,
        'signals': list(signals) if signals else [],
        'planned_episodes': int(generation_stats.get('planned_demos', num_episodes)),
        'num_episodes': num_episodes,
        'valid_episodes': valid_episodes,
        'phase_counts': phase_counts,
        'avg_phases': float(np.mean(phase_counts)) if phase_counts else 0.0,
        'status': generation_stats.get('status', 'completed' if num_episodes == valid_episodes else 'partial_failed'),
        'generation_stats': {
            'success_demos': int(generation_stats.get('success_demos', num_episodes)),
            'phase_valid_demos': int(generation_stats.get('phase_valid_demos', valid_episodes)),
            'timeout_demos': int(generation_stats.get('timeout_demos', 0)),
            'failed_exception_demos': int(generation_stats.get('failed_exception_demos', 0)),
            'phase_invalid_demos': int(generation_stats.get('phase_invalid_demos', 0)),
            'failed_demos': int(generation_stats.get('failed_demos', 0)),
        },
        'episode_summaries': episode_summaries,
    }

    path = os.path.join(variation_path, 'variation_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_task_metadata(task_path, task_name, fixed_phase_num=None):
    """
    保存 task 层级元数据。

    Args:
        task_path:        str，task 文件夹路径
        task_name:        str，任务名称（snake_case）
        fixed_phase_num:  int，期望的阶段数（可选）
    """
    variation_folders = sorted([
        d for d in os.listdir(task_path)
        if d.startswith('variation') and os.path.isdir(os.path.join(task_path, d))
    ])

    total_episodes = 0
    all_phase_counts = []
    valid_episodes = 0

    for var_folder in variation_folders:
        var_meta_path = os.path.join(task_path, var_folder, 'variation_metadata.json')
        if os.path.exists(var_meta_path):
            with open(var_meta_path, 'r') as f:
                var_meta = json.load(f)
            phase_counts = var_meta.get('phase_counts', [])
            all_phase_counts.extend(phase_counts)
            total_episodes += var_meta.get('num_episodes', 0)
            valid_episodes += var_meta.get('valid_episodes', 0)

    task_class = ''.join(w.title() for w in task_name.split('_'))

    metadata = {
        'task_name': task_name,
        'task_class': task_class,
        'num_variations': len(variation_folders),
        'total_episodes': total_episodes,
        'valid_episodes': valid_episodes,
        'avg_phases': float(np.mean(all_phase_counts)) if all_phase_counts else 0.0,
        'total_phases': int(sum(all_phase_counts)),
    }
    if fixed_phase_num is not None:
        metadata['fixed_phase_num'] = fixed_phase_num

    path = os.path.join(task_path, 'task_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_dataset_metadata(output_path, started_at, finished_at, args,
                          task_names, progress, variation_stats):
    """
    保存数据集根目录级的全局元数据。

    Args:
        output_path: str，数据集根目录
        started_at: datetime，开始时间
        finished_at: datetime，结束时间
        args: argparse.Namespace，运行参数
        task_names: list[str]，本次处理的任务名
        progress: Mapping，本次全局进度统计
        variation_stats: Mapping，variation 级统计
    """
    os.makedirs(output_path, exist_ok=True)

    variation_values = list(variation_stats.values())
    phase_valid_episodes = sum(int(v.get('phase_valid_demos', 0)) for v in variation_values)
    phase_invalid_episodes = sum(int(v.get('phase_invalid_demos', 0)) for v in variation_values)
    total_variations = len(variation_values)

    signals = list(args.signals) if getattr(args, 'signals', None) else []
    metadata = {
        'started_at': started_at.isoformat(),
        'finished_at': finished_at.isoformat(),
        'duration_seconds': round((finished_at - started_at).total_seconds(), 3),
        'output_path': os.path.abspath(output_path),
        'tasks': task_names,
        'num_tasks': len(task_names),
        'num_variations': total_variations,
        'planned_episodes': int(progress.get('planned_episodes', 0)),
        'done_episodes': int(progress.get('done_episodes', 0)),
        'success_episodes': int(progress.get('success_episodes', 0)),
        'failed_episodes': int(progress.get('failed_episodes', 0)),
        'timeout_episodes': int(progress.get('timeout_episodes', 0)),
        'phase_invalid_episodes': phase_invalid_episodes,
        'phase_valid_episodes': phase_valid_episodes,
        'config': {
            'image_size': list(args.image_size),
            'renderer': args.renderer,
            'processes': int(args.processes),
            'episodes_per_task': int(args.episodes_per_task),
            'variations': int(args.variations),
            'arm_max_velocity': float(args.arm_max_velocity),
            'arm_max_acceleration': float(args.arm_max_acceleration),
            'demo_timeout': int(args.demo_timeout),
            'worker_stuck_timeout': int(args.worker_stuck_timeout),
            'min_phase_len': int(args.min_phase_len),
            'save_mode': args.save_mode,
            'fixed_phase_csv': args.fixed_phase_csv,
            'signals': signals,
        },
    }

    path = os.path.join(output_path, 'dataset_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

# -*- coding: utf-8 -*-
"""
元数据管理模块

保存 variation 和 task 层级的元数据。
"""

import os
import json
import pickle
import numpy as np

from rlbench.backend.const import VARIATION_DESCRIPTIONS


def save_variation_metadata(variation_path, variation_index, descriptions,
                            episode_stats):
    """
    保存 variation 层级元数据。

    Args:
        variation_path:  str，variation 文件夹路径
        variation_index: int，变体序号
        descriptions:    list，该变体的语言描述列表
        episode_stats:   list，各 episode 的统计信息
    """
    num_episodes = len(episode_stats)
    phase_counts = [s.get('num_phases', 0) for s in episode_stats]
    success_phase_counts = [s.get('num_phases', 0) for s in episode_stats
                            if s.get('phase_valid', True)]

    metadata = {
        'variation_index': variation_index,
        'variation_descriptions': descriptions,
        'num_episodes': num_episodes,
        'phase_counts': phase_counts,
        'avg_phases': float(np.mean(phase_counts)) if phase_counts else 0,
        'success_episodes': len(success_phase_counts),
        'episode_stats': episode_stats,
    }
    path = os.path.join(variation_path, 'variation_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_descriptions(variation_path, descriptions):
    """保存 variation 描述文件。"""
    with open(os.path.join(variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
        pickle.dump(descriptions, f)


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
            valid_episodes += var_meta.get('success_episodes', 0)

    task_class = ''.join(w.title() for w in task_name.split('_'))

    metadata = {
        'task_name': task_name,
        'task_class': task_class,
        'num_variations': len(variation_folders),
        'total_episodes': total_episodes,
        'valid_episodes': valid_episodes,
        'avg_phases': float(np.mean(all_phase_counts)) if all_phase_counts else 0,
        'total_phases': int(sum(all_phase_counts)),
    }
    if fixed_phase_num is not None:
        metadata['fixed_phase_num'] = fixed_phase_num

    path = os.path.join(task_path, 'task_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_split_summary(var_out_path, task, var_dir, descriptions,
                       save_mode, signals, episode_summaries):
    """
    保存 variation 层级的分割摘要。

    Args:
        var_out_path:     str，variation 输出路径
        task:             str，任务名
        var_dir:          str，variation 目录名
        descriptions:     list，任务描述
        save_mode:        str，保存模式
        signals:          set，启用的信号
        episode_summaries: list，episode 摘要列表
    """
    os.makedirs(var_out_path, exist_ok=True)
    with open(os.path.join(var_out_path, 'split_summary.json'), 'w') as f:
        json.dump({
            'task': task,
            'variation': var_dir,
            'descriptions': descriptions,
            'save_mode': save_mode,
            'signals': list(signals) if signals else [],
            'episodes': episode_summaries,
        }, f, ensure_ascii=False, indent=2)

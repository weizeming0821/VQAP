# -*- coding: utf-8 -*-
"""
元数据管理模块

保存 dataset、variation 和 task 层级的元数据。
"""

import os
import json
import numpy as np


def _load_json_if_exists(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as file_obj:
            return json.load(file_obj)
    except Exception:
        return None


def _parse_episode_index(value):
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.startswith('episode'):
        text = text[len('episode'):]
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _phase_count_from_metadata(phase_metadata):
    num_phases = phase_metadata.get('num_phases')
    if num_phases is None:
        num_phases = len(list(phase_metadata.get('phases', [])))
    return int(num_phases)


def _variation_index_from_path(variation_path, fallback=0):
    variation_name = os.path.basename(os.path.normpath(variation_path))
    if variation_name.startswith('variation'):
        try:
            return int(variation_name[len('variation'):])
        except (TypeError, ValueError):
            pass
    return int(fallback)


def _sorted_episode_entries(episodes_root):
    episode_entries = []
    if not os.path.isdir(episodes_root):
        return episode_entries
    for entry in os.listdir(episodes_root):
        episode_index = _parse_episode_index(entry)
        if episode_index is None:
            continue
        episode_entries.append((entry, int(episode_index)))
    return sorted(episode_entries, key=lambda item: item[1])


def _iter_task_names(output_path):
    if not os.path.isdir(output_path):
        return []
    task_names = []
    for entry in sorted(os.listdir(output_path)):
        task_path = os.path.join(output_path, entry)
        if entry.startswith('.') or not os.path.isdir(task_path):
            continue
        variation_entries = [
            child for child in os.listdir(task_path)
            if child.startswith('variation') and os.path.isdir(os.path.join(task_path, child))
        ]
        if variation_entries or os.path.exists(os.path.join(task_path, 'task_metadata.json')):
            task_names.append(entry)
    return task_names


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
        summary = {
            'episode': f'episode{episode_index}',
            'num_phases': int(stat.get('num_phases', 0)),
            'phase_valid': bool(stat.get('phase_valid', True)),
            'phase_metadata_path': os.path.join('episodes', f'episode{episode_index}', 'phase_metadata.json'),
        }
        requested_episode = stat.get('requested_episode')
        if requested_episode is not None:
            summary['requested_episode'] = int(requested_episode)
        episode_summaries.append(summary)

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
            'demo_timeout_demos': int(generation_stats.get('demo_timeout_demos', 0)),
            'watchdog_timeout_demos': int(generation_stats.get('watchdog_timeout_demos', 0)),
            'exception_demos': int(generation_stats.get('exception_demos', 0)),
            'phase_invalid_attempts': int(generation_stats.get('phase_invalid_attempts', 0)),
            'phase_invalid_demos': int(generation_stats.get('phase_invalid_demos', 0)),
            'aborted_demos': int(generation_stats.get('aborted_demos', 0)),
            'failed_demos': int(generation_stats.get('failed_demos', 0)),
        },
        'episode_summaries': episode_summaries,
    }

    path = os.path.join(variation_path, 'variation_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def normalize_variation_episode_indices(variation_path, dry_run=False):
    variation_path = os.path.abspath(variation_path)
    episodes_root = os.path.join(variation_path, 'episodes')
    variation_metadata = _load_json_if_exists(os.path.join(variation_path, 'variation_metadata.json')) or {}

    episode_records = []
    seen_saved_episode_indices = set()
    next_requested_episode = 0
    metadata_requested_missing = False

    for summary in list(variation_metadata.get('episode_summaries', [])):
        saved_episode_index = _parse_episode_index(summary.get('episode'))
        if saved_episode_index is None or saved_episode_index in seen_saved_episode_indices:
            continue
        phase_metadata = _load_json_if_exists(
            os.path.join(episodes_root, f'episode{saved_episode_index}', 'phase_metadata.json')
        )
        if phase_metadata is None:
            relative_phase_path = summary.get('phase_metadata_path')
            if relative_phase_path:
                phase_metadata = _load_json_if_exists(os.path.join(variation_path, relative_phase_path))
        if phase_metadata is None:
            continue

        requested_episode = _parse_episode_index(summary.get('requested_episode'))
        if requested_episode is None:
            requested_episode = next_requested_episode
            metadata_requested_missing = True
        next_requested_episode = max(next_requested_episode, int(requested_episode) + 1)

        episode_records.append({
            'saved_episode': int(saved_episode_index),
            'requested_episode': int(requested_episode),
            'num_phases': _phase_count_from_metadata(phase_metadata),
            'phase_valid': bool(summary.get('phase_valid', True)),
        })
        seen_saved_episode_indices.add(int(saved_episode_index))

    for entry, saved_episode_index in _sorted_episode_entries(episodes_root):
        if saved_episode_index in seen_saved_episode_indices:
            continue
        phase_metadata = _load_json_if_exists(os.path.join(episodes_root, entry, 'phase_metadata.json'))
        if phase_metadata is None:
            continue
        episode_records.append({
            'saved_episode': int(saved_episode_index),
            'requested_episode': int(next_requested_episode),
            'num_phases': _phase_count_from_metadata(phase_metadata),
            'phase_valid': True,
        })
        seen_saved_episode_indices.add(int(saved_episode_index))
        next_requested_episode += 1

    expected_rename_count = sum(
        1
        for new_episode_index, record in enumerate(episode_records)
        if int(record['saved_episode']) != int(new_episode_index)
    )
    rename_count = 0
    if not dry_run and os.path.isdir(episodes_root):
        temp_paths = []
        for new_episode_index, record in enumerate(episode_records):
            old_episode_index = int(record['saved_episode'])
            if old_episode_index == new_episode_index:
                continue
            old_path = os.path.join(episodes_root, f'episode{old_episode_index}')
            if not os.path.isdir(old_path):
                continue
            temp_path = os.path.join(
                episodes_root,
                f'.episode_reindex_tmp_{old_episode_index}_{new_episode_index}',
            )
            os.replace(old_path, temp_path)
            temp_paths.append((temp_path, os.path.join(episodes_root, f'episode{new_episode_index}')))
            rename_count += 1
        for temp_path, final_path in temp_paths:
            os.replace(temp_path, final_path)

    metadata_rewritten = bool(episode_records) and (
        expected_rename_count > 0
        or metadata_requested_missing
        or len(episode_records) != len(list(variation_metadata.get('episode_summaries', [])))
    )
    if not dry_run and episode_records:
        variation_index = _variation_index_from_path(
            variation_path,
            variation_metadata.get('variation_index', 0),
        )
        generation_stats = dict(variation_metadata.get('generation_stats', {}))
        if 'planned_demos' not in generation_stats and 'planned_episodes' in variation_metadata:
            generation_stats['planned_demos'] = int(variation_metadata.get('planned_episodes', len(episode_records)))
        if variation_metadata.get('status') is not None:
            generation_stats.setdefault('status', variation_metadata.get('status'))
        save_variation_metadata(
            variation_path,
            variation_index,
            list(variation_metadata.get('descriptions', [])),
            [
                {
                    'episode': int(new_episode_index),
                    'requested_episode': int(record['requested_episode']),
                    'num_phases': int(record['num_phases']),
                    'phase_valid': bool(record.get('phase_valid', True)),
                }
                for new_episode_index, record in enumerate(episode_records)
            ],
            variation_metadata.get('save_mode', 'keyframe_only'),
            list(variation_metadata.get('signals', [])),
            generation_stats=generation_stats,
        )

    return {
        'changed': bool(expected_rename_count > 0 or metadata_rewritten),
        'episode_count': len(episode_records),
        'rename_count': int(expected_rename_count if dry_run else rename_count),
        'max_saved_episode_index': max(
            (int(record['saved_episode']) for record in episode_records),
            default=-1,
        ),
    }


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
    planned_episodes = sum(int(v.get('planned_demos', 0)) for v in variation_values)
    success_episodes = sum(int(v.get('success_demos', 0)) for v in variation_values)
    failed_episodes = sum(int(v.get('failed_demos', 0)) for v in variation_values)
    demo_timeout_episodes = sum(int(v.get('demo_timeout_demos', 0)) for v in variation_values)
    watchdog_timeout_episodes = sum(int(v.get('watchdog_timeout_demos', 0)) for v in variation_values)
    exception_episodes = sum(int(v.get('exception_demos', 0)) for v in variation_values)
    phase_invalid_attempts = sum(int(v.get('phase_invalid_attempts', 0)) for v in variation_values)
    phase_valid_episodes = sum(int(v.get('phase_valid_demos', 0)) for v in variation_values)
    phase_invalid_episodes = sum(int(v.get('phase_invalid_demos', 0)) for v in variation_values)
    aborted_episodes = sum(int(v.get('aborted_demos', 0)) for v in variation_values)
    total_variations = len(variation_values)
    done_episodes = success_episodes + failed_episodes

    signals = list(args.signals) if getattr(args, 'signals', None) else []
    metadata = {
        'started_at': started_at.isoformat(),
        'finished_at': finished_at.isoformat(),
        'duration_seconds': round((finished_at - started_at).total_seconds(), 3),
        'output_path': os.path.abspath(output_path),
        'tasks': task_names,
        'num_tasks': len(task_names),
        'num_variations': total_variations,
        'planned_episodes': planned_episodes,
        'done_episodes': done_episodes,
        'success_episodes': success_episodes,
        'failed_episodes': failed_episodes,
        'timeout_episodes': demo_timeout_episodes + watchdog_timeout_episodes,
        'demo_timeout_episodes': demo_timeout_episodes,
        'watchdog_timeout_episodes': watchdog_timeout_episodes,
        'exception_episodes': exception_episodes,
        'phase_invalid_attempts': phase_invalid_attempts,
        'phase_invalid_episodes': phase_invalid_episodes,
        'aborted_episodes': aborted_episodes,
        'phase_valid_episodes': phase_valid_episodes,
        'config': {
            'image_size': list(args.image_size),
            'renderer': args.renderer,
            'processes': int(args.processes),
            'episodes_per_task': int(args.episodes_per_task),
            'variations': int(args.variations),
            'variation_index': list(getattr(args, 'variation_index', []) or []),
            'arm_max_velocity': float(args.arm_max_velocity),
            'arm_max_acceleration': float(args.arm_max_acceleration),
            'demo_timeout': int(args.demo_timeout),
            'min_phase_len': int(args.min_phase_len),
            'save_mode': args.save_mode,
            'fixed_phase_csv': args.fixed_phase_csv,
            'signals': signals,
        },
    }

    path = os.path.join(output_path, 'dataset_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_dataset_metadata_from_disk(output_path, started_at, finished_at, args, extra_metadata=None):
    os.makedirs(output_path, exist_ok=True)

    task_names = _iter_task_names(output_path)
    planned_episodes = 0
    success_episodes = 0
    failed_episodes = 0
    demo_timeout_episodes = 0
    watchdog_timeout_episodes = 0
    exception_episodes = 0
    phase_invalid_attempts = 0
    phase_invalid_episodes = 0
    aborted_episodes = 0
    phase_valid_episodes = 0
    total_variations = 0

    for task_name in task_names:
        task_path = os.path.join(output_path, task_name)
        for entry in sorted(os.listdir(task_path)):
            variation_path = os.path.join(task_path, entry)
            if not entry.startswith('variation') or not os.path.isdir(variation_path):
                continue
            variation_meta = _load_json_if_exists(os.path.join(variation_path, 'variation_metadata.json'))
            if variation_meta is None:
                continue
            generation_stats = dict(variation_meta.get('generation_stats', {}))
            total_variations += 1
            planned_episodes += int(variation_meta.get('planned_episodes', generation_stats.get('planned_demos', 0)))
            success_episodes += int(generation_stats.get('success_demos', variation_meta.get('num_episodes', 0)))
            failed_episodes += int(generation_stats.get('failed_demos', 0))
            demo_timeout_episodes += int(generation_stats.get('demo_timeout_demos', 0))
            watchdog_timeout_episodes += int(generation_stats.get('watchdog_timeout_demos', 0))
            exception_episodes += int(generation_stats.get('exception_demos', 0))
            phase_invalid_attempts += int(generation_stats.get('phase_invalid_attempts', 0))
            phase_invalid_episodes += int(generation_stats.get('phase_invalid_demos', 0))
            aborted_episodes += int(generation_stats.get('aborted_demos', 0))
            phase_valid_episodes += int(generation_stats.get('phase_valid_demos', variation_meta.get('valid_episodes', 0)))

    signals = list(args.signals) if getattr(args, 'signals', None) else []
    metadata = {
        'started_at': started_at.isoformat(),
        'finished_at': finished_at.isoformat(),
        'duration_seconds': round((finished_at - started_at).total_seconds(), 3),
        'output_path': os.path.abspath(output_path),
        'tasks': task_names,
        'num_tasks': len(task_names),
        'num_variations': int(total_variations),
        'planned_episodes': int(planned_episodes),
        'done_episodes': int(success_episodes + failed_episodes),
        'success_episodes': int(success_episodes),
        'failed_episodes': int(failed_episodes),
        'timeout_episodes': int(demo_timeout_episodes + watchdog_timeout_episodes),
        'demo_timeout_episodes': int(demo_timeout_episodes),
        'watchdog_timeout_episodes': int(watchdog_timeout_episodes),
        'exception_episodes': int(exception_episodes),
        'phase_invalid_attempts': int(phase_invalid_attempts),
        'phase_invalid_episodes': int(phase_invalid_episodes),
        'aborted_episodes': int(aborted_episodes),
        'phase_valid_episodes': int(phase_valid_episodes),
        'config': {
            'image_size': list(getattr(args, 'image_size', [])),
            'renderer': getattr(args, 'renderer', None),
            'processes': int(getattr(args, 'processes', 0)),
            'episodes_per_task': int(getattr(args, 'episodes_per_task', 0)),
            'variations': int(getattr(args, 'variations', 0)),
            'variation_index': list(getattr(args, 'variation_index', []) or []),
            'arm_max_velocity': float(getattr(args, 'arm_max_velocity', 0.0)),
            'arm_max_acceleration': float(getattr(args, 'arm_max_acceleration', 0.0)),
            'demo_timeout': int(getattr(args, 'demo_timeout', 0)),
            'min_phase_len': int(getattr(args, 'min_phase_len', 0)),
            'save_mode': getattr(args, 'save_mode', None),
            'fixed_phase_csv': getattr(args, 'fixed_phase_csv', None),
            'signals': signals,
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    path = os.path.join(output_path, 'dataset_metadata.json')
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    return metadata

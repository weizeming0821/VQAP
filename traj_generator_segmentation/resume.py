# -*- coding: utf-8 -*-
"""断点续跑辅助工具。"""

import json
import os
import shutil
from datetime import datetime

from rlbench.backend.const import VARIATIONS_FOLDER


def _planned_variation_indices(task_variation_targets, task_name):
    target = task_variation_targets.get(task_name, [])
    if isinstance(target, int):
        return list(range(max(0, int(target))))
    if not target:
        return []
    return [int(index) for index in target]


def append_timestamped_log(log_path, message):
    if not log_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as file_obj:
        file_obj.write(f'[{timestamp}] {message}\n')


def load_json_if_exists(path):
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


def _next_available_episode_index(used_episode_indices):
    candidate = 0
    used_indices = {int(index) for index in used_episode_indices if int(index) >= 0}
    while candidate in used_indices:
        candidate += 1
    return candidate


def _load_phase_metadata(variation_path, relative_phase_path, episode_index):
    candidate_paths = []
    if relative_phase_path:
        candidate_paths.append(os.path.join(variation_path, relative_phase_path))
    if episode_index is not None:
        candidate_paths.append(
            os.path.join(variation_path, 'episodes', f'episode{int(episode_index)}', 'phase_metadata.json')
        )
    for candidate_path in candidate_paths:
        metadata = load_json_if_exists(candidate_path)
        if metadata is not None:
            return metadata
    return None


def load_existing_variation_payload(variation_path):
    metadata_path = os.path.join(variation_path, 'variation_metadata.json')
    metadata = load_json_if_exists(metadata_path) or {}
    descriptions = list(metadata.get('descriptions', []))

    episode_stats = []
    seen_episodes = set()
    next_requested_episode = 0
    for summary in list(metadata.get('episode_summaries', [])):
        episode_index = _parse_episode_index(summary.get('episode'))
        phase_metadata = _load_phase_metadata(
            variation_path,
            summary.get('phase_metadata_path'),
            episode_index,
        )
        if episode_index is None or phase_metadata is None:
            continue
        num_phases = phase_metadata.get('num_phases')
        if num_phases is None:
            num_phases = len(list(phase_metadata.get('phases', [])))
        requested_episode = _parse_episode_index(summary.get('requested_episode'))
        if requested_episode is None:
            requested_episode = next_requested_episode
        next_requested_episode = max(next_requested_episode, int(requested_episode) + 1)
        episode_stats.append({
            'episode': int(episode_index),
            'requested_episode': int(requested_episode),
            'num_phases': int(num_phases),
            'phase_valid': bool(summary.get('phase_valid', True)),
        })
        seen_episodes.add(int(episode_index))

    episodes_root = os.path.join(variation_path, 'episodes')
    if os.path.isdir(episodes_root):
        for entry in sorted(os.listdir(episodes_root)):
            episode_path = os.path.join(episodes_root, entry)
            if not entry.startswith('episode') or not os.path.isdir(episode_path):
                continue
            episode_index = _parse_episode_index(entry)
            if episode_index is None or int(episode_index) in seen_episodes:
                continue
            phase_metadata = load_json_if_exists(os.path.join(episode_path, 'phase_metadata.json'))
            if phase_metadata is None:
                continue
            num_phases = phase_metadata.get('num_phases')
            if num_phases is None:
                num_phases = len(list(phase_metadata.get('phases', [])))
            episode_stats.append({
                'episode': int(episode_index),
                'requested_episode': int(next_requested_episode),
                'num_phases': int(num_phases),
                'phase_valid': True,
            })
            seen_episodes.add(int(episode_index))
            next_requested_episode += 1

    episode_stats.sort(key=lambda item: (
        int(item.get('requested_episode', -1)),
        int(item.get('episode', -1)),
    ))
    next_episode_index = _next_available_episode_index(seen_episodes)

    return {
        'metadata': metadata,
        'descriptions': descriptions,
        'episode_stats': episode_stats,
        'next_episode_index': int(next_episode_index),
    }


def is_variation_complete(variation_path, planned_episodes):
    metadata_path = os.path.join(variation_path, 'variation_metadata.json')
    metadata = load_json_if_exists(metadata_path)
    if not metadata:
        return False, None

    if int(metadata.get('planned_episodes', planned_episodes)) != int(planned_episodes):
        return False, metadata
    if int(metadata.get('num_episodes', 0)) != int(planned_episodes):
        return False, metadata
    if metadata.get('status') != 'completed':
        return False, metadata

    generation_stats = dict(metadata.get('generation_stats', {}))
    if int(generation_stats.get('success_demos', 0)) < int(planned_episodes):
        return False, metadata
    if int(generation_stats.get('failed_demos', 0)) != 0:
        return False, metadata

    episode_summaries = list(metadata.get('episode_summaries', []))
    if len(episode_summaries) != int(planned_episodes):
        return False, metadata
    for summary in episode_summaries:
        relative_phase_path = summary.get('phase_metadata_path')
        if not relative_phase_path:
            return False, metadata
        phase_metadata_path = os.path.join(variation_path, relative_phase_path)
        if not os.path.exists(phase_metadata_path):
            return False, metadata

    return True, metadata


def build_variation_stats_from_metadata(task_name, variation_index, planned_episodes, metadata):
    generation_stats = dict(metadata.get('generation_stats', {}))
    success_demos = int(generation_stats.get('success_demos', metadata.get('num_episodes', 0)))
    phase_valid_demos = int(generation_stats.get('phase_valid_demos', metadata.get('valid_episodes', 0)))
    demo_timeout_demos = int(generation_stats.get('demo_timeout_demos', 0))
    watchdog_timeout_demos = int(generation_stats.get('watchdog_timeout_demos', 0))
    exception_demos = int(generation_stats.get('exception_demos', 0))
    phase_invalid_attempts = int(generation_stats.get('phase_invalid_attempts', 0))
    phase_invalid_demos = int(generation_stats.get('phase_invalid_demos', 0))
    aborted_demos = int(generation_stats.get('aborted_demos', 0))
    failed_demos = int(generation_stats.get(
        'failed_demos',
        demo_timeout_demos + watchdog_timeout_demos + exception_demos + phase_invalid_demos + aborted_demos,
    ))
    status = metadata.get('status', 'completed' if failed_demos == 0 else 'partial_failed')

    return {
        'task_name': task_name,
        'variation_index': int(variation_index),
        'planned_demos': int(planned_episodes),
        'success_demos': success_demos,
        'phase_valid_demos': phase_valid_demos,
        'demo_timeout_demos': demo_timeout_demos,
        'watchdog_timeout_demos': watchdog_timeout_demos,
        'exception_demos': exception_demos,
        'phase_invalid_attempts': phase_invalid_attempts,
        'phase_invalid_demos': phase_invalid_demos,
        'aborted_demos': aborted_demos,
        'failed_demos': failed_demos,
        'failure_details': [],
        'status': status,
    }


def build_variation_stats_for_complete(task_name, variation_index, planned_episodes, payload):
    episode_stats = list(payload.get('episode_stats', []))
    success_demos = len(episode_stats)
    phase_valid_demos = sum(1 for stat in episode_stats if stat.get('phase_valid', True))
    status = 'completed' if success_demos >= int(planned_episodes) else 'in_progress'
    return {
        'task_name': task_name,
        'variation_index': int(variation_index),
        'planned_demos': int(planned_episodes),
        'success_demos': int(success_demos),
        'phase_valid_demos': int(phase_valid_demos),
        'demo_timeout_demos': 0,
        'watchdog_timeout_demos': 0,
        'exception_demos': 0,
        'phase_invalid_attempts': 0,
        'phase_invalid_demos': 0,
        'aborted_demos': 0,
        'failed_demos': 0,
        'failure_details': [],
        'status': status,
    }


def build_progress_from_variation_stats(variation_stats):
    progress = {
        'planned_episodes': 0,
        'done_episodes': 0,
        'success_episodes': 0,
        'failed_episodes': 0,
        'demo_timeout_episodes': 0,
        'watchdog_timeout_episodes': 0,
        'exception_episodes': 0,
        'phase_invalid_episodes': 0,
        'aborted_episodes': 0,
    }
    for stats in variation_stats.values():
        progress['planned_episodes'] += int(stats.get('planned_demos', 0))
        progress['success_episodes'] += int(stats.get('success_demos', 0))
        progress['failed_episodes'] += int(stats.get('failed_demos', 0))
        progress['demo_timeout_episodes'] += int(stats.get('demo_timeout_demos', 0))
        progress['watchdog_timeout_episodes'] += int(stats.get('watchdog_timeout_demos', 0))
        progress['exception_episodes'] += int(stats.get('exception_demos', 0))
        progress['phase_invalid_episodes'] += int(stats.get('phase_invalid_demos', 0))
        progress['aborted_episodes'] += int(stats.get('aborted_demos', 0))
    progress['done_episodes'] = progress['success_episodes'] + progress['failed_episodes']
    return progress


def inspect_existing_variations(output_path, task_names, task_variation_targets, planned_episodes,
                                reset_incomplete=False, log_message=None):
    completed_variations = {task_name: set() for task_name in task_names}
    variation_stats = {}
    reset_variations = []
    incomplete_variations = []

    for task_name in task_names:
        task_path = os.path.join(output_path, task_name)
        if not os.path.isdir(task_path):
            continue

        for variation_index in _planned_variation_indices(task_variation_targets, task_name):
            variation_path = os.path.join(task_path, VARIATIONS_FOLDER % variation_index)
            if not os.path.isdir(variation_path):
                continue

            is_complete, metadata = is_variation_complete(variation_path, planned_episodes)
            if is_complete:
                completed_variations[task_name].add(int(variation_index))
                variation_stats[f'{task_name}::{variation_index}'] = build_variation_stats_from_metadata(
                    task_name,
                    variation_index,
                    planned_episodes,
                    metadata,
                )
                continue

            incomplete_variations.append((task_name, int(variation_index), variation_path))
            if reset_incomplete:
                shutil.rmtree(variation_path, ignore_errors=True)
                reset_variations.append((task_name, int(variation_index), variation_path))
                if callable(log_message):
                    log_message(
                        f'reset incomplete variation task={task_name} variation={variation_index} '
                        f'path={variation_path}')

    return {
        'completed_variations': completed_variations,
        'variation_stats': variation_stats,
        'progress': build_progress_from_variation_stats(variation_stats),
        'reset_variations': reset_variations,
        'incomplete_variations': incomplete_variations,
    }


def inspect_existing_variations_for_complete(output_path, task_names, task_variation_targets,
                                            planned_episodes, log_message=None):
    completed_variations = {task_name: set() for task_name in task_names}
    variation_stats = {}
    incomplete_variations = []

    for task_name in task_names:
        task_path = os.path.join(output_path, task_name)
        if not os.path.isdir(task_path):
            continue

        for variation_index in _planned_variation_indices(task_variation_targets, task_name):
            variation_path = os.path.join(task_path, VARIATIONS_FOLDER % variation_index)
            if not os.path.isdir(variation_path):
                continue

            payload = load_existing_variation_payload(variation_path)
            recovered_episodes = len(payload.get('episode_stats', []))
            if recovered_episodes <= 0:
                incomplete_variations.append((task_name, int(variation_index), variation_path))
                if callable(log_message):
                    log_message(
                        f'complete mode detected variation without recoverable episodes '
                        f'task={task_name} variation={variation_index} path={variation_path}'
                    )
                continue

            stats = build_variation_stats_for_complete(
                task_name,
                variation_index,
                planned_episodes,
                payload,
            )
            variation_stats[f'{task_name}::{variation_index}'] = stats
            if recovered_episodes >= int(planned_episodes):
                completed_variations[task_name].add(int(variation_index))
            else:
                incomplete_variations.append((task_name, int(variation_index), variation_path))
                if callable(log_message):
                    log_message(
                        f'complete mode will append variation task={task_name} '
                        f'variation={variation_index} existing_episodes={recovered_episodes} '
                        f'planned_episodes={planned_episodes} path={variation_path}'
                    )

    return {
        'completed_variations': completed_variations,
        'variation_stats': variation_stats,
        'progress': build_progress_from_variation_stats(variation_stats),
        'reset_variations': [],
        'incomplete_variations': incomplete_variations,
    }


def merge_task_directory_contents(source_task_path, destination_task_path, log_message=None):
    if not os.path.isdir(source_task_path):
        return
    os.makedirs(destination_task_path, exist_ok=True)

    for entry in sorted(os.listdir(source_task_path)):
        source_entry = os.path.join(source_task_path, entry)
        destination_entry = os.path.join(destination_task_path, entry)

        if not os.path.exists(destination_entry):
            shutil.move(source_entry, destination_entry)
            continue

        if not entry.startswith('variation') or not os.path.isdir(source_entry):
            if callable(log_message):
                log_message(
                    f'skip duplicate task entry entry={entry} source={source_entry} '
                    f'destination={destination_entry}')
            if os.path.isdir(source_entry):
                shutil.rmtree(source_entry, ignore_errors=True)
            else:
                try:
                    os.remove(source_entry)
                except OSError:
                    pass
            continue

        source_complete, source_metadata = is_variation_complete(
            source_entry,
            int(load_json_if_exists(os.path.join(source_entry, 'variation_metadata.json')) or {}
                .get('planned_episodes', 0) or 0),
        )
        destination_complete, destination_metadata = is_variation_complete(
            destination_entry,
            int(load_json_if_exists(os.path.join(destination_entry, 'variation_metadata.json')) or {}
                .get('planned_episodes', 0) or 0),
        )
        source_mtime = os.path.getmtime(source_entry)
        destination_mtime = os.path.getmtime(destination_entry)

        replace_destination = False
        if source_complete and not destination_complete:
            replace_destination = True
        elif not source_complete and destination_complete:
            replace_destination = False
        elif source_complete and destination_complete:
            replace_destination = source_mtime > destination_mtime
        else:
            replace_destination = source_mtime > destination_mtime

        if replace_destination:
            shutil.rmtree(destination_entry, ignore_errors=True)
            shutil.move(source_entry, destination_entry)
            if callable(log_message):
                log_message(
                    f'replace duplicate variation destination={destination_entry} source={source_entry}')
        else:
            shutil.rmtree(source_entry, ignore_errors=True)
            if callable(log_message):
                log_message(
                    f'keep existing variation destination={destination_entry} discard={source_entry}')

    if os.path.isdir(source_task_path) and not os.listdir(source_task_path):
        try:
            os.rmdir(source_task_path)
        except OSError:
            pass
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验 segmented dataset 的编号与 metadata 一致性。"""

import argparse
import json
import math
import os
import sys


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def _isclose(left, right, tol=1e-6):
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def _parse_prefixed_index(name, prefix):
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _is_task_dir(path):
    if not os.path.isdir(path):
        return False
    entries = os.listdir(path)
    if 'task_metadata.json' in entries:
        return True
    return any(_parse_prefixed_index(entry, 'variation') is not None for entry in entries)


class ValidationCollector:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)


def _check_contiguous(indices):
    if not indices:
        return True
    expected = list(range(min(indices), max(indices) + 1))
    return list(indices) == expected


def _validate_phase_metadata(episode_path, collector):
    phase_meta_path = os.path.join(episode_path, 'phase_metadata.json')
    if not os.path.exists(phase_meta_path):
        collector.error(f'Missing phase_metadata.json: {phase_meta_path}')
        return None

    phase_meta = _load_json(phase_meta_path)
    num_phases = int(phase_meta.get('num_phases', -1))
    phases = phase_meta.get('phases', [])
    keyframe_inds = phase_meta.get('keyframe_inds', [])

    if num_phases != len(phases):
        collector.error(
            f'phase_metadata num_phases mismatch: {phase_meta_path} '
            f'num_phases={num_phases} len(phases)={len(phases)}')

    if num_phases != len(keyframe_inds):
        collector.error(
            f'phase_metadata keyframe count mismatch: {phase_meta_path} '
            f'num_phases={num_phases} len(keyframe_inds)={len(keyframe_inds)}')

    phase_indices = []
    for position, phase in enumerate(phases):
        phase_index = int(phase.get('phase_index', -1))
        phase_indices.append(phase_index)
        if phase_index != position:
            collector.error(
                f'Non-sequential phase_index in {phase_meta_path}: '
                f'expected={position} actual={phase_index}')
        if int(phase.get('length', 0)) <= 0:
            collector.error(f'Non-positive phase length in {phase_meta_path}: phase_index={phase_index}')

    return {
        'path': phase_meta_path,
        'num_phases': num_phases,
        'phase_indices': phase_indices,
    }


def _validate_variation_dir(task_name, variation_path, collector):
    variation_name = os.path.basename(variation_path)
    variation_index = _parse_prefixed_index(variation_name, 'variation')
    if variation_index is None:
        collector.error(f'Invalid variation directory name: {variation_path}')
        return None

    variation_meta_path = os.path.join(variation_path, 'variation_metadata.json')
    if not os.path.exists(variation_meta_path):
        collector.error(f'Missing variation_metadata.json: {variation_meta_path}')
        return None

    variation_meta = _load_json(variation_meta_path)
    if int(variation_meta.get('variation_index', -1)) != variation_index:
        collector.error(
            f'variation_index mismatch in {variation_meta_path}: '
            f'dir={variation_index} meta={variation_meta.get("variation_index")}')

    episodes_path = os.path.join(variation_path, 'episodes')
    if not os.path.isdir(episodes_path):
        collector.error(f'Missing episodes directory: {episodes_path}')
        return None

    episode_dirs = []
    episode_indices = []
    for entry in sorted(os.listdir(episodes_path)):
        episode_index = _parse_prefixed_index(entry, 'episode')
        if episode_index is None:
            continue
        episode_dir_path = os.path.join(episodes_path, entry)
        if os.path.isdir(episode_dir_path):
            episode_dirs.append((entry, episode_dir_path, episode_index))
            episode_indices.append(episode_index)

    if episode_indices and not _check_contiguous(episode_indices):
        collector.warning(
            f'Non-contiguous episode ids in {variation_path}: {episode_indices}. '
            'This can happen when some episodes fail after retries.')

    episode_summaries = variation_meta.get('episode_summaries', [])
    summary_map = {}
    for item in episode_summaries:
        episode_name = item.get('episode')
        if episode_name in summary_map:
            collector.error(f'Duplicate episode summary in {variation_meta_path}: {episode_name}')
            continue
        summary_map[episode_name] = item

    actual_episode_names = [item[0] for item in episode_dirs]
    actual_episode_set = set(actual_episode_names)
    summary_episode_set = set(summary_map.keys())
    missing_summary = sorted(actual_episode_set - summary_episode_set)
    extra_summary = sorted(summary_episode_set - actual_episode_set)
    if missing_summary:
        collector.error(
            f'Episode directories missing from episode_summaries in {variation_meta_path}: '
            + ', '.join(missing_summary))
    if extra_summary:
        collector.error(
            f'Episode summaries missing directories in {variation_meta_path}: '
            + ', '.join(extra_summary))

    actual_phase_counts = []
    valid_episode_count = 0
    for episode_name, episode_dir_path, episode_index in episode_dirs:
        phase_info = _validate_phase_metadata(episode_dir_path, collector)
        if phase_info is None:
            continue

        actual_phase_counts.append(int(phase_info['num_phases']))
        summary = summary_map.get(episode_name)
        if summary is None:
            continue

        if bool(summary.get('phase_valid', False)):
            valid_episode_count += 1

        if int(summary.get('num_phases', -1)) != int(phase_info['num_phases']):
            collector.error(
                f'num_phases mismatch for {episode_name} in {variation_meta_path}: '
                f'summary={summary.get("num_phases")} actual={phase_info["num_phases"]}')

        expected_rel_path = os.path.join('episodes', episode_name, 'phase_metadata.json')
        if summary.get('phase_metadata_path') != expected_rel_path:
            collector.error(
                f'phase_metadata_path mismatch for {episode_name} in {variation_meta_path}: '
                f'expected={expected_rel_path} actual={summary.get("phase_metadata_path")}')

        if episode_index != _parse_prefixed_index(episode_name, 'episode'):
            collector.error(
                f'Episode directory/index mismatch in {variation_path}: '
                f'dir={episode_index} summary={episode_name}')

    if int(variation_meta.get('num_episodes', -1)) != len(episode_dirs):
        collector.error(
            f'num_episodes mismatch in {variation_meta_path}: '
            f'meta={variation_meta.get("num_episodes")} actual={len(episode_dirs)}')

    if int(variation_meta.get('valid_episodes', -1)) != valid_episode_count:
        collector.error(
            f'valid_episodes mismatch in {variation_meta_path}: '
            f'meta={variation_meta.get("valid_episodes")} actual={valid_episode_count}')

    phase_counts = [int(value) for value in variation_meta.get('phase_counts', [])]
    if phase_counts != actual_phase_counts:
        collector.error(
            f'phase_counts mismatch in {variation_meta_path}: '
            f'meta={phase_counts} actual={actual_phase_counts}')

    expected_avg = float(sum(actual_phase_counts)) / len(actual_phase_counts) if actual_phase_counts else 0.0
    if not _isclose(variation_meta.get('avg_phases', 0.0), expected_avg):
        collector.error(
            f'avg_phases mismatch in {variation_meta_path}: '
            f'meta={variation_meta.get("avg_phases")} actual={expected_avg}')

    generation_stats = variation_meta.get('generation_stats', {})
    if int(generation_stats.get('success_demos', -1)) != len(episode_dirs):
        collector.error(
            f'success_demos mismatch in {variation_meta_path}: '
            f'meta={generation_stats.get("success_demos")} actual={len(episode_dirs)}')

    if int(generation_stats.get('phase_valid_demos', -1)) != valid_episode_count:
        collector.error(
            f'phase_valid_demos mismatch in {variation_meta_path}: '
            f'meta={generation_stats.get("phase_valid_demos")} actual={valid_episode_count}')

    return {
        'task_name': task_name,
        'variation_name': variation_name,
        'variation_index': variation_index,
        'num_episodes': len(episode_dirs),
        'valid_episodes': valid_episode_count,
        'phase_counts': actual_phase_counts,
        'generation_stats': generation_stats,
        'planned_episodes': int(variation_meta.get('planned_episodes', 0)),
    }


def _validate_task_dir(task_path, collector):
    task_name = os.path.basename(task_path)
    variation_entries = []
    variation_indices = []
    for entry in sorted(os.listdir(task_path)):
        variation_index = _parse_prefixed_index(entry, 'variation')
        if variation_index is None:
            continue
        full_path = os.path.join(task_path, entry)
        if os.path.isdir(full_path):
            variation_entries.append((entry, full_path, variation_index))
            variation_indices.append(variation_index)

    if variation_indices and not _check_contiguous(variation_indices):
        collector.warning(
            f'Non-contiguous variation ids in {task_path}: {variation_indices}')

    variation_reports = []
    for _, variation_path, _ in variation_entries:
        report = _validate_variation_dir(task_name, variation_path, collector)
        if report is not None:
            variation_reports.append(report)

    task_meta_path = os.path.join(task_path, 'task_metadata.json')
    if not os.path.exists(task_meta_path):
        collector.warning(f'Missing task_metadata.json: {task_meta_path}')
        task_meta = None
    else:
        task_meta = _load_json(task_meta_path)

    total_episodes = sum(item['num_episodes'] for item in variation_reports)
    valid_episodes = sum(item['valid_episodes'] for item in variation_reports)
    all_phase_counts = [count for item in variation_reports for count in item['phase_counts']]
    total_phases = sum(all_phase_counts)
    avg_phases = float(total_phases) / len(all_phase_counts) if all_phase_counts else 0.0

    if task_meta is not None:
        if task_meta.get('task_name') != task_name:
            collector.error(
                f'task_name mismatch in {task_meta_path}: '
                f'meta={task_meta.get("task_name")} actual={task_name}')

        if int(task_meta.get('num_variations', -1)) != len(variation_reports):
            collector.error(
                f'num_variations mismatch in {task_meta_path}: '
                f'meta={task_meta.get("num_variations")} actual={len(variation_reports)}')

        if int(task_meta.get('total_episodes', -1)) != total_episodes:
            collector.error(
                f'total_episodes mismatch in {task_meta_path}: '
                f'meta={task_meta.get("total_episodes")} actual={total_episodes}')

        if int(task_meta.get('valid_episodes', -1)) != valid_episodes:
            collector.error(
                f'valid_episodes mismatch in {task_meta_path}: '
                f'meta={task_meta.get("valid_episodes")} actual={valid_episodes}')

        if int(task_meta.get('total_phases', -1)) != total_phases:
            collector.error(
                f'total_phases mismatch in {task_meta_path}: '
                f'meta={task_meta.get("total_phases")} actual={total_phases}')

        if not _isclose(task_meta.get('avg_phases', 0.0), avg_phases):
            collector.error(
                f'avg_phases mismatch in {task_meta_path}: '
                f'meta={task_meta.get("avg_phases")} actual={avg_phases}')

    return {
        'task_name': task_name,
        'num_variations': len(variation_reports),
        'total_episodes': total_episodes,
        'valid_episodes': valid_episodes,
        'total_phases': total_phases,
        'avg_phases': avg_phases,
        'variation_reports': variation_reports,
    }


def _validate_single_dataset_root(root_path, collector):
    task_reports = []
    task_dirs = []
    for entry in sorted(os.listdir(root_path)):
        full_path = os.path.join(root_path, entry)
        if _is_task_dir(full_path):
            task_dirs.append(full_path)

    if not task_dirs:
        collector.error(f'No task directories found under {root_path}')
        return {
            'mode': 'single_dataset',
            'root_path': os.path.abspath(root_path),
            'task_reports': [],
        }

    for task_path in task_dirs:
        task_reports.append(_validate_task_dir(task_path, collector))

    dataset_meta_path = os.path.join(root_path, 'dataset_metadata.json')
    if os.path.exists(dataset_meta_path):
        dataset_meta = _load_json(dataset_meta_path)
        actual_task_names = sorted(item['task_name'] for item in task_reports)
        actual_num_variations = sum(item['num_variations'] for item in task_reports)
        actual_success = sum(item['total_episodes'] for item in task_reports)
        actual_valid = sum(item['valid_episodes'] for item in task_reports)
        variation_reports = [
            variation
            for task_report in task_reports
            for variation in task_report['variation_reports']
        ]
        actual_planned = sum(int(item.get('planned_episodes', 0)) for item in variation_reports)
        actual_timeout = sum(int(item['generation_stats'].get('timeout_demos', 0)) for item in variation_reports)
        actual_failed_non_phase = sum(int(item['generation_stats'].get('failed_demos', 0)) for item in variation_reports)
        actual_phase_invalid = sum(int(item['generation_stats'].get('phase_invalid_demos', 0)) for item in variation_reports)
        actual_phase_invalid_attempts = sum(
            int(item['generation_stats'].get('phase_invalid_attempts', 0))
            for item in variation_reports)
        actual_failed = actual_failed_non_phase + actual_phase_invalid
        actual_done = actual_success + actual_failed

        if sorted(dataset_meta.get('tasks', [])) != actual_task_names:
            collector.error(
                f'dataset tasks mismatch in {dataset_meta_path}: '
                f'meta={sorted(dataset_meta.get("tasks", []))} actual={actual_task_names}')

        if int(dataset_meta.get('num_tasks', -1)) != len(actual_task_names):
            collector.error(
                f'num_tasks mismatch in {dataset_meta_path}: '
                f'meta={dataset_meta.get("num_tasks")} actual={len(actual_task_names)}')

        if int(dataset_meta.get('num_variations', -1)) != actual_num_variations:
            collector.error(
                f'num_variations mismatch in {dataset_meta_path}: '
                f'meta={dataset_meta.get("num_variations")} actual={actual_num_variations}')

        if int(dataset_meta.get('planned_episodes', -1)) != actual_planned:
            collector.error(
                f'planned_episodes mismatch in {dataset_meta_path}: '
                f'meta={dataset_meta.get("planned_episodes")} actual={actual_planned}')

        if int(dataset_meta.get('done_episodes', -1)) != actual_done:
            collector.error(
                f'done_episodes mismatch in {dataset_meta_path}: '
                f'meta={dataset_meta.get("done_episodes")} actual={actual_done}')

        if int(dataset_meta.get('success_episodes', -1)) != actual_success:
            collector.error(
                f'success_episodes mismatch in {dataset_meta_path}: '
                f'meta={dataset_meta.get("success_episodes")} actual={actual_success}')

        if int(dataset_meta.get('phase_valid_episodes', -1)) != actual_valid:
            collector.error(
                f'phase_valid_episodes mismatch in {dataset_meta_path}: '
                f'meta={dataset_meta.get("phase_valid_episodes")} actual={actual_valid}')

        if int(dataset_meta.get('timeout_episodes', -1)) != actual_timeout:
            collector.error(
                f'timeout_episodes mismatch in {dataset_meta_path}: '
                f'meta={dataset_meta.get("timeout_episodes")} actual={actual_timeout}')

        if int(dataset_meta.get('failed_episodes', -1)) != actual_failed:
            collector.error(
                f'failed_episodes mismatch in {dataset_meta_path}: '
                f'meta={dataset_meta.get("failed_episodes")} actual={actual_failed}')

        phase_invalid_attempts = dataset_meta.get('phase_invalid_attempts')
        if phase_invalid_attempts is not None and int(phase_invalid_attempts) != actual_phase_invalid_attempts:
            collector.error(
                f'phase_invalid_attempts mismatch in {dataset_meta_path}: '
                f'meta={phase_invalid_attempts} actual={actual_phase_invalid_attempts}')

        if int(dataset_meta.get('phase_invalid_episodes', -1)) != actual_phase_invalid:
            collector.error(
                f'phase_invalid_episodes mismatch in {dataset_meta_path}: '
                f'meta={dataset_meta.get("phase_invalid_episodes")} actual={actual_phase_invalid}')

    else:
        collector.warning(f'Missing dataset_metadata.json: {dataset_meta_path}')

    return {
        'mode': 'single_dataset',
        'root_path': os.path.abspath(root_path),
        'task_reports': task_reports,
    }


def _resolve_existing_path(base_path, candidate):
    if candidate is None:
        return None
    if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    joined = os.path.join(base_path, os.path.basename(candidate))
    if os.path.exists(joined):
        return os.path.abspath(joined)
    return os.path.abspath(candidate)


def validate_target_path(target_path):
    collector = ValidationCollector()
    target_path = os.path.abspath(target_path)
    launcher_summary_path = os.path.join(target_path, 'multi_gpu_launcher_summary.json')

    if os.path.exists(launcher_summary_path):
        launcher_summary = _load_json(launcher_summary_path)
        job_reports = []
        seen_tasks = {}
        for job in launcher_summary.get('jobs', []):
            output_path = _resolve_existing_path(target_path, job.get('output_path'))
            if not os.path.exists(output_path):
                collector.error(f'Launcher job output path not found: {output_path}')
                continue
            shard_report = _validate_single_dataset_root(output_path, collector)
            task_names = [item['task_name'] for item in shard_report.get('task_reports', [])]
            for task_name in task_names:
                if task_name in seen_tasks:
                    collector.error(
                        f'Task appears in multiple shards: {task_name} '
                        f'({seen_tasks[task_name]} and {output_path})')
                else:
                    seen_tasks[task_name] = output_path
            job_reports.append({
                'job_index': job.get('job_index'),
                'display': job.get('display'),
                'output_path': output_path,
                'returncode': job.get('returncode'),
                'status': job.get('status'),
                'dataset_report': shard_report,
            })

        report = {
            'mode': 'multi_shard',
            'target_path': target_path,
            'job_reports': job_reports,
            'errors': collector.errors,
            'warnings': collector.warnings,
            'ok': len(collector.errors) == 0,
        }
        return report

    single_report = _validate_single_dataset_root(target_path, collector)
    return {
        'mode': single_report.get('mode'),
        'target_path': target_path,
        'task_reports': single_report.get('task_reports', []),
        'errors': collector.errors,
        'warnings': collector.warnings,
        'ok': len(collector.errors) == 0,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='校验 segmented dataset 的编号与 metadata 一致性')
    parser.add_argument('target_path', help='数据集根目录，或 multi-GPU launcher 的根目录')
    parser.add_argument('--report_path', type=str, default='',
                        help='可选，若提供则将 JSON 报告写到该路径')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    report = validate_target_path(args.target_path)

    print(f'[Info] Validation mode: {report.get("mode")}')
    print(f'[Info] Target path: {report.get("target_path")}')
    print(f'[Info] Errors: {len(report.get("errors", []))}')
    print(f'[Info] Warnings: {len(report.get("warnings", []))}')

    for message in report.get('warnings', []):
        print(f'[Warn] {message}')
    for message in report.get('errors', []):
        print(f'[Error] {message}')

    if args.report_path:
        report_path = os.path.abspath(args.report_path)
        with open(report_path, 'w', encoding='utf-8') as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)
        print(f'[Info] Validation report saved to {report_path}')

    return 0 if report.get('ok') else 1


if __name__ == '__main__':
    raise SystemExit(main())
# -*- coding: utf-8 -*-
"""
多 DISPLAY / 多 GPU 启动器实现。

该模块承载任务分片、子作业拉起、跨 shard 进度聚合、
部分失败后的合并与 launcher 汇总日志逻辑。
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime

import rlbench.backend.task as task
from tqdm import tqdm

from .validation import (
    load_fixed_phase_config,
    resolve_fixed_phase_csv_path,
    split_tasks_by_fixed_phase_config,
)


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
SEGMENTATION_ENTRY = os.path.join(REPO_ROOT, 'scripts', 'generate_segmented_dataset.py')
RESERVED_FORWARD_ARGS = {
    '--output_path',
    '--base_output_path',
    '--tasks',
    '--processes',
    '--fixed_phase_csv',
    '--base_seed',
    '--progress_file',
    '--log_path',
    '--execution_mode',
}


def _append_text(log_path, message):
    with open(log_path, 'a', encoding='utf-8') as file_obj:
        file_obj.write(message)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


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


def _collect_failure_details(snapshot):
    if not snapshot:
        return []

    details = []
    for stat in snapshot.get('variation_stats', {}).values():
        task_name = stat.get('task_name')
        variation_index = int(stat.get('variation_index', -1))
        for detail in stat.get('failure_details', []):
            details.append({
                'task_name': detail.get('task_name', task_name),
                'variation_index': int(detail.get('variation_index', variation_index)),
                'requested_episode': int(detail.get('requested_episode', detail.get('episode', -1))),
                'failure_type': detail.get('failure_type', 'unknown'),
                'reason': detail.get('reason', ''),
                'stage': detail.get('stage', 'unknown'),
                'disposition': detail.get('disposition', 'failed'),
                'observed_phases': detail.get('observed_phases'),
                'expected_phases': detail.get('expected_phases'),
                'retries': detail.get('retries'),
                'trigger_failure_type': detail.get('trigger_failure_type'),
                'trigger_stage': detail.get('trigger_stage'),
                'trigger_requested_episode': detail.get('trigger_requested_episode'),
            })

    return sorted(
        details,
        key=lambda item: (item['task_name'], item['variation_index'], item['requested_episode'], item['failure_type']))


def _append_launcher_summary(master_log_path, summary, aggregate, jobs):
    job_map = {job['job_index']: job for job in jobs}
    lines = [
        '',
        '===== Launcher Summary =====',
        f'started_at={summary.get("started_at")}',
        f'finished_at={summary.get("finished_at")}',
        f'status={summary.get("status")}',
        f'output_path={summary.get("output_path")}',
        f'log_path={summary.get("master_log_path")}',
        f'fixed_phase_csv={summary.get("fixed_phase_csv")}',
        f'planned_episodes={int(aggregate.get("planned_episodes", 0))}',
        f'done_episodes={int(aggregate.get("done_episodes", 0))}',
        f'success_episodes={int(aggregate.get("success_episodes", 0))}',
        f'timeout_episodes={int(aggregate.get("timeout_episodes", 0))}',
        f'failed_episodes={int(aggregate.get("failed_episodes", 0))}',
        f'demo_timeout_episodes={int(aggregate.get("demo_timeout_episodes", 0))}',
        f'watchdog_timeout_episodes={int(aggregate.get("watchdog_timeout_episodes", 0))}',
        f'exception_episodes={int(aggregate.get("exception_episodes", 0))}',
        f'phase_invalid_episodes={int(aggregate.get("phase_invalid_episodes", 0))}',
        f'aborted_episodes={int(aggregate.get("aborted_episodes", 0))}',
    ]

    for job_summary in summary.get('jobs', []):
        job_index = job_summary['job_index']
        job_state = job_map.get(job_index, {})
        snapshot = job_state.get('progress_snapshot')
        progress = snapshot.get('progress', {}) if snapshot else {}
        lines.append(
            f'[Job] job={job_index} status={job_summary.get("status")} returncode={job_summary.get("returncode")} '
            f'display={job_summary.get("display")} gpu={job_summary.get("cuda_visible_devices")} '
            f'processes={job_summary.get("processes")}')
        lines.append(f'  tasks={", ".join(job_summary.get("tasks", []))}')
        lines.append(f'  shard_output_path={job_summary.get("output_path")}')
        if job_summary.get('command'):
            lines.append('  command=' + shlex.join(job_summary['command']))
        if progress:
            lines.append(
                f'  progress planned={int(progress.get("planned_episodes", 0))} '
                f'accounted={int(progress.get("done_episodes", 0))} '
                f'ok={int(progress.get("success_episodes", 0))} '
                f'fail={int(progress.get("failed_episodes", 0))}')
            lines.append(
                f'  failure_breakdown demo_timeout={int(progress.get("demo_timeout_episodes", 0))} '
                f'watchdog_timeout={int(progress.get("watchdog_timeout_episodes", 0))} '
                f'exception={int(progress.get("exception_episodes", 0))} '
                f'phase_invalid={int(progress.get("phase_invalid_episodes", 0))} '
                f'aborted={int(progress.get("aborted_episodes", 0))}')
        for detail in _collect_failure_details(snapshot):
            extra_parts = []
            if detail.get('observed_phases') is not None and detail.get('expected_phases') is not None:
                extra_parts.append(
                    f'phases={detail["observed_phases"]}/{detail["expected_phases"]}')
            if detail.get('retries') is not None:
                extra_parts.append(f'retries={detail["retries"]}')
            if detail.get('trigger_failure_type') is not None:
                extra_parts.append(f'trigger_type={detail["trigger_failure_type"]}')
            if detail.get('trigger_stage') is not None:
                extra_parts.append(f'trigger_stage={detail["trigger_stage"]}')
            if detail.get('trigger_requested_episode') is not None:
                extra_parts.append(f'trigger_requested_episode={detail["trigger_requested_episode"]}')
            extra_suffix = (' ' + ' '.join(extra_parts)) if extra_parts else ''
            lines.append(
                f'  failure task={detail["task_name"]} variation={detail["variation_index"]} '
                f'requested_episode={detail["requested_episode"]} type={detail["failure_type"]} '
                f'stage={detail["stage"]} disposition={detail["disposition"]} '
                f'reason={detail["reason"]}{extra_suffix}')

    _append_text(master_log_path, '\n'.join(lines) + '\n')


def _append_job_logs(master_log_path, jobs):
    suppressed_prefixes = (
        '[Launcher] ',
        '[Info] Explicit tasks',
        '[Info] Start collecting.',
        '[Info] Log saved:',
        'Data collection & segmentation done!',
        '===== Segmented Collection Summary =====',
        'Started at:',
        'Finished at:',
        'Total duration:',
        'Generation speed:',
        'Planned tasks:',
        'Processed variations:',
        'Success variations:',
        'Failed variations:',
        'Planned demos:',
        'Done demos:',
        'Accounted demos:',
        'Success demos:',
        'Total failed demos:',
        'Timeout/exception failed demos:',
        'Timeout demos:',
        'Demo timeout demos:',
        'Watchdog timeout demos:',
        'Exception demos:',
        'Phase invalid demos:',
        'Aborted demos:',
        'Phase invalid attempts:',
        'Phase valid demos:',
        'Phase valid rate:',
        'Failed task breakdown:',
        'Failed episode details:',
        '- task=',
    )

    for job in jobs:
        pipeline_log_path = job.get('pipeline_log_path')
        console_log_path = job.get('console_log_path')

        if pipeline_log_path and os.path.exists(pipeline_log_path):
            _append_text(master_log_path, f'\n===== job-{job["job_index"]} traj_gen_seg log =====\n')
            with open(pipeline_log_path, 'r', encoding='utf-8') as file_obj:
                _append_text(master_log_path, file_obj.read())

        if console_log_path and os.path.exists(console_log_path):
            with open(console_log_path, 'r', encoding='utf-8') as file_obj:
                console_lines = [line.rstrip() for line in file_obj.readlines()]

            filtered_lines = []
            for line in console_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('Collecting & Segmenting:') or stripped.startswith('Launcher Progress:'):
                    continue
                if stripped.startswith('qt.qpa.xcb:'):
                    continue
                if stripped.startswith(suppressed_prefixes):
                    continue
                filtered_lines.append(line)

            if filtered_lines:
                _append_text(master_log_path, f'\n===== job-{job["job_index"]} console diagnostics =====\n')
                _append_text(master_log_path, '\n'.join(filtered_lines) + '\n')


def _get_available_task_names():
    return sorted(
        file_name[:-3]
        for file_name in os.listdir(task.TASKS_PATH)
        if file_name.endswith('.py') and file_name != '__init__.py'
    )


def _load_fixed_phase_tasks(csv_path):
    resolved_csv_path = resolve_fixed_phase_csv_path(csv_path)
    if not os.path.exists(resolved_csv_path):
        raise ValueError(f'Fixed-phase CSV not found: {resolved_csv_path}')

    fixed_phase_config = load_fixed_phase_config(resolved_csv_path, warn_if_missing=False)
    task_names = sorted(fixed_phase_config)

    if not task_names:
        raise ValueError(f'No fixed-phase tasks found in {resolved_csv_path}')

    return resolved_csv_path, task_names


def _resolve_task_names(fixed_phase_csv, fixed_phase_only, requested_tasks):
    available_tasks = _get_available_task_names()
    available_set = set(available_tasks)

    resolved_csv_path = resolve_fixed_phase_csv_path(fixed_phase_csv)
    csv_tasks = None
    if fixed_phase_only:
        resolved_csv_path, csv_tasks = _load_fixed_phase_tasks(fixed_phase_csv)

    if requested_tasks:
        invalid_tasks = sorted(task_name for task_name in requested_tasks if task_name not in available_set)
        if invalid_tasks:
            raise ValueError(f'Task(s) not recognised: {", ".join(invalid_tasks)}')

        if csv_tasks is not None:
            not_in_csv = sorted(task_name for task_name in requested_tasks if task_name not in csv_tasks)
            if not_in_csv:
                raise ValueError(
                    'Task(s) not listed in TASK_FIXED_PHASE_NUM.csv: ' + ', '.join(not_in_csv))

        if fixed_phase_only:
            fixed_phase_tasks = list(requested_tasks)
            normal_tasks = []
        else:
            fixed_phase_config = load_fixed_phase_config(resolved_csv_path, warn_if_missing=False)
            fixed_phase_tasks, normal_tasks = split_tasks_by_fixed_phase_config(
                requested_tasks, fixed_phase_config)
            if not os.path.exists(resolved_csv_path):
                print(f'[Info] Fixed phase CSV not found for explicit tasks: {resolved_csv_path}')
                print('[Info] Explicit tasks will use normal segmentation: ' + ', '.join(requested_tasks))
            else:
                if fixed_phase_tasks:
                    print('[Info] Explicit tasks using fixed phase config: ' + ', '.join(fixed_phase_tasks))
                if normal_tasks:
                    print(
                        '[Info] Explicit tasks not listed in TASK_FIXED_PHASE_NUM.csv '
                        'and will use normal segmentation: ' + ', '.join(normal_tasks)
                    )

        return resolved_csv_path, requested_tasks

    if fixed_phase_only:
        valid_csv_tasks = sorted(task_name for task_name in csv_tasks if task_name in available_set)
        missing_tasks = sorted(task_name for task_name in csv_tasks if task_name not in available_set)
        if missing_tasks:
            print('[Warn] These TASK_FIXED_PHASE_NUM.csv tasks are not available in RLBench and will be skipped:')
            print('  ' + ', '.join(missing_tasks))
        if not valid_csv_tasks:
            raise ValueError('No valid RLBench tasks remain after filtering TASK_FIXED_PHASE_NUM.csv')
        return resolved_csv_path, valid_csv_tasks

    return resolved_csv_path, available_tasks


def _normalize_per_display(values, expected_len, name):
    if not values:
        return [None] * expected_len

    if len(values) == 1:
        return list(values) * expected_len

    if len(values) != expected_len:
        raise ValueError(f'{name} expects either 1 value or {expected_len} values, got {len(values)}')

    return list(values)


def _chunk_tasks_round_robin(task_names, shard_count):
    shards = [[] for _ in range(shard_count)]
    for index, task_name in enumerate(task_names):
        shards[index % shard_count].append(task_name)
    return shards


def _sanitize_display(display_value):
    return display_value.replace(':', '').replace('.', '_')


def _extract_conflicting_passthrough_args(passthrough):
    conflicts = []
    for token in passthrough:
        option = token.split('=', 1)[0]
        if option in RESERVED_FORWARD_ARGS:
            conflicts.append(token)
    return conflicts


def _build_child_command(python_executable, output_path, fixed_phase_csv,
                         task_names, processes, passthrough_args,
                         base_seed, progress_file, log_path):
    command = [
        python_executable,
        SEGMENTATION_ENTRY,
        '--execution_mode', 'single',
        '--output_path', output_path,
        '--fixed_phase_csv', fixed_phase_csv,
        '--processes', str(processes),
        '--base_seed', str(base_seed),
        '--progress_file', progress_file,
        '--log_path', log_path,
        '--tasks',
        *task_names,
        *passthrough_args,
    ]
    return command


def _read_progress_snapshot(progress_file):
    if not progress_file or not os.path.exists(progress_file):
        return None
    try:
        return _load_json(progress_file)
    except Exception:
        return None


def _aggregate_progress(jobs):
    totals = {
        'planned_episodes': 0,
        'done_episodes': 0,
        'success_episodes': 0,
        'timeout_episodes': 0,
        'failed_episodes': 0,
        'demo_timeout_episodes': 0,
        'watchdog_timeout_episodes': 0,
        'exception_episodes': 0,
        'phase_invalid_episodes': 0,
        'aborted_episodes': 0,
        'active_jobs': 0,
    }
    for job in jobs:
        snapshot = _read_progress_snapshot(job.get('progress_file'))
        job['progress_snapshot'] = snapshot
        if snapshot is None:
            continue
        progress = snapshot.get('progress', {})
        totals['planned_episodes'] += int(progress.get('planned_episodes', 0))
        totals['done_episodes'] += int(progress.get('done_episodes', 0))
        totals['success_episodes'] += int(progress.get('success_episodes', 0))
        totals['timeout_episodes'] += int(progress.get('timeout_episodes', 0))
        totals['failed_episodes'] += int(progress.get('failed_episodes', 0))
        totals['demo_timeout_episodes'] += int(progress.get('demo_timeout_episodes', 0))
        totals['watchdog_timeout_episodes'] += int(progress.get('watchdog_timeout_episodes', 0))
        totals['exception_episodes'] += int(progress.get('exception_episodes', 0))
        totals['phase_invalid_episodes'] += int(progress.get('phase_invalid_episodes', 0))
        totals['aborted_episodes'] += int(progress.get('aborted_episodes', 0))
        if not snapshot.get('finished', False):
            totals['active_jobs'] += 1
    return totals


def _progress_snapshot_to_dataset_meta(snapshot, started_at):
    if not snapshot:
        return None

    progress = dict(snapshot.get('progress', {}))
    variation_stats = dict(snapshot.get('variation_stats', {}))
    tasks = sorted({
        stat.get('task_name') for stat in variation_stats.values()
        if stat.get('task_name')
    })
    config = dict(snapshot.get('config', {}))
    phase_invalid_attempts = sum(int(stat.get('phase_invalid_attempts', 0)) for stat in variation_stats.values())
    phase_invalid_episodes = sum(int(stat.get('phase_invalid_demos', 0)) for stat in variation_stats.values())
    phase_valid_episodes = sum(int(stat.get('phase_valid_demos', 0)) for stat in variation_stats.values())
    demo_timeout_episodes = sum(int(stat.get('demo_timeout_demos', 0)) for stat in variation_stats.values())
    watchdog_timeout_episodes = sum(int(stat.get('watchdog_timeout_demos', 0)) for stat in variation_stats.values())
    exception_episodes = sum(int(stat.get('exception_demos', 0)) for stat in variation_stats.values())
    aborted_episodes = sum(int(stat.get('aborted_demos', 0)) for stat in variation_stats.values())

    return {
        'started_at': snapshot.get('started_at', started_at.isoformat()),
        'finished_at': snapshot.get('updated_at', datetime.now().isoformat()),
        'duration_seconds': round((datetime.now() - started_at).total_seconds(), 3),
        'tasks': tasks or list(config.get('tasks', [])),
        'num_tasks': len(tasks or list(config.get('tasks', []))),
        'num_variations': len(variation_stats),
        'planned_episodes': int(progress.get('planned_episodes', 0)),
        'done_episodes': int(progress.get('done_episodes', 0)),
        'success_episodes': int(progress.get('success_episodes', 0)),
        'failed_episodes': int(progress.get('failed_episodes', 0)),
        'timeout_episodes': int(progress.get('timeout_episodes', 0)),
        'demo_timeout_episodes': int(progress.get('demo_timeout_episodes', demo_timeout_episodes)),
        'watchdog_timeout_episodes': int(progress.get('watchdog_timeout_episodes', watchdog_timeout_episodes)),
        'exception_episodes': int(progress.get('exception_episodes', exception_episodes)),
        'phase_invalid_attempts': int(phase_invalid_attempts),
        'phase_invalid_episodes': int(phase_invalid_episodes),
        'aborted_episodes': int(progress.get('aborted_episodes', aborted_episodes)),
        'phase_valid_episodes': int(phase_valid_episodes),
        'config': config,
    }


def _merge_dataset_metadata(dataset_metas, merged_output_path, jobs, started_at, merged_task_names=None):
    if not dataset_metas:
        return None

    tasks = []
    merged_meta = {
        'started_at': min(meta.get('started_at', started_at.isoformat()) for meta in dataset_metas),
        'finished_at': datetime.now().isoformat(),
        'duration_seconds': round((datetime.now() - started_at).total_seconds(), 3),
        'output_path': os.path.abspath(merged_output_path),
        'tasks': tasks,
        'num_tasks': 0,
        'num_variations': 0,
        'planned_episodes': 0,
        'done_episodes': 0,
        'success_episodes': 0,
        'failed_episodes': 0,
        'timeout_episodes': 0,
        'demo_timeout_episodes': 0,
        'watchdog_timeout_episodes': 0,
        'exception_episodes': 0,
        'phase_invalid_attempts': 0,
        'phase_invalid_episodes': 0,
        'aborted_episodes': 0,
        'phase_valid_episodes': 0,
        'config': dict(dataset_metas[0].get('config', {})),
        'launcher': {
            'displays': [job['display'] for job in jobs],
            'gpu_ids': [job['cuda_visible_devices'] for job in jobs],
            'processes_per_display': [job['processes'] for job in jobs],
            'seed_base': jobs[0].get('seed_base') if jobs else None,
        },
    }

    for meta in dataset_metas:
        tasks.extend(meta.get('tasks', []))
        merged_meta['num_variations'] += int(meta.get('num_variations', 0))
        merged_meta['planned_episodes'] += int(meta.get('planned_episodes', 0))
        merged_meta['done_episodes'] += int(meta.get('done_episodes', 0))
        merged_meta['success_episodes'] += int(meta.get('success_episodes', 0))
        merged_meta['failed_episodes'] += int(meta.get('failed_episodes', 0))
        merged_meta['timeout_episodes'] += int(meta.get('timeout_episodes', 0))
        merged_meta['demo_timeout_episodes'] += int(meta.get('demo_timeout_episodes', 0))
        merged_meta['watchdog_timeout_episodes'] += int(meta.get('watchdog_timeout_episodes', 0))
        merged_meta['exception_episodes'] += int(meta.get('exception_episodes', 0))
        merged_meta['phase_invalid_attempts'] += int(meta.get('phase_invalid_attempts', 0))
        merged_meta['phase_invalid_episodes'] += int(meta.get('phase_invalid_episodes', 0))
        merged_meta['aborted_episodes'] += int(meta.get('aborted_episodes', 0))
        merged_meta['phase_valid_episodes'] += int(meta.get('phase_valid_episodes', 0))

    merged_meta['tasks'] = sorted(merged_task_names) if merged_task_names else sorted(tasks)
    merged_meta['num_tasks'] = len(merged_meta['tasks'])
    return merged_meta


def _merge_shards(merged_output_path, jobs, master_log_path):
    os.makedirs(merged_output_path, exist_ok=True)

    dataset_metas = []
    merged_task_names = set()
    for job in jobs:
        shard_output_path = job['output_path']
        dataset_meta_path = os.path.join(shard_output_path, 'dataset_metadata.json')
        if os.path.exists(dataset_meta_path):
            dataset_metas.append(_load_json(dataset_meta_path))
        else:
            snapshot = _read_progress_snapshot(job.get('progress_file'))
            fallback_meta = _progress_snapshot_to_dataset_meta(
                snapshot,
                datetime.fromisoformat(job['started_at']),
            )
            if fallback_meta is not None:
                dataset_metas.append(fallback_meta)

        for entry in sorted(os.listdir(shard_output_path)):
            if entry.startswith('.'):
                continue
            source_path = os.path.join(shard_output_path, entry)
            if not os.path.isdir(source_path):
                continue
            if entry in {'launcher_logs'}:
                continue
            destination_path = os.path.join(merged_output_path, entry)
            if entry in merged_task_names or os.path.exists(destination_path):
                raise ValueError(f'Duplicate task directory while merging shards: {entry}')
            shutil.move(source_path, destination_path)
            merged_task_names.add(entry)

    merged_meta = _merge_dataset_metadata(
        dataset_metas,
        merged_output_path,
        jobs,
        datetime.fromisoformat(jobs[0]['started_at']),
        merged_task_names=merged_task_names,
    )
    if merged_meta is not None:
        _write_json(os.path.join(merged_output_path, 'dataset_metadata.json'), merged_meta)

    _append_text(master_log_path, f'[Merge] merged_output_path={merged_output_path}\n')
    return merged_output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description='按多 DISPLAY / 多 GPU 分片启动统一数据生成脚本')
    parser.add_argument('--output_path', type=str, default='',
                        help='最终合并后的数据集输出目录；例如 ./smoke_dataset')
    parser.add_argument('--base_output_path', type=str, default='',
                        help=argparse.SUPPRESS)
    parser.add_argument('--displays', nargs='+', required=True,
                        help='要使用的 DISPLAY 列表，例如 :99.0 :99.1')
    parser.add_argument('--gpu_ids', nargs='*', default=[],
                        help='可选，与 displays 对齐的 CUDA_VISIBLE_DEVICES 列表')
    parser.add_argument('--processes_per_display', nargs='+', type=int, default=[1],
                        help='每个 DISPLAY 的 worker 数；可填 1 个值广播，或逐 DISPLAY 指定')
    parser.add_argument('--tasks', nargs='*', default=[],
                        help='仅运行这些任务；为空时由 fixed_phase_only 决定使用全部任务还是 CSV 中的任务')
    parser.add_argument('--fixed_phase_csv', type=str, default='./TASK_FIXED_PHASE_NUM.csv',
                        help='固定阶段数 CSV 路径')
    parser.add_argument('--fixed_phase_only', action='store_true',
                        help='若不显式提供 tasks，则只运行 TASK_FIXED_PHASE_NUM.csv 中的任务')
    parser.add_argument('--seed_base', type=int, default=-1,
                        help='可选，launcher 层基础随机种子；每个作业会从这里派生自己的 base_seed')
    parser.add_argument('--keep_workdirs', action='store_true',
                        help='保留 shard 工作目录与中间日志，便于调试')
    parser.add_argument('--python_executable', type=str, default=sys.executable,
                        help='启动子作业时使用的 Python 可执行文件')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印分片与命令，不真正启动子作业')
    return parser


def parse_args(argv=None):
    return build_parser().parse_known_args(argv)


def main(argv=None):
    args, passthrough_args = parse_args(argv)

    args.output_path = args.output_path or args.base_output_path
    if not args.output_path:
        raise ValueError('Please provide --output_path')
    args.output_path = os.path.abspath(args.output_path)

    conflicts = _extract_conflicting_passthrough_args(passthrough_args)
    if conflicts:
        raise ValueError(
            'These arguments are managed by the launcher and must not be forwarded: '
            + ', '.join(conflicts))

    resolved_csv_path, task_names = _resolve_task_names(
        fixed_phase_csv=args.fixed_phase_csv,
        fixed_phase_only=args.fixed_phase_only,
        requested_tasks=args.tasks,
    )
    if not task_names:
        raise ValueError('No tasks selected for launch')

    display_count = len(args.displays)
    processes_per_display = _normalize_per_display(
        args.processes_per_display, display_count, 'processes_per_display')
    gpu_ids = _normalize_per_display(args.gpu_ids, display_count, 'gpu_ids')
    seed_base = int(args.seed_base) if int(args.seed_base) >= 0 else int(time.time() * 1_000_000)

    task_shards = _chunk_tasks_round_robin(task_names, display_count)

    output_parent = os.path.dirname(args.output_path)
    os.makedirs(output_parent, exist_ok=True)

    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    work_root = os.path.join(output_parent, f'.{os.path.basename(args.output_path)}_launcher_work')
    if os.path.exists(work_root):
        shutil.rmtree(work_root, ignore_errors=True)
    shard_root = os.path.join(work_root, 'shards')
    progress_root = os.path.join(work_root, 'progress')
    pipeline_log_dir = os.path.join(work_root, 'pipeline_logs')
    console_log_dir = os.path.join(work_root, 'console_logs')
    log_root = os.path.join(REPO_ROOT, 'log')
    master_log_path = os.path.join(log_root, f'traj_gen_seg_{run_stamp}.log')
    os.makedirs(shard_root, exist_ok=True)
    os.makedirs(progress_root, exist_ok=True)
    os.makedirs(pipeline_log_dir, exist_ok=True)
    os.makedirs(console_log_dir, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    started_at = datetime.now()
    with open(master_log_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(f'[Launcher] started_at={started_at.isoformat()}\n')

    jobs = []
    for index, display in enumerate(args.displays):
        shard_tasks = task_shards[index]
        if not shard_tasks:
            continue

        shard_name = f'shard_{index:02d}_{_sanitize_display(display)}'
        shard_output_path = os.path.join(shard_root, shard_name)
        os.makedirs(shard_output_path, exist_ok=True)

        child_log_path = os.path.join(console_log_dir, f'{shard_name}.log')
        pipeline_log_path = os.path.join(pipeline_log_dir, f'{shard_name}.log')
        progress_file = os.path.join(progress_root, f'{shard_name}.json')
        child_seed = seed_base + index * 1_000_003
        command = _build_child_command(
            python_executable=args.python_executable,
            output_path=shard_output_path,
            fixed_phase_csv=resolved_csv_path,
            task_names=shard_tasks,
            processes=processes_per_display[index],
            passthrough_args=passthrough_args,
            base_seed=child_seed,
            progress_file=progress_file,
            log_path=pipeline_log_path,
        )
        jobs.append({
            'job_index': index,
            'started_at': started_at.isoformat(),
            'display': display,
            'cuda_visible_devices': gpu_ids[index],
            'processes': int(processes_per_display[index]),
            'tasks': shard_tasks,
            'output_path': shard_output_path,
            'console_log_path': child_log_path,
            'traj_gen_seg_log_path': pipeline_log_path,
            'log_path': child_log_path,
            'pipeline_log_path': pipeline_log_path,
            'progress_file': progress_file,
            'seed_base': child_seed,
            'command': command,
        })

    if not jobs:
        raise ValueError('No non-empty task shards to launch')

    print(f'[Info] Selected {len(task_names)} task(s), split into {len(jobs)} job(s).')
    for job in jobs:
        gpu_desc = job['cuda_visible_devices'] if job['cuda_visible_devices'] is not None else 'inherit'
        print(
            f'[Plan] job={job["job_index"]} display={job["display"]} '
            f'gpu={gpu_desc} processes={job["processes"]} '
            f'tasks={len(job["tasks"])} output={job["output_path"]}')

    summary = {
        'started_at': started_at.isoformat(),
        'output_path': args.output_path,
        'master_log_path': os.path.abspath(master_log_path),
        'fixed_phase_csv': resolved_csv_path,
        'fixed_phase_only': bool(args.fixed_phase_only),
        'seed_base': seed_base,
        'work_root': os.path.abspath(work_root),
        'jobs': [],
    }

    if args.dry_run:
        for job in jobs:
            print('[Dry Run] ' + shlex.join(job['command']))
            summary['jobs'].append({
                'job_index': job['job_index'],
                'display': job['display'],
                'cuda_visible_devices': job['cuda_visible_devices'],
                'processes': job['processes'],
                'tasks': job['tasks'],
                'output_path': job['output_path'],
                'console_log_path': job['console_log_path'],
                'traj_gen_seg_log_path': job['traj_gen_seg_log_path'],
                'log_path': job['log_path'],
                'pipeline_log_path': job['pipeline_log_path'],
                'progress_file': job['progress_file'],
                'seed_base': job['seed_base'],
                'command': job['command'],
                'status': 'dry_run',
                'returncode': None,
            })
        _append_launcher_summary(master_log_path, summary, _aggregate_progress(jobs), jobs)
        print(f'[Info] Dry-run log saved to {master_log_path}')
        return 0

    processes = []
    for job in jobs:
        env = os.environ.copy()
        env['DISPLAY'] = job['display']
        if job['cuda_visible_devices'] is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(job['cuda_visible_devices'])

        log_file = open(job['console_log_path'], 'w', encoding='utf-8')
        log_file.write('[Launcher] ' + shlex.join(job['command']) + '\n')
        log_file.flush()
        process = subprocess.Popen(
            job['command'],
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((job, process, log_file))
        print(
            f'[Launch] job={job["job_index"]} pid={process.pid} '
            f'display={job["display"]} dataset_shard={job["output_path"]}')
        _append_text(master_log_path,
                     f'[Launch] job={job["job_index"]} display={job["display"]} '
                     f'gpu={job["cuda_visible_devices"]} processes={job["processes"]} '
                     f'seed_base={job["seed_base"]} shard_output={job["output_path"]}\n')

    exit_code = 0
    progress_bar = tqdm(
        total=None,
        desc='Launcher Progress',
        dynamic_ncols=True,
        disable=_disable_tqdm_output(),
    )
    last_done = 0
    while any(process.poll() is None for _, process, _ in processes):
        aggregate = _aggregate_progress(jobs)
        done = int(aggregate.get('done_episodes', 0))
        total = int(aggregate.get('planned_episodes', 0))
        if total > 0 and progress_bar.total != total:
            progress_bar.total = total
            progress_bar.refresh()
        delta = done - last_done
        if delta > 0:
            progress_bar.update(delta)
            last_done = done
        progress_bar.set_postfix({
            'ok': int(aggregate.get('success_episodes', 0)),
            'demo_timeout': int(aggregate.get('demo_timeout_episodes', 0)),
            'watchdog': int(aggregate.get('watchdog_timeout_episodes', 0)),
            'fail': int(aggregate.get('failed_episodes', 0)),
            'jobs': int(aggregate.get('active_jobs', 0)),
        })
        time.sleep(0.5)

    for job, process, log_file in processes:
        returncode = process.wait()
        log_file.close()
        status = 'completed' if returncode == 0 else 'failed'
        if returncode != 0:
            exit_code = returncode
        print(
            f'[Done] job={job["job_index"]} display={job["display"]} '
            f'returncode={returncode}')
        _append_text(master_log_path,
                     f'[Done] job={job["job_index"]} display={job["display"]} '
                     f'returncode={returncode}\n')
        summary['jobs'].append({
            'job_index': job['job_index'],
            'display': job['display'],
            'cuda_visible_devices': job['cuda_visible_devices'],
            'processes': job['processes'],
            'tasks': job['tasks'],
            'output_path': job['output_path'],
            'console_log_path': job['console_log_path'],
            'traj_gen_seg_log_path': job['traj_gen_seg_log_path'],
            'log_path': job['log_path'],
            'pipeline_log_path': job['pipeline_log_path'],
            'progress_file': job['progress_file'],
            'seed_base': job['seed_base'],
            'command': job['command'],
            'status': status,
            'returncode': returncode,
        })

    aggregate = _aggregate_progress(jobs)
    final_done = int(aggregate.get('done_episodes', 0))
    if final_done > last_done:
        progress_bar.update(final_done - last_done)
    if progress_bar.total is None:
        progress_bar.total = int(aggregate.get('planned_episodes', 0))
    progress_bar.refresh()
    progress_bar.close()

    merged_output_path = _merge_shards(args.output_path, jobs, master_log_path)
    if exit_code != 0:
        _append_text(master_log_path, '[Launcher] partial merge completed despite one or more job failures.\n')

    _append_job_logs(master_log_path, jobs)

    summary['finished_at'] = datetime.now().isoformat()
    summary['status'] = 'completed' if exit_code == 0 else 'partial_failed'
    summary['output_path'] = os.path.abspath(merged_output_path) if merged_output_path else args.output_path
    summary['master_log_path'] = os.path.abspath(master_log_path)
    _append_launcher_summary(master_log_path, summary, aggregate, jobs)

    if not args.keep_workdirs:
        shutil.rmtree(work_root, ignore_errors=True)

    print(f'[Info] Launcher log saved to {master_log_path}')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分段采集 benchmark 启动器。"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

from validate_segmented_dataset import validate_target_path


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LAUNCHER_ENTRY = os.path.join(REPO_ROOT, 'traj_generator_segmentation_multi_gpu.py')
BENCHMARK_RESERVED_ARGS = {
    '--output_path',
    '--base_output_path',
    '--displays',
    '--gpu_ids',
    '--processes_per_display',
    '--tasks',
    '--fixed_phase_csv',
    '--fixed_phase_only',
    '--python_executable',
    '--dry_run',
}


def _normalize_per_display(values, expected_len, name):
    if not values:
        return []
    if len(values) == 1:
        return list(values) * expected_len
    if len(values) != expected_len:
        raise ValueError(f'{name} expects either 1 value or {expected_len} values, got {len(values)}')
    return list(values)


def _extract_conflicting_passthrough_args(passthrough_args):
    conflicts = []
    for token in passthrough_args:
        option = token.split('=', 1)[0]
        if option in BENCHMARK_RESERVED_ARGS:
            conflicts.append(token)
    return conflicts


def _resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(REPO_ROOT, path_value))


def _build_launcher_command(args, run_output_path, worker_count, passthrough_args):
    command = [
        args.python_executable,
        LAUNCHER_ENTRY,
        '--output_path', run_output_path,
        '--displays', *args.displays,
        '--processes_per_display', *([str(worker_count)] * len(args.displays)),
        '--fixed_phase_csv', args.fixed_phase_csv,
    ]

    if args.gpu_ids:
        command.extend(['--gpu_ids', *args.gpu_ids])

    if args.tasks:
        command.extend(['--tasks', *args.tasks])
    elif args.fixed_phase_only:
        command.append('--fixed_phase_only')

    command.extend(passthrough_args)
    return command


def _collect_dataset_metrics(run_output_path):
    dataset_paths = [run_output_path]

    aggregate = {
        'num_shards': len(dataset_paths),
        'num_tasks': 0,
        'num_variations': 0,
        'planned_episodes': 0,
        'done_episodes': 0,
        'success_episodes': 0,
        'failed_episodes': 0,
        'timeout_episodes': 0,
        'phase_invalid_episodes': 0,
        'phase_valid_episodes': 0,
        'dataset_paths': dataset_paths,
    }

    for dataset_path in dataset_paths:
        dataset_meta_path = os.path.join(dataset_path, 'dataset_metadata.json')
        if not os.path.exists(dataset_meta_path):
            continue
        with open(dataset_meta_path, 'r', encoding='utf-8') as file_obj:
            dataset_meta = json.load(file_obj)
        aggregate['num_tasks'] += int(dataset_meta.get('num_tasks', 0))
        aggregate['num_variations'] += int(dataset_meta.get('num_variations', 0))
        aggregate['planned_episodes'] += int(dataset_meta.get('planned_episodes', 0))
        aggregate['done_episodes'] += int(dataset_meta.get('done_episodes', 0))
        aggregate['success_episodes'] += int(dataset_meta.get('success_episodes', 0))
        aggregate['failed_episodes'] += int(dataset_meta.get('failed_episodes', 0))
        aggregate['timeout_episodes'] += int(dataset_meta.get('timeout_episodes', 0))
        aggregate['phase_invalid_episodes'] += int(dataset_meta.get('phase_invalid_episodes', 0))
        aggregate['phase_valid_episodes'] += int(dataset_meta.get('phase_valid_episodes', 0))

    return aggregate


def _format_float(value):
    return f'{float(value):.3f}'


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='自动 benchmark 不同 worker 数下的 segmented collection 吞吐与稳定性')
    parser.add_argument('--benchmark_root', required=True,
                        help='benchmark 输出根目录')
    parser.add_argument('--displays', nargs='+', required=True,
                        help='参与 benchmark 的 DISPLAY 列表，例如 :99.0 :99.1')
    parser.add_argument('--gpu_ids', nargs='*', default=[],
                        help='可选，与 displays 对齐的 CUDA_VISIBLE_DEVICES 列表')
    parser.add_argument('--worker_counts', nargs='+', type=int, default=[1, 2, 4],
                        help='要测试的每 DISPLAY worker 数列表')
    parser.add_argument('--repeats', type=int, default=1,
                        help='每个 worker 配置重复运行次数')
    parser.add_argument('--tasks', nargs='*', default=[],
                        help='只 benchmark 这些任务')
    parser.add_argument('--fixed_phase_csv', type=str, default='./TASK_FIXED_PHASE_NUM.csv',
                        help='固定阶段数 CSV 路径')
    parser.add_argument('--fixed_phase_only', action='store_true',
                        help='若不显式提供 tasks，则只跑 CSV 中任务')
    parser.add_argument('--python_executable', type=str, default=sys.executable,
                        help='用于启动 launcher 的 Python 可执行文件')
    parser.add_argument('--skip_validation', action='store_true',
                        help='跳过每轮 benchmark 后的数据编号与 metadata 校验')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印 benchmark 计划，不真正运行')
    return parser.parse_known_args(argv)


def main(argv=None):
    args, passthrough_args = _parse_args(argv)

    conflicts = _extract_conflicting_passthrough_args(passthrough_args)
    if conflicts:
        raise ValueError(
            'These arguments are managed by the benchmark script and must not be forwarded: '
            + ', '.join(conflicts))

    if not args.tasks and not args.fixed_phase_only:
        raise ValueError('Please provide --tasks or enable --fixed_phase_only for benchmark runs')

    if any(worker_count <= 0 for worker_count in args.worker_counts):
        raise ValueError('worker_counts must all be positive integers')

    display_count = len(args.displays)
    normalized_gpu_ids = _normalize_per_display(args.gpu_ids, display_count, 'gpu_ids')
    benchmark_root = _resolve_path(args.benchmark_root)
    os.makedirs(benchmark_root, exist_ok=True)

    planned_runs = []
    for worker_count in args.worker_counts:
        for repeat_index in range(args.repeats):
            run_name = f'workers_{worker_count:02d}_repeat_{repeat_index:02d}'
            run_output_path = os.path.join(benchmark_root, run_name)
            command = _build_launcher_command(args, run_output_path, worker_count, passthrough_args)
            planned_runs.append({
                'run_name': run_name,
                'worker_count': int(worker_count),
                'repeat_index': int(repeat_index),
                'run_output_path': run_output_path,
                'command': command,
            })

    if not planned_runs:
        raise ValueError('No benchmark runs scheduled')

    print(f'[Info] Planned {len(planned_runs)} benchmark run(s).')
    print(f'[Info] Displays: {args.displays}')
    if normalized_gpu_ids:
        print(f'[Info] GPU ids: {normalized_gpu_ids}')
    print(f'[Info] Worker counts: {args.worker_counts}')
    print(f'[Info] Repeats: {args.repeats}')

    benchmark_summary = {
        'started_at': datetime.now().isoformat(),
        'benchmark_root': benchmark_root,
        'displays': args.displays,
        'gpu_ids': normalized_gpu_ids,
        'worker_counts': args.worker_counts,
        'repeats': int(args.repeats),
        'runs': [],
    }

    if args.dry_run:
        for item in planned_runs:
            print('[Dry Run] ' + shlex.join(item['command']))
            benchmark_summary['runs'].append({
                'run_name': item['run_name'],
                'worker_count': item['worker_count'],
                'repeat_index': item['repeat_index'],
                'status': 'dry_run',
                'command': item['command'],
            })
        summary_path = os.path.join(benchmark_root, 'benchmark_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as file_obj:
            json.dump(benchmark_summary, file_obj, ensure_ascii=False, indent=2)
        print(f'[Info] Dry-run benchmark summary saved to {summary_path}')
        return 0

    exit_code = 0
    total_runs = len(planned_runs)
    for run_index, item in enumerate(planned_runs, start=1):
        run_name = item['run_name']
        run_output_path = item['run_output_path']

        os.makedirs(run_output_path, exist_ok=True)
        start_time = time.perf_counter()
        print(f'[Run] {run_index}/{total_runs} {run_name}: ' + shlex.join(item['command']))
        process = subprocess.run(
            item['command'],
            cwd=REPO_ROOT,
            check=False,
        )
        wall_seconds = time.perf_counter() - start_time

        metrics = _collect_dataset_metrics(run_output_path)
        validation_report = None
        if not args.skip_validation:
            validation_report = validate_target_path(run_output_path)

        success_episodes = int(metrics.get('success_episodes', 0))
        done_episodes = int(metrics.get('done_episodes', 0))
        failed_episodes = int(metrics.get('failed_episodes', 0))
        success_per_min = (success_episodes / wall_seconds * 60.0) if wall_seconds > 0 else 0.0
        failure_rate = (failed_episodes / done_episodes) if done_episodes > 0 else 0.0

        run_result = {
            'run_name': run_name,
            'worker_count': item['worker_count'],
            'repeat_index': item['repeat_index'],
            'status': 'completed' if process.returncode == 0 else 'failed',
            'returncode': int(process.returncode),
            'command': item['command'],
            'run_output_path': run_output_path,
            'master_log_path': None,
            'merged_output_path': run_output_path,
            'wall_seconds': round(wall_seconds, 3),
            'success_episodes_per_min': round(success_per_min, 3),
            'failure_rate': round(failure_rate, 6),
            'metrics': metrics,
            'validation': validation_report,
        }
        benchmark_summary['runs'].append(run_result)

        validation_desc = 'skipped'
        if validation_report is not None:
            validation_desc = (
                f'ok={validation_report.get("ok")} '
                f'errors={len(validation_report.get("errors", []))} '
                f'warnings={len(validation_report.get("warnings", []))}')

        print(
            f'[Result] {run_name} workers/display={item["worker_count"]} '
            f'returncode={process.returncode} wall={_format_float(wall_seconds)}s '
            f'success/min={_format_float(success_per_min)} '
            f'failure_rate={_format_float(failure_rate)} validation={validation_desc}')

        if process.returncode != 0 or (validation_report is not None and not validation_report.get('ok')):
            exit_code = 1

    benchmark_summary['finished_at'] = datetime.now().isoformat()
    benchmark_summary['status'] = 'completed' if exit_code == 0 else 'partial_failed'

    successful_runs = [
        run for run in benchmark_summary['runs']
        if run.get('returncode') == 0 and (run.get('validation') is None or run['validation'].get('ok'))
    ]
    successful_runs.sort(key=lambda item: item.get('success_episodes_per_min', 0.0), reverse=True)
    benchmark_summary['ranking'] = [
        {
            'run_name': run['run_name'],
            'worker_count': run['worker_count'],
            'repeat_index': run['repeat_index'],
            'success_episodes_per_min': run['success_episodes_per_min'],
            'failure_rate': run['failure_rate'],
        }
        for run in successful_runs
    ]

    summary_path = os.path.join(benchmark_root, 'benchmark_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as file_obj:
        json.dump(benchmark_summary, file_obj, ensure_ascii=False, indent=2)
    print(f'[Info] Benchmark summary saved to {summary_path}')

    if successful_runs:
        best_run = successful_runs[0]
        print(
            f'[Best] run={best_run["run_name"]} workers/display={best_run["worker_count"]} '
            f'success/min={best_run["success_episodes_per_min"]} '
            f'failure_rate={best_run["failure_rate"]}')
    else:
        print('[Best] No fully successful and validated benchmark run found.')

    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
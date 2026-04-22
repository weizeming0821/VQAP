# -*- coding: utf-8 -*-
"""仅处理 TASK_FIXED_PHASE_NUM.csv 中任务的包装入口。"""

import argparse
import csv
import os
import subprocess
import sys


def _resolve_csv_path(csv_path):
	if os.path.isabs(csv_path):
		return csv_path

	cwd_path = os.path.abspath(csv_path)
	if os.path.exists(cwd_path):
		return cwd_path

	script_dir = os.path.dirname(os.path.abspath(__file__))
	script_path = os.path.join(script_dir, csv_path)
	if os.path.exists(script_path):
		return script_path

	return cwd_path


def _load_fixed_phase_tasks(csv_path):
	resolved_csv_path = _resolve_csv_path(csv_path)
	if not os.path.exists(resolved_csv_path):
		raise ValueError(f'Fixed-phase CSV not found: {resolved_csv_path}')

	task_names = []
	with open(resolved_csv_path, 'r', encoding='utf-8-sig', newline='') as file_obj:
		reader = csv.DictReader(file_obj)
		for row in reader:
			task_name = (row.get('task_name') or '').strip()
			fixed_phase_num = (row.get('fixed_phase_num') or '').strip()
			if task_name and fixed_phase_num:
				task_names.append(task_name)

	if not task_names:
		raise ValueError(f'No fixed-phase tasks found in {resolved_csv_path}')

	return resolved_csv_path, sorted(set(task_names))


def _get_available_task_names():
	import rlbench.backend.task as task

	return sorted(
		file_name[:-3]
		for file_name in os.listdir(task.TASKS_PATH)
		if file_name.endswith('.py') and file_name != '__init__.py'
	)


def _resolve_tasks_from_fixed_phase_csv(csv_path, requested_tasks):
	resolved_csv_path, csv_tasks = _load_fixed_phase_tasks(csv_path)
	available_tasks = set(_get_available_task_names())

	missing_tasks = sorted(task_name for task_name in csv_tasks if task_name not in available_tasks)
	valid_csv_tasks = sorted(task_name for task_name in csv_tasks if task_name in available_tasks)

	if missing_tasks:
		print('[Warn] These TASK_FIXED_PHASE_NUM.csv tasks are not available in RLBench and will be skipped:')
		print('  ' + ', '.join(missing_tasks))

	if requested_tasks:
		invalid_tasks = sorted(task_name for task_name in requested_tasks if task_name not in available_tasks)
		if invalid_tasks:
			raise ValueError(f'Task(s) not recognised: {", ".join(invalid_tasks)}')

		not_in_csv = sorted(task_name for task_name in requested_tasks if task_name not in csv_tasks)
		if not_in_csv:
			raise ValueError('Task(s) not listed in TASK_FIXED_PHASE_NUM.csv: ' + ', '.join(not_in_csv))

		return resolved_csv_path, requested_tasks

	if not valid_csv_tasks:
		raise ValueError('No valid RLBench tasks remain after filtering TASK_FIXED_PHASE_NUM.csv')

	return resolved_csv_path, valid_csv_tasks


def _parse_wrapper_args(argv):
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--fixed_phase_csv', default='./TASK_FIXED_PHASE_NUM.csv')
	parser.add_argument('--tasks', nargs='*', default=[])
	return parser.parse_known_args(argv)


def main(argv=None):
	argv = list(sys.argv[1:] if argv is None else argv)
	known_args, passthrough = _parse_wrapper_args(argv)
	resolved_csv_path, task_names = _resolve_tasks_from_fixed_phase_csv(
		known_args.fixed_phase_csv,
		known_args.tasks,
	)

	command = [
		sys.executable,
		os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traj_generator_segmentation.py'),
		'--fixed_phase_csv', resolved_csv_path,
		'--tasks',
		*task_names,
		*passthrough,
	]

	print(f'[Info] Restricted to {len(task_names)} task(s) from {resolved_csv_path}')
	raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
	main()

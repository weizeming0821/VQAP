# -*- coding: utf-8 -*-
"""
阶段数验证模块

根据 TASK_FIXED_PHASE_NUM.csv 验证每个任务的分割结果是否符合要求。
"""

import csv
import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_fixed_phase_csv_path(csv_path):
    """解析固定 phase CSV 路径，优先使用当前工作目录，其次回退到仓库根目录。"""
    if os.path.isabs(csv_path):
        return csv_path

    cwd_path = os.path.abspath(csv_path)
    if os.path.exists(cwd_path):
        return cwd_path

    repo_path = os.path.join(REPO_ROOT, csv_path)
    if os.path.exists(repo_path):
        return repo_path

    return cwd_path


def load_fixed_phase_config(csv_path, warn_if_missing=True):
    """
    从 CSV 文件加载每个任务的固定阶段数配置。

    Args:
        csv_path: str，CSV 文件路径

    Returns:
        dict: task_name -> fixed_phase_num
    """
    config = {}
    resolved_csv_path = resolve_fixed_phase_csv_path(csv_path)
    if not os.path.exists(resolved_csv_path):
        if warn_if_missing:
            print(f'[Warning] Fixed phase config not found: {resolved_csv_path}')
        return config

    with open(resolved_csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row.get('task_name', '').strip()
            fixed_phase_num = row.get('fixed_phase_num', '').strip()
            if task_name and fixed_phase_num:
                config[task_name] = int(fixed_phase_num)

    return config


def split_tasks_by_fixed_phase_config(task_names, fixed_phase_config):
    """按是否命中固定 phase 配置拆分任务，保留原始顺序。"""
    fixed_phase_tasks = []
    normal_tasks = []
    for task_name in task_names:
        if task_name in fixed_phase_config:
            fixed_phase_tasks.append(task_name)
        else:
            normal_tasks.append(task_name)
    return fixed_phase_tasks, normal_tasks


def validate_phase_count(task_name, num_phases, fixed_phase_config):
    """
    验证分割后的阶段数是否符合要求。

    Args:
        task_name:          str，任务名
        num_phases:         int，实际阶段数
        fixed_phase_config: dict，固定阶段数配置

    Returns:
        tuple: (valid, expected)
            valid: bool，是否有效
            expected: int，期望的阶段数（如果配置了）
    """
    if task_name not in fixed_phase_config:
        # 未配置该任务的固定阶段数，默认有效
        return True, None

    expected = fixed_phase_config[task_name]
    return (num_phases == expected), expected


def get_expected_phase_count(task_name, fixed_phase_config):
    """
    获取任务期望的阶段数。

    Args:
        task_name:          str，任务名
        fixed_phase_config: dict，固定阶段数配置

    Returns:
        int or None: 期望的阶段数
    """
    return fixed_phase_config.get(task_name, None)

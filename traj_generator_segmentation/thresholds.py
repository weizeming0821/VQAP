# -*- coding: utf-8 -*-
"""
自动阈值估计模块

从 demo 数据分布自动估计各信号阈值，适配不同任务。
"""

import numpy as np


def auto_thresholds(demo):
    """
    从 demo 数据分布自动估计各信号阈值，适配不同任务。

    Args:
        demo: List[Observation]，RLBench 演示轨迹

    Returns:
        dict: 信号名 → 阈值（仅包含可从数据推断的信号）
    """
    thresholds = {}
    n = len(demo)

    # ── vel：速度近零阈值 = max(均值 - 0.5×std, 1e-3) ─────────────────────
    vel_norms = []
    for obs in demo:
        if obs is not None and obs.joint_velocities is not None:
            vel_norms.append(np.linalg.norm(obs.joint_velocities))
    if vel_norms:
        v_arr = np.array(vel_norms)
        thresholds['vel'] = float(np.mean(v_arr) - 0.5 * np.std(v_arr))
    else:
        thresholds['vel'] = 0.02

    # ── dir：末端方向突变阈值 = 85 百分位位移变化量 ─────────────────────
    cos_changes = []
    for i in range(1, n):
        if demo[i] is None or demo[i - 1] is None:
            continue
        p = getattr(demo[i], 'gripper_pose', None)
        pp = getattr(demo[i - 1], 'gripper_pose', None)
        if p is None or pp is None:
            continue
        d_curr = np.array(p[:3])
        d_prev = np.array(pp[:3])
        diff = d_curr - d_prev
        norm = np.linalg.norm(diff)
        if norm > 1e-6:
            cos_changes.append(norm)
    if len(cos_changes) >= 10:
        thresholds['dir'] = float(np.percentile(cos_changes, 85))
    else:
        thresholds['dir'] = 0.05

    # ── contact：夹爪接触力突变阈值 = 均值 + 1.5×std ────────────────────
    contact_vals = []
    for obs in demo:
        if obs is None:
            continue
        t = getattr(obs, 'gripper_touch_forces', None)
        if t is not None:
            contact_vals.append(float(np.linalg.norm(t)))
    if contact_vals:
        c_arr = np.array(contact_vals)
        thresholds['contact'] = float(np.mean(c_arr) + 1.5 * np.std(c_arr))
    else:
        thresholds['contact'] = 0.1

    # ── force：关节力矩突变阈值 = 差分序列均值 + 1.5×std ─────────────────
    force_deltas = []
    for i in range(1, n):
        if demo[i] is None or demo[i - 1] is None:
            continue
        f_curr = getattr(demo[i], 'joint_forces', None)
        f_prev = getattr(demo[i - 1], 'joint_forces', None)
        if f_curr is not None and f_prev is not None:
            force_deltas.append(
                float(np.linalg.norm(np.array(f_curr) - np.array(f_prev))))
    if force_deltas:
        fd = np.array(force_deltas)
        thresholds['force'] = float(np.mean(fd) + 1.5 * np.std(fd))
    else:
        thresholds['force'] = 5.0

    # ── acc：关节加速度突变阈值 = 差分序列均值 + 1.5×std ─────────────────
    acc_vals = []
    for i in range(1, n):
        if demo[i] is None or demo[i - 1] is None:
            continue
        v_curr = getattr(demo[i], 'joint_velocities', None)
        v_prev = getattr(demo[i - 1], 'joint_velocities', None)
        if v_curr is not None and v_prev is not None:
            acc_vals.append(
                float(np.linalg.norm(np.array(v_curr) - np.array(v_prev))))
    if acc_vals:
        a_arr = np.array(acc_vals)
        thresholds['acc'] = float(np.mean(a_arr) + 1.5 * np.std(a_arr))
    else:
        thresholds['acc'] = 1.0

    return thresholds

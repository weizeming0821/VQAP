# -*- coding: utf-8 -*-
"""
信号候选帧提取模块

支持 6 种信号，每种信号提取候选关键帧：
    S1 gripper  : 夹爪开合状态变化
    S2 vel      : 关节速度由动变静（到达 waypoint）
    S3 dir      : 末端运动方向突变
    S4 contact  : 夹爪接触力突变
    S5 force    : 关节力矩突变
    S6 acc      : 关节加速度突变
"""

import numpy as np
from .config import GRIPPER_OPEN_THR

# 当前启用的所有信号
ALL_SIGNALS = ('gripper', 'vel', 'contact')


# ─────────────────────────────────────────────────────────────────────────────
# S1: 夹爪开合变化
# ─────────────────────────────────────────────────────────────────────────────

def _candidates_gripper(demo):
    """S1：夹爪开合变化帧（取变化后第一帧）。"""
    cands = set()
    n = len(demo)
    for i in range(1, n):
        if demo[i] is None or demo[i - 1] is None:
            continue
        prev = float(demo[i - 1].gripper_open) > GRIPPER_OPEN_THR
        curr = float(demo[i].gripper_open) > GRIPPER_OPEN_THR
        if prev != curr:
            cands.add(i)
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# S2: 速度由动变静
# ─────────────────────────────────────────────────────────────────────────────

def _candidates_vel(demo, threshold, window=3):
    """S2：速度由动变静（连续 window 帧低于阈值且之前在运动）。"""
    cands = set()
    n = len(demo)
    low_count = 0
    for i in range(n):
        if demo[i] is None or demo[i].joint_velocities is None:
            low_count = 0
            continue
        vnorm = np.linalg.norm(demo[i].joint_velocities)
        if vnorm < threshold:
            low_count += 1
            if low_count == window:
                stop_idx = i - window + 1
                if stop_idx > 0 and demo[stop_idx - 1] is not None:
                    pv = np.linalg.norm(demo[stop_idx - 1].joint_velocities)
                    if pv > threshold:
                        cands.add(stop_idx)
        else:
            low_count = 0
    return cands


def _candidates_vel_start(demo, threshold, still_window=3, move_window=3,
                          hysteresis=1.15, prior_move_window=5):
    """S2b：速度由静变动（仅保留"运动→静止→再运动"中的再启动帧）。"""
    cands = set()
    n = len(demo)
    if n <= 1:
        return cands

    vnorms = []
    for obs in demo:
        if obs is None or obs.joint_velocities is None:
            vnorms.append(None)
        else:
            vnorms.append(float(np.linalg.norm(obs.joint_velocities)))

    move_thr = threshold * hysteresis
    for i in range(still_window, n - move_window + 1):
        if i < still_window + prior_move_window:
            continue

        prior_vals = vnorms[i - still_window - prior_move_window:i - still_window]
        prev_vals = vnorms[i - still_window:i]
        next_vals = vnorms[i:i + move_window]
        if (any(v is None for v in prior_vals) or
                any(v is None for v in prev_vals) or
                any(v is None for v in next_vals)):
            continue
        if (all(v > threshold for v in prior_vals) and
                all(v < threshold for v in prev_vals) and
                all(v > move_thr for v in next_vals)):
            cands.add(i)

    return cands


# ─────────────────────────────────────────────────────────────────────────────
# S3: 末端方向突变
# ─────────────────────────────────────────────────────────────────────────────

def _candidates_dir(demo, threshold):
    """S3：末端 XYZ 位移方向突变。"""
    cands = set()
    n = len(demo)
    for i in range(2, n):
        if any(demo[j] is None for j in (i, i - 1, i - 2)):
            continue
        p0 = getattr(demo[i - 2], 'gripper_pose', None)
        p1 = getattr(demo[i - 1], 'gripper_pose', None)
        p2 = getattr(demo[i], 'gripper_pose', None)
        if p0 is None or p1 is None or p2 is None:
            continue
        d1 = np.array(p1[:3]) - np.array(p0[:3])
        d2 = np.array(p2[:3]) - np.array(p1[:3])
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        change = np.linalg.norm(d2 / n2 - d1 / n1)
        if change > threshold:
            cands.add(i)
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# S4: 夹爪接触力突变
# ─────────────────────────────────────────────────────────────────────────────

def _candidates_contact(demo, threshold, window=3):
    """S4：夹爪接触力突变（当前帧力范数与前 window 帧均值之差 > threshold）。"""
    cands = set()
    n = len(demo)
    force_norms = []
    for obs in demo:
        if obs is None:
            force_norms.append(None)
            continue
        t = getattr(obs, 'gripper_touch_forces', None)
        force_norms.append(float(np.linalg.norm(t)) if t is not None else None)

    for i in range(window, n):
        if force_norms[i] is None:
            continue
        prev_vals = [force_norms[j] for j in range(i - window, i)
                     if force_norms[j] is not None]
        if not prev_vals:
            continue
        if abs(force_norms[i] - float(np.mean(prev_vals))) > threshold:
            cands.add(i)
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# S5: 关节力矩突变
# ─────────────────────────────────────────────────────────────────────────────

def _candidates_force(demo, threshold):
    """S5：关节力矩突变（相邻帧力矩向量差范数 > threshold）。"""
    cands = set()
    n = len(demo)
    for i in range(1, n):
        if demo[i] is None or demo[i - 1] is None:
            continue
        fc = getattr(demo[i], 'joint_forces', None)
        fp = getattr(demo[i - 1], 'joint_forces', None)
        if fc is None or fp is None:
            continue
        if np.linalg.norm(np.array(fc) - np.array(fp)) > threshold:
            cands.add(i)
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# S6: 关节加速度突变
# ─────────────────────────────────────────────────────────────────────────────

def _candidates_acc(demo, threshold, persist=2):
    """S6：关节加速度突变（速度差分范数 > threshold，且持续 persist 帧）。"""
    cands = set()
    n = len(demo)
    acc_norms = []
    for i in range(1, n):
        if demo[i] is None or demo[i - 1] is None:
            acc_norms.append(0.0)
            continue
        vc = getattr(demo[i], 'joint_velocities', None)
        vp = getattr(demo[i - 1], 'joint_velocities', None)
        if vc is None or vp is None:
            acc_norms.append(0.0)
        else:
            acc_norms.append(float(np.linalg.norm(np.array(vc) - np.array(vp))))

    for i in range(len(acc_norms) - persist + 1):
        if all(acc_norms[i + j] > threshold for j in range(persist)):
            cands.add(i + 1)
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# 候选帧汇总
# ─────────────────────────────────────────────────────────────────────────────

def collect_stage1_candidates(demo, signals, thresholds):
    """
    用所有启用信号提取候选帧，并直接汇总去重。

    Args:
        demo:       List[Observation]
        signals:    启用的信号名称集合（子集 of ALL_SIGNALS）
        thresholds: dict，信号名 → 阈值（由 auto_thresholds 生成）

    Returns:
        candidates: List[int]，汇总候选帧（已排序）
        frame_signals: dict[int, list[str]]，帧索引 → 触发信号列表
        signal_candidates: dict[str, list[int]]，每个信号对应的候选帧列表
    """
    n = len(demo)
    candidate_set = set()
    frame_signals = {}
    signal_candidates = {}

    def _add(cands, sig):
        clean_cands = sorted(idx for idx in cands if 0 <= idx < n)
        signal_candidates[sig] = clean_cands
        for idx in clean_cands:
            candidate_set.add(idx)
            frame_signals.setdefault(idx, []).append(sig)

    if 'gripper' in signals:
        _add(_candidates_gripper(demo), 'gripper')
    if 'vel' in signals:
        vel_cands = set(_candidates_vel(demo, thresholds['vel']))
        vel_cands.update(_candidates_vel_start(demo, thresholds['vel']))
        _add(vel_cands, 'vel')
    if 'dir' in signals:
        _add(_candidates_dir(demo, thresholds['dir']), 'dir')
    if 'contact' in signals:
        _add(_candidates_contact(demo, thresholds['contact']), 'contact')
    if 'force' in signals:
        _add(_candidates_force(demo, thresholds['force']), 'force')
    if 'acc' in signals:
        _add(_candidates_acc(demo, thresholds['acc']), 'acc')

    return sorted(candidate_set), frame_signals, signal_candidates

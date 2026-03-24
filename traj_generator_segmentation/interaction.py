# -*- coding: utf-8 -*-
"""
交互段判定模块

判断每个分割段是否处于"交互状态"（机械臂正与物体作用），
用于交互感知合并。
"""

import numpy as np
from .config import (
    LOW_CONTACT_FORCE, CONTACT_PERSIST, BOUNDARY_INTERACT_WINDOW,
    MID_INTERACT_RUN_THR, OPEN_FORCE_DELTA_SCALE, OPEN_VEL_SCALE,
    OPEN_HOLD_VEL_SCALE, OPEN_HOLD_FORCE_DELTA_SCALE,
    INTERACT_HOLD_GAP_FRAMES, OPEN_D_ENTER_PERSIST,
    OPEN_AFTER_RELEASE_D_COOLDOWN, CLOSED_PRESS_ENTER_PERSIST,
    GRIPPER_OPEN_THR,
)


# ─────────────────────────────────────────────────────────────────────────────
# 帧级交互判定
# ─────────────────────────────────────────────────────────────────────────────

def label_interacting_frames(demo, contact_thr, vel_thr, force_delta_thr,
                             persist=CONTACT_PERSIST, return_debug=False):
    """
    逐帧判断是否处于"与物体交互"状态，返回 bool 列表。

    帧证据 OR 逻辑：
      A. touch_force_norm > contact_thr
      B. gripper_open==0 AND grasp_likely AND vel_norm>vel_thr
      C. gripper_open==1 AND touch_force_norm>LOW_CONTACT_FORCE
      D. gripper_open==1 AND force_delta_norm > force_delta_thr * OPEN_FORCE_DELTA_SCALE
          AND vel_norm > vel_thr * OPEN_VEL_SCALE
      E. gripper_open==0 AND grasp_likely==0 AND touch_force_norm>LOW_CONTACT_FORCE
          AND force_delta_norm > force_delta_thr * OPEN_FORCE_DELTA_SCALE

    Args:
        demo:        List[Observation]
        contact_thr: 接触力阈值
        vel_thr:     速度阈值
        force_delta_thr: 力矩变化阈值
        persist:     接触持续帧数要求

    Returns:
        List[bool]，长度与 demo 相同
        若 return_debug=True，额外返回调试信息 dict
    """
    n = len(demo)
    fn_list = [0.0] * n
    vn_list = [0.0] * n
    fd_list = [0.0] * n
    g_open_list = [True] * n
    valid = [False] * n

    for i, obs in enumerate(demo):
        if obs is None:
            continue
        valid[i] = True
        touch = getattr(obs, 'gripper_touch_forces', None)
        fn_list[i] = float(np.linalg.norm(touch)) if touch is not None else 0.0
        g_open_list[i] = float(getattr(obs, 'gripper_open', 1.0)) > GRIPPER_OPEN_THR
        vel = getattr(obs, 'joint_velocities', None)
        vn_list[i] = float(np.linalg.norm(vel)) if vel is not None else 0.0
        if i > 0 and demo[i - 1] is not None:
            f_prev = getattr(demo[i - 1], 'joint_forces', None)
            f_curr = getattr(obs, 'joint_forces', None)
            if f_prev is not None and f_curr is not None:
                fd_list[i] = float(np.linalg.norm(np.array(f_curr) - np.array(f_prev)))

    # 推断"疑似持物"状态
    grasp_likely = [False] * n
    latched = False
    touch_thr = max(LOW_CONTACT_FORCE, contact_thr * 0.35)
    for i in range(n):
        if not valid[i]:
            continue
        if i > 0 and valid[i - 1]:
            if g_open_list[i - 1] and (not g_open_list[i]):
                ws = max(0, i - 3)
                we = min(n, i + 4)
                near_touch = any(valid[j] and (fn_list[j] > touch_thr)
                                 for j in range(ws, we))
                near_open_load = any(
                    valid[j] and g_open_list[j] and
                    (fd_list[j] > force_delta_thr * OPEN_FORCE_DELTA_SCALE) and
                    (vn_list[j] > vel_thr * OPEN_VEL_SCALE)
                    for j in range(ws, we)
                )
                latched = bool(near_touch or near_open_load)
            elif (not g_open_list[i - 1]) and g_open_list[i]:
                latched = False
        grasp_likely[i] = latched

    raw = [False] * n
    raw_reasons = {}
    for i, obs in enumerate(demo):
        if not valid[i]:
            continue
        fn = fn_list[i]
        g_open = g_open_list[i]
        vn = vn_list[i]

        cond_a = fn > contact_thr
        cond_b = (not g_open) and grasp_likely[i] and (vn > vel_thr)
        cond_c = g_open and (fn > LOW_CONTACT_FORCE)

        fd = fd_list[i]
        cond_d = (
            g_open and
            (fd > force_delta_thr * OPEN_FORCE_DELTA_SCALE) and
            (vn > vel_thr * OPEN_VEL_SCALE)
        )
        cond_e = (
            (not g_open) and
            (not grasp_likely[i]) and
            (fn > LOW_CONTACT_FORCE) and
            (fd > force_delta_thr * OPEN_FORCE_DELTA_SCALE)
        )

        reasons = []
        if cond_a:
            reasons.append('A_touch_force')
        if cond_b:
            reasons.append('B_closed_gripper_transport')
        if cond_c:
            reasons.append('C_open_gripper_light_touch')
        if cond_d:
            reasons.append('D_open_gripper_force_delta_motion')
        if cond_e:
            reasons.append('E_closed_gripper_pressing')

        raw[i] = bool(reasons)
        if reasons:
            raw_reasons[i] = reasons

    # 帧级状态机平滑
    result = [False] * n
    active = False
    strong_streak = 0
    d_streak = 0
    e_streak = 0
    gap = 0
    last_release_idx = -10**9

    for i, obs in enumerate(demo):
        if not valid[i]:
            if active:
                gap += 1
                if gap <= INTERACT_HOLD_GAP_FRAMES:
                    result[i] = True
                else:
                    active = False
                    gap = 0
            strong_streak = 0
            continue

        fn = fn_list[i]
        g_open = g_open_list[i]
        vn = vn_list[i]
        fd = fd_list[i]

        if i > 0 and valid[i - 1] and (not g_open_list[i - 1]) and g_open:
            last_release_idx = i
        in_release_cooldown = (i - last_release_idx) <= OPEN_AFTER_RELEASE_D_COOLDOWN

        cond_a = fn > contact_thr
        cond_b = (not g_open) and grasp_likely[i] and (vn > vel_thr)
        cond_c = g_open and (fn > LOW_CONTACT_FORCE)
        cond_d_raw = (
            g_open and
            (fd > force_delta_thr * OPEN_FORCE_DELTA_SCALE) and
            (vn > vel_thr * OPEN_VEL_SCALE)
        )
        cond_d = cond_d_raw and (not in_release_cooldown)
        cond_e = (
            (not g_open) and
            (not grasp_likely[i]) and
            (fn > LOW_CONTACT_FORCE) and
            (fd > force_delta_thr * OPEN_FORCE_DELTA_SCALE)
        )
        strong = cond_a or cond_b or cond_c or cond_d or cond_e

        if strong:
            strong_streak += 1
        else:
            strong_streak = 0
        if cond_d:
            d_streak += 1
        else:
            d_streak = 0
        if cond_e:
            e_streak += 1
        else:
            e_streak = 0

        if not active:
            bc = cond_b or cond_c
            if (cond_a or
                    (bc and strong_streak >= persist) or
                    (d_streak >= OPEN_D_ENTER_PERSIST) or
                    (e_streak >= CLOSED_PRESS_ENTER_PERSIST)):
                active = True
                gap = 0

        hold_open = (
            g_open and
            (vn > vel_thr * OPEN_HOLD_VEL_SCALE) and
            (fd > force_delta_thr * OPEN_HOLD_FORCE_DELTA_SCALE)
        )
        hold_closed = cond_b or cond_e
        hold = strong or hold_open or hold_closed

        if active:
            if hold:
                result[i] = True
                gap = 0
            else:
                gap += 1
                if gap <= INTERACT_HOLD_GAP_FRAMES:
                    result[i] = True
                else:
                    active = False
                    gap = 0

    accepted_runs = []
    i = 0
    while i < n:
        if result[i]:
            j = i
            while j < n and result[j]:
                j += 1
            accepted_runs.append((i, j - 1))
            i = j
        else:
            i += 1

    if not return_debug:
        return result

    interacting_frames = [idx for idx, flag in enumerate(result) if flag]
    frame_debug = {
        'raw_true_frames': sorted(raw_reasons.keys()),
        'raw_reasons': {int(k): v for k, v in raw_reasons.items()},
        'accepted_runs': [[int(s), int(e)] for s, e in accepted_runs],
        'interacting_frames': interacting_frames,
        'persist': int(persist),
    }
    return result, frame_debug


# ─────────────────────────────────────────────────────────────────────────────
# 段级交互判定
# ─────────────────────────────────────────────────────────────────────────────

def label_interacting_segments(keyframes, demo, contact_thr, vel_thr, force_thr,
                               return_debug=False):
    """
    根据每段首尾帧是否交互，判断该段是否为"交互段"。

    Args:
        keyframes: List[int]，关键帧索引列表
        demo:      List[Observation]
        contact_thr, vel_thr, force_thr: 各类阈值

    Returns:
        List[bool]，长度 = len(keyframes)
        若 return_debug=True，额外返回段级调试信息 dict
    """
    n = len(demo)
    frame_result = label_interacting_frames(
        demo, contact_thr, vel_thr,
        force_delta_thr=force_thr,
        return_debug=return_debug,
    )
    if return_debug:
        frame_labels, frame_debug = frame_result
    else:
        frame_labels = frame_result
        frame_debug = None

    boundaries = [0]
    for kf in keyframes:
        nxt = kf + 1
        if nxt < n:
            boundaries.append(nxt)
    if boundaries[-1] != n:
        boundaries.append(n)

    def _find_first_valid_label(start, end):
        for i in range(start, end):
            if demo[i] is not None:
                return frame_labels[i]
        return False

    def _find_last_valid_label(start, end):
        for i in range(end - 1, start - 1, -1):
            if demo[i] is not None:
                return frame_labels[i]
        return False

    def _max_true_run(start, end):
        run = 0
        best = 0
        for i in range(start, end):
            if frame_labels[i]:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        return best

    # 夹爪开合切换检测
    fn_list = [0.0] * n
    vn_list = [0.0] * n
    fd_list = [0.0] * n
    g_open_list = [True] * n
    valid = [False] * n
    for i, obs in enumerate(demo):
        if obs is None:
            continue
        valid[i] = True
        touch = getattr(obs, 'gripper_touch_forces', None)
        fn_list[i] = float(np.linalg.norm(touch)) if touch is not None else 0.0
        g_open_list[i] = float(getattr(obs, 'gripper_open', 1.0)) > GRIPPER_OPEN_THR
        vel = getattr(obs, 'joint_velocities', None)
        vn_list[i] = float(np.linalg.norm(vel)) if vel is not None else 0.0
        if i > 0 and demo[i - 1] is not None:
            f_prev = getattr(demo[i - 1], 'joint_forces', None)
            f_curr = getattr(obs, 'joint_forces', None)
            if f_prev is not None and f_curr is not None:
                fd_list[i] = float(np.linalg.norm(np.array(f_curr) - np.array(f_prev)))

    def _is_gripper_toggle(i):
        if i <= 0 or i >= n:
            return False
        if not (valid[i] and valid[i - 1]):
            return False
        prev_open = g_open_list[i - 1]
        curr_open = g_open_list[i]
        if prev_open == curr_open:
            return False
        ws = max(0, i - 2)
        we = min(n, i + 3)
        touch_thr = max(LOW_CONTACT_FORCE, contact_thr * 0.35)
        has_touch = any(valid[j] and (fn_list[j] > touch_thr) for j in range(ws, we))
        has_open_load = any(
            valid[j] and g_open_list[j] and
            (fd_list[j] > force_thr * OPEN_FORCE_DELTA_SCALE) and
            (vn_list[j] > vel_thr * OPEN_VEL_SCALE)
            for j in range(ws, we)
        )
        has_frame_interact = any(frame_labels[j] for j in range(ws, we))
        return bool(has_touch or has_open_load or has_frame_interact)

    seg_labels = []
    seg_debug = []
    for p in range(len(keyframes)):
        start = boundaries[p]
        end = boundaries[p + 1]
        if end <= start:
            seg_labels.append(False)
            seg_debug.append({
                'segment_index': int(p),
                'keyframe': int(keyframes[p]),
                'start': int(start),
                'end': int(end),
                'is_interacting': False,
                'reason': 'empty_segment',
            })
            continue

        head_win_end = min(end, start + BOUNDARY_INTERACT_WINDOW)
        tail_win_start = max(start, end - BOUNDARY_INTERACT_WINDOW)

        head_window_hit = any(frame_labels[i] for i in range(start, head_win_end))
        tail_window_hit = any(frame_labels[i] for i in range(tail_win_start, end))

        head_interact = (
            _find_first_valid_label(start, end) or
            _is_gripper_toggle(start) or
            head_window_hit
        )
        tail_interact = (
            _find_last_valid_label(start, end) or
            _is_gripper_toggle(end - 1) or
            tail_window_hit
        )

        endpoint_interact = bool(head_interact or tail_interact)
        if endpoint_interact:
            seg_labels.append(True)
            seg_debug.append({
                'segment_index': int(p),
                'keyframe': int(keyframes[p]),
                'start': int(start),
                'end': int(end),
                'is_interacting': True,
                'reason': 'endpoint_interact',
            })
            continue

        mid_run = _max_true_run(start, end)
        is_interacting = mid_run >= MID_INTERACT_RUN_THR
        seg_labels.append(is_interacting)
        seg_debug.append({
            'segment_index': int(p),
            'keyframe': int(keyframes[p]),
            'start': int(start),
            'end': int(end),
            'is_interacting': bool(is_interacting),
            'reason': 'mid_run_fallback' if is_interacting else 'non_interacting',
        })

    if not return_debug:
        return seg_labels

    return seg_labels, {
        'frame_debug': frame_debug,
        'segment_debug': seg_debug,
    }

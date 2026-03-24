# -*- coding: utf-8 -*-
"""
关键帧提取主逻辑模块

三阶段关键帧提取：
    阶段 1 — 候选帧提取与汇总
    阶段 2 — 距离合并
    阶段 3 — 交互感知合并
    阶段 4 — 静止段丢弃
"""

import numpy as np
from .config import (
    RUN_MIN_PHASE_LEN, RUN_DEBUG_TRACE, RUN_SHOW_SEG_TRACE, RUN_TRACE_MAX_ITEMS,
    STATIC_RUN_DROP_THR, STATIC_POSE_EPS, STATIC_ROT_EPS_RAD,
    STATIC_RUN_JITTER_POSE_SCALE, STATIC_RUN_JITTER_ROT_SCALE,
    STATIC_SLOW_MOTION_POSE_SCALE, STATIC_SLOW_MOTION_ROT_SCALE,
    GRIPPER_OPEN_THR,
)
from .thresholds import auto_thresholds
from .signals import collect_stage1_candidates, ALL_SIGNALS
from .interaction import label_interacting_segments


# ─────────────────────────────────────────────────────────────────────────────
# 距离合并
# ─────────────────────────────────────────────────────────────────────────────

def merge_by_distance_with_trace(keyframes, min_phase_len):
    """合并距离 < min_phase_len 的相邻关键帧（保留靠后的帧），并返回合并追踪。"""
    if not keyframes:
        return keyframes, []

    merged = [keyframes[0]]
    merge_trace = []
    for idx in keyframes[1:]:
        gap = idx - merged[-1]
        if gap < min_phase_len:
            dropped = merged[-1]
            merged[-1] = idx
            merge_trace.append({
                'stage': 'distance_merge',
                'dropped': int(dropped),
                'kept': int(idx),
                'gap': int(gap),
                'min_phase_len': int(min_phase_len),
            })
        else:
            merged.append(idx)
    return merged, merge_trace


def merge_by_distance(keyframes, min_phase_len):
    """合并距离 < min_phase_len 的相邻关键帧（保留靠后的帧）。"""
    merged, _ = merge_by_distance_with_trace(keyframes, min_phase_len)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 交互感知合并
# ─────────────────────────────────────────────────────────────────────────────

def merge_non_interacting_blocks(keyframes, seg_labels, return_trace=False):
    """
    对非交互段的连续块做方向感知合并。

    块在轨迹中的位置决定保留策略：
      左无邻 / 右有交互  (approach 块) → 仅保留块最后一段
      左有交互 / 右有交互 (transit 块) → 仅保留块最后一段
      左有交互 / 右无邻  (reset 块)   → 全部删除
      左无邻 / 右无邻   (全无交互)   → 保留所有
    """
    n_seg = len(keyframes)
    if n_seg == 0:
        if return_trace:
            return keyframes, []
        return keyframes

    any_interact = any(seg_labels)
    kept = []
    merge_trace = []
    i = 0

    while i < n_seg:
        if seg_labels[i]:
            kept.append(keyframes[i])
            i += 1
        else:
            j = i
            while j < n_seg and not seg_labels[j]:
                j += 1
            block_kf = keyframes[i:j]

            if not any_interact:
                keep_block = list(block_kf)
                rule = 'all_non_interact_keep_all'
            else:
                has_left = (i > 0)
                has_right = (j < n_seg)

                if not has_left and has_right:
                    keep_block = [block_kf[-1]]
                    rule = 'approach_keep_last'
                elif has_left and has_right:
                    keep_block = [block_kf[-1]]
                    rule = 'transit_keep_last'
                elif has_left and not has_right:
                    keep_block = []
                    rule = 'reset_drop_all'
                else:
                    keep_block = list(block_kf)
                    rule = 'fallback_keep_all'

            kept.extend(keep_block)
            dropped = [kf for kf in block_kf if kf not in keep_block]
            for d in dropped:
                merge_trace.append({
                    'stage': 'non_interacting_merge',
                    'dropped': int(d),
                    'block': [int(x) for x in block_kf],
                    'kept_in_block': [int(x) for x in keep_block],
                    'rule': rule,
                })
            i = j

    result = sorted(set(kept))
    if return_trace:
        return result, merge_trace
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 静止段检测与丢弃
# ─────────────────────────────────────────────────────────────────────────────

def _quat_angle_delta(q_prev, q_curr):
    """返回两个四元数间的最小夹角（弧度）。"""
    a = np.asarray(q_prev, dtype=float).reshape(-1)
    b = np.asarray(q_curr, dtype=float).reshape(-1)
    if a.shape[0] != 4 or b.shape[0] != 4:
        return None
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    a = a / na
    b = b / nb
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _max_static_run_in_segment(demo, start, end,
                               pose_eps=STATIC_POSE_EPS,
                               rot_eps=STATIC_ROT_EPS_RAD):
    """返回 [start, end) 内最长静止连续段信息。"""
    if end - start <= 1:
        return 0, None, None

    trans_strict = []
    trans_relaxed = []
    for i in range(start + 1, end):
        prev = demo[i - 1]
        curr = demo[i]
        if prev is None or curr is None:
            trans_strict.append(False)
            trans_relaxed.append(False)
            continue

        pp = getattr(prev, 'gripper_pose', None)
        cp = getattr(curr, 'gripper_pose', None)
        if pp is None or cp is None:
            trans_strict.append(False)
            trans_relaxed.append(False)
            continue

        prev_open = float(getattr(prev, 'gripper_open', 1.0)) > GRIPPER_OPEN_THR
        curr_open = float(getattr(curr, 'gripper_open', 1.0)) > GRIPPER_OPEN_THR
        pose_delta = float(np.linalg.norm(np.array(cp[:3]) - np.array(pp[:3])))
        rot_delta = _quat_angle_delta(pp[3:7], cp[3:7])
        if rot_delta is None:
            trans_strict.append(False)
            trans_relaxed.append(False)
            continue

        strict_static = (
            pose_delta <= pose_eps and
            rot_delta <= rot_eps and
            prev_open == curr_open
        )
        relaxed_static = (
            pose_delta <= pose_eps * STATIC_RUN_JITTER_POSE_SCALE and
            rot_delta <= rot_eps * STATIC_RUN_JITTER_ROT_SCALE and
            prev_open == curr_open
        )
        trans_strict.append(bool(strict_static))
        trans_relaxed.append(bool(relaxed_static))

    effective = list(trans_strict)
    for t in range(1, len(effective) - 1):
        if (not effective[t] and trans_relaxed[t] and
                trans_strict[t - 1] and trans_strict[t + 1]):
            effective[t] = True

    run = 1
    run_start = start
    best = 1
    best_start = start
    best_end = start
    for t, is_static in enumerate(effective):
        i = start + 1 + t
        if is_static:
            run += 1
            if run > best:
                best = run
                best_start = run_start
                best_end = i
        else:
            run = 1
            run_start = i

    return int(best), int(best_start), int(best_end)


def _run_span_pose_rot_delta(demo, run_start, run_end):
    """计算静止连续段首尾帧的累计位置/姿态变化。"""
    if run_start is None or run_end is None or run_end <= run_start:
        return None, None
    if run_start < 0 or run_end >= len(demo):
        return None, None

    obs_s = demo[run_start]
    obs_e = demo[run_end]
    if obs_s is None or obs_e is None:
        return None, None

    ps = getattr(obs_s, 'gripper_pose', None)
    pe = getattr(obs_e, 'gripper_pose', None)
    if ps is None or pe is None:
        return None, None

    pose_span = float(np.linalg.norm(np.array(pe[:3]) - np.array(ps[:3])))
    rot_span = _quat_angle_delta(ps[3:7], pe[3:7])
    return pose_span, rot_span


def drop_static_segments(kept_segments, demo,
                         run_thr=STATIC_RUN_DROP_THR,
                         pose_eps=STATIC_POSE_EPS,
                         rot_eps=STATIC_ROT_EPS_RAD,
                         return_trace=False):
    """丢弃段内出现长时间静止的分段。"""
    filtered = []
    trace = []
    for seg in kept_segments:
        start = int(seg['start'])
        end = int(seg['end'])
        max_run, run_start, run_end = _max_static_run_in_segment(
            demo, start, end, pose_eps=pose_eps, rot_eps=rot_eps)

        span_pose_delta, span_rot_delta = _run_span_pose_rot_delta(
            demo, run_start, run_end)
        has_slow_motion = (
            (span_pose_delta is not None and
             span_pose_delta > pose_eps * STATIC_SLOW_MOTION_POSE_SCALE) or
            (span_rot_delta is not None and
             span_rot_delta > rot_eps * STATIC_SLOW_MOTION_ROT_SCALE)
        )

        if max_run >= run_thr:
            if has_slow_motion:
                filtered.append(seg)
                trace.append({
                    'stage': 'static_segment_keep_slow_motion',
                    'kept_keyframe': int(seg['keyframe']),
                    'start': start,
                    'end': end,
                    'max_static_run': int(max_run),
                    'rule': 'keep_if_max_run_span_exceeds_threshold',
                })
                continue
            trace.append({
                'stage': 'static_segment_drop',
                'dropped_keyframe': int(seg['keyframe']),
                'start': start,
                'end': end,
                'max_static_run': int(max_run),
                'rule': 'drop_long_static_segment',
            })
            continue
        filtered.append(seg)

    if return_trace:
        return filtered, trace
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# 关键帧提取主函数
# ─────────────────────────────────────────────────────────────────────────────

def extract_keyframes(demo,
                      signals=None,
                      min_phase_len=RUN_MIN_PHASE_LEN,
                      vel_threshold=None,
                      dir_threshold=None,
                      contact_threshold=None,
                      force_threshold=None,
                      acc_threshold=None):
    """
    三阶段关键帧提取：候选提取与汇总 → 距离合并 → 交互感知合并 → 静止段丢弃。

    Args:
        demo:             List[Observation]
        signals:          启用的信号名称列表，默认全部启用
        min_phase_len:    相邻关键帧最小帧距（距离合并）
        vel/dir/contact/force/acc_threshold:
                          手动覆盖对应信号的阈值

    Returns:
        keyframe_inds:  List[int]
        debug_info:     dict（阈值、各帧触发信号、交互段标记）
        num_phases:     int（最终阶段数）
    """
    if not demo:
        return [], {}, 0

    n = len(demo)
    if signals is None:
        signals = set(ALL_SIGNALS)
    else:
        signals = set(signals)

    # 阶段 1a：自动阈值
    auto_thr = auto_thresholds(demo)
    thresholds = {
        'vel': vel_threshold if vel_threshold is not None else auto_thr['vel'],
        'dir': dir_threshold if dir_threshold is not None else auto_thr['dir'],
        'contact': contact_threshold if contact_threshold is not None else auto_thr['contact'],
        'force': force_threshold if force_threshold is not None else auto_thr['force'],
        'acc': acc_threshold if acc_threshold is not None else auto_thr['acc'],
    }

    # 阶段 1b：候选帧汇总
    candidates, frame_signals, signal_candidates = collect_stage1_candidates(
        demo, signals, thresholds)

    if not candidates:
        return [n - 1], {
            'thresholds': thresholds,
            'frame_signals': {},
            'seg_interacting': {n - 1: False},
            'trace': {
                'stage1_signal_candidates': signal_candidates,
                'final_keyframes': [n - 1],
            },
        }, 1

    filtered = list(candidates)
    forced_last = None
    if not filtered or filtered[-1] != n - 1:
        filtered.append(n - 1)
        forced_last = n - 1

    # 阶段 2：距离合并
    keyframes, stage2_trace = merge_by_distance_with_trace(filtered, min_phase_len)

    # 阶段 3：交互感知筛选
    if RUN_DEBUG_TRACE:
        seg_labels_all, seg_debug_all = label_interacting_segments(
            keyframes, demo,
            contact_thr=thresholds['contact'],
            vel_thr=thresholds['vel'],
            force_thr=thresholds['force'],
            return_debug=True,
        )
    else:
        seg_labels_all = label_interacting_segments(
            keyframes, demo,
            contact_thr=thresholds['contact'],
            vel_thr=thresholds['vel'],
            force_thr=thresholds['force'],
            return_debug=False,
        )
        seg_debug_all = {'frame_debug': {}, 'segment_debug': []}

    keyframes_stage2 = list(keyframes)
    keyframes, stage3_trace = merge_non_interacting_blocks(
        keyframes_stage2, seg_labels_all, return_trace=True)

    # 构建保留段
    boundaries_stage2 = [0]
    for kf in keyframes_stage2:
        nxt = kf + 1
        if nxt < n:
            boundaries_stage2.append(nxt)
    if boundaries_stage2[-1] != n:
        boundaries_stage2.append(n)

    stage3_initial_segments = []
    for p, kf in enumerate(keyframes_stage2):
        stage3_initial_segments.append({
            'segment_index': int(p),
            'keyframe': int(kf),
            'start': int(boundaries_stage2[p]),
            'end': int(boundaries_stage2[p + 1]),
            'is_interacting': bool(seg_labels_all[p]),
        })

    kept_set = set(keyframes)
    kept_segments = []
    for p, kf in enumerate(keyframes_stage2):
        if kf not in kept_set:
            continue
        kept_segments.append({
            'segment_index': int(p),
            'keyframe': int(kf),
            'start': int(boundaries_stage2[p]),
            'end': int(boundaries_stage2[p + 1]),
            'is_interacting': bool(seg_labels_all[p]),
        })

    # 阶段 4：静止段丢弃
    kept_segments, stage4_trace = drop_static_segments(
        kept_segments, demo, return_trace=True)

    keyframes = [int(seg['keyframe']) for seg in kept_segments]
    num_phases = len(keyframes)

    debug_info = {
        'thresholds': thresholds,
        'frame_signals': {k: v for k, v in frame_signals.items()},
        'seg_interacting': {seg['keyframe']: seg['is_interacting'] for seg in kept_segments},
        'trace': {
            'stage1_signal_candidates': signal_candidates,
            'stage1_candidates_merged': filtered,
            'stage1_forced_last': forced_last,
            'stage2_distance_merge_trace': stage2_trace,
            'stage3_non_interacting_merge_trace': stage3_trace,
            'stage3_initial_segment_labels': stage3_initial_segments,
            'stage4_static_drop_trace': stage4_trace,
            'final_keyframes': keyframes,
            'stage3_kept_segments': kept_segments,
        },
    }

    return keyframes, debug_info, num_phases

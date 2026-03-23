"""
子阶段 demo 数据分割器（keyframe 方案）

支持 6 种信号，算法分三个阶段：

    阶段 1 — 候选帧提取与汇总（各信号阈值自动从数据分布估计）
    S1 gripper  : 夹爪开合状态变化
    S2 vel      : 关节速度由动变静（到达 waypoint）
    S3 dir      : 末端运动方向突变
    S4 contact  : 夹爪接触力突变
    S5 force    : 关节力矩突变
    S6 acc      : 关节加速度突变
    当前使用信号（gripper/vel/contact）等权，候选直接汇总；
    末帧（n-1）作为强制边界补入候选，避免尾部微小分段。

    阶段 2 — 距离合并
    合并距离 < min_phase_len 的相邻候选帧（保留更靠后的帧）。

    阶段 3 — 交互感知合并
    判断每个分割段是否处于"交互状态"（机械臂正与物体作用）并对
    非交互段（approach / reset 等）做方向感知合并：
      - approach 块（左无/右有交互）：仅保留最后一段
      - transit 块（左右均有交互）  ：保留首段 + 尾段
            - reset 块（左有/右无交互）   ：全部删除（轨迹以交互段结尾）
      - 全无交互（兜底）            ：保留所有段

自动阈值
  vel      : 全局速度序列均值 - 0.5×std
  dir      : 85 百分位位移变化量
  contact  : max(均值 + 1.5×std, MIN_CONTACT_FORCE)
  force    : 均值 + 1.5×std
  acc      : 均值 + 1.5×std

保存模式（--save_mode）：
  full          保留关键帧之间的完整子轨迹
  keyframe_only 仅保留每段关键帧本身（默认）

用法：
  python test_traj_segmentation.py --task open_drawer
  python test_traj_segmentation.py --save_mode full
  python test_traj_segmentation.py --min_phase_len 10
"""

import argparse
import os
import pickle
import json
import shutil
import numpy as np
from rlbench.backend.const import LOW_DIM_PICKLE, VARIATION_DESCRIPTIONS

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

# 推/按类任务的松散接触阈值（open=1 时仍有轻微接触力）
LOW_CONTACT_FORCE  = 0.05
# 接触力需持续多少帧才算稳定接触（防单帧噪声）
CONTACT_PERSIST    = 3
# 端点判定向段内扩展的窗口（帧）
BOUNDARY_INTERACT_WINDOW = 2
# 首尾都未命中时，中段连续交互兜底长度（帧）
MID_INTERACT_RUN_THR = 3
# 开夹爪非夹持交互：关节力矩变化阈值缩放（越小越敏感）
OPEN_FORCE_DELTA_SCALE = 0.6
# 开夹爪非夹持交互：速度阈值缩放（需有一定运动）
OPEN_VEL_SCALE = 0.5
# 帧级交互状态机：开夹爪维持交互的弱速度阈值缩放
OPEN_HOLD_VEL_SCALE = 0.25
# 帧级交互状态机：开夹爪维持交互的弱受载阈值缩放
OPEN_HOLD_FORCE_DELTA_SCALE = 0.08
# 帧级交互状态机：允许短暂丢证据的最大帧数
INTERACT_HOLD_GAP_FRAMES = 2
# 开夹爪受载交互（D）进入所需连续命中帧数，抑制单帧惯性尖峰误判
OPEN_D_ENTER_PERSIST = 2
# 开爪（closed->open）后，D 证据进入冷却窗口（帧数）。
# 用于抑制释放物体后的惯性受载尖峰把 reset 误判成交互。
OPEN_AFTER_RELEASE_D_COOLDOWN = 8
# 闭夹爪非抓持下压（E）进入所需连续命中帧数
CLOSED_PRESS_ENTER_PERSIST = 2
# 静止段判定：末端位置变化阈值（米）
STATIC_POSE_EPS = 5e-4
# 静止段判定：末端姿态变化阈值（弧度）
STATIC_ROT_EPS_RAD = 0.05
# 静止段判定：若段内出现 >= 该阈值的连续静止帧，则整段丢弃
STATIC_RUN_DROP_THR = 8
# 静止 run 计算：允许单帧轻微抖动桥接（位置阈值放宽倍数）
STATIC_RUN_JITTER_POSE_SCALE = 2.5
# 静止 run 计算：允许单帧轻微抖动桥接（姿态阈值放宽倍数）
STATIC_RUN_JITTER_ROT_SCALE = 1.5
# 慢运动保留：仅当最长静止 run 首尾累计位移显著超阈值才保留
STATIC_SLOW_MOTION_POSE_SCALE = 4.0
# 慢运动保留：仅当最长静止 run 首尾累计姿态变化显著超阈值才保留
STATIC_SLOW_MOTION_ROT_SCALE = 3.0
# gripper_open 二值化阈值：> 该值判 open，否则判 closed
GRIPPER_OPEN_THR = 0.5

# ─── 运行参数（直接在此修改，无需命令行） ─────────────────────────────────
# 启用的信号，None = 全部启用
RUN_SIGNALS        = None
# 相邻关键帧最小帧距（距离合并阈值）
RUN_MIN_PHASE_LEN  = 5
# 打印每个 episode 的分割追踪信息（候选/合并/最终关键帧）
RUN_SHOW_SEG_TRACE = True
# 是否打印交互帧/交互段的详细判定依据（debug 模式）
RUN_DEBUG_TRACE    = False
# 追踪输出中每项列表最多打印多少个索引（None = 不限制）
RUN_TRACE_MAX_ITEMS = 120
# 是否导出每帧传感器值到 episode 输出目录下的 sensors.txt
RUN_DUMP_SENSORS = True
# 是否在分割过程中同步把每帧传感器值打印到终端
RUN_PRINT_SENSOR_VALUES = False

# ─────────────────────────────────────────────────────────────────────────────
# 信号定义
# ─────────────────────────────────────────────────────────────────────────────


# ALL_SIGNALS = ('gripper', 'vel', 'dir', 'contact', 'force', 'acc')
ALL_SIGNALS = ('gripper', 'vel', 'contact') 
# ─────────────────────────────────────────────────────────────────────────────
# 阶段 1：自动阈值估计
# ─────────────────────────────────────────────────────────────────────────────

def auto_thresholds(demo):
    """
    从 demo 数据分布自动估计各信号阈值，适配不同任务。

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

    # ── dir：末端方向突变阈值 = 90 百分位 cos 变化量 ─────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 1：逐信号候选帧提取
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
    """S2b：速度由静变动（仅保留“运动→静止→再运动”中的再启动帧）。

    通过 `prior_move_window` 要求静止窗口前存在持续运动，避免把轨迹起始的
    初始启动（静止→首次起动）误判为关键帧。
    """
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
    # i 是“开始运动”的首帧索引
    for i in range(still_window, n - move_window + 1):
        # 忽略轨迹开头：没有“先运动再静止”的上下文，不应记为语义再启动。
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


def _candidates_dir(demo, threshold):
    """S3：末端 XYZ 位移方向突变（连续差分向量夹角 > threshold 对应的位移量变化）。"""
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
        # 用位移量变化（当前步与前一步的差）衡量方向变化严重程度
        change = np.linalg.norm(d2 / n2 - d1 / n1)
        if change > threshold:
            cands.add(i)
    return cands


def _candidates_contact(demo, threshold, window=3):
    """S4：夹爪接触力突变（当前帧力范数与前 window 帧均值之差 > threshold）。

    使用窗口均值而非相邻帧差分，可捕获缓慢渐变的接触力变化，同时对单帧噪声更鲁棒。
    """
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


def _candidates_acc(demo, threshold, persist=2):
    """
    S6：关节加速度突变（速度差分范数 > threshold，且持续 persist 帧）。
    持续性要求避免单帧噪声误判。
    """
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
            cands.add(i + 1)   # +1 因为 acc_norms[i] 对应 demo[i+1]
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 1b：候选汇总
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
        frame_signals: dict[int, list[str]]，帧索引 → 触发信号列表（用于调试）
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


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 2：距离合并  |  阶段 3：交互感知合并
# ─────────────────────────────────────────────────────────────────────────────


def label_interacting_frames(demo, contact_thr, vel_thr, force_delta_thr,
                             persist=CONTACT_PERSIST, return_debug=False):
    """
    逐帧判断是否处于"与物体交互"状态，返回 bool 列表。

                帧证据 OR 逻辑：
      A. touch_force_norm > contact_thr  （接触力超阈值：抓、托、夹）
            B. gripper_open==0 AND grasp_likely AND vel_norm>vel_thr  （疑似持物运输）
      C. gripper_open==1 AND touch_force_norm>LOW_CONTACT_FORCE
                                              （推/按类，开夹仍有轻接触）
              D. gripper_open==1 AND force_delta_norm > force_delta_thr * OPEN_FORCE_DELTA_SCALE
                  AND vel_norm > vel_thr * OPEN_VEL_SCALE
                                                                              （非夹爪接触：末端受载推动/压门）
            E. gripper_open==0 AND grasp_likely==0 AND touch_force_norm>LOW_CONTACT_FORCE
                                    AND force_delta_norm > force_delta_thr * OPEN_FORCE_DELTA_SCALE
                                                                                                                                                            （闭夹爪预闭后下压/按压，非抓持交互）

        再做帧级状态机平滑：
            - 进入交互：A 单帧可进入；B/C 仍受 persist 约束；D/E 受各自持续约束；
            - 维持交互：允许使用更弱的开夹爪受载+运动条件保持，避免压门尾段漏判；
            - 退出交互：连续若干帧无证据后退出，抑制短时抖动。

    Args:
        demo:        List[Observation]
        contact_thr: 接触力阈值（来自 auto_thresholds['contact']）
        vel_thr:     速度阈值（来自 auto_thresholds['vel']）
        force_delta_thr: 力矩变化阈值（来自 auto_thresholds['force']）
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

    # 推断“疑似持物”状态：仅当开->闭附近存在接触/受载证据时才允许 B 运输信号触发。
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
        cond_b = (not g_open) and grasp_likely[i] and (vn > vel_thr)  # 疑似持物运输
        cond_c = g_open and (fn > LOW_CONTACT_FORCE)    # 推/按（有触碰）

        # 末端（非夹爪）接触时，touch_force 往往不显著；用关节受载变化补充检测。
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

    # 帧级状态机平滑：避免“压门尾段接触持续但差分不显著”被漏判。
    result = [False] * n
    active = False
    strong_streak = 0
    d_streak = 0
    e_streak = 0
    gap = 0
    last_release_idx = -10**9
    for i, obs in enumerate(demo):
        if not valid[i]:
            # 丢观测时视作无新证据，按短暂空洞处理。
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

        # 记录最近一次开爪时刻（closed -> open）。
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
        # 开爪后的短窗口内禁止 D 进入，避免 release 惯性峰值误触发。
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

        # 进入规则：
        # 1) A（接触力超阈值）允许单帧进入，避免抓取接触起始被延迟；
        # 2) B/C 连续达到 persist（抗噪）；
        # 3) D 需连续命中若干帧，避免松手后离开物体时的单帧受载尖峰误判；
        # 4) E 需连续命中，识别“提前闭爪后下压/按压”的非抓持交互。
        if not active:
            bc = cond_b or cond_c
            if (cond_a or
                    (bc and strong_streak >= persist) or
                    (d_streak >= OPEN_D_ENTER_PERSIST) or
                    (e_streak >= CLOSED_PRESS_ENTER_PERSIST)):
                active = True
                gap = 0

        # 维持规则：开夹爪下允许更弱的“受载+运动”维持，覆盖压门尾段。
        hold_open = (
            g_open and
            (vn > vel_thr * OPEN_HOLD_VEL_SCALE) and
            (fd > force_delta_thr * OPEN_HOLD_FORCE_DELTA_SCALE)
        )
        # 闭夹爪维持仅依赖强证据，避免“空夹爪关闭后移动”被误判成交互。
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


def label_interacting_segments(keyframes, demo, contact_thr, vel_thr, force_thr,
                               return_debug=False):
    """
    根据每段首尾帧是否交互，判断该段是否为"交互段"。

    段边界由 keyframes 决定：第 p 段 = demo[boundaries[p]:boundaries[p+1]]。
        判定规则：
            1) 若该段"首帧"或"尾帧"（含边界窗口）为交互帧，则该段为交互段；
            2) 否则，若段内存在长度 >= MID_INTERACT_RUN_THR 的连续交互帧，
                 也判为交互段（兜底，覆盖首尾都停稳但中段在交互的情况）。
    说明：这里的尾帧是 end-1（因为分段区间右端是开区间）。

    为了避免边界帧无效（None）造成误判，若首/尾位置无效，会向段内
    线性搜索最近的有效帧标签作为替代。

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
        """返回 [start, end) 内首个有效观测帧的交互标签；若不存在则 False。"""
        for i in range(start, end):
            if demo[i] is not None:
                return frame_labels[i]
        return False

    def _find_last_valid_label(start, end):
        """返回 [start, end) 内最后一个有效观测帧的交互标签；若不存在则 False。"""
        for i in range(end - 1, start - 1, -1):
            if demo[i] is not None:
                return frame_labels[i]
        return False

    def _max_true_run(start, end):
        """返回 [start, end) 内连续 True 的最大长度。"""
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

    # 为了避免“移动中空夹爪预闭合/松手后撤离”被误判，
    # 开合事件仅在局部存在接触/受载证据时才作为交互边界证据。
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
        """夹爪开合切换且邻域有接触/受载证据时，才视为交互边界证据。"""
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
        end   = boundaries[p + 1]
        if end <= start:
            seg_labels.append(False)
            seg_debug.append({
                'segment_index': int(p),
                'keyframe': int(keyframes[p]),
                'start': int(start),
                'end': int(end),
                'head_window_hit': False,
                'tail_window_hit': False,
                'head_interact': False,
                'tail_interact': False,
                'mid_true_run': 0,
                'is_interacting': False,
                'reason': 'empty_segment',
            })
            continue

        head_idx = start
        tail_idx = end - 1

        # 端点窗口证据：允许在段首/段尾附近若干帧内命中交互。
        head_win_end = min(end, start + BOUNDARY_INTERACT_WINDOW)
        tail_win_start = max(start, end - BOUNDARY_INTERACT_WINDOW)

        head_window_hit = any(frame_labels[i] for i in range(start, head_win_end))
        tail_window_hit = any(frame_labels[i] for i in range(tail_win_start, end))

        head_interact = (
            _find_first_valid_label(start, end) or
            _is_gripper_toggle(head_idx) or
            head_window_hit
        )
        tail_interact = (
            _find_last_valid_label(start, end) or
            _is_gripper_toggle(tail_idx) or
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
                'head_window_hit': bool(head_window_hit),
                'tail_window_hit': bool(tail_window_hit),
                'head_interact': bool(head_interact),
                'tail_interact': bool(tail_interact),
                'mid_true_run': int(_max_true_run(start, end)),
                'is_interacting': True,
                'reason': 'endpoint_interact',
            })
            continue

        # 兜底：首尾都未命中，但中段持续交互（如抓持运输/推门中段）仍判交互。
        mid_run = _max_true_run(start, end)
        is_interacting = mid_run >= MID_INTERACT_RUN_THR
        seg_labels.append(is_interacting)
        seg_debug.append({
            'segment_index': int(p),
            'keyframe': int(keyframes[p]),
            'start': int(start),
            'end': int(end),
            'head_window_hit': bool(head_window_hit),
            'tail_window_hit': bool(tail_window_hit),
            'head_interact': bool(head_interact),
            'tail_interact': bool(tail_interact),
            'mid_true_run': int(mid_run),
            'is_interacting': bool(is_interacting),
            'reason': 'mid_run_fallback' if is_interacting else 'non_interacting',
        })

    if not return_debug:
        return seg_labels

    return seg_labels, {
        'frame_debug': frame_debug,
        'segment_debug': seg_debug,
    }


def merge_non_interacting_blocks(keyframes, seg_labels, return_trace=False):
    """
    对非交互段的连续块做方向感知合并。

        块在轨迹中的位置决定保留策略：
      左无邻 / 右有交互  (approach 块) → 仅保留块最后一段的 keyframe
      左有交互 / 右有交互 (transit 块) → 仅保留块最后一段的 keyframe（靠近右侧交互段）
            左有交互 / 右无邻  (reset 块)   → 全部删除（确保轨迹末尾停在交互段）
      左无邻 / 右无邻   (全无交互兜底) → 保留块所有 keyframe

    交互段的 keyframe 始终保留，不受影响。

    Args:
        keyframes:  List[int]，距离合并后的关键帧列表
        seg_labels: List[bool]，长度同 keyframes，True = 交互段

    Returns:
        List[int]，合并后的关键帧列表（已排序去重）
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
            # 收集连续非交互段块 [i, j)
            j = i
            while j < n_seg and not seg_labels[j]:
                j += 1
            block_kf = keyframes[i:j]

            if not any_interact:
                # 全轨迹无交互段 → 兜底保留所有
                keep_block = list(block_kf)
                rule = 'all_non_interact_keep_all'
            else:
                has_left  = (i > 0)      # 左侧存在交互段
                has_right = (j < n_seg)  # 右侧存在交互段

                if not has_left and has_right:
                    # approach 块：仅保留最后一段（最靠近交互的段）
                    keep_block = [block_kf[-1]]
                    rule = 'approach_keep_last'
                elif has_left and has_right:
                    # transit 块：仅保留尾段（最靠近右侧交互段）
                    keep_block = [block_kf[-1]]
                    rule = 'transit_keep_last'
                elif has_left and not has_right:
                    # reset 块：全部删除，确保最终轨迹以交互段结尾。
                    keep_block = []
                    rule = 'reset_drop_all'
                else:
                    # 逻辑上不可达（any_interact=True 但左右均无），兜底保留
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


def merge_by_distance_with_trace(keyframes, min_phase_len):
    """
    合并距离 < min_phase_len 的相邻关键帧（保留靠后的帧），并返回合并追踪。
    """
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
                'rule': 'gap_lt_min_phase_len_keep_later',
            })
        else:
            merged.append(idx)
    return merged, merge_trace


def merge_by_distance(keyframes, min_phase_len):
    """
    合并距离 < min_phase_len 的相邻关键帧（保留靠后的帧）。
    """
    merged, _ = merge_by_distance_with_trace(keyframes, min_phase_len)
    return merged


def _quat_angle_delta(q_prev, q_curr):
    """返回两个四元数间的最小夹角（弧度，范围 [0, pi]）。"""
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
    """返回 [start, end) 内最长静止连续段信息。

    Returns:
        (max_run, run_start, run_end)
        - max_run: 最长连续静止帧长度
        - run_start: 该连续段起始帧索引（含）
        - run_end: 该连续段结束帧索引（含）
    """
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

        # 严格静止：位置变化小 + 姿态变化小 + 夹爪状态未变化。
        strict_static = (
            pose_delta <= pose_eps and
            rot_delta <= rot_eps and
            prev_open == curr_open
        )
        # 放宽静止：用于桥接单帧抖动，避免 run 被偶发噪声打断。
        relaxed_static = (
            pose_delta <= pose_eps * STATIC_RUN_JITTER_POSE_SCALE and
            rot_delta <= rot_eps * STATIC_RUN_JITTER_ROT_SCALE and
            prev_open == curr_open
        )
        trans_strict.append(bool(strict_static))
        trans_relaxed.append(bool(relaxed_static))

    effective = list(trans_strict)
    # 允许桥接“单帧轻微抖动”：仅当该非严格帧两侧均为严格静止时生效。
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
    """丢弃段内出现长时间静止（位置+姿态+夹爪状态不变）的分段。"""
    filtered = []
    trace = []
    for seg in kept_segments:
        start = int(seg['start'])
        end = int(seg['end'])
        max_run, run_start, run_end = _max_static_run_in_segment(
            demo, start, end,
            pose_eps=pose_eps,
            rot_eps=rot_eps,
        )

        span_pose_delta, span_rot_delta = _run_span_pose_rot_delta(
            demo, run_start, run_end)
        # 若最长“静止”连续段跨时段累计位姿变化已超过阈值，
        # 说明是每帧缓慢移动而非真正静止，不应丢弃该段。
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
                    'run_start': int(run_start) if run_start is not None else None,
                    'run_end': int(run_end) if run_end is not None else None,
                    'run_span_pose_delta': None if span_pose_delta is None else float(span_pose_delta),
                    'run_span_rot_delta_rad': None if span_rot_delta is None else float(span_rot_delta),
                    'run_threshold': int(run_thr),
                    'pose_eps': float(pose_eps),
                    'rot_eps_rad': float(rot_eps),
                    'rule': 'keep_if_max_run_span_exceeds_threshold',
                })
                continue
            trace.append({
                'stage': 'static_segment_drop',
                'dropped_keyframe': int(seg['keyframe']),
                'start': start,
                'end': end,
                'max_static_run': int(max_run),
                'run_start': int(run_start) if run_start is not None else None,
                'run_end': int(run_end) if run_end is not None else None,
                'run_span_pose_delta': None if span_pose_delta is None else float(span_pose_delta),
                'run_span_rot_delta_rad': None if span_rot_delta is None else float(span_rot_delta),
                'run_threshold': int(run_thr),
                'pose_eps': float(pose_eps),
                'rot_eps_rad': float(rot_eps),
                'rule': 'drop_long_static_segment',
            })
            continue
        filtered.append(seg)

    if return_trace:
        return filtered, trace
    return filtered


def _truncate_list(values, limit):
    if limit is None or len(values) <= limit:
        return values
    return values[:limit]


def print_segmentation_trace(ep_label, debug_info):
    """打印单个 episode 的关键帧提取与合并全过程。"""
    trace = debug_info.get('trace', {})
    if not trace:
        return

    print(f'\n  [Trace] {ep_label}')

    # 阶段 1：按信号候选
    s1 = trace.get('stage1_signal_candidates', {})
    for sig in sorted(s1.keys()):
        vals = [int(x) for x in s1[sig]]
        show = _truncate_list(vals, RUN_TRACE_MAX_ITEMS)
        suffix = '' if len(show) == len(vals) else f' ...(+{len(vals)-len(show)})'
        print(f'    S1 {sig:8s}: {show}{suffix}')

    # 阶段 1：候选汇总
    cand = trace.get('stage1_candidates_merged', [])
    forced = trace.get('stage1_forced_last', None)
    print(f'    S1 merged: candidates={len(cand)}, forced_last={forced}')

    # 阶段 2 + 阶段 3：合并追踪
    s2 = trace.get('stage2_distance_merge_trace', [])
    if s2:
        print(f'    S2 distance merges ({len(s2)}):')
        for m in _truncate_list(s2, RUN_TRACE_MAX_ITEMS):
            print(f'      drop {m["dropped"]} -> keep {m["kept"]} '
                  f'(gap={m["gap"]} < {m["min_phase_len"]})')
    else:
        print('    S2 distance merges: none')

    s3 = trace.get('stage3_non_interacting_merge_trace', [])
    if s3:
        print(f'    S3 non-interacting merges ({len(s3)}):')
        for m in _truncate_list(s3, RUN_TRACE_MAX_ITEMS):
            print(f'      drop {m["dropped"]} in block {m["block"]} '
                  f'by {m["rule"]}, keep {m["kept_in_block"]}')
    else:
        print('    S3 non-interacting merges: none')

    s3_init = trace.get('stage3_initial_segment_labels', [])
    if s3_init:
        print(f'    S3 initial segment labels ({len(s3_init)}):')
        for seg in _truncate_list(s3_init, RUN_TRACE_MAX_ITEMS):
            print(f'      seg_{seg["segment_index"]} '
                  f'kf={seg["keyframe"]} '
                  f'seg[{seg["start"]},{seg["end"]}) '
                  f'=> {"INTERACT" if seg["is_interacting"] else "non-interact"}')

    s3_drop = trace.get('stage3_dropped_segments', [])
    if s3_drop:
        print(f'    S3 dropped segments ({len(s3_drop)}):')
        for seg in _truncate_list(s3_drop, RUN_TRACE_MAX_ITEMS):
            print(f'      drop seg_{seg["segment_index"]} '
                  f'kf={seg["keyframe"]} '
                  f'seg[{seg["start"]},{seg["end"]}) '
                  f'({"INTERACT" if seg["is_interacting"] else "non-interact"}) '
                  f'by {seg.get("rule", "-")}')
    else:
        print('    S3 dropped segments: none')

    final_kf = [int(x) for x in trace.get('final_keyframes', [])]
    show_final = _truncate_list(final_kf, RUN_TRACE_MAX_ITEMS)
    suffix = '' if len(show_final) == len(final_kf) else f' ...(+{len(final_kf)-len(show_final)})'
    print(f'    Final keyframes ({len(final_kf)}): {show_final}{suffix}')

    s4 = trace.get('stage4_static_drop_trace', [])
    if s4:
        print(f'    S4 static decisions ({len(s4)}):')
        for m in _truncate_list(s4, RUN_TRACE_MAX_ITEMS):
            stage = m.get('stage', '')
            run_span_pose = m.get('run_span_pose_delta', None)
            run_span_rot = m.get('run_span_rot_delta_rad', None)
            span_str = (
                f'span_pos={"None" if run_span_pose is None else f"{run_span_pose:.6f}"}, '
                f'span_rot={"None" if run_span_rot is None else f"{run_span_rot:.6f}"}rad'
            )
            if stage == 'static_segment_keep_slow_motion':
                print(
                    f'      keep kf {m.get("kept_keyframe")} '
                    f'seg[{m["start"]},{m["end"]}) '
                    f'max_static_run={m["max_static_run"]} >= {m["run_threshold"]}, '
                    f'{span_str} (slow motion)'
                )
            else:
                print(
                    f'      drop kf {m.get("dropped_keyframe")} '
                    f'seg[{m["start"]},{m["end"]}) '
                    f'max_static_run={m["max_static_run"]} >= {m["run_threshold"]}, '
                    f'{span_str}'
                )
    else:
        print('    S4 static drops: none')

    if RUN_DEBUG_TRACE:
        # 交互帧与交互段判定依据（仅 debug 模式打印）
        frame_dbg = trace.get('interaction_frame_debug', {})
        if frame_dbg:
            inter_frames = [int(x) for x in frame_dbg.get('interacting_frames', [])]
            show_inter = _truncate_list(inter_frames, RUN_TRACE_MAX_ITEMS)
            suffix_inter = '' if len(show_inter) == len(inter_frames) else f' ...(+{len(inter_frames)-len(show_inter)})'
            print(f'    Interacting frames ({len(inter_frames)}): {show_inter}{suffix_inter}')

            accepted_runs = frame_dbg.get('accepted_runs', [])
            if accepted_runs:
                show_runs = _truncate_list(accepted_runs, RUN_TRACE_MAX_ITEMS)
                suffix_runs = '' if len(show_runs) == len(accepted_runs) else f' ...(+{len(accepted_runs)-len(show_runs)})'
                print(f'    Interacting runs (persist>={frame_dbg.get("persist")}): {show_runs}{suffix_runs}')

            raw_reasons = frame_dbg.get('raw_reasons', {})
            if raw_reasons:
                print('    Interacting frame reasons:')
                sorted_items = sorted(raw_reasons.items(), key=lambda x: int(x[0]))
                for idx, reasons in _truncate_list(sorted_items, RUN_TRACE_MAX_ITEMS):
                    print(f'      frame {int(idx)}: {" + ".join(reasons)}')

        seg_dbg = trace.get('interaction_segment_debug', [])
        if seg_dbg:
            print(f'    Segment interaction labels ({len(seg_dbg)}):')
            for item in _truncate_list(seg_dbg, RUN_TRACE_MAX_ITEMS):
                print(
                    '      '
                    f"seg_{item['segment_index']} "
                    f"kf={item['keyframe']} "
                    f"[{item['start']},{item['end']}) "
                    f"=> {'INTERACT' if item['is_interacting'] else 'non-interact'} "
                    f"(reason={item['reason']}, "
                    f"head={item['head_interact']}, tail={item['tail_interact']}, "
                    f"head_win={item['head_window_hit']}, tail_win={item['tail_window_hit']}, "
                    f"mid_run={item['mid_true_run']})"
                )


def _fmt_array(arr, precision=6):
    """将 ndarray/list 转为稳定可读字符串，便于写入 sensors.txt。"""
    if arr is None:
        return 'None'
    try:
        flat = np.asarray(arr).reshape(-1)
    except Exception:
        return str(arr)
    vals = ', '.join(f'{float(x):.{precision}f}' for x in flat)
    return '[' + vals + ']'


def _gripper_width_from_joint_positions(gjpos):
    """由 gripper_joint_positions 估计夹爪宽度标量。"""
    if gjpos is None:
        return None
    try:
        flat = np.asarray(gjpos, dtype=float).reshape(-1)
    except Exception:
        return None
    if flat.size == 0:
        return None
    if flat.size >= 2:
        # RLBench 常见为双指两关节，宽度取两指位置之和。
        return float(flat[0] + flat[1])
    return float(flat[0])


def dump_episode_sensors(demo, out_ep_path, debug_info, ep_label=None,
                         print_to_console=False):
    """导出逐帧传感器值与分割判定中间量到 sensors.txt。"""
    os.makedirs(out_ep_path, exist_ok=True)
    out_file = os.path.join(out_ep_path, 'sensors.txt')

    trace = debug_info.get('trace', {}) if debug_info else {}
    thresholds = debug_info.get('thresholds', {}) if debug_info else {}
    frame_signals = debug_info.get('frame_signals', {}) if debug_info else {}
    final_keyframes = set(int(x) for x in trace.get('final_keyframes', []))

    static_drops = trace.get('stage4_static_drop_trace', [])
    static_ranges = [
        (int(item['start']), int(item['end']), int(item['dropped_keyframe']))
        for item in static_drops
        if item.get('stage') == 'static_segment_drop'
    ]

    inter_frames = label_interacting_frames(
        demo,
        contact_thr=float(thresholds.get('contact', 0.0)),
        vel_thr=float(thresholds.get('vel', 0.0)),
        force_delta_thr=float(thresholds.get('force', 0.0)),
        return_debug=False,
    ) if demo else []

    header_lines = [
        f'# episode: {ep_label or out_ep_path}',
        f'# num_frames: {len(demo)}',
        '# thresholds: '
        + ', '.join(f'{k}={float(v):.6f}' for k, v in sorted(thresholds.items())),
        '# columns: frame | valid | gripper_open | gripper_width | vel_norm | force_delta_norm '
        '| touch_norm | pose_delta | rot_delta_rad | static_pair | interacting_frame '
        '| trigger_signals | final_keyframe | in_static_dropped_segment '
        '| joint_positions | joint_velocities | joint_forces '
        '| gripper_pose | gripper_joint_positions | gripper_touch_forces',
        '',
    ]

    lines = list(header_lines)
    for i, obs in enumerate(demo):
        if obs is None:
            line = (
                f'frame={i:04d} | valid=False | gripper_open=None | gripper_width=None | vel_norm=None '
                f'| force_delta_norm=None | touch_norm=None | pose_delta=None '
                f'| rot_delta_rad=None '
                f'| static_pair=False | interacting_frame=False '
                f'| trigger_signals={frame_signals.get(i, [])} '
                f'| final_keyframe={i in final_keyframes} '
                f'| in_static_dropped_segment=False '
                f'| joint_positions=None | joint_velocities=None | joint_forces=None '
                f'| gripper_pose=None | gripper_joint_positions=None '
                f'| gripper_touch_forces=None'
            )
            lines.append(line)
            continue

        g_open = float(getattr(obs, 'gripper_open', 1.0)) > GRIPPER_OPEN_THR

        jpos = getattr(obs, 'joint_positions', None)
        jvel = getattr(obs, 'joint_velocities', None)
        jfor = getattr(obs, 'joint_forces', None)
        gpose = getattr(obs, 'gripper_pose', None)
        gjpos = getattr(obs, 'gripper_joint_positions', None)
        gtouch = getattr(obs, 'gripper_touch_forces', None)
        gwidth = _gripper_width_from_joint_positions(gjpos)

        vel_norm = float(np.linalg.norm(jvel)) if jvel is not None else None
        touch_norm = float(np.linalg.norm(gtouch)) if gtouch is not None else None

        force_delta = None
        pose_delta = None
        rot_delta = None
        static_pair = False
        if i > 0 and demo[i - 1] is not None:
            prev = demo[i - 1]
            p_for = getattr(prev, 'joint_forces', None)
            if jfor is not None and p_for is not None:
                force_delta = float(np.linalg.norm(np.array(jfor) - np.array(p_for)))

            prev_pose = getattr(prev, 'gripper_pose', None)
            if gpose is not None and prev_pose is not None:
                pose_delta = float(np.linalg.norm(np.array(gpose[:3]) - np.array(prev_pose[:3])))
                rot_delta = _quat_angle_delta(prev_pose[3:7], gpose[3:7])
                prev_open = float(getattr(prev, 'gripper_open', 1.0)) > GRIPPER_OPEN_THR
                static_pair = (
                    pose_delta <= STATIC_POSE_EPS and
                    rot_delta is not None and
                    rot_delta <= STATIC_ROT_EPS_RAD and
                    prev_open == g_open
                )

        in_static_drop = False
        for s, e, _kf in static_ranges:
            if s <= i < e:
                in_static_drop = True
                break

        line = (
            f'frame={i:04d} | valid=True | gripper_open={"open" if g_open else "closed"} '
            f'| gripper_width={"None" if gwidth is None else f"{gwidth:.6f}"} '
            f'| vel_norm={"None" if vel_norm is None else f"{vel_norm:.6f}"} '
            f'| force_delta_norm={"None" if force_delta is None else f"{force_delta:.6f}"} '
            f'| touch_norm={"None" if touch_norm is None else f"{touch_norm:.6f}"} '
            f'| pose_delta={"None" if pose_delta is None else f"{pose_delta:.6f}"} '
            f'| rot_delta_rad={"None" if rot_delta is None else f"{rot_delta:.6f}"} '
            f'| static_pair={static_pair} '
            f'| interacting_frame={bool(inter_frames[i]) if i < len(inter_frames) else False} '
            f'| trigger_signals={frame_signals.get(i, [])} '
            f'| final_keyframe={i in final_keyframes} '
            f'| in_static_dropped_segment={in_static_drop} '
            f'| joint_positions={_fmt_array(jpos)} '
            f'| joint_velocities={_fmt_array(jvel)} '
            f'| joint_forces={_fmt_array(jfor)} '
            f'| gripper_pose={_fmt_array(gpose)} '
            f'| gripper_joint_positions={_fmt_array(gjpos)} '
            f'| gripper_touch_forces={_fmt_array(gtouch)}'
        )
        lines.append(line)

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    if print_to_console:
        for ln in lines:
            print(ln)


# ─────────────────────────────────────────────────────────────────────────────
# 总入口：关键帧提取
# ─────────────────────────────────────────────────────────────────────────────

def extract_keyframes(demo,
                      signals=None,
                      min_phase_len=5,
                      vel_threshold=None,
                      dir_threshold=None,
                      contact_threshold=None,
                      force_threshold=None,
                      acc_threshold=None):
    """
    三阶段关键帧提取：候选提取与汇总 → 距离合并 → 交互感知合并。

    Args:
        demo:             List[Observation]
        signals:          启用的信号名称列表，默认全部启用
        min_phase_len:    相邻关键帧最小帧距（距离合并）
        vel/dir/contact/force/acc_threshold:
                          手动覆盖对应信号的阈值（None 则使用自动估计值）

    Returns:
        keyframe_inds:  List[int]
        debug_info:     dict（阈值、各帧触发信号、交互段标记）
    """
    if not demo:
        return [], {}

    n = len(demo)
    if signals is None:
        signals = set(ALL_SIGNALS)
    else:
        signals = set(signals)

    # ── 阶段 1a：自动阈值 ────────────────────────────────────────────────
    auto_thr = auto_thresholds(demo)
    thresholds = {
        'vel':     vel_threshold     if vel_threshold     is not None else auto_thr['vel'],
        'dir':     dir_threshold     if dir_threshold     is not None else auto_thr['dir'],
        'contact': contact_threshold if contact_threshold is not None else auto_thr['contact'],
        'force':   force_threshold   if force_threshold   is not None else auto_thr['force'],
        'acc':     acc_threshold     if acc_threshold     is not None else auto_thr['acc'],
    }

    # ── 阶段 1b：候选帧汇总（无打分） ───────────────────────────────────
    candidates, frame_signals, signal_candidates = collect_stage1_candidates(
        demo, signals, thresholds)

    if not candidates:
        return [n - 1], {
            'thresholds': thresholds,
            'frame_signals': {},
            'scores': {},
            'seg_interacting': {n - 1: False},
            'trace': {
                'stage1_signal_candidates': signal_candidates,
                'stage1_candidates_merged': [],
                'stage1_forced_last': n - 1,
                'stage2_distance_merge_trace': [],
                'stage3_non_interacting_merge_trace': [],
                'final_keyframes': [n - 1],
            },
        }

    # ── 阶段 1c：候选直通（无筛选） ─────────────────────────────────────
    filtered = list(candidates)

    # 末帧作为强制边界插入，使邻近冗余帧可在阶段 2 合并入末帧。
    forced_last = None
    if not filtered or filtered[-1] != n - 1:
        filtered.append(n - 1)
        forced_last = n - 1

    # ── 阶段 2：距离合并 ─────────────────────────────────────────────────
    keyframes, stage2_trace = merge_by_distance_with_trace(filtered, min_phase_len)

    # ── 阶段 3：交互感知筛选（不做关键帧连续合并） ───────────────────────
    # 复用已计算的 contact/vel 阈值判断各段是否为交互状态。
    # 注意：这里保留/删除的是“原始分段”，而不是把被删段并到相邻段。
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

    # 用 stage2 的原始段边界构建“保留段”，避免删除段后被连续合并。
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

    stage3_dropped_segments = []
    seg_by_kf = {int(seg['keyframe']): seg for seg in stage3_initial_segments}
    for m in stage3_trace:
        dropped_kf = int(m['dropped'])
        base = seg_by_kf.get(dropped_kf, {'segment_index': -1, 'start': -1, 'end': -1, 'is_interacting': False})
        stage3_dropped_segments.append({
            'segment_index': int(base['segment_index']),
            'keyframe': dropped_kf,
            'start': int(base['start']),
            'end': int(base['end']),
            'is_interacting': bool(base['is_interacting']),
            'rule': m.get('rule', '-'),
            'block': [int(x) for x in m.get('block', [])],
            'kept_in_block': [int(x) for x in m.get('kept_in_block', [])],
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

    kept_segments, stage4_trace = drop_static_segments(
        kept_segments,
        demo,
        run_thr=STATIC_RUN_DROP_THR,
        pose_eps=STATIC_POSE_EPS,
        rot_eps=STATIC_ROT_EPS_RAD,
        return_trace=True,
    )

    keyframes = [int(seg['keyframe']) for seg in kept_segments]
    seg_labels_final = [bool(s['is_interacting']) for s in kept_segments]
    seg_debug_final = {
        'frame_debug': seg_debug_all.get('frame_debug', {}),
        'segment_debug': seg_debug_all.get('segment_debug', []),
    }

    debug_info = {
        'thresholds':      thresholds,
        'frame_signals':   {k: v for k, v in frame_signals.items()},
        'scores':          {},
        'seg_interacting': {keyframes[p]: seg_labels_final[p]
                            for p in range(len(keyframes))},
        'trace': {
            'stage1_signal_candidates': signal_candidates,
            'stage1_candidates_merged': filtered,
            'stage1_forced_last': forced_last,
            'stage2_distance_merge_trace': stage2_trace,
            'stage3_non_interacting_merge_trace': stage3_trace,
            'stage3_initial_segment_labels': stage3_initial_segments,
            'stage3_dropped_segments': stage3_dropped_segments,
            'stage4_static_drop_trace': stage4_trace,
            'final_keyframes': keyframes,
            'stage3_kept_segments': kept_segments,
        },
    }
    if RUN_DEBUG_TRACE:
        debug_info['trace']['interaction_frame_debug'] = seg_debug_final.get('frame_debug', {})
        debug_info['trace']['interaction_segment_debug'] = seg_debug_final.get('segment_debug', [])

    return keyframes, debug_info


# ─────────────────────────────────────────────
# 保存子阶段 demo
# ─────────────────────────────────────────────

def save_subphase_demo(subphase_obs, out_path, src_ep_path, frame_indices):
    """
    将子阶段观测列表和对应的图像文件保存到 out_path，
    使输出目录结构与原始 episode 完全一致。

    图像文件名保留原始帧索引（不重编号），与原始 episode 一致。
    """
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(subphase_obs, f)

    image_dirs = [
        d for d in os.listdir(src_ep_path)
        if os.path.isdir(os.path.join(src_ep_path, d))
    ]
    for img_dir in image_dirs:
        src_img_dir = os.path.join(src_ep_path, img_dir)
        dst_img_dir = os.path.join(out_path, img_dir)
        os.makedirs(dst_img_dir, exist_ok=True)
        for orig_idx in frame_indices:
            src_file = os.path.join(src_img_dir, f'{orig_idx}.png')
            dst_file = os.path.join(dst_img_dir, f'{orig_idx}.png')
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)


def process_episode(ep_path, out_ep_path, descriptions,
                    signals=None, min_phase_len=5,
                    save_mode='full', ep_label=None):
    """
    对单个 episode 执行三阶段关键帧分割并保存结果。

    Returns:
        phase_info: List[dict]
    """
    low_dim_file = os.path.join(ep_path, LOW_DIM_PICKLE)
    if not os.path.exists(low_dim_file):
        return []

    with open(low_dim_file, 'rb') as f:
        demo = pickle.load(f)

    keyframe_inds, debug_info = extract_keyframes(
        demo,
        signals=signals,
        min_phase_len=min_phase_len,
    )

    if RUN_SHOW_SEG_TRACE:
        print_segmentation_trace(ep_label or ep_path, debug_info)

    if RUN_DUMP_SENSORS:
        dump_episode_sensors(
            demo,
            out_ep_path,
            debug_info,
            ep_label=ep_label or ep_path,
            print_to_console=RUN_PRINT_SENSOR_VALUES,
        )

    kept_segments = debug_info.get('trace', {}).get('stage3_kept_segments', None)
    if kept_segments:
        phase_ranges = [
            (int(seg['start']), int(seg['end']), int(seg['keyframe']))
            for seg in kept_segments
        ]
    else:
        # 兼容旧逻辑：按连续边界分段
        boundaries = [0]
        for kf in keyframe_inds:
            next_start = kf + 1
            if next_start < len(demo):
                boundaries.append(next_start)
        if boundaries[-1] != len(demo):
            boundaries.append(len(demo))
        phase_ranges = [
            (boundaries[p], boundaries[p + 1], keyframe_inds[p])
            for p in range(len(keyframe_inds))
        ]

    num_phases = len(phase_ranges)

    phase_info = []
    for p, (start, end, kf_idx) in enumerate(phase_ranges):
        if save_mode == 'keyframe_only':
            frame_indices = [start, kf_idx] if start != kf_idx else [kf_idx]
            phase_obs = [demo[i] for i in frame_indices]
        else:
            frame_indices = list(range(start, end))
            phase_obs = demo[start:end]

        phase_out = os.path.join(out_ep_path, f'phase_{p}')
        save_subphase_demo(phase_obs, phase_out, ep_path, frame_indices)

        gripper_states = [float(o.gripper_open) > GRIPPER_OPEN_THR
                          for o in phase_obs if o is not None]
        # 触发信号列表（用于调试输出）
        trigger = debug_info.get('frame_signals', {}).get(kf_idx, [])
        is_interacting = debug_info.get('seg_interacting', {}).get(kf_idx, None)
        phase_info.append({
            'phase_index':        p,
            'keyframe_index':     kf_idx,
            'start_frame':        start,
            'end_frame':          end,
            'length':             end - start,
            'saved_frames':       len(frame_indices),
            'gripper_open_ratio': float(np.mean(gripper_states)) if gripper_states else 0.0,
            'trigger_signals':    trigger,
            'is_interacting':     is_interacting,
        })

    ep_meta = {
        'total_frames':      len(demo),
        'save_mode':         save_mode,
        'num_phases':        num_phases,
        'keyframe_inds':     [int(p['keyframe_index']) for p in phase_info],
        'boundaries':        [int(p['start_frame']) for p in phase_info] +
                             ([int(phase_info[-1]['end_frame'])] if phase_info else []),
        'phase_ranges':      [
            {
                'start_frame': int(p['start_frame']),
                'end_frame': int(p['end_frame']),
                'keyframe_index': int(p['keyframe_index']),
            }
            for p in phase_info
        ],
        'task_descriptions': descriptions,
        'auto_thresholds':   {k: float(v) for k, v in
                              debug_info.get('thresholds', {}).items()},
        'segmentation_trace': debug_info.get('trace', {}),
        'phases':            phase_info,
    }
    with open(os.path.join(out_ep_path, 'phase_metadata.json'), 'w') as f:
        json.dump(ep_meta, f, ensure_ascii=False, indent=2)

    return phase_info


# ─────────────────────────────────────────────
# 主流程：遍历所有任务 / 变体 / episode
# ─────────────────────────────────────────────

def split_all_demos(save_path, output_path, task_name=None,
                    signals=None, min_phase_len=5,
                    save_mode='full'):
    """
    遍历 demos 文件夹，对每个 episode 进行三阶段关键帧分割，
    结果保存到 output_path（目录结构与原始 demos 一致）。
    """
    if not os.path.exists(save_path):
        print(f'Error: {save_path} does not exist.')
        return

    task_dirs = sorted([
        d for d in os.listdir(save_path)
        if os.path.isdir(os.path.join(save_path, d))
    ])

    for task in task_dirs:
        if task_name and task != task_name:
            continue

        task_path     = os.path.join(save_path, task)
        task_out_path = os.path.join(output_path, task)

        variation_dirs = sorted([
            d for d in os.listdir(task_path)
            if d.startswith('variation') and
            os.path.isdir(os.path.join(task_path, d))
        ])

        print(f'\nTask: {task}  ({len(variation_dirs)} variations)')

        for var_dir in variation_dirs:
            var_path     = os.path.join(task_path, var_dir)
            var_out_path = os.path.join(task_out_path, var_dir)
            episodes_path     = os.path.join(var_path, 'episodes')
            episodes_out_path = os.path.join(var_out_path, 'episodes')

            descriptions = []
            var_desc_file = os.path.join(var_path, VARIATION_DESCRIPTIONS)
            if os.path.exists(var_desc_file):
                with open(var_desc_file, 'rb') as f:
                    descriptions = pickle.load(f)

            if not os.path.exists(episodes_path):
                continue

            episode_dirs = sorted([
                d for d in os.listdir(episodes_path)
                if d.startswith('episode') and
                os.path.isdir(os.path.join(episodes_path, d))
            ])

            var_summary = []
            for ep_dir in episode_dirs:
                ep_path     = os.path.join(episodes_path, ep_dir)
                ep_out_path = os.path.join(episodes_out_path, ep_dir)
                os.makedirs(ep_out_path, exist_ok=True)

                phase_info = process_episode(
                    ep_path, ep_out_path, descriptions,
                    signals=signals,
                    min_phase_len=min_phase_len,
                    save_mode=save_mode,
                    ep_label=f'{task}/{var_dir}/{ep_dir}',
                )

                if phase_info:
                    phases_str = ' | '.join(
                        f"phase_{p['phase_index']}"
                        f"(kf={p['keyframe_index']}, "
                        f"len={p['length']}f, "
                        f"sig={'+'.join(p['trigger_signals']) or '-'}, "
                        f"{'INTERACT' if p.get('is_interacting') else 'non-interact'}, "
                        f"gripper={'open' if p['gripper_open_ratio'] > GRIPPER_OPEN_THR else 'closed'})"
                        for p in phase_info)
                    print(f'  {var_dir}/{ep_dir}: {phases_str}')
                    var_summary.append({
                        'episode':    ep_dir,
                        'num_phases': len(phase_info),
                        'phases':     phase_info,
                    })

            os.makedirs(var_out_path, exist_ok=True)
            with open(os.path.join(var_out_path, 'split_summary.json'), 'w') as f:
                json.dump({
                    'task':         task,
                    'variation':    var_dir,
                    'descriptions': descriptions,
                    'save_mode':    save_mode,
                    'signals':      list(signals) if signals else list(ALL_SIGNALS),
                    'episodes':     var_summary,
                }, f, ensure_ascii=False, indent=2)

    print('\nDone.')


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='将 RLBench demo 数据按关键帧分割（三阶段流程）',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--save_path',   default='demos_all_0323',
                        help='原始 demos 文件夹路径 (默认: demos)')
    parser.add_argument('--output_path', default='test_demos_subphase_all_0323',
                        help='输出子阶段 demos 的根目录 (默认: demos_subphase)')
    parser.add_argument('--task',        default=None,
                        help='只处理指定任务，不填则处理所有任务')
    parser.add_argument('--save_mode',   default='keyframe_only',
                        choices=['full', 'keyframe_only'],
                        help='保存模式 (默认: keyframe_only)')

    args = parser.parse_args()

    split_all_demos(
        save_path=args.save_path,
        output_path=args.output_path,
        task_name=args.task,
        signals=RUN_SIGNALS,
        min_phase_len=RUN_MIN_PHASE_LEN,
        save_mode=args.save_mode,
    )


if __name__ == '__main__':
    main()

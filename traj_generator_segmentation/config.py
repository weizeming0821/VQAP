# -*- coding: utf-8 -*-
"""
全局配置常量

所有可调参数集中在此文件，便于统一修改。
"""

# ─────────────────────────────────────────────────────────────────────────────
# 信号检测常量
# ─────────────────────────────────────────────────────────────────────────────

# 推/按类任务的松散接触阈值（open=1 时仍有轻微接触力）
LOW_CONTACT_FORCE = 0.05
# 接触力需持续多少帧才算稳定接触（防单帧噪声）
CONTACT_PERSIST = 3
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
# 开爪（closed->open）后，D 证据进入冷却窗口（帧数）
OPEN_AFTER_RELEASE_D_COOLDOWN = 8
# 闭夹爪非抓持下压（E）进入所需连续命中帧数
CLOSED_PRESS_ENTER_PERSIST = 2

# gripper_open 二值化阈值：> 该值判 open，否则判 closed
GRIPPER_OPEN_THR = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# 静止段判定常量
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# 运行参数（可直接修改）
# ─────────────────────────────────────────────────────────────────────────────

# 启用的信号，None = 使用 ALL_SIGNALS
RUN_SIGNALS = None
# 相邻关键帧最小帧距（距离合并阈值）
RUN_MIN_PHASE_LEN = 5
# 打印每个 episode 的分割追踪信息
RUN_SHOW_SEG_TRACE = True
# 是否打印交互帧/交互段的详细判定依据（debug 模式）
RUN_DEBUG_TRACE = False
# 追踪输出中每项列表最多打印多少个索引（None = 不限制）
RUN_TRACE_MAX_ITEMS = 120
# 是否导出每帧传感器值
RUN_DUMP_SENSORS = True
# 是否在分割过程中打印每帧传感器值
RUN_PRINT_SENSOR_VALUES = False

# ─────────────────────────────────────────────────────────────────────────────
# 采集配置
# ─────────────────────────────────────────────────────────────────────────────

# 默认图像分辨率
DEFAULT_IMAGE_SIZE = [128, 128]
# 默认渲染器
DEFAULT_RENDERER = 'opengl3'
# 默认进程数
DEFAULT_PROCESSES = 8
# 默认每个变体的 episode 数
DEFAULT_EPISODES_PER_TASK = 100
# 默认变体数量
DEFAULT_VARIATIONS = 5
# 默认最大手臂速度
DEFAULT_ARM_MAX_VELOCITY = 1.0
# 默认最大手臂加速度
DEFAULT_ARM_MAX_ACCELERATION = 4.0
# 单条 demo 采集超时（秒）
DEFAULT_DEMO_TIMEOUT = 90
# Worker 卡住判定秒数
DEFAULT_WORKER_STUCK_TIMEOUT = 180
# 最大重试次数
MAX_DEMO_ATTEMPTS = 10
# 验证阶段数不匹配时的最大重试次数
MAX_PHASE_VALIDATION_RETRIES = 5

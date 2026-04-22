# VQAP: Vector-Quantized Action Primitives via Intrinsic Curiosity-driven World Models

---

## 一体化 Demo 生成与关键帧分割流水线

直接在内存中采集 RLBench 演示轨迹，立即进行关键帧分割，仅保存分割后的子阶段结果。

### 目录结构

```
traj_generator_segmentation/
├── config.py          # 全局配置常量
├── thresholds.py      # 自动阈值估计
├── signals.py         # 信号候选帧提取
├── interaction.py     # 交互段判定
├── keyframe.py        # 关键帧提取主逻辑
├── demo_io.py         # 子阶段保存
├── metadata.py        # 元数据管理
├── validation.py      # 阶段数验证
└── pipeline.py        # 主流水线

traj_generator_segmentation.py  # 入口脚本
config/
└── traj_generator_segmentation.yaml  # 分割与采集配置
```

说明：
`traj_generator_segmentation/config.py` 现在负责从 `config/traj_generator_segmentation.yaml` 读取配置并导出兼容常量；日常调参请直接修改 YAML 文件。

### 使用方法

```bash
# 默认配置
python traj_generator_segmentation.py

# 指定任务
python traj_generator_segmentation.py --tasks open_drawer close_drawer

# 指定输出路径
python traj_generator_segmentation.py --output_path ./my_output

# 调整采集参数
python traj_generator_segmentation.py --episodes_per_task 5 --variations 10 --processes 8

# 完整模式（保留完整子轨迹，而非仅关键帧）
python traj_generator_segmentation.py --save_mode full
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_path` | `./demos_subphase` | 输出目录 |
| `--tasks` | 全部任务 | 要采集的任务列表 |
| `--episodes_per_task` | 2 | 每个变体的 episode 数 |
| `--variations` | 5 | 每个任务的变体数 |
| `--processes` | 8 | 并行进程数 |
| `--min_phase_len` | 5 | 相邻关键帧最小帧距 |
| `--save_mode` | `keyframe_only` | 保存模式 (`full`/`keyframe_only`) |
| `--fixed_phase_csv` | `./TASK_FIXED_PHASE_NUM.csv` | 固定阶段数配置 |

### 输出结构

```
output_path/
  task_name/
    task_metadata.json
    variationN/
      variation_metadata.json
      split_summary.json
      episodes/
        episodeN/
          phase_metadata.json
          phase_0/
            low_dim_obs.pkl
            front_rgb/*.png
            ...
```

### 阶段数验证

根据 `TASK_FIXED_PHASE_NUM.csv` 验证分割结果，阶段数不匹配时自动重试采集（最多 3 次）。

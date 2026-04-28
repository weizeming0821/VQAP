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

### 多 GPU / 多 DISPLAY 分片启动

当服务器上可同时使用多个 DISPLAY / GPU 时，建议不要单纯增大单次运行的 `--processes`，而是启动多个独立作业，把任务按 task 维度分片到不同 DISPLAY。仓库提供了独立入口：

```bash
# 例：使用两个 DISPLAY，并为每个 DISPLAY 启动 2 个 worker
python traj_generator_segmentation_multi_gpu.py \
  --output_path ./demos_multi_gpu \
  --displays :99.0 :99.1 \
  --gpu_ids 2 3 \
  --processes_per_display 2 2 \
  --fixed_phase_only \
  --episodes_per_task 10 \
  --variations 3 \
  --demo_timeout 120
```

说明：
- 每个 DISPLAY 会启动一个独立的 `traj_generator_segmentation.py` 子作业。
- 启动器会在终端显示跨所有 DISPLAY 的总体进度，而不是只打印子作业启动信息。
- `--output_path` 直接指定最终数据集根目录；例如 `--output_path ./smoke_dataset`，完成后目录结构就是 `smoke_dataset/task/variation...`。
- 子作业会先写入数据集旁边的固定内部 shard 工作目录 `.输出目录名_launcher_work/`，随后自动合并到 `--output_path` 指定的数据集目录；每次启动会先重建该目录，默认即使部分任务失败也会清理这类中间目录。
- 启动器最终只保留一份总日志，保存在仓库级 `VQAP/log/` 下，文件名形如 `traj_gen_seg_时间.log`。
- `--gpu_ids` 主要用于同步约束子进程环境变量；RLBench / PyRep 的渲染绑定仍以 `DISPLAY` 为主。
- 默认会清理中间 shard 工作目录；若要保留它们用于排查，可加 `--keep_workdirs`。

如果你只想运行指定任务，也可以显式传入 `--tasks`：

```bash
python traj_generator_segmentation_multi_gpu.py \
  --output_path ./demos_multi_gpu \
  --displays :99.0 :99.1 \
  --processes_per_display 1 1 \
  --tasks open_drawer close_drawer insert_onto_square_peg \
  --episodes_per_task 20 \
  --variations 5
```

### 自动 benchmark 不同 worker 配置

如果你想自动比较每个 DISPLAY 下 `1 / 2 / 4` 个 worker 的吞吐、失败率，并在每轮结束后校验输出编号与 metadata 一致性，可以使用：

```bash
python traj_generator_segmentation_benchmark.py \
  --benchmark_root ./benchmark_runs \
  --displays :99.0 :99.1 \
  --gpu_ids 2 3 \
  --worker_counts 1 2 4 \
  --tasks open_drawer close_drawer \
  --episodes_per_task 10 \
  --variations 2 \
  --demo_timeout 120
```

说明：
- benchmark 脚本会为每个 worker 配置单独创建一个 run 目录。
- 每轮 benchmark 会调用 `traj_generator_segmentation_multi_gpu.py` 启动作业。
- 每轮结束后默认会调用 `validate_segmented_dataset.py`，检查 variation / episode 编号与各层 metadata 聚合是否一致。
- 结果会汇总到 `benchmark_root/benchmark_summary.json`，并输出吞吐排名。
- 如果只想先看命令矩阵，不真正运行，可加 `--dry_run`。

### 校验生成结果的编号与 metadata

可以单独对已有输出目录做一致性校验：

```bash
# 校验单个数据集根目录
python validate_segmented_dataset.py ./demos_test

# 校验 multi-GPU launcher 生成出的最终数据集目录
python validate_segmented_dataset.py ./demos_multi_gpu
```

校验内容包括：
- `variationN` 目录名与 `variation_metadata.json` 中的 `variation_index` 是否一致。
- `episodeN` 目录、`episode_summaries`、`phase_metadata_path` 是否一一对应。
- `num_episodes`、`valid_episodes`、`phase_counts`、`avg_phases` 是否与实际 episode/phase 文件一致。
- `task_metadata.json` 和 `dataset_metadata.json` 是否能由 variation 层 metadata 正确聚合得到。
- 兼容旧版 multi-shard launcher 输出时，也会检查 task 是否重复落到多个 shard。

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
  dataset_metadata.json
  task_name/
    task_metadata.json
    variationN/
      variation_metadata.json
      episodes/
        episodeN/
          phase_metadata.json
          phase_0/
            low_dim_obs.pkl
            front_rgb/*.png
            ...
```

其中：
- `dataset_metadata.json` 记录本次整体生成统计与关键运行参数。
- `variation_metadata.json` 记录变体描述、variation 级统计以及每个 episode 的摘要索引。
- `phase_metadata.json` 仅保留单个 episode 的分割细节，避免跨层重复记录 descriptions。

### 阶段数验证

根据 `TASK_FIXED_PHASE_NUM.csv` 验证分割结果，阶段数不匹配时自动重试采集（最多 3 次）。

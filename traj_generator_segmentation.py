#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一体化 Demo 生成与关键帧分割流水线

直接在内存中采集 RLBench 演示轨迹，立即进行关键帧分割，
仅将分割后的子阶段结果写入磁盘，避免先保存完整 demo 再分割带来的巨大磁盘开销。

特性：
  - 内存中采集，立即分割，节省磁盘空间
  - 自动验证分割阶段数是否符合 TASK_FIXED_PHASE_NUM.csv 要求
  - 阶段数不匹配时自动重试采集
  - 模块化代码结构，易于维护

输出目录结构：
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
              front_rgb/0.png, 1.png, ...
              wrist_rgb/...
            phase_1/
            ...

用法：
  python traj_generator_segmentation.py --output_path ./demos_subphase
  python traj_generator_segmentation.py --tasks open_drawer close_drawer
  python traj_generator_segmentation.py --episodes_per_task 5 --variations 10
"""

from traj_generator_segmentation.pipeline import main

if __name__ == '__main__':
    main()

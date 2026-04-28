# -*- coding: utf-8 -*-
"""
一体化 Demo 生成与关键帧分割流水线

直接在内存中采集 RLBench 演示轨迹，立即进行关键帧分割，
仅将分割后的子阶段结果写入磁盘，避免先保存完整 demo 再分割带来的巨大磁盘开销。
"""

from .config import *
from .thresholds import auto_thresholds
from .signals import collect_stage1_candidates, ALL_SIGNALS
from .interaction import label_interacting_frames, label_interacting_segments
from .keyframe import extract_keyframes
from .demo_io import save_subphase_demo_from_memory, save_phase_metadata, process_demo_in_memory
from .metadata import save_variation_metadata, save_task_metadata
from .validation import load_fixed_phase_config, validate_phase_count
from .collection import run_segmented_collection
from .cli import main

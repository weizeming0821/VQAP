# -*- coding: utf-8 -*-
"""兼容封装，实际实现已迁移到 traj_generator_segmentation.collection。"""

from .collection import build_parser, parse_args, run_segmented_collection, main

__all__ = ['build_parser', 'parse_args', 'run_segmented_collection', 'main']


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一数据生成入口。"""

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from traj_generator_segmentation.cli import main


if __name__ == '__main__':
    raise SystemExit(main())

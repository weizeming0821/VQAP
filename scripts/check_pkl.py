# -*- coding: utf-8 -*-
"""检查现有 pkl 文件的内容"""

import pickle
import os

episode_path = '/home/fanfan/proj/VQAP/demos_subphase/phone_on_base/variation0/episodes/episode0'

print("检查现有 pkl 文件内容：")
print("-" * 50)

for phase_idx in range(3):
    pkl_path = os.path.join(episode_path, f'phase_{phase_idx}', 'low_dim_obs.pkl')
    with open(pkl_path, 'rb') as f:
        obs_list = pickle.load(f)

    # 检查第一个观测的字段
    if obs_list:
        obs = obs_list[0]
        print(f"phase_{phase_idx}:")
        print(f"  pkl 帧数: {len(obs_list)}")
        print(f"  gripper_open: {obs.gripper_open}")
        print(f"  joint_positions shape: {obs.joint_positions.shape if obs.joint_positions is not None else None}")
        print()

# VQAP

VQAP 是面向 VLA 模型的**离散动作码本预训练**项目：以 RLBench 专家演示中的原子动作片段为数据，通过双码本（全局 $K_g{=}36$ / 细节 $K_d{=}256$）NSVQ 量化 + Flow Matching 重构 + 视觉语义对齐，学习一组带原子动作语义的离散码本，供下游 VLA 复用。

> 本 README 只记录**如何执行脚本、复刻实验、跑通流程**。模型结构、损失设计、维度推导等请见 [VQAP_Design.md](VQAP_Design.md)（数据集 / 双码本模型 / 预训练）；下游 VLA 集成设计见 [VLA_Design.md](VLA_Design.md)。

---

## 1. 环境依赖

- Python 3.13、PyTorch（CUDA）、`peft`、`pyyaml`、`tqdm`、`tensorboard`
- 视角选择 / 视觉编码用 DINOv2，经 `torch.hub` 加载 `facebookresearch/dinov2`（首次需联网下载到 hub cache）
- 码本可视化额外需要 `matplotlib`、`scikit-learn`
- 数据集生成额外需要 RLBench / PyRep（仅在重新采集数据时需要）

---

## 2. 目录结构

```
config/        训练 / 模型 / 数据集 / 数据生成配置（YAML）
data/          AtomActionDataset、collate、归一化、视角选择
model/         VQAP 模型与各子模块
utils/         损失函数、logger / TensorBoard 工具
scripts/       train_vqap.py（训练）、generate_dataset.py（数据生成）
tools/         visualize_codebook.py（码本可视化）
traj_generator_segmentation/   RLBench 采集 + 分割
train_vqap.sh        训练启动脚本（torchrun）
run_tensorboard.sh   TensorBoard 启动脚本
checkpoints/   训练输出权重（按实验名 / stage 分目录）
log/           文本日志
tensorboard/   TensorBoard 事件文件
AtomAction_Dataset/   训练数据集
```

---

## 3. 数据集

训练消费的是 `AtomAction_Dataset/`：每条样本为一个**原子动作片段（phase）**的低维轨迹 + 多视角 RGB 起止帧，按 `action/task/variation/phase` 组织。归一化统计量存于数据集根目录的 `dataset_metadata.json::traj_stats`，在 `__getitem__` 阶段直接读取。详细数据流见 [VQAP_Design.md](VQAP_Design.md) §二。

**重新生成数据集**（需 RLBench 环境，已有数据集可跳过）：

```bash
# 查看全部参数
python scripts/generate_dataset.py --help

# 单机采集
python scripts/generate_dataset.py --output_path ./demos_subphase

# 多 DISPLAY 并行采集
python scripts/generate_dataset.py --displays :99.0 :99.1 --output_path ./demos_multi_gpu
```

---

## 4. 配置文件（控制参数）

训练读取三份 YAML，可用命令行覆盖路径（见 §5）：

| 文件 | 作用 | 常调参数 |
|---|---|---|
| `config/train.yaml` | 训练超参 | `data.batch_size`、`stage.stage0_epochs/stage1_epochs`、`lr.*`、`loss.*`、`codebook.*`、`checkpoint.save_every_epochs`、`runtime.precision` |
| `config/model.yaml` | 模型结构 | 码本大小、各模块维度、RoPE 等 |
| `config/global.yaml` | 数据集 / 全局 | `atomactiondataset.dataset_root`、`top_k`、`train_actions`（动作白名单）、`train_act_dim`（输入字段白名单） |

关键控制点：

- **训练时长 / 阶段**：`stage.stage0_epochs`（默认 60）+ `stage.stage1_epochs`（默认 40）= 总 epoch。Stage 0 联合预热（含 DINOv2 LoRA + $L_{AG}$），Stage 1 视觉冻结、只训码本与未来帧支路。
- **学习率**：`lr.stage0_main` / `lr.stage0_lora` / `lr.stage1_main`；线性 warmup + cosine（`scheduler.warmup_epochs`、`scheduler.min_lr_ratio`）。
- **损失权重**：`loss.lambda_ap/ag/rot/grip/sep`，未来帧权重调度 `loss.future_stage0_max`、`future_stage1_max`、`future_stage1_ramp_epochs`。
- **码本防坍缩**：`codebook.perplexity_g_threshold` / `perplexity_d_threshold` / `replace_interval_epochs`（每 epoch 判定，设为 `inf` 则只保留 perplexity 触发）。
- **精度**：`runtime.precision`，`bf16`（默认）/ `fp16`（启用 GradScaler）/ `fp32`。
- **动作白名单**：`global.yaml::train_actions` 决定参与训练的动作类型（`pose-adjust` 永久排除）。

---

## 5. 训练

单机多卡用 `torchrun` 启动。快捷脚本 `train_vqap.sh`：

```bash
bash train_vqap.sh
# 内容等价于：
# CUDA_VISIBLE_DEVICES="1,5" torchrun --nproc_per_node=2 scripts/train_vqap.py
```

直接调用并指定参数：

```bash
# 全新训练（2 卡），自定义实验名
CUDA_VISIBLE_DEVICES="1,5" torchrun --nproc_per_node=2 \
    scripts/train_vqap.py --exp-name vqap_run1
```

命令行参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | `config/train.yaml` | 训练配置 |
| `--model-config` | `config/model.yaml` | 模型结构配置 |
| `--global-config` | `config/global.yaml` | 数据集 / 全局配置 |
| `--exp-name` | 取 yaml 中 `exp_name` | 实验名（决定输出子目录） |
| `--resume-stage` | 无 | 续训阶段 `{0,1}`；省略=全新训练 |
| `--resume-path` | 无 | 显式 checkpoint 路径，**必须**与 `--resume-stage` 同用 |
| `--disable-tensorboard` | 关 | 即使 yaml 启用也强制关闭 TensorBoard |

**断点续训**：仅当传 `--resume-stage` 时才续训；不指定 `--resume-path` 时默认读 `checkpoints/<exp>/stage{N}/latest.pth`。

```bash
# 从 Stage 0 最新 checkpoint 续训（跑到 stage0_epochs 后自动切入 Stage 1）
torchrun --nproc_per_node=2 scripts/train_vqap.py --exp-name vqap_run1 --resume-stage 0

# 从指定 Stage 1 checkpoint 续训
torchrun --nproc_per_node=2 scripts/train_vqap.py --exp-name vqap_run1 \
    --resume-stage 1 --resume-path checkpoints/vqap_run1/stage1/best_lap.pth
```

> 续训会校验配置一致性：进入 Stage 1 后 `stage0_epochs` 不可改；当前 stage 的 epoch 数只能增大不能改小（延长训练请增大对应 `stage*_epochs`）。

**输出产物**（`checkpoints/<exp-name>/stage{0,1}/`）：

| 文件 | 说明 |
|---|---|
| `latest.pth` | 最新权重，按 `save_every_epochs` 与 stage 末尾保存，用于续训 |
| `best_lap.pth` / `best_ltotal.pth` | $L_{AP}$ / $L_{total}$ 最优权重（每 epoch 判定） |
| `codebook.pth` | 仅码本权重，供下游 VLA 迁移 |

文本日志写入 `log/`，TensorBoard 事件写入 `tensorboard/<exp-name>/`。

---

## 6. TensorBoard

```bash
bash run_tensorboard.sh          # 默认端口 6006
bash run_tensorboard.sh 6007     # 指定端口
```

监控曲线（x 轴 = epoch）：`loss/*`、`loss_ap/*`、`loss_ag/*`、`train_state/*`（grad_norm、lambda_future、stage）、`lr/*`、`codebook/*`（perplexity_g/d、replaced_g/d）、`best/*`。码本健康判据见 [VQAP_Design.md](VQAP_Design.md) §5.5。

---

## 7. 码本可视化

训练后用 `tools/visualize_codebook.py` 统计与可视化全局 / 细节码本使用情况（t-SNE 等）：

```bash
python tools/visualize_codebook.py \
    --checkpoint checkpoints/vqap_pretrain/stage1/best_lap.pth \
    --output-dir viz/codebook/stage1
```

常用参数：`--output-dir`（建议显式指定输出目录）、`--batch-size`、`--max-batches`（0=全量）、`--max-points`（t-SNE 采样上限）、`--device`。

---

## 8. 复刻实验最小流程

```bash
# 1. 确认数据集就位：AtomAction_Dataset/ 及其 dataset_metadata.json
# 2. 按需调整 config/train.yaml、config/global.yaml
# 3. 启动训练
CUDA_VISIBLE_DEVICES="1,5" torchrun --nproc_per_node=2 scripts/train_vqap.py --exp-name my_run
# 4. 另开终端看曲线
bash run_tensorboard.sh
# 5. 训练完成后可视化码本
python tools/visualize_codebook.py --checkpoint checkpoints/my_run/stage1/best_lap.pth
```

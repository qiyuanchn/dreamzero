# DreamZero 训练说明

这份文档参考 `docs/INFERENCE_SERVICE_GUIDE.md` 的写法，整理了当前仓库里 DreamZero 1.3B 训练的实际用法，包含：

- 只在前四卡启动 LoRA
- full fine-tune 的启动方式
- 如何实时看 loss
- 如何用 TensorBoard 看当前和后续训练
- 训练结果如何统一放到 `outputs/`

适用仓库路径：`/home/zqy/ws/dreamzero`

## 1. 基本结论

当前建议的 1.3B 训练路线：

- 第一阶段：LoRA，优先验证训练通路和初始收敛
- 第二阶段：full fine-tune，作为正式主线
- backbone: `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP`
- tokenizer: `umt5-xxl`
- 数据：`GEAR-Dreams/DreamZero-DROID-Data`

当前仓库里的训练脚本默认会：

- 自动识别普通 LeRobot 数据目录
- 自动识别 Hugging Face cache/snapshot 数据目录
- 默认把输出放到 `outputs/training/`
- 默认开启 `report_to=tensorboard`

## 2. 环境准备

进入环境：

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
export PYTHONPATH=/home/zqy/ws/dreamzero
export DREAMZERO_OUTPUT_ROOT=/home/zqy/ws/dreamzero/outputs
```

常用路径：

```bash
WAN_CKPT_DIR=/home/zqy/ws/dreamzero/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP
TOKENIZER_DIR=/home/zqy/ws/dreamzero/checkpoints/umt5-xxl
DROID_DATA_ROOT=/data2/zqy/datasets--GEAR-Dreams--DreamZero-DROID-Data
OUTPUT_ROOT=/home/zqy/ws/dreamzero/outputs
```

如果 `DROID_DATA_ROOT` 指向 Hugging Face cache 根目录，训练脚本会自动解析到最新的 `snapshots/<hash>`。

## 3. LoRA 训练

### 3.1 只用前四卡启动

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
PYTHONPATH=/home/zqy/ws/dreamzero \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
DROID_DATA_ROOT=/data2/zqy/datasets--GEAR-Dreams--DreamZero-DROID-Data \
OUTPUT_DIR=/home/zqy/ws/dreamzero/outputs/training/droid_lora \
TB_LOGDIR=/home/zqy/ws/dreamzero/outputs/training/droid_lora/tensorboard \
bash /home/zqy/ws/dreamzero/scripts/train/droid_training_lora.sh
```

默认行为：

- `report_to=tensorboard`
- checkpoint 输出到 `outputs/training/droid_lora`
- TensorBoard 日志输出到 `outputs/training/droid_lora/tensorboard`
- `save_lora_only=true`

常用覆盖项：

```bash
MAX_STEPS=20000
SAVE_STEPS=1000
GLOBAL_BATCH_SIZE=4
PER_DEVICE_BS=1
NUM_WORKERS=4
```

## 4. Full Fine-Tune

### 4.1 启动 full

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
PYTHONPATH=/home/zqy/ws/dreamzero \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
DROID_DATA_ROOT=/data2/zqy/datasets--GEAR-Dreams--DreamZero-DROID-Data \
OUTPUT_DIR=/home/zqy/ws/dreamzero/outputs/training/droid_full \
TB_LOGDIR=/home/zqy/ws/dreamzero/outputs/training/droid_full/tensorboard \
bash /home/zqy/ws/dreamzero/scripts/train/droid_training_full_finetune.sh
```

默认行为：

- `report_to=tensorboard`
- checkpoint 输出到 `outputs/training/droid_full`
- TensorBoard 日志输出到 `outputs/training/droid_full/tensorboard`
- `save_lora_only=false`
- deepspeed 配置：`zero2_offload`

## 5. 实时看训练效果

### 5.1 最直接：看 JSONL loss

每个 run 会写一个 `loss_log.jsonl`，包含：

- `loss`
- `dynamics_loss_avg`
- `action_loss_avg`
- `learning_rate`

例如：

```bash
tail -f /home/zqy/ws/dreamzero/outputs/training/droid_lora/loss_log.jsonl
```

如果当前 run 在别的磁盘目录，也可以直接看那个目录下的 `loss_log.jsonl`。

### 5.2 用 TensorBoard 看后续训练

脚本默认已经开启 `report_to=tensorboard`，直接启动：

```bash
tensorboard --logdir /home/zqy/ws/dreamzero/outputs/training --port 6006
```

浏览器里打开：

```text
http://127.0.0.1:6006
```

### 5.3 给“当前已经在跑的老 run”补 TensorBoard

如果某个 run 启动时 `report_to=none`，但已经有 `loss_log.jsonl`，可以用下面的桥接脚本实时同步：

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero

python scripts/train/loss_jsonl_to_tensorboard.py \
  --input /data2/zqy/training_runs/droid_lora_1p3b_from_base_20260326_191733/loss_log.jsonl \
  --logdir /home/zqy/ws/dreamzero/outputs/tensorboard/droid_lora_1p3b_from_base_20260326_191733
```

然后启动：

```bash
tensorboard --logdir /home/zqy/ws/dreamzero/outputs/tensorboard --port 6006
```

## 6. 只保留前四卡 LoRA

如果你要清空其它卡上的训练，只保留 `CUDA_VISIBLE_DEVICES=0,1,2,3` 的 LoRA：

```bash
ps -ef | rg "train_architecture=full|output_dir="
pkill -TERM -f "train_architecture=full"
```

更稳妥的做法是按 `output_dir` 定位后再停：

```bash
pkill -TERM -f "output_dir=/path/to/that/run"
```

## 7. 输出目录约定

推荐统一放到：

```text
/home/zqy/ws/dreamzero/outputs/training/
```

建议结构：

- `outputs/training/droid_lora`
- `outputs/training/droid_full`
- `outputs/tensorboard/<run_name>`

## 8. 常见监控命令

看最新 loss：

```bash
tail -n 20 /home/zqy/ws/dreamzero/outputs/training/droid_lora/loss_log.jsonl
```

看 checkpoint：

```bash
find /home/zqy/ws/dreamzero/outputs/training/droid_lora -maxdepth 1 -type d -name 'checkpoint-*'
```

看 GPU：

```bash
nvidia-smi
```

看训练进程：

```bash
ps -ef | rg "groot/vla/experiment/experiment.py|torch.distributed.run"
```

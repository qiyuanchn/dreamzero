#!/bin/bash
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DREAMZERO_ROOT=${DREAMZERO_ROOT:-"$REPO_ROOT"}
source "$REPO_ROOT/scripts/lib/output_layout.sh"
DREAMZERO_OUTPUT_ROOT="$(dz_default_output_root "$DREAMZERO_ROOT")"

DRONE_DATA_ROOT=${DRONE_DATA_ROOT:-"$REPO_ROOT/data/drone_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$(dz_default_train_dir "$DREAMZERO_OUTPUT_ROOT" "drone" "lora")"}
TB_LOGDIR=${TB_LOGDIR:-"$OUTPUT_DIR/tensorboard"}
REPORT_TO=${REPORT_TO:-tensorboard}
NUM_GPUS=${NUM_GPUS:-1}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
NUM_STEPS_PER_SHARD=${NUM_STEPS_PER_SHARD:-2000}
MIN_NUM_SHARDS_TO_SAMPLE=${MIN_NUM_SHARDS_TO_SAMPLE:-4096}
ESTIMATED_NUM_SHARDS_TO_SAMPLE=$(( (MAX_STEPS * GLOBAL_BATCH_SIZE + NUM_STEPS_PER_SHARD - 1) / NUM_STEPS_PER_SHARD ))
ESTIMATED_NUM_SHARDS_TO_SAMPLE=$(( ESTIMATED_NUM_SHARDS_TO_SAMPLE * 4 ))
if [ "$ESTIMATED_NUM_SHARDS_TO_SAMPLE" -lt "$MIN_NUM_SHARDS_TO_SAMPLE" ]; then
    ESTIMATED_NUM_SHARDS_TO_SAMPLE="$MIN_NUM_SHARDS_TO_SAMPLE"
fi
NUM_SHARDS_TO_SAMPLE=${NUM_SHARDS_TO_SAMPLE:-$ESTIMATED_NUM_SHARDS_TO_SAMPLE}
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-true}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
PYTHON_BIN=${DREAMZERO_PYTHON:-}

if [ -z "$PYTHON_BIN" ]; then
    if [ -x "/home/zqy/miniconda3/envs/dreamzero/bin/python" ]; then
        PYTHON_BIN="/home/zqy/miniconda3/envs/dreamzero/bin/python"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi

if [ ! -d "$DRONE_DATA_ROOT" ]; then
    echo "ERROR: drone dataset not found at $DRONE_DATA_ROOT"
    exit 1
fi

if [ ! -f "$DRONE_DATA_ROOT/info.json" ] || [ ! -f "$DRONE_DATA_ROOT/modality.json" ]; then
    echo "ERROR: dataset metadata missing in $DRONE_DATA_ROOT"
    exit 1
fi

"$PYTHON_BIN" - "$DRONE_DATA_ROOT" "$NUM_STEPS_PER_SHARD" "$NUM_SHARDS_TO_SAMPLE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
num_steps_per_shard = sys.argv[2]
summary_path = root / "conversion_summary.json"
info_path = root / "info.json"

print(f"Dataset root: {root}")
print(f"num_steps_per_shard: {num_steps_per_shard}")
print(f"num_shards_to_sample: {sys.argv[3]}")
if info_path.exists():
    info = json.loads(info_path.read_text())
    print(
        "Dataset summary: "
        f"episodes={info.get('total_episodes')} "
        f"frames={info.get('total_frames')} "
        f"chunks={info.get('total_chunks')}"
    )
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    print(f"Source formats: {summary.get('source_formats', ['unknown'])}")
    for key in ["observation_source", "action_source", "timestamp_source", "language_source"]:
        if key in summary:
            print(f"{key}: {summary[key]}")
PY

mkdir -p "$OUTPUT_DIR" "$TB_LOGDIR"
mkdir -p "$DREAMZERO_OUTPUT_ROOT/train"
ln -sfn "$OUTPUT_DIR" "$DREAMZERO_OUTPUT_ROOT/train/latest"
ln -sfn "$OUTPUT_DIR" "$DREAMZERO_OUTPUT_ROOT/train/latest_drone_lora"
cd "$DREAMZERO_ROOT"

EXTRA_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    EXTRA_ARGS+=("resume_from_checkpoint=$RESUME_FROM_CHECKPOINT")
fi

exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone \
    groot/vla/experiment/experiment.py \
    report_to="$REPORT_TO" \
    data=dreamzero/drone_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=1 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan13 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    max_state_dim=12 \
    max_action_dim=72 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    +training_args.logging_dir="$TB_LOGDIR" \
    save_steps="$SAVE_STEPS" \
    logging_steps="$LOGGING_STEPS" \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size="$PER_DEVICE_BS" \
    global_batch_size="$GLOBAL_BATCH_SIZE" \
    max_steps="$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit=5 \
    +training_args.save_only_model="$SAVE_ONLY_MODEL" \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers="$NUM_WORKERS" \
    drone_num_steps_per_shard="$NUM_STEPS_PER_SHARD" \
    +train_dataset.mixture_kwargs.num_shards_to_sample="$NUM_SHARDS_TO_SAMPLE" \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=220 \
    save_strategy=steps \
    drone_data_root="$DRONE_DATA_ROOT" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR" \
    "${EXTRA_ARGS[@]}"

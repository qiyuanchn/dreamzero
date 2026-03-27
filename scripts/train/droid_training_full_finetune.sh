#!/bin/bash
# DreamZero DROID Full Fine-Tuning Script (Wan2.1 1.3B)
#
# Usage:
#   source /home/zqy/miniconda3/etc/profile.d/conda.sh
#   conda activate dreamzero
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train/droid_training_full_finetune.sh

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DREAMZERO_ROOT=${DREAMZERO_ROOT:-"$REPO_ROOT"}
source "$REPO_ROOT/scripts/lib/output_layout.sh"
DREAMZERO_OUTPUT_ROOT="$(dz_default_output_root "$DREAMZERO_ROOT")"

resolve_data_root() {
    local candidate="$1"
    if [ -d "$candidate/data" ] && [ -d "$candidate/meta" ]; then
        echo "$candidate"
        return 0
    fi
    if [ -d "$candidate/snapshots" ]; then
        local latest_snapshot
        latest_snapshot="$(find "$candidate/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
        if [ -n "$latest_snapshot" ] && [ -d "$latest_snapshot/data" ] && [ -d "$latest_snapshot/meta" ]; then
            echo "$latest_snapshot"
            return 0
        fi
    fi
    return 1
}

default_data_root="./data/droid_lerobot"
if [ -d "/data2/zqy/datasets--GEAR-Dreams--DreamZero-DROID-Data" ]; then
    default_data_root="/data2/zqy/datasets--GEAR-Dreams--DreamZero-DROID-Data"
fi

DROID_DATA_ROOT=${DROID_DATA_ROOT:-"$default_data_root"}
OUTPUT_DIR=${OUTPUT_DIR:-"$(dz_default_train_dir "$DREAMZERO_OUTPUT_ROOT" "droid" "full")"}
TB_LOGDIR=${TB_LOGDIR:-"$OUTPUT_DIR/tensorboard"}
REPORT_TO=${REPORT_TO:-tensorboard}
NUM_GPUS=${NUM_GPUS:-4}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_STEPS=${SAVE_STEPS:-2000}
LOGGING_STEPS=${LOGGING_STEPS:-10}
NUM_WORKERS=${NUM_WORKERS:-4}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
PYTHON_BIN=${DREAMZERO_PYTHON:-}

if [ -z "$PYTHON_BIN" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    elif [ -x "/home/zqy/miniconda3/envs/dreamzero/bin/python" ]; then
        PYTHON_BIN="/home/zqy/miniconda3/envs/dreamzero/bin/python"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi

if ! RESOLVED_DROID_DATA_ROOT="$(resolve_data_root "$DROID_DATA_ROOT")"; then
    echo "ERROR: DROID dataset not found or not in expected layout: $DROID_DATA_ROOT"
    echo "Expected either:"
    echo "  - <root>/data and <root>/meta"
    echo "  - Hugging Face cache root with snapshots/<hash>/data and snapshots/<hash>/meta"
    exit 1
fi
DROID_DATA_ROOT="$RESOLVED_DROID_DATA_ROOT"

if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-Fun-V1.1-1.3B-InP not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

mkdir -p "$OUTPUT_DIR" "$TB_LOGDIR"
mkdir -p "$DREAMZERO_OUTPUT_ROOT/train"
ln -sfn "$OUTPUT_DIR" "$DREAMZERO_OUTPUT_ROOT/train/latest"
ln -sfn "$OUTPUT_DIR" "$DREAMZERO_OUTPUT_ROOT/train/latest_droid_full"
cd "$DREAMZERO_ROOT"

exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone \
    groot/vla/experiment/experiment.py \
    report_to="$REPORT_TO" \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan13 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json" \
    +training_args.logging_dir="$TB_LOGDIR" \
    save_steps="$SAVE_STEPS" \
    logging_steps="$LOGGING_STEPS" \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size="$PER_DEVICE_BS" \
    global_batch_size="$GLOBAL_BATCH_SIZE" \
    max_steps="$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers="$NUM_WORKERS" \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    droid_data_root="$DROID_DATA_ROOT" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR"

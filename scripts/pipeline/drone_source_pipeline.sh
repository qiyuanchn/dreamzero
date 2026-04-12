#!/bin/bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-}
DATASET_OUT=${DATASET_OUT:-"$REPO_ROOT/data/drone_lerobot"}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-"$REPO_ROOT/outputs/train/drone_source_smoke"}
INFER_OUTPUT_DIR=${INFER_OUTPUT_DIR:-"$REPO_ROOT/outputs/inference/drone_source_smoke"}
MODEL_PATH=${MODEL_PATH:-"$TRAIN_OUTPUT_DIR/checkpoint-1"}
RUN_TRAIN=${RUN_TRAIN:-true}
RUN_INFER=${RUN_INFER:-true}
REUSE_DATASET=${REUSE_DATASET:-true}
REUSE_CHECKPOINT=${REUSE_CHECKPOINT:-true}
MAX_STEPS=${MAX_STEPS:-1}
INFER_INDEX=${INFER_INDEX:-0}
DEVICE=${DEVICE:-cuda:0}
CONVERT_TASK_VARIANTS=${CONVERT_TASK_VARIANTS:-1}
SKIP_VIDEO_DECODE=${SKIP_VIDEO_DECODE:-false}
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

if [ -z "$SOURCE_RUN_DIR" ]; then
    echo "ERROR: set SOURCE_RUN_DIR to source/outputs/test/YYYY-MM-DD/HH-MM-SS or a trajs directory"
    exit 1
fi

cd "$REPO_ROOT"

if [ "$REUSE_DATASET" != "true" ] || [ ! -f "$DATASET_OUT/info.json" ]; then
    "$PYTHON_BIN" scripts/data/convert_drone_tar_to_lerobot.py \
        --input "$SOURCE_RUN_DIR" \
        --output "$DATASET_OUT" \
        --task-variants "$CONVERT_TASK_VARIANTS"
else
    echo "Reusing existing dataset at $DATASET_OUT"
fi

if [ "$RUN_TRAIN" = "true" ] && { [ "$REUSE_CHECKPOINT" != "true" ] || [ ! -f "$MODEL_PATH/model.safetensors" ]; }; then
    DRONE_DATA_ROOT="$DATASET_OUT" \
    OUTPUT_DIR="$TRAIN_OUTPUT_DIR" \
    REPORT_TO=none \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_STEPS=1 \
    LOGGING_STEPS=1 \
    NUM_WORKERS=0 \
    SAVE_ONLY_MODEL=true \
    bash scripts/train/drone_training.sh
elif [ "$RUN_TRAIN" = "true" ]; then
    echo "Reusing existing checkpoint at $MODEL_PATH"
fi

if [ "$RUN_INFER" = "true" ]; then
    INFER_ARGS=()
    if [ "$SKIP_VIDEO_DECODE" = "true" ]; then
        INFER_ARGS+=(--skip_video_decode)
    fi

    "$PYTHON_BIN" scripts/open_loop_drone.py \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_OUT" \
        --device "$DEVICE" \
        --index "$INFER_INDEX" \
        --output_dir "$INFER_OUTPUT_DIR" \
        "${INFER_ARGS[@]}"
fi

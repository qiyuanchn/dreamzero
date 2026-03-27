#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/output_layout.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(dz_default_output_root "$ROOT_DIR")}"
PORT="${PORT:-5999}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/checkpoints/DreamZero-DROID}"
CONDA_ENV="${CONDA_ENV:-dreamzero}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-50000}"
PYTORCH_ALLOC="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DREAMZERO_SAVE_RESET_VIDEO="${DREAMZERO_SAVE_RESET_VIDEO:-0}"
ENABLE_DIT_CACHE="${ENABLE_DIT_CACHE:-0}"
ENABLE_TENSORRT="${ENABLE_TENSORRT:-true}"
DREAMZERO_OUTPUT_ROOT="${DREAMZERO_OUTPUT_ROOT:-$OUTPUT_ROOT}"
RUN_DATE="$(dz_output_timestamp_date)"
RUN_TIME="$(dz_output_timestamp_time)"
DREAMZERO_OUTPUT_DATE="${DREAMZERO_OUTPUT_DATE:-$RUN_DATE}"
DREAMZERO_OUTPUT_TIME="${DREAMZERO_OUTPUT_TIME:-$RUN_TIME}"
DREAMZERO_INFERENCE_OUTPUT_DIR="${DREAMZERO_INFERENCE_OUTPUT_DIR:-$(dz_default_inference_dir "$DREAMZERO_OUTPUT_ROOT" "$(basename "$MODEL_PATH")")}"
LOG_DIR="${LOG_DIR:-$DREAMZERO_INFERENCE_OUTPUT_DIR}"
LOG_FILE="$LOG_DIR/dreamzero_service_port_${PORT}.log"
PID_FILE="$LOG_DIR/dreamzero_service_port_${PORT}.pid"
SESSION_FILE="$LOG_DIR/dreamzero_service_port_${PORT}.screen"
SCREEN_SESSION="${SCREEN_SESSION:-dreamzero_service}"
SCREEN_WINDOW="${SCREEN_WINDOW:-port_${PORT}}"

mkdir -p "$LOG_DIR" "$DREAMZERO_OUTPUT_ROOT" "$DREAMZERO_INFERENCE_OUTPUT_DIR"
mkdir -p "$DREAMZERO_OUTPUT_ROOT/inference"
ln -sfn "$DREAMZERO_INFERENCE_OUTPUT_DIR" "$DREAMZERO_OUTPUT_ROOT/inference/latest"

if [[ -f "$SESSION_FILE" ]]; then
  IFS=':' read -r OLD_SESSION OLD_WINDOW < "$SESSION_FILE" || true
  if [[ -n "${OLD_SESSION:-}" ]] && screen -ls | grep -q "[.]${OLD_SESSION}[[:space:]]"; then
    echo "DreamZero service metadata already exists for port $PORT"
    echo "  Screen session: ${OLD_SESSION:-unknown}"
    echo "  Screen window: ${OLD_WINDOW:-unknown}"
    echo "  To watch output: screen -r ${OLD_SESSION:-dreamzero_service}"
    exit 1
  fi
fi

IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_DEVICES"
NUM_GPUS="${#GPU_ARRAY[@]}"

echo "Launching DreamZero service..."
echo "  Session: $SCREEN_SESSION"
echo "  Window: $SCREEN_WINDOW"
echo "  Port: $PORT"
echo "  GPUs: $CUDA_DEVICES"
echo "  Model: $MODEL_PATH"
echo "  Save reset video: $DREAMZERO_SAVE_RESET_VIDEO"
echo "  Output root: $DREAMZERO_OUTPUT_ROOT"
echo "  Inference dir: $DREAMZERO_INFERENCE_OUTPUT_DIR"
echo "  Enable TensorRT: $ENABLE_TENSORRT"

screen -S "$SCREEN_SESSION" -X select . >/dev/null 2>&1 || screen -dmS "$SCREEN_SESSION"

PYTHON_CMD="python -m torch.distributed.run --standalone --nproc_per_node=$NUM_GPUS socket_test_optimized_AR.py --port $PORT --timeout-seconds $TIMEOUT_SECONDS --model-path $MODEL_PATH"
if [[ "$ENABLE_DIT_CACHE" == "1" ]]; then
  PYTHON_CMD+=" --enable-dit-cache"
fi

SCREEN_CMD=$(cat <<EOF
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd $ROOT_DIR
export PYTHONPATH=$ROOT_DIR
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_ALLOC
export DREAMZERO_SAVE_RESET_VIDEO=$DREAMZERO_SAVE_RESET_VIDEO
export ENABLE_TENSORRT=$ENABLE_TENSORRT
export DREAMZERO_OUTPUT_ROOT=$DREAMZERO_OUTPUT_ROOT
export DREAMZERO_OUTPUT_DATE=$DREAMZERO_OUTPUT_DATE
export DREAMZERO_OUTPUT_TIME=$DREAMZERO_OUTPUT_TIME
export DREAMZERO_INFERENCE_OUTPUT_DIR=$DREAMZERO_INFERENCE_OUTPUT_DIR
$PYTHON_CMD
EOF
)
SCREEN_CMD+=" 2>&1 | tee $LOG_FILE; exec bash"

screen -S "$SCREEN_SESSION" -X screen -t "$SCREEN_WINDOW" bash -lc "$SCREEN_CMD"
sleep 2

SCREEN_PID="$(screen -ls | awk -v name="$SCREEN_SESSION" '$0 ~ ("[.]" name "[[:space:]]") {split($1,a,"."); print a[1]; exit}')"
if [[ -n "$SCREEN_PID" ]]; then
  echo "$SCREEN_PID" > "$PID_FILE"
fi
printf '%s:%s\n' "$SCREEN_SESSION" "$SCREEN_WINDOW" > "$SESSION_FILE"

echo "Started DreamZero service in screen"
echo "  Screen session: $SCREEN_SESSION"
echo "  Screen window: $SCREEN_WINDOW"
echo "  Port: $PORT"
echo "  GPUs: $CUDA_DEVICES"
echo "  Model: $MODEL_PATH"
echo "  Save reset video: $DREAMZERO_SAVE_RESET_VIDEO"
echo "  Output root: $DREAMZERO_OUTPUT_ROOT"
echo "  Inference dir: $DREAMZERO_INFERENCE_OUTPUT_DIR"
echo "  Enable TensorRT: $ENABLE_TENSORRT"
echo "  Log: $LOG_FILE"
echo "  To watch output: screen -r $SCREEN_SESSION"
echo "  In screen, switch to window: $SCREEN_WINDOW"

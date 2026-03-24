#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${PORT:-5999}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/checkpoints/DreamZero-DROID}"
CONDA_ENV="${CONDA_ENV:-dreamzero}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
LOG_FILE="$LOG_DIR/dreamzero_service_port_${PORT}.log"
PID_FILE="$LOG_DIR/dreamzero_service_port_${PORT}.pid"
SESSION_FILE="$LOG_DIR/dreamzero_service_port_${PORT}.screen"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-50000}"
PYTORCH_ALLOC="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DREAMZERO_SAVE_RESET_VIDEO="${DREAMZERO_SAVE_RESET_VIDEO:-0}"
ENABLE_DIT_CACHE="${ENABLE_DIT_CACHE:-0}"
SCREEN_SESSION="${SCREEN_SESSION:-dreamzero_service}"
SCREEN_WINDOW="${SCREEN_WINDOW:-port_${PORT}}"

mkdir -p "$LOG_DIR"

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

screen -S "$SCREEN_SESSION" -X select . >/dev/null 2>&1 || screen -dmS "$SCREEN_SESSION"

SCREEN_CMD=$(cat <<EOF
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd $ROOT_DIR
export PYTHONPATH=$ROOT_DIR
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_ALLOC
export DREAMZERO_SAVE_RESET_VIDEO=$DREAMZERO_SAVE_RESET_VIDEO
python -m torch.distributed.run --standalone --nproc_per_node=$NUM_GPUS socket_test_optimized_AR.py --port $PORT --timeout-seconds $TIMEOUT_SECONDS --model-path $MODEL_PATH
EOF
)

if [[ "$ENABLE_DIT_CACHE" == "1" ]]; then
  SCREEN_CMD+=" --enable-dit-cache"
fi
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
echo "  Log: $LOG_FILE"
echo "  To watch output: screen -r $SCREEN_SESSION"
echo "  In screen, switch to window: $SCREEN_WINDOW"

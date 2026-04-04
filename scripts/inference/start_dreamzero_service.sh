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
NUM_DIT_STEPS="${NUM_DIT_STEPS:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
ENABLE_TENSORRT="${ENABLE_TENSORRT:-true}"
LOAD_TRT_ENGINE="${LOAD_TRT_ENGINE:-}"
TRT_MODEL_TYPE="${TRT_MODEL_TYPE:-ar_1.3B_droid}"
TENSORRT_PRECISION="${TENSORRT_PRECISION:-nvfp4}"
PREWARM_ON_START="${PREWARM_ON_START:-0}"
PREWARM_NUM_CHUNKS="${PREWARM_NUM_CHUNKS:-1}"
PREWARM_USE_ZERO_IMAGES="${PREWARM_USE_ZERO_IMAGES:-1}"
PREWARM_TIMEOUT_SECONDS="${PREWARM_TIMEOUT_SECONDS:-120}"
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

if [[ -z "$LOAD_TRT_ENGINE" ]] && [[ "$ENABLE_TENSORRT" == "true" ]]; then
  DEFAULT_TRT_ENGINE="$MODEL_PATH/tensorrt/wan/WanModel_${TENSORRT_PRECISION}.trt"
  if [[ -f "$DEFAULT_TRT_ENGINE" ]]; then
    LOAD_TRT_ENGINE="$DEFAULT_TRT_ENGINE"
  fi
fi

echo "Launching DreamZero service..."
echo "  Session: $SCREEN_SESSION"
echo "  Window: $SCREEN_WINDOW"
echo "  Port: $PORT"
echo "  GPUs: $CUDA_DEVICES"
echo "  Model: $MODEL_PATH"
echo "  Save reset video: $DREAMZERO_SAVE_RESET_VIDEO"
if [[ -n "$NUM_INFERENCE_STEPS" ]]; then
  echo "  Num inference steps: $NUM_INFERENCE_STEPS"
fi
if [[ -n "$NUM_DIT_STEPS" ]]; then
  echo "  Num DiT steps: $NUM_DIT_STEPS"
fi
echo "  Output root: $DREAMZERO_OUTPUT_ROOT"
echo "  Inference dir: $DREAMZERO_INFERENCE_OUTPUT_DIR"
echo "  Enable TensorRT: $ENABLE_TENSORRT"
if [[ -n "$LOAD_TRT_ENGINE" ]]; then
  echo "  TRT engine: $LOAD_TRT_ENGINE"
  echo "  TRT model type: $TRT_MODEL_TYPE"
else
  echo "  TRT engine: <not set>"
fi

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
export NUM_DIT_STEPS=$NUM_DIT_STEPS
export NUM_INFERENCE_STEPS=$NUM_INFERENCE_STEPS
export ENABLE_TENSORRT=$ENABLE_TENSORRT
export DREAMZERO_OUTPUT_ROOT=$DREAMZERO_OUTPUT_ROOT
export DREAMZERO_OUTPUT_DATE=$DREAMZERO_OUTPUT_DATE
export DREAMZERO_OUTPUT_TIME=$DREAMZERO_OUTPUT_TIME
export DREAMZERO_INFERENCE_OUTPUT_DIR=$DREAMZERO_INFERENCE_OUTPUT_DIR
export LOAD_TRT_ENGINE=$LOAD_TRT_ENGINE
export TRT_MODEL_TYPE=$TRT_MODEL_TYPE
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
if [[ -n "$NUM_INFERENCE_STEPS" ]]; then
  echo "  Num inference steps: $NUM_INFERENCE_STEPS"
fi
if [[ -n "$NUM_DIT_STEPS" ]]; then
  echo "  Num DiT steps: $NUM_DIT_STEPS"
fi
echo "  Output root: $DREAMZERO_OUTPUT_ROOT"
echo "  Inference dir: $DREAMZERO_INFERENCE_OUTPUT_DIR"
echo "  Enable TensorRT: $ENABLE_TENSORRT"
if [[ -n "$LOAD_TRT_ENGINE" ]]; then
  echo "  TRT engine: $LOAD_TRT_ENGINE"
  echo "  TRT model type: $TRT_MODEL_TYPE"
else
  echo "  TRT engine: <not set>"
fi
echo "  Log: $LOG_FILE"
echo "  To watch output: screen -r $SCREEN_SESSION"
echo "  In screen, switch to window: $SCREEN_WINDOW"

if [[ "$PREWARM_ON_START" == "1" ]]; then
  echo "Running prewarm request..."
  PREWARM_ARGS=(--host 127.0.0.1 --port "$PORT" --num-chunks "$PREWARM_NUM_CHUNKS" --no-report)
  if [[ "$PREWARM_USE_ZERO_IMAGES" == "1" ]]; then
    PREWARM_ARGS+=(--use-zero-images)
  fi

  source /home/zqy/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
  cd "$ROOT_DIR"
  export PYTHONPATH="$ROOT_DIR"

  PREWARM_STARTED=0
  for ((attempt=1; attempt<=PREWARM_TIMEOUT_SECONDS; attempt++)); do
    if python - <<PY >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", ${PORT}))
finally:
    s.close()
PY
    then
      PREWARM_STARTED=1
      break
    fi
    sleep 1
  done

  if [[ "$PREWARM_STARTED" == "1" ]]; then
    python test_client_AR.py "${PREWARM_ARGS[@]}"
    echo "Prewarm finished."
  else
    echo "Prewarm skipped: service did not open port $PORT within ${PREWARM_TIMEOUT_SECONDS}s"
  fi
fi

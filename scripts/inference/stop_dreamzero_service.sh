#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
PORT="${PORT:-5999}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
PID_FILE="$LOG_DIR/dreamzero_service_port_${PORT}.pid"
SESSION_FILE="$LOG_DIR/dreamzero_service_port_${PORT}.screen"

if [[ ! -f "$SESSION_FILE" ]]; then
  echo "No DreamZero screen metadata found for port $PORT"
  rm -f "$PID_FILE"
  exit 0
fi

IFS=':' read -r SCREEN_SESSION SCREEN_WINDOW < "$SESSION_FILE"

if screen -ls | grep -q "[.]${SCREEN_SESSION}[[:space:]]"; then
  screen -S "$SCREEN_SESSION" -p "$SCREEN_WINDOW" -X stuff $'\003'
  sleep 1
  screen -S "$SCREEN_SESSION" -p "$SCREEN_WINDOW" -X kill >/dev/null 2>&1 || true
  if ! screen -S "$SCREEN_SESSION" -Q windows 2>/dev/null | grep -q .; then
    screen -S "$SCREEN_SESSION" -X quit >/dev/null 2>&1 || true
  fi
  echo "Stopped DreamZero screen window $SCREEN_WINDOW in session $SCREEN_SESSION"
else
  echo "Screen session $SCREEN_SESSION is not running"
fi

rm -f "$SESSION_FILE" "$PID_FILE"

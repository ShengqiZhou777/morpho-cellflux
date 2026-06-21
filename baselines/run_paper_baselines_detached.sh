#!/usr/bin/env bash
set -euo pipefail

# Detached launcher for the full paper baseline queue.
# If the first-pass diet queue is still running, this waits for it and then
# resumes missing outputs instead of launching competing GPU jobs.

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/miniconda3/bin/conda}"
LOG_DIR="$PROJECT/outputs/baselines/logs"
LOG="$LOG_DIR/paper_baselines_$(date -u +%Y%m%dT%H%M%SZ).log"
PID_FILE="$LOG_DIR/paper_baselines.pid"
PRIMARY_PID_FILE="$LOG_DIR/primary_diet.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Paper baseline queue already appears to be running: pid=$old_pid" >&2
    echo "log: $(readlink -f "$LOG_DIR/paper_baselines.latest.log" 2>/dev/null || true)" >&2
    exit 2
  fi
fi

WAIT_FOR_PID="${WAIT_FOR_PID:-}"
if [[ -z "$WAIT_FOR_PID" && -f "$PRIMARY_PID_FILE" ]]; then
  primary_pid="$(cat "$PRIMARY_PID_FILE")"
  if [[ -n "$primary_pid" ]] && kill -0 "$primary_pid" 2>/dev/null; then
    WAIT_FOR_PID="$primary_pid"
  fi
fi

ln -sfn "$(basename "$LOG")" "$LOG_DIR/paper_baselines.latest.log"

setsid nohup bash -c '
set -eo pipefail
PROJECT="$1"
LOG="$2"
WAIT_FOR_PID="$3"
CONDA_BIN="$4"
cd "$PROJECT"
{
  echo "[$(date -Is)] START detached paper baseline queue"
  echo "project=$PROJECT"
  echo "wait_for_pid=${WAIT_FOR_PID:-none}"
  if [[ -n "$WAIT_FOR_PID" ]]; then
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
      echo "[$(date -Is)] waiting for existing baseline pid=$WAIT_FOR_PID"
      sleep 60
    done
  fi
  echo "[$(date -Is)] running in conda env pmf"
  set +e
  "$CONDA_BIN" run --no-capture-output -n pmf bash baselines/run_paper_baselines.sh
  code=$?
  set -e
  echo "[$(date -Is)] DONE detached paper baseline queue exit=$code"
  exit "$code"
} >> "$LOG" 2>&1
' _ "$PROJECT" "$LOG" "${WAIT_FOR_PID:-}" "$CONDA_BIN" >/dev/null 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"
echo "Started paper baseline queue"
echo "pid: $pid"
echo "log: $LOG"
if [[ -n "$WAIT_FOR_PID" ]]; then
  echo "waiting for existing pid before running: $WAIT_FOR_PID"
fi

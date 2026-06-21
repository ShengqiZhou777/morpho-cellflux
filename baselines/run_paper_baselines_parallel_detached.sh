#!/usr/bin/env bash
set -euo pipefail

# Detached launcher for the PARALLEL paper baseline queue
# (run_paper_baselines_parallel.sh: one benchmark per GPU, run concurrently).

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/miniconda3/bin/conda}"
LOG_DIR="$PROJECT/outputs/baselines/logs"
LOG="$LOG_DIR/paper_baselines_parallel_$(date -u +%Y%m%dT%H%M%SZ).log"
PID_FILE="$LOG_DIR/paper_baselines_parallel.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Parallel baseline queue already running: pid=$old_pid" >&2
    exit 2
  fi
fi

ln -sfn "$(basename "$LOG")" "$LOG_DIR/paper_baselines_parallel.latest.log"

setsid nohup bash -c '
set -uo pipefail
PROJECT="$1"; LOG="$2"; CONDA_BIN="$3"
cd "$PROJECT"
{
  echo "[$(date -Is)] START detached parallel paper baseline queue"
  echo "project=$PROJECT"
  "$CONDA_BIN" run --no-capture-output -n pmf bash baselines/run_paper_baselines_parallel.sh
  code=$?
  echo "[$(date -Is)] DONE detached parallel paper baseline queue exit=$code"
  exit "$code"
} >> "$LOG" 2>&1
' _ "$PROJECT" "$LOG" "$CONDA_BIN" >/dev/null 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"
echo "Started PARALLEL paper baseline queue"
echo "pid: $pid"
echo "log: $LOG"

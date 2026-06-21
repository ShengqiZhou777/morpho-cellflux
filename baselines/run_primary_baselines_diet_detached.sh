#!/usr/bin/env bash
set -euo pipefail

# Detached launcher for the first paper baseline queue:
#   export shared baseline data -> PhenDiff diet_v3 -> IMPA diet_v3 -> collect tables.
#
# Usage:
#   bash baselines/run_primary_baselines_diet_detached.sh
#
# Optional overrides are forwarded to run_primary_baselines_diet.sh:
#   PHENDIFF_EPOCHS=2 IMPA_EPOCHS=2 bash baselines/run_primary_baselines_diet_detached.sh

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-morpho-cellflux}"
if [[ -z "${CONDA_SH:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Set CONDA_SH=/path/to/conda.sh or activate the environment manually." >&2
    exit 127
  fi
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
LOG_DIR="$PROJECT/outputs/baselines/logs"
LOG="$LOG_DIR/primary_diet_$(date -u +%Y%m%dT%H%M%SZ).log"
PID_FILE="$LOG_DIR/primary_diet.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Baseline diet queue already appears to be running: pid=$old_pid" >&2
    echo "log: $(readlink -f "$LOG_DIR/primary_diet.latest.log" 2>/dev/null || true)" >&2
    exit 2
  fi
fi

ln -sfn "$(basename "$LOG")" "$LOG_DIR/primary_diet.latest.log"

(
  cd "$PROJECT"
  {
    echo "[$(date -Is)] START primary diet baselines"
    echo "project=$PROJECT"
    echo "phendiff_epochs=${PHENDIFF_EPOCHS:-8} impa_epochs=${IMPA_EPOCHS:-8}"
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    bash baselines/run_primary_baselines_diet.sh
    code=$?
    echo "[$(date -Is)] DONE primary diet baselines exit=$code"
    exit "$code"
  } >> "$LOG" 2>&1
) &

pid=$!
echo "$pid" > "$PID_FILE"
echo "Started primary diet baseline queue"
echo "pid: $pid"
echo "log: $LOG"

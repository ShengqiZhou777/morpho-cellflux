#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
B="${BENCHMARK:-diet}"
G="${CUDA_VISIBLE_DEVICES:-0}"
LOGDIR="outputs/baselines/logs"
mkdir -p "$LOGDIR"
echo "=== Serial Baselines $(date '+%m/%d %H:%M') === GPU=$G Bench=$B"

run_impa()       { CUDA_VISIBLE_DEVICES=$G BENCHMARK=$B BATCH=40 EPOCHS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash baselines/run_impa.sh; }
run_stargan()    { CUDA_VISIBLE_DEVICES=$G BENCHMARK=$B BATCH=128 NUM_ITERS=30000 bash baselines/run_stargan.sh; }
run_morphodiff() { CUDA_VISIBLE_DEVICES=$G BENCHMARK=$B BATCH=128 GRAD_ACCUM=1 EPOCHS=5 bash baselines/run_morphodiff.sh; }

for m in impa stargan morphodiff; do
  [ -f "outputs/baselines/$m/$B/aggregate_eval_summary.json" ] && { echo "[skip] $m"; continue; }
  echo "[$(date '+%H:%M')] START $m"
  "run_$m" 2>&1 | tee "$LOGDIR/${m}_${B}.log"
  echo "[$(date '+%H:%M')] DONE $m"
done
echo "=== ALL DONE $(date '+%m/%d %H:%M') ==="

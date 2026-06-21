#!/usr/bin/env bash
# Parallel paper-baseline queue: one benchmark per GPU, run concurrently.
#
# Rationale: each training job only uses ~7-14 GiB of a 32 GiB card, so the
# sequential queue (run_paper_baselines.sh) left a whole GPU idle. This variant
# pins each benchmark to one GPU and runs the two benchmarks side by side.
#
# Training recipes are preserved, NOT changed:
#   - PhenDiff runs single-process (NPROC=1) at batch 32, which is the SAME
#     global batch as the 2-GPU NPROC=2 batch-16 setup.
#   - IMPA's global batch is unchanged (DataParallel over 1 visible GPU == the
#     same nominal batch, just on one card).
#
# Resumable: any method/benchmark with aggregate_eval_summary.json is skipped.
set -uo pipefail   # deliberately no -e: one lane must not silently kill the other

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

INCLUDE_STARGAN="${INCLUDE_STARGAN:-0}"   # StarGAN stays a separate supplement
PHENDIFF_EPOCHS="${PHENDIFF_EPOCHS:-8}"
IMPA_EPOCHS="${IMPA_EPOCHS:-8}"
PHENDIFF_BATCH="${PHENDIFF_BATCH:-32}"    # single-GPU == 2-GPU(batch16) global batch
IMPA_BATCH="${IMPA_BATCH:-16}"
# Lane assignment: "<gpu>:<benchmark>" pairs.
LANE0="${LANE0:-0:diet_v3}"
LANE1="${LANE1:-1:crispr_v8}"

# Concurrency hardening (two PhenDiff jobs at once previously raced on startup
# network calls and a shared accelerate rendezvous port):
#   - run fully offline; all weights (incl. torch-fidelity InceptionV3) are cached
#   - stagger lane1 so lane0 warms shared caches / passes startup first
#   - each lane gets its own accelerate main_process_port (set per-lane below)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export GIT_TERMINAL_PROMPT=0
LANE_STAGGER_SECONDS="${LANE_STAGGER_SECONDS:-120}"

summary_exists() { [[ -f "$PROJECT/$1/aggregate_eval_summary.json" ]]; }

benchmark_config() {
  case "$1" in
    diet_v3) echo "configs/diet_id_v3.yaml" ;;
    crispr_v8) echo "configs/perturbmulti_train_id.yaml" ;;
    *) echo "Unknown benchmark: $1" >&2; return 2 ;;
  esac
}

ensure_export() {
  local benchmark="$1" config out
  config="$(benchmark_config "$benchmark")"
  out="outputs/baselines/_data/$benchmark"
  if [[ -f "$out/manifest.json" ]]; then
    echo "[$(date -Is)] SKIP export $benchmark: manifest exists"
    return 0
  fi
  echo "[$(date -Is)] START export $benchmark"
  python baselines/export_baseline_data.py \
    --config "$config" --benchmark "$benchmark" --output "$out" \
    --splits train,test --workers "${EXPORT_WORKERS:-8}"
  echo "[$(date -Is)] DONE export $benchmark"
}

run_copy_control_for() {
  local benchmark="$1" config out
  config="$(benchmark_config "$benchmark")"
  out="outputs/baselines/copy_control/$benchmark"
  if summary_exists "$out"; then
    echo "[$(date -Is)] SKIP copy_control $benchmark: summary exists"
    return 0
  fi
  echo "[$(date -Is)] START copy_control $benchmark"
  python baselines/copy_control.py --config "$config" --output "$out" --split test \
    && python scripts/aggregate_eval.py "$out" 5 0
  echo "[$(date -Is)] DONE copy_control $benchmark"
}

# A lane owns one GPU and runs the full per-benchmark chain on it, sequentially.
lane() {
  local spec="$1" gpu benchmark
  gpu="${spec%%:*}"
  benchmark="${spec#*:}"
  export CUDA_VISIBLE_DEVICES="$gpu"   # subshell-local: pins every step in this lane
  export MAIN_PROCESS_PORT=$((29500 + gpu))   # distinct accelerate rendezvous per lane

  echo "[$(date -Is)] [gpu=$gpu] LANE START $benchmark"

  run_copy_control_for "$benchmark"

  if summary_exists "outputs/baselines/phendiff/$benchmark"; then
    echo "[$(date -Is)] [gpu=$gpu] SKIP PhenDiff $benchmark: summary exists"
  else
    echo "[$(date -Is)] [gpu=$gpu] START PhenDiff $benchmark (NPROC=1 BATCH=$PHENDIFF_BATCH)"
    NPROC=1 BENCHMARK="$benchmark" EPOCHS="$PHENDIFF_EPOCHS" BATCH="$PHENDIFF_BATCH" \
      bash baselines/run_phendiff.sh \
      || echo "[$(date -Is)] [gpu=$gpu] FAIL PhenDiff $benchmark"
  fi

  if summary_exists "outputs/baselines/impa/$benchmark"; then
    echo "[$(date -Is)] [gpu=$gpu] SKIP IMPA $benchmark: summary exists"
  else
    echo "[$(date -Is)] [gpu=$gpu] START IMPA $benchmark (BATCH=$IMPA_BATCH)"
    BENCHMARK="$benchmark" EPOCHS="$IMPA_EPOCHS" BATCH="$IMPA_BATCH" DEVICES=1 \
      bash baselines/run_impa.sh \
      || echo "[$(date -Is)] [gpu=$gpu] FAIL IMPA $benchmark"
  fi

  echo "[$(date -Is)] [gpu=$gpu] LANE DONE $benchmark"
}

echo "[$(date -Is)] START parallel paper baseline queue"
echo "project=$PROJECT"
echo "lane0=$LANE0 lane1=$LANE1 include_stargan=$INCLUDE_STARGAN"
echo "phendiff_epochs=$PHENDIFF_EPOCHS phendiff_batch=$PHENDIFF_BATCH impa_epochs=$IMPA_EPOCHS impa_batch=$IMPA_BATCH"

# Exports are CPU/IO; do them up front so both GPU lanes start clean.
for spec in "$LANE0" "$LANE1"; do
  ensure_export "${spec#*:}"
done

lane "$LANE0" &
p0=$!
sleep "$LANE_STAGGER_SECONDS"   # let lane0 pass startup before lane1 begins
lane "$LANE1" &
p1=$!
wait "$p0"; r0=$?
wait "$p1"; r1=$?
echo "[$(date -Is)] lanes finished: lane0=$r0 lane1=$r1"

python baselines/collect_paper_metrics.py
echo "[$(date -Is)] DONE parallel paper baseline queue"

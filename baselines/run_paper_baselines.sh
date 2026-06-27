#!/usr/bin/env bash
set -euo pipefail

# Sequential, resumable baseline queue for paper method-comparison tables.
# This does not train Morpho-CellFlux / ours. It only prepares shared exports,
# runs external or null baselines, and then collects existing method summaries.

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

BENCHMARKS="${BENCHMARKS:-diet crispr_paper}"
INCLUDE_STARGAN="${INCLUDE_STARGAN:-1}"
INCLUDE_COPY_CONTROL="${INCLUDE_COPY_CONTROL:-0}"

summary_exists() {
  [[ -f "$PROJECT/$1/aggregate_eval_summary.json" ]]
}

benchmark_config() {
  case "$1" in
    diet) echo "configs/diet_id.yaml" ;;
    crispr_paper) echo "configs/crispr_paper_core.yaml" ;;
    *)
      echo "Unknown benchmark: $1" >&2
      return 2
      ;;
  esac
}

ensure_export() {
  local benchmark="$1"
  local config
  config="$(benchmark_config "$benchmark")"
  local out="outputs/baselines/_data/$benchmark"

  if [[ -f "$out/manifest.json" ]]; then
    echo "[$(date -Is)] SKIP export $benchmark: $out/manifest.json exists"
    return
  fi

  echo "[$(date -Is)] START export $benchmark"
  python baselines/export_baseline_data.py \
    --config "$config" \
    --benchmark "$benchmark" \
    --output "$out" \
    --splits train,test \
    --workers "${EXPORT_WORKERS:-8}"
  echo "[$(date -Is)] DONE export $benchmark"
}

run_phendiff_for() {
  local benchmark="$1"
  local out="outputs/baselines/phendiff/$benchmark"

  if summary_exists "$out"; then
    echo "[$(date -Is)] SKIP PhenDiff $benchmark: summary exists"
    return
  fi

  echo "[$(date -Is)] START PhenDiff $benchmark"
  BENCHMARK="$benchmark" \
    EPOCHS="${PHENDIFF_EPOCHS:-8}" \
    BATCH="${PHENDIFF_BATCH:-16}" \
    bash baselines/run_phendiff.sh
  echo "[$(date -Is)] DONE PhenDiff $benchmark"
}

run_impa_for() {
  local benchmark="$1"
  local out="outputs/baselines/impa/$benchmark"

  if summary_exists "$out"; then
    echo "[$(date -Is)] SKIP IMPA $benchmark: summary exists"
    return
  fi

  echo "[$(date -Is)] START IMPA $benchmark"
  BENCHMARK="$benchmark" \
    EPOCHS="${IMPA_EPOCHS:-8}" \
    BATCH="${IMPA_BATCH:-16}" \
    DEVICES="${IMPA_DEVICES:-1}" \
    bash baselines/run_impa.sh
  echo "[$(date -Is)] DONE IMPA $benchmark"
}

run_stargan_for() {
  local benchmark="$1"
  local out="outputs/baselines/stargan/$benchmark"

  if [[ "$INCLUDE_STARGAN" != "1" ]]; then
    echo "[$(date -Is)] SKIP StarGAN $benchmark: INCLUDE_STARGAN=$INCLUDE_STARGAN"
    return
  fi

  if summary_exists "$out"; then
    echo "[$(date -Is)] SKIP StarGAN $benchmark: summary exists"
    return
  fi

  echo "[$(date -Is)] START StarGAN $benchmark"
  BENCHMARK="$benchmark" \
    NUM_ITERS="${STARGAN_NUM_ITERS:-50000}" \
    BATCH="${STARGAN_BATCH:-16}" \
    bash baselines/run_stargan.sh
  echo "[$(date -Is)] DONE StarGAN $benchmark"
}

echo "[$(date -Is)] START paper baseline queue"
echo "project=$PROJECT"
echo "benchmarks=$BENCHMARKS"

for benchmark in $BENCHMARKS; do
  ensure_export "$benchmark"
  run_phendiff_for "$benchmark"
  run_impa_for "$benchmark"
  run_stargan_for "$benchmark"
done

python baselines/collect_paper_metrics.py
echo "[$(date -Is)] DONE paper baseline queue"

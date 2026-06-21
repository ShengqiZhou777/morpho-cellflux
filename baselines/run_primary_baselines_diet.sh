#!/usr/bin/env bash
set -euo pipefail

# First-pass paper baselines on the diet benchmark. Run CRISPR only after these
# two adapters have produced valid aggregate_eval summaries.

python baselines/export_baseline_data.py \
  --config configs/diet_id_v3.yaml \
  --benchmark diet_v3 \
  --output outputs/baselines/_data/diet_v3 \
  --splits train,test \
  --workers "${EXPORT_WORKERS:-8}"

BENCHMARK=diet_v3 EPOCHS="${PHENDIFF_EPOCHS:-8}" BATCH="${PHENDIFF_BATCH:-16}" \
  bash baselines/run_phendiff.sh

BENCHMARK=diet_v3 EPOCHS="${IMPA_EPOCHS:-8}" BATCH="${IMPA_BATCH:-16}" DEVICES="${IMPA_DEVICES:-1}" \
  bash baselines/run_impa.sh

python baselines/collect_paper_metrics.py

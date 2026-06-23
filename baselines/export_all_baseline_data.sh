#!/usr/bin/env bash
set -euo pipefail

# Export shared data for external baseline adapters. This produces PNG
# imagefolders and IMPA-compatible `.npy` files from the same Morpho-CellFlux
# configs used by the proposed method.

BASELINE_DATA_ROOT="${BASELINE_DATA_ROOT:-outputs/baselines/_data}"

python baselines/export_baseline_data.py \
  --config configs/diet_id.yaml \
  --benchmark diet \
  --output "$BASELINE_DATA_ROOT/diet" \
  --splits train,test \
  --workers "${EXPORT_WORKERS:-8}"

python baselines/export_baseline_data.py \
  --config configs/crispr_paper_core.yaml \
  --benchmark crispr_paper \
  --output "$BASELINE_DATA_ROOT/crispr_paper" \
  --splits train,test \
  --workers "${EXPORT_WORKERS:-8}"

#!/usr/bin/env bash
set -euo pipefail

# Export shared data for external baseline adapters. This produces PNG
# imagefolders and IMPA-compatible `.npy` files from the same Morpho-CellFlux
# configs used by the proposed method.

python baselines/export_baseline_data.py \
  --config configs/diet_id_v3.yaml \
  --benchmark diet_v3 \
  --output outputs/baselines/_data/diet_v3 \
  --splits train,test \
  --workers "${EXPORT_WORKERS:-8}"

python baselines/export_baseline_data.py \
  --config configs/perturbmulti_train_id.yaml \
  --benchmark crispr_v8 \
  --output outputs/baselines/_data/crispr_v8 \
  --splits train,test \
  --workers "${EXPORT_WORKERS:-8}"

#!/usr/bin/env bash
set -euo pipefail

# CPU/file-I/O baseline. This intentionally does not allocate GPUs.

python baselines/copy_control.py \
  --config configs/diet_id_v3.yaml \
  --output outputs/baselines/copy_control/diet_v3 \
  --split test

python scripts/aggregate_eval.py outputs/baselines/copy_control/diet_v3 5 0

python baselines/copy_control.py \
  --config configs/perturbmulti_train_id.yaml \
  --output outputs/baselines/copy_control/crispr_v8 \
  --split test

python scripts/aggregate_eval.py outputs/baselines/copy_control/crispr_v8 5 0

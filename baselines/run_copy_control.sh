#!/usr/bin/env bash
set -euo pipefail

# CPU/file-I/O baseline. This intentionally does not allocate GPUs.

python baselines/copy_control.py \
  --config configs/diet_id.yaml \
  --output outputs/baselines/copy_control/diet \
  --split test

python scripts/aggregate_eval.py outputs/baselines/copy_control/diet 5 0

python baselines/copy_control.py \
  --config configs/perturbmulti_train_id.yaml \
  --output outputs/baselines/copy_control/crispr \
  --split test

python scripts/aggregate_eval.py outputs/baselines/copy_control/crispr 5 0

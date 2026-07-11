#!/usr/bin/env bash
# Quick balanced-subset training for microalgae timepoint data.
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TORCHRUN=${TORCHRUN:-/data/conda_envs/pmf/bin/torchrun}

if [[ ! -x "$TORCHRUN" ]]; then
  echo "torchrun not found or not executable: $TORCHRUN" >&2
  exit 127
fi

cd "$PROJECT_DIR"

/data/conda_envs/pmf/bin/python scripts/build_timepoint_subset.py \
  --train-per-label "${TRAIN_PER_LABEL:-256}" \
  --test-per-label "${TEST_PER_LABEL:-64}" \
  --seed "${SEED:-42}"

OUT=${OUT:-$PROJECT_DIR/outputs/runs/microalgae/timepoint_quick_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TORCHRUN" --standalone --nproc_per_node="${NPROC:-1}" -m phenoflux.train \
  --dataset phenoflux \
  --config microalgae_timepoint_quick \
  --device cuda \
  --batch_size "${BATCH:-8}" \
  --accum_iter "${ACCUM:-1}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --epochs "${EPOCHS:-5}" \
  --use_initial 1 \
  --use_ema \
  --skewed_timesteps \
  --class_drop_prob 0.2 \
  --cfg_scale "${CFG:-0.2}" \
  --eval_frequency "${EVAL_FREQ:--1}" \
  --fid_samples "${FID_SAMPLES:-256}" \
  --early_stop_patience "${EARLY_STOP:-0}" \
  --wandb_project "${WANDB_PROJECT:-phenoflux}" \
  --wandb_run_name "${WANDB_RUN_NAME:-microalgae_timepoint_quick_$(basename "$OUT")}" \
  --output_dir "$OUT" 2>&1 | tee -a "$OUT/train_stdout.log"

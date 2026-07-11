#!/usr/bin/env bash
# Single-GPU ablation launcher to diagnose identity-collapse fixes.
# Runs ONE arm of the collapse ablation on the 62-d omics condition (same as the
# collapsed run) using a stratified subset index for fast turnaround.
#
# Arms (set via env):
#   A: USE_INITIAL=0 GAN_WEIGHT=0      (noise start, no GAN)
#   B: USE_INITIAL=1 GAN_WEIGHT=0.1    (control start + GAN breaks identity)
#   C: USE_INITIAL=0 GAN_WEIGHT=0.1    (noise start + GAN)
#   baseline reference = the already-collapsed run (USE_INITIAL=1 GAN_WEIGHT=0)
#
# Example:
#   OUT=outputs/runs/microalgae/ablate/A USE_INITIAL=0 GAN_WEIGHT=0 bash scripts/ablate_collapse.sh
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TORCHRUN=${TORCHRUN:-$(command -v torchrun || true)}
if [[ -z "$TORCHRUN" ]]; then
  echo "torchrun not found. Activate the pmf env or set TORCHRUN=/path/to/torchrun." >&2
  exit 127
fi

OUT=${OUT:-$PROJECT_DIR/outputs/runs/microalgae/ablate/arm}
CONFIG=${CONFIG:-microalgae_timepoint_512_62d}
DATASET=${DATASET:-phenoflux}
DATA_INDEX=${DATA_INDEX:-data/processed/microalgae_v1/views/timepoint_512/index_ablation_subset.csv}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-4}
EVAL_FREQ=${EVAL_FREQ:-4}      # eval at the last epoch
FID_SAMPLES=${FID_SAMPLES:-512}
USE_INITIAL=${USE_INITIAL:-0}
GAN_WEIGHT=${GAN_WEIGHT:-0.0}
CFG=${CFG:-0.2}
NPROC=${NPROC:-1}             # single 4090
ODE_OPTIONS=${ODE_OPTIONS:-'{"step_size": 0.02}'}

mkdir -p "$OUT"
cd "$PROJECT_DIR"
echo "[ablate] OUT=$OUT USE_INITIAL=$USE_INITIAL GAN_WEIGHT=$GAN_WEIGHT EPOCHS=$EPOCHS" >&2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
  --dataset "$DATASET" --config "$CONFIG" --device cuda \
  --data_index "$DATA_INDEX" \
  --batch_size "$BATCH" --accum_iter 1 --num_workers 10 --epochs "$EPOCHS" \
  --use_initial "$USE_INITIAL" --gan_weight "$GAN_WEIGHT" --use_ema --skewed_timesteps \
  --class_drop_prob 0.2 --cfg_scale "$CFG" \
  --eval_frequency "$EVAL_FREQ" --compute_fid --fid_samples "$FID_SAMPLES" \
  --ode_options "$ODE_OPTIONS" --save_fid_samples \
  --early_stop_patience 0 \
  --output_dir "$OUT" 2>&1 | tee -a "$OUT/train_stdout.log"

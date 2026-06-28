#!/usr/bin/env bash
# Launch PhenoFlux flow-matching training on N GPUs.
# Per-step stdout is persisted to $OUT/train_stdout.log.
#
# Override defaults via env vars, e.g.:
#   OUT=outputs/my_run BATCH=16 ACCUM=2 EPOCHS=60 bash scripts/train.sh
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TORCHRUN=${TORCHRUN:-$(command -v torchrun || true)}
if [[ -z "$TORCHRUN" ]]; then
  echo "torchrun not found. Activate the project environment or set TORCHRUN=/path/to/torchrun." >&2
  exit 127
fi

OUT=${OUT:-$PROJECT_DIR/outputs/runs/crispr/phenoflux_crispr_v1}
BATCH=${BATCH:-16}          # per-GPU batch size. 16 uses about 25GB and is safe through FID eval on a 32GB card.
ACCUM=${ACCUM:-1}           # gradient accumulation. Effective batch = BATCH * ACCUM * NPROC, no extra memory.
EPOCHS=${EPOCHS:-40}
EVAL_FREQ=${EVAL_FREQ:-10}
FID_SAMPLES=${FID_SAMPLES:-1024}
NPROC=${NPROC:-2}
USE_INITIAL=${USE_INITIAL:-1}   # 0 = noise to target, 1 = control init, 2 = control plus noise.
NOISE_LEVEL=${NOISE_LEVEL:-0.2} # noise added to the control image when USE_INITIAL=2.
CFG=${CFG:-0.2}                 # classifier-free guidance scale at sampling.
CONFIG=${CONFIG:-phenoflux_crispr}            # config name under configs/, selects the data index and embedding.
DATASET=${DATASET:-phenoflux}                 # dataset name passed to --dataset.
WANDB_PROJECT="${WANDB_PROJECT:-phenoflux}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-crispr_$(basename "$OUT")}"
ODE_OPTIONS=${ODE_OPTIONS:-'{"step_size": 0.02}'}
FOREGROUND_LOSS=${FOREGROUND_LOSS:-0}
FOREGROUND_THRESHOLD=${FOREGROUND_THRESHOLD:-0.05}
FOREGROUND_WEIGHT=${FOREGROUND_WEIGHT:-5.0}
BACKGROUND_WEIGHT=${BACKGROUND_WEIGHT:-0.1}
EARLY_STOP=${EARLY_STOP:-5}   # stop if loss doesn't improve for N epochs. 0 = disabled.
TEST_RUN=${TEST_RUN:-0}

EXTRA_ARGS=()
if [[ "$FOREGROUND_LOSS" == "1" ]]; then
  EXTRA_ARGS+=(
    --foreground_loss
    --foreground_threshold "$FOREGROUND_THRESHOLD"
    --foreground_weight "$FOREGROUND_WEIGHT"
    --background_weight "$BACKGROUND_WEIGHT"
  )
fi
if [[ "$TEST_RUN" == "1" ]]; then
  EXTRA_ARGS+=(--test_run)
fi

mkdir -p "$OUT"
cd "$PROJECT_DIR"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
  --dataset "$DATASET" --config "$CONFIG" --device cuda \
  --batch_size "$BATCH" --accum_iter "$ACCUM" --num_workers 10 --epochs "$EPOCHS" \
  --use_initial "$USE_INITIAL" --noise_level "$NOISE_LEVEL" --use_ema --skewed_timesteps \
  --class_drop_prob 0.2 --cfg_scale "$CFG" \
  --eval_frequency "$EVAL_FREQ" --compute_fid --fid_samples "$FID_SAMPLES" \
  --ode_options "$ODE_OPTIONS" --save_fid_samples \
  --early_stop_patience "$EARLY_STOP" \
  --wandb_project "$WANDB_PROJECT" --wandb_run_name "$WANDB_RUN_NAME" \
  --output_dir "$OUT" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$OUT/train_stdout.log"

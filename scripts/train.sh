#!/usr/bin/env bash
# Launch Morpho-CellFlux flow-matching training on N GPUs.
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

OUT=${OUT:-$PROJECT_DIR/outputs/runs/microalgae/timepoint_512_genes_v1}
BATCH=${BATCH:-16}          # per-GPU batch size. 16 uses about 25GB and is safe through FID eval on a 32GB card.
ACCUM=${ACCUM:-1}           # gradient accumulation. Effective batch = BATCH * ACCUM * NPROC, no extra memory.
EPOCHS=${EPOCHS:-40}
EVAL_FREQ=${EVAL_FREQ:-5}
FID_SAMPLES=${FID_SAMPLES:-2048}
NPROC=${NPROC:-2}
USE_INITIAL=${USE_INITIAL:-0}   # 0 = noise to target (anti-collapse), 1 = control init (collapses), 2 = control plus noise.
NOISE_LEVEL=${NOISE_LEVEL:-0.2}              # stddev of noise added to control when USE_INITIAL=2.
NOISE_PROB=${NOISE_PROB:-0.5}                # probability of adding noise when USE_INITIAL=2.
CENTER_NOISE_SIGMA=${CENTER_NOISE_SIGMA:-0.4}  # center-weighted noise envelope for USE_INITIAL=0 (>0 = one centered cell).
GAN_WEIGHT=${GAN_WEIGHT:-0.1}                # PatchGAN adversarial loss weight (only working distribution-matching signal).
MMD_WEIGHT=${MMD_WEIGHT:-0.5}                # MMD distribution-matching loss weight (scDFM 2026). NOTE: verify .detach() is removed in train_loop.
CFG=${CFG:-0.2}                 # classifier-free guidance scale at sampling.
CONFIG=${CONFIG:-microalgae_timepoint_512_genes}   # config name under configs/, selects the data index and embedding.
DATASET=${DATASET:-phenoflux}                 # dataset name passed to --dataset.
WANDB_PROJECT="${WANDB_PROJECT:-phenoflux}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-microalgae_$(basename "$OUT")}"
ODE_OPTIONS=${ODE_OPTIONS:-'{"step_size": 0.02}'}
EARLY_STOP=${EARLY_STOP:-5}   # stop if loss doesn't improve for N epochs. 0 = disabled.
RESUME=${RESUME:-}            # path to a checkpoint-N.pth to resume from (auto-sets start_epoch = N+1). empty = fresh.
TEST_RUN=${TEST_RUN:-0}

EXTRA_ARGS=()
if [[ "$TEST_RUN" == "1" ]]; then
  EXTRA_ARGS+=(--test_run)
fi

mkdir -p "$OUT"
cd "$PROJECT_DIR"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
  --dataset "$DATASET" --config "$CONFIG" --device cuda \
  --batch_size "$BATCH" --accum_iter "$ACCUM" --num_workers 10 --epochs "$EPOCHS" \
  --use_initial "$USE_INITIAL" --noise_level "$NOISE_LEVEL" --noise_prob "$NOISE_PROB" \
  --center_noise_sigma "$CENTER_NOISE_SIGMA" --gan_weight "$GAN_WEIGHT" --mmd_weight "$MMD_WEIGHT" --use_ema --skewed_timesteps \
  --class_drop_prob 0.2 --cfg_scale "$CFG" \
  --eval_frequency "$EVAL_FREQ" --compute_fid --fid_samples "$FID_SAMPLES" \
  --ode_options "$ODE_OPTIONS" --save_fid_samples \
  --early_stop_patience "$EARLY_STOP" \
  --resume "$RESUME" \
  --wandb_project "$WANDB_PROJECT" --wandb_run_name "$WANDB_RUN_NAME" \
  --output_dir "$OUT" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$OUT/train_stdout.log"

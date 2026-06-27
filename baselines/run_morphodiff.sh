#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${BENCHMARK:-diet}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-8}"
LR="${LR:-1e-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GUIDANCE="${GUIDANCE:-1.5}"
FRAC_DIFFUSION_SKIPPED="${FRAC_DIFFUSION_SKIPPED:-0.55}"
INFERENCE_STEPS="${INFERENCE_STEPS:-50}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

case "$BENCHMARK" in
  diet)
    CONFIG="configs/diet_id.yaml"
    OUT="outputs/baselines/morphodiff/diet"
    ;;
  crispr_paper)
    CONFIG="configs/crispr_paper_core.yaml"
    OUT="outputs/baselines/morphodiff/crispr_paper"
    ;;
  *)
    echo "Unknown BENCHMARK=$BENCHMARK; expected diet or crispr_paper" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR_OVERRIDE:-$REPO_ROOT/outputs/baselines/_data/$BENCHMARK}"

if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  echo "Missing $DATA_DIR/manifest.json; run: bash baselines/export_all_baseline_data.sh" >&2
  exit 2
fi

CKPT_DIR="$REPO_ROOT/$OUT/external_checkpoints/final"

# ---------- Step 1: Train ----------
echo "[$(date -Is)] START MorphoDiff training ($BENCHMARK)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$REPO_ROOT/baselines/morphodiff_train.py" \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --output "$OUT" \
  --benchmark "$BENCHMARK" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --mixed-precision "$MIXED_PRECISION" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --num-workers 6 \
  --seed 42 \
  --use-ema \
  --proba-uncond 0.1
echo "[$(date -Is)] DONE MorphoDiff training"

# ---------- Step 2: Export generated samples ----------
if [[ ! -d "$CKPT_DIR/unet" ]]; then
  echo "Missing checkpoint $CKPT_DIR/unet" >&2
  exit 2
fi

MAX_SAMPLES_ARG=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARG=(--max-samples "$MAX_SAMPLES")
fi

python "$REPO_ROOT/baselines/morphodiff_export_fid.py" \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CKPT_DIR" \
  --output "$OUT" \
  --benchmark "$BENCHMARK" \
  --guidance "$GUIDANCE" \
  --frac-diffusion-skipped "$FRAC_DIFFUSION_SKIPPED" \
  --num-inference-steps "$INFERENCE_STEPS" \
  "${MAX_SAMPLES_ARG[@]}"

# ---------- Step 3: Evaluate ----------
python "$REPO_ROOT/scripts/aggregate_eval.py" "$REPO_ROOT/$OUT" 5 0
echo "[$(date -Is)] DONE MorphoDiff pipeline"

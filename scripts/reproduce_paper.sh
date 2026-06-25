#!/usr/bin/env bash
# ============================================================================
# PhenoFlux Paper — Reproduction Script (cleaned)
# ============================================================================
#
# Three models — one command each:
#
#   CellFlux baseline (one-hot only):
#     CONFIG=diet_id DATASET=diet_id bash scripts/train.sh
#
#   MSA (Marker Self-Attention):
#     CONFIG=diet_id_msa DATASET=diet_id_msa bash scripts/train.sh
#
#   MSA + PCD (final model):
#     CONFIG=diet_id_msa_pcd DATASET=diet_id_msa_pcd bash scripts/train.sh
#
# Eval (all models, CFG=3.0):
#   bash scripts/reproduce_paper.sh eval
#
# Usage:
#   bash scripts/reproduce_paper.sh all
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NPROC=${NPROC:-2}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-20}
EVAL_FREQ=${EVAL_FREQ:-5}
FID_SAMPLES=${FID_SAMPLES:-1000}
CFG_EVAL=${CFG_EVAL:-3.0}

train_one() {
  local NAME="$1" CONFIG="$2" DATASET="$3"
  local OUT="$REPO_ROOT/outputs/paper/$NAME"
  echo "=== TRAIN: $NAME ==="
  mkdir -p "$OUT"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node="$NPROC" \
    -m morphoflux.engine.train \
    --dataset "$DATASET" --config "$CONFIG" --device cuda \
    --batch_size "$BATCH" --accum_iter 1 --num_workers 10 \
    --epochs "$EPOCHS" --eval_frequency "$EVAL_FREQ" \
    --use_initial 1 --use_ema --skewed_timesteps \
    --class_drop_prob 0.2 --cfg_scale 0.2 \
    --compute_fid --fid_samples "$FID_SAMPLES" --save_fid_samples \
    --foreground_loss \
    --foreground_threshold 0.05 --foreground_weight 5.0 --background_weight 0.1 \
    --ode_options '{"step_size": 0.02}' \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/train_stdout.log"
}

eval_one() {
  local NAME="$1" CONFIG="$2" DATASET="$3" CKPT_EPOCH="$4"
  local OUT="$REPO_ROOT/outputs/paper/$NAME"
  local CKPT="$OUT/checkpoint-${CKPT_EPOCH}.pth"
  local EVAL_DIR="$OUT/eval_cfg${CFG_EVAL}_ep${CKPT_EPOCH}"

  if [ ! -f "$CKPT" ]; then
    echo "SKIP $NAME: checkpoint not found: $CKPT"
    return
  fi

  echo "=== EVAL: $NAME CFG=$CFG_EVAL ==="
  torchrun --standalone --nproc_per_node="$NPROC" \
    -m morphoflux.engine.train \
    --dataset "$DATASET" --config "$CONFIG" --device cuda \
    --eval_only --resume "$CKPT" \
    --use_initial 1 --cfg_scale "$CFG_EVAL" --use_ema \
    --fid_samples "$FID_SAMPLES" --compute_fid --save_fid_samples \
    --output_dir "$EVAL_DIR"

  # gap_closed
  local EPOCH_DIR=$(ls -d "$EVAL_DIR"/fid_samples/epoch-* | head -1)
  local EVAL_EPOCH=$(basename "$EPOCH_DIR" | sed 's/epoch-//')
  python scripts/diet_marker_distribution_figure.py \
    --run-dir "$EVAL_DIR" --epoch "$EVAL_EPOCH" \
    --out-dir "$EVAL_DIR" --prefix "${NAME}_cfg${CFG_EVAL}"

  # MoA
  python src/morphoflux/engine/moa/train_moa.py \
    --config_path "configs/${CONFIG}.yaml" --mode eval \
    --img_root_path "$EPOCH_DIR" \
    --ckpt_path outputs/baselines/moa/diet/condition_classifier.pth \
    --out_json "$EVAL_DIR/moa_result.json"
}

cmd_train() {
  train_one "cellflux_baseline" "diet_id" "diet_id"
  train_one "msa" "diet_id_msa" "diet_id_msa"
  train_one "msa_pcd" "diet_id_msa_pcd" "diet_id_msa_pcd"
}

cmd_eval() {
  eval_one "cellflux_baseline" "diet_id" "diet_id" 19
  eval_one "msa" "diet_id_msa" "diet_id_msa" 19
  eval_one "msa_pcd" "diet_id_msa_pcd" "diet_id_msa_pcd" 19
}

cmd_all() {
  cmd_train
  cmd_eval
}

case "${1:-all}" in
  train) cmd_train ;;
  eval)  cmd_eval ;;
  all)   cmd_all ;;
  *)
    echo "Usage: bash scripts/reproduce_paper.sh {train|eval|all}"
    ;;
esac

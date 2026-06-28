#!/usr/bin/env bash
# Reproduce all PhenoFlux paper experiments.
# Run after data preparation (build_diet_data.py, build_crispr_paper_data.py).
# Uses 2 GPUs with automatic parallel scheduling.
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

# --- Configuration ---
EPOCHS=${EPOCHS:-20}
EVAL_FREQ=${EVAL_FREQ:-5}
FID_SAMPLES=${FID_SAMPLES:-5120}
BATCH=${BATCH:-16}
NPROC=${NPROC:-2}
SUBSET=${SUBSET:-5k}   # 5k for ablation, full for final results
SKIP_DONE=${SKIP_DONE:-1}

echo "=== PhenoFlux Paper Experiments ==="
echo "Subset: $SUBSET  Epochs: $EPOCHS  GPUs: $NPROC"

SUFFIX=""
INDEX="index_diet.csv"
if [[ "$SUBSET" == "5k" ]]; then
  SUFFIX="_5k"
  INDEX="index_diet_5k.csv"
elif [[ "$SUBSET" == "10k" ]]; then
  SUFFIX="_10k"
  INDEX="index_diet_10k.csv"
elif [[ "$SUBSET" == "mini" ]]; then
  SUFFIX="_mini"
  INDEX="index_diet_mini.csv"
fi

run_experiment() {
  local NAME="$1" CONFIG="$2" DATASET="$3" OUT="$4"
  shift 4
  local EXTRA=("$@")

  OUT="outputs/runs/diet/${NAME}${SUFFIX}_v1"
  if [[ "$SKIP_DONE" == "1" ]] && [[ -f "$OUT/checkpoint-${EPOCHS}.pth" ]]; then
    echo "[skip] $NAME — checkpoint already exists at $OUT"
    return
  fi
  echo "=== Training: $NAME ==="
  mkdir -p "$OUT"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
    --dataset "$DATASET" --config "$CONFIG" --device cuda \
    --batch_size "$BATCH" --accum_iter 1 --num_workers 10 --epochs "$EPOCHS" \
    --use_initial 1 --use_ema --skewed_timesteps \
    --class_drop_prob 0.2 --cfg_scale 0.2 \
    --eval_frequency "$EVAL_FREQ" --compute_fid --fid_samples "$FID_SAMPLES" \
    --save_fid_samples --foreground_loss --foreground_threshold 0.05 \
    --foreground_weight 5.0 --background_weight 0.1 \
    --output_dir "$OUT" "${EXTRA[@]}" 2>&1 | tee "$OUT/train.log"
}

# --- Diet Experiments ---
echo ""
echo "=== Phase 1: Diet Ablation ==="

run_experiment "diet_id"           "diet_id${SUFFIX}"        "diet_id"
run_experiment "diet_id_18ch"      "diet_id_18ch${SUFFIX}"   "diet_id_18ch"
run_experiment "diet_id_msa"       "diet_id_msa${SUFFIX}"    "diet_id_msa"
run_experiment "diet_id_msa_pcd"   "diet_id_msa_pcd${SUFFIX}" "diet_id_msa_pcd"

# --- CRISPR Experiments ---
echo ""
echo "=== Phase 2: CRISPR Experiments ==="

run_experiment_crispr() {
  local NAME="$1" CONFIG="$2" DATASET="$3"
  local OUT="outputs/runs/crispr/${NAME}_v1"

  if [[ "$SKIP_DONE" == "1" ]] && [[ -f "$OUT/checkpoint-${EPOCHS}.pth" ]]; then
    echo "[skip] $NAME — checkpoint already exists at $OUT"
    return
  fi
  echo "=== Training: $NAME ==="
  mkdir -p "$OUT"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
    --dataset "$DATASET" --config "$CONFIG" --device cuda \
    --batch_size "$BATCH" --accum_iter 1 --num_workers 10 --epochs "$EPOCHS" \
    --use_initial 1 --use_ema --skewed_timesteps \
    --class_drop_prob 0.2 --cfg_scale 0.2 \
    --eval_frequency "$EVAL_FREQ" --compute_fid --fid_samples "$FID_SAMPLES" \
    --save_fid_samples --foreground_loss --foreground_threshold 0.05 \
    --foreground_weight 5.0 --background_weight 0.1 \
    --output_dir "$OUT" 2>&1 | tee "$OUT/train.log"
}

run_experiment_crispr "perturbmulti_id"     "perturbmulti_id"     "perturbmulti_id"
run_experiment_crispr "perturbmulti_idsig"  "perturbmulti_idsig"  "perturbmulti_idsig"

echo ""
echo "=== All experiments launched ==="
echo "Evaluate with:"
echo "  python scripts/aggregate_eval.py outputs/runs/diet/<name>${SUFFIX}_v1 5"
echo "  python scripts/aggregate_eval.py outputs/runs/crispr/<name>_v1 5"

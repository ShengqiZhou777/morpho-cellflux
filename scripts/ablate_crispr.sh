#!/bin/bash
# CRISPR ablation: baseline -> MSA -> MSA+PCD
# 3 configs x 40 epochs on 40-gene paper-core subset.
# Train WITHOUT intermediate eval (saves ~1.5h per config).
# After training: eval_only on best-loss checkpoint, then aggregate.
#
# Usage:
#   bash scripts/ablate_crispr.sh
#
# Env overrides:
#   NPROC=2 EPOCHS=40 CONFIGS_OVERRIDE=baseline,msa

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate pmf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TZ=Asia/Shanghai

NPROC="${NPROC:-2}"
EPOCHS="${EPOCHS:-40}"
DATA_INDEX="${DATA_INDEX:-data/processed/crispr/index_paper_40.csv}"
TAG="${TAG:-ablate_$(date +%m%d_%H%M)}"
FID_EVAL="${FID_EVAL:-2048}"
WANDB_PROJECT="${WANDB_PROJECT:-phenoflux}"

declare -A CONFIGS=(
    [baseline]="phenoflux_crispr"
    [msa]="phenoflux_crispr_msa"
    [msa_pcd]="phenoflux_crispr_msa_pcd"
)

if [[ -n "${CONFIGS_OVERRIDE:-}" ]]; then
    IFS=',' read -ra RUN_LIST <<< "$CONFIGS_OVERRIDE"
else
    RUN_LIST=(baseline msa msa_pcd)
fi

echo "=============================================="
echo "CRISPR Ablation: ${#RUN_LIST[@]} config(s) x ${EPOCHS} epochs"
echo "  GPUs=$NPROC  dopri5  bs=16  eval_freq=0 (post-hoc only)"
echo "  Data=$DATA_INDEX  Tag=$TAG"
echo "  Configs: ${RUN_LIST[*]}"
echo "  Started: $(date '+%F %T %Z')"
echo "=============================================="

for NAME in "${RUN_LIST[@]}"; do
    CONFIG="${CONFIGS[$NAME]}"
    OUT="outputs/ablate_crispr/${TAG}/${NAME}"
    mkdir -p "$OUT"

    # ---- Step 1: Train (no intermediate eval) ----
    echo ""
    echo "====== [$NAME] TRAIN config=$CONFIG  $(date '+%T') ======"

    torchrun --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
        --dataset phenoflux --config "$CONFIG" --device cuda \
        --batch_size 16 --epochs "$EPOCHS" --use_initial 1 --cfg_scale 0.2 \
        --use_ema --skewed_timesteps --class_drop_prob 0.2 \
        --eval_frequency 0 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "crispr_${NAME}_${TAG}" \
        --wandb_tags "crispr,ablation,${NAME}" \
        --data_index "$DATA_INDEX" \
        --output_dir "$OUT" \
        2>&1 | tee "$OUT/train.log"

    # ---- Step 2: Eval on best-loss checkpoint ----
    CKPT="$OUT/checkpoint-best_loss.pth"
    if [[ ! -f "$CKPT" ]]; then
        echo "WARNING: best_loss checkpoint not found, using last epoch"
        CKPT=$(ls "$OUT"/checkpoint-*.pth 2>/dev/null | sort -V | tail -1)
    fi
    echo "====== [$NAME] EVAL  ckpt=$(basename $CKPT)  $(date '+%T') ======"

    torchrun --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
        --dataset phenoflux --config "$CONFIG" --device cuda \
        --batch_size 16 --epochs 1 --start_epoch 0 \
        --use_initial 1 --cfg_scale 0.2 \
        --use_ema --skewed_timesteps --class_drop_prob 0.2 \
        --eval_frequency 1 --fid_samples "$FID_EVAL" --compute_fid --save_fid_samples \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "crispr_${NAME}_eval_${TAG}" \
        --wandb_tags "crispr,eval,${NAME}" \
        --eval_only --resume "$CKPT" \
        --data_index "$DATA_INDEX" \
        --output_dir "$OUT" \
        2>&1 | tee "$OUT/eval.log"

    # ---- Step 3: Aggregate ----
    LAST_EPOCH_DIR=$(find "$OUT/fid_samples" -maxdepth 1 -type d -name "epoch-*" 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [[ -n "$LAST_EPOCH_DIR" ]]; then
        EPOCH_NUM=$(basename "$LAST_EPOCH_DIR" | sed 's/epoch-//')
        echo "====== [$NAME] AGGREGATE epoch=$EPOCH_NUM  $(date '+%T') ======"
        python phenoflux/eval/aggregate.py "$OUT" 5 "$EPOCH_NUM" \
            2>&1 | tee "$OUT/aggregate.log" || true
    fi
done

echo ""
echo "=== CRISPR ablation complete  $(date '+%F %T %Z') ==="

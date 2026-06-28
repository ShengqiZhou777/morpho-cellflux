#!/usr/bin/env bash
# Focused re-test: MSA vs MSA+PCD on diet, after the PCD v2 fix
# (tanh-bounded magnitude + per-condition sigmoid gate).
# Goal: confirm fixed-PCD no longer causes fasted weak-perturbation overshoot.
# Panel [9,5,10] (current configs). Fresh dir — does NOT touch old ablations.
set -euo pipefail

EPOCHS="${EPOCHS:-20}"
DATA_INDEX="${DATA_INDEX:-data/processed/diet/index_diet_5k.csv}"
OUT_ROOT="${OUT_ROOT:-outputs/ablate_diet_pcdfix}"
BATCH_SIZE="${BATCH_SIZE:-16}"   # 16/GPU: bs=32 OOMs on 32GB under DDP here

eval "$(conda shell.bash hook)"
conda activate pmf
# reduce fragmentation (suggested by the OOM error)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Diet PCD-fix re-test: msa vs msa_pcd x ${EPOCHS}ep  bs=${BATCH_SIZE}"
echo "Data: ${DATA_INDEX}  Out: ${OUT_ROOT}"
echo "Started: $(TZ=Asia/Shanghai date '+%F %T %Z')"
echo "=============================================="

for NAME in msa msa_pcd; do
    CONFIG="phenoflux_diet_${NAME}"
    OUT="${OUT_ROOT}/${NAME}"
    echo ""
    echo "====== [$NAME] config=$CONFIG  $(TZ=Asia/Shanghai date '+%T') ======"
    mkdir -p "$OUT"

    torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
        --dataset phenoflux --config "$CONFIG" --device cuda \
        --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --use_initial 1 --cfg_scale 0.2 \
        --use_ema --skewed_timesteps --class_drop_prob 0.2 \
        --eval_frequency 5 --fid_samples 5120 --compute_fid --save_fid_samples \
        --data_index "$DATA_INDEX" \
        --output_dir "$OUT"

    # eval fires every 5 epochs -> last dir is epoch-(EPOCHS-1)
    python phenoflux/eval/aggregate.py "$OUT" 5 "$((EPOCHS-1))" || true
done

echo ""
echo "=== DONE $(TZ=Asia/Shanghai date '+%F %T %Z') ==="

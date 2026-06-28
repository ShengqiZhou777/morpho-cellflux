#!/bin/bash
# Diet ablation experiment: baseline → +18ch → +MSA → +MSA+PCD
# Quick 5-epoch run on 5k subset to validate molecular prior contributions.
#
# Usage:
#   bash scripts/ablate_diet.sh
#
# Environment overrides:
#   NPROC=2 EPOCHS=5 FID_SAMPLES=512

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export PATH=/home/ubuntu/miniconda3/envs/pmf/bin:$PATH

NPROC="${NPROC:-2}"
EPOCHS="${EPOCHS:-5}"
FID_SAMPLES="${FID_SAMPLES:-512}"
DATA_INDEX="data/processed/diet/index_diet_5k.csv"
TAG="ablate_$(date +%m%d_%H%M)"

declare -A CONFIGS=(
    [baseline]="phenoflux_diet"
    [naive]="phenoflux_diet_18ch"
    [msa]="phenoflux_diet_msa"
    [msa_pcd]="phenoflux_diet_msa_pcd"
)

echo "=========================================="
echo "Diet Ablation: 4 configs × ${EPOCHS} epochs"
echo "  GPUs=$NPROC  FID=$FID_SAMPLES  Data=$DATA_INDEX"
echo "  Tag: $TAG"
echo "=========================================="

for NAME in baseline naive msa msa_pcd; do
    CONFIG="${CONFIGS[$NAME]}"
    OUT="outputs/ablate_diet/${TAG}/${NAME}"
    mkdir -p "$OUT"

    echo ""
    echo "====== [$NAME] config=$CONFIG ======"
    echo "  Output: $OUT"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
        --dataset phenoflux --config "$CONFIG" --device cuda \
        --batch_size 32 --num_workers 4 --epochs "$EPOCHS" \
        --use_initial 1 --cfg_scale 0.2 --use_ema \
        --skewed_timesteps --class_drop_prob 0.2 \
        --eval_frequency "$EPOCHS" --fid_samples "$FID_SAMPLES" \
        --compute_fid --save_fid_samples \
        --eval_batch_size 128 \
        --data_index "$DATA_INDEX" \
        --output_dir "$OUT" \
        2>&1 | tee "$OUT/train.log"

    # --- Aggregate Eval ---
    FID_DIR=$(find "$OUT/fid_samples" -maxdepth 1 -type d -name "epoch-*" 2>/dev/null | sort | tail -1)
    if [[ -n "$FID_DIR" ]]; then
        EPOCH_NUM=$(basename "$FID_DIR" | sed 's/epoch-//')
        python3 scripts/aggregate_eval.py "$OUT" 5 "$EPOCH_NUM" \
            2>&1 | tee "$OUT/aggregate_eval.log"
    fi
done

echo ""
echo "=========================================="
echo "Ablation complete. Summary:"
echo "=========================================="
for NAME in baseline naive msa msa_pcd; do
    OUT="outputs/ablate_diet/${TAG}/${NAME}"
    echo ""
    echo "--- $NAME ---"
    if [[ -f "$OUT/aggregate_eval_summary.json" ]]; then
        python3 -c "
import json
with open('$OUT/aggregate_eval_summary.json') as f:
    data = json.load(f)
if 'per_condition_dist' in data:
    for cond, chs in data['per_condition_dist'].items():
        for ch, m in chs.items():
            print(f'  PGC {cond:8s} {ch:15s} = {m[\"pgc_wd\"]:+.3f}')
elif 'dist_pooled' in data:
    for ch, m in data['dist_pooled'].items():
        print(f'  PGC pooled {ch:15s} = {m[\"pgc_wd\"]:+.3f}')
"
    else
        echo "  (no aggregate results)"
    fi
    if [[ -f "$OUT/log.txt" ]]; then
        FID=$(python3 -c "
import json
with open('$OUT/log.txt') as f:
    for line in f:
        d = json.loads(line)
        if 'eval_fid' in d:
            print(f'{d[\"eval_fid\"]:.1f}')
" 2>/dev/null | tail -1)
        if [[ -n "$FID" ]]; then
            echo "  FID = $FID"
        fi
    fi
done

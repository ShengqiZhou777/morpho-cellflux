#!/bin/bash
# Quick validation pipeline for PhenoFlux experiments.
# Runs a complete train→eval→metrics cycle on a mini subset to validate
# that a config works before committing to full-scale training.
#
# Usage:
#   bash scripts/quick_validate.sh <config> [dataset] [data_index]
#
# Examples:
#   bash scripts/quick_validate.sh diet_id_msa_pcd diet_id_msa_pcd
#   bash scripts/quick_validate.sh diet_id diet_id
#
# Environment overrides:
#   NPROC=1              # number of GPUs (default: 1)
#   EPOCHS=2             # training epochs (default: 2)
#   FID_SAMPLES=64       # images to generate (default: 64)
#   SKIP_TRAIN=1         # skip training, only eval existing checkpoint

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="${1:?Usage: $0 <config> [dataset] [data_index]}"
DATASET="${2:-$CONFIG}"
# Default to mini subset for fast validation; override for larger subsets
DATA_INDEX="${3:-data/processed/diet/index_diet_2k.csv}"

# --- Config ---
NPROC="${NPROC:-1}"
EPOCHS="${EPOCHS:-2}"
FID_SAMPLES="${FID_SAMPLES:-64}"
EVAL_FREQ=1
OUT_DIR="${OUT_DIR:-outputs/quick_validate/${CONFIG}_$(date +%Y%m%d_%H%M%S)}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-phenoflux}"

# --- Conda setup ---
if [[ -f "/home/ubuntu/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
    conda activate pmf
fi

TORCHRUN="${TORCHRUN:-$(command -v torchrun 2>/dev/null || echo 'torchrun')}"

echo "=========================================="
echo "Quick Validate: config=$CONFIG dataset=$DATASET"
echo "  GPUs=$NPROC  Epochs=$EPOCHS  FID Samples=$FID_SAMPLES"
echo "  Output: $OUT_DIR"
echo "=========================================="

mkdir -p "$OUT_DIR"

# --- Step 1: Training ---
if [[ "$SKIP_TRAIN" != "1" ]]; then
    echo ""
    echo "--- [1/3] Training ($EPOCHS epochs) ---"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
        --dataset "$DATASET" --config "$CONFIG" --device cuda \
        --batch_size 32 --num_workers 4 --epochs "$EPOCHS" \
        --use_initial 1 --cfg_scale 0.2 \
        --skewed_timesteps --class_drop_prob 0.2 \
        --eval_frequency "$EVAL_FREQ" --fid_samples "$FID_SAMPLES" \
        --compute_fid --save_fid_samples \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "qv_${CONFIG}_$(date +%m%d_%H%M)" \
        --wandb_tags "quick-validate,${CONFIG}" \
        --data_index "$DATA_INDEX" \
        --test_run \
        --output_dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/train.log"
    echo "Training done."
else
    echo "--- [1/3] Skipping training (SKIP_TRAIN=1) ---"
fi

# --- Step 2: Find generated images ---
echo ""
echo "--- [2/3] Checking generated images ---"
FID_DIR=$(find "$OUT_DIR/fid_samples" -maxdepth 1 -type d -name "epoch-*" 2>/dev/null | sort | tail -1)
if [[ -z "$FID_DIR" ]]; then
    echo "ERROR: No generated images found in $OUT_DIR/fid_samples/"
    echo "Check $OUT_DIR/train.log for errors."
    exit 1
fi
echo "Found generated images: $FID_DIR"

# Count images per condition
for cond_dir in "$FID_DIR"/*/; do
    cond=$(basename "$cond_dir")
    n=$(find "$cond_dir" -name "*.png" 2>/dev/null | wc -l)
    echo "  $cond: $n images"
done

# Check images are valid (not all black/white)
FIRST_PNG=$(find "$FID_DIR" -name "*.png" 2>/dev/null | head -1)
if [[ -z "$FIRST_PNG" ]]; then
    echo "ERROR: No PNG files found."
    exit 1
fi

python3 -c "
import numpy as np
from PIL import Image
p = '$FIRST_PNG'
img = np.array(Image.open(p))
mean_val = img.mean()
min_val = img.min()
max_val = img.max()
if min_val == max_val:
    print(f'ERROR: {p} is constant (value={min_val}). Model may have collapsed.')
    exit(1)
print(f'  Sample check: mean={mean_val:.3f}, min={min_val}, max={max_val} -> OK')
" || { echo "Image validation failed!"; exit 1; }

# --- Step 3: Aggregate evaluation ---
echo ""
echo "--- [3/3] Computing aggregate metrics ---"
EPOCH_NUM=$(basename "$FID_DIR" | sed 's/epoch-//')

# Run aggregate_eval
python3 phenoflux/eval/aggregate.py "$OUT_DIR" 5 "$EPOCH_NUM" \
    2>&1 | tee "$OUT_DIR/aggregate_eval.log"

# Check for results
if [[ -f "$OUT_DIR/aggregate_eval_summary.json" ]]; then
    echo ""
    echo "=== Aggregate Results ==="
    python3 -c "
import json
with open('$OUT_DIR/aggregate_eval_summary.json') as f:
    data = json.load(f)
# Print channel-level PGC
if 'by_channel' in data:
    for ch, metrics in data['by_channel'].items():
        gc = metrics.get('pgc', {})
        for cond, val in gc.items():
            print(f'  PGC[{ch}][{cond}] = {val:.4f}')
elif isinstance(data, list):
    for item in data:
        print(f'  {item}')
else:
    print(json.dumps(data, indent=2)[:2000])
"
fi

echo ""
if grep -q "pgc" "$OUT_DIR/aggregate_eval.log" 2>/dev/null; then
    echo "Quick validation PASSED: PGC computed successfully."
else
    echo "Quick validation COMPLETE. See $OUT_DIR/ for results."
fi

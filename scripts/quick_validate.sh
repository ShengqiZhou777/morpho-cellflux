#!/usr/bin/env bash
# Quick validation for the single active Morpho-CellFlux path.
#
# Active config:
#   microalgae_timepoint
#
# Usage:
#   bash scripts/quick_validate.sh
#   bash scripts/quick_validate.sh microalgae_timepoint
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="${1:-microalgae_timepoint}"
if [[ "$CONFIG" != "microalgae_timepoint" ]]; then
    echo "Only microalgae_timepoint is active. Got: $CONFIG" >&2
    exit 2
fi

DATASET="phenoflux"
DATA_INDEX="data/processed/microalgae_v1/views/timepoint/index.csv"

NPROC="${NPROC:-1}"
EPOCHS="${EPOCHS:-1}"
FID_SAMPLES="${FID_SAMPLES:-8}"
BATCH="${BATCH:-4}"
OUT_DIR="${OUT_DIR:-outputs/quick_validate/timepoint_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

if [[ ! -f "$DATA_INDEX" ]]; then
    echo "Missing active data index: $DATA_INDEX" >&2
    echo "Build it with: python scripts/build_microalgae_dataset.py --version microalgae_v1 --views timepoint" >&2
    exit 1
fi

if [[ ! -f "data/processed/microalgae_v1/views/timepoint/embedding.csv" ]]; then
    echo "Missing active embedding: data/processed/microalgae_v1/views/timepoint/embedding.csv" >&2
    exit 1
fi

TORCHRUN="${TORCHRUN:-$(command -v torchrun 2>/dev/null || echo 'torchrun')}"

echo "=========================================="
echo "Quick Validate: microalgae_timepoint"
echo "  Dataset: $DATASET"
echo "  Data index: $DATA_INDEX"
echo "  GPUs: $NPROC  Epochs: $EPOCHS  Batch: $BATCH  FID Samples: $FID_SAMPLES"
echo "  Output: $OUT_DIR"
echo "=========================================="

mkdir -p "$OUT_DIR"

WANDB_ARGS=()
if [[ -n "$WANDB_PROJECT" ]]; then
    WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT" --wandb_run_name "qv_timepoint_$(date +%m%d_%H%M)")
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
    --dataset "$DATASET" --config "$CONFIG" --device cuda \
    --batch_size "$BATCH" --num_workers 4 --epochs "$EPOCHS" \
    --use_initial 1 --cfg_scale 0.2 \
    --skewed_timesteps --class_drop_prob 0.2 \
    --eval_frequency 1 --fid_samples "$FID_SAMPLES" \
    --compute_fid --save_fid_samples \
    --data_index "$DATA_INDEX" \
    --test_run \
    --output_dir "$OUT_DIR" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/train.log"

echo "Quick validation complete: $OUT_DIR"

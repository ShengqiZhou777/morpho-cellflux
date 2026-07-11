#!/usr/bin/env bash
# Quick 1-GPU sanity run for the active Morpho-CellFlux timepoint path.
#
# Active configs (512px timepoint view):
#   microalgae_timepoint_512_genes   (primary, 476-dim gene/protein condition)   <- default
#   microalgae_timepoint_512       (4d baseline, ablation counterpart)
#
# Usage:
#   bash scripts/quick_validate.sh
#   bash scripts/quick_validate.sh microalgae_timepoint_512
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="${1:-microalgae_timepoint_512_genes}"
case "$CONFIG" in
    microalgae_timepoint_512_genes) EMBEDDING="embedding_genes.csv" ;;
    microalgae_timepoint_512)     EMBEDDING="embedding.csv" ;;
    *)
        echo "Active timepoint configs: microalgae_timepoint_512_genes | microalgae_timepoint_512. Got: $CONFIG" >&2
        exit 2
        ;;
esac

DATASET="phenoflux"
VIEW_DIR="data/processed/microalgae_v1/views/timepoint_512"
DATA_INDEX="$VIEW_DIR/index.csv"

NPROC="${NPROC:-1}"
EPOCHS="${EPOCHS:-1}"
FID_SAMPLES="${FID_SAMPLES:-8}"
BATCH="${BATCH:-4}"
OUT_DIR="${OUT_DIR:-outputs/quick_validate/${CONFIG}_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

if [[ ! -f "$DATA_INDEX" ]]; then
    echo "Missing timepoint_512 index: $DATA_INDEX" >&2
    echo "The 512px timepoint view must be prepared under $VIEW_DIR before running." >&2
    exit 1
fi

if [[ ! -f "$VIEW_DIR/$EMBEDDING" ]]; then
    echo "Missing embedding for $CONFIG: $VIEW_DIR/$EMBEDDING" >&2
    if [[ "$EMBEDDING" == "embedding_genes.csv" ]]; then
        echo "Build it with: python scripts/build_gene_condition.py" >&2
    fi
    exit 1
fi

TORCHRUN="${TORCHRUN:-$(command -v torchrun 2>/dev/null || echo 'torchrun')}"

echo "=========================================="
echo "Quick Validate: $CONFIG"
echo "  Dataset: $DATASET"
echo "  Data index: $DATA_INDEX"
echo "  Embedding: $VIEW_DIR/$EMBEDDING"
echo "  GPUs: $NPROC  Epochs: $EPOCHS  Batch: $BATCH  FID Samples: $FID_SAMPLES"
echo "  Output: $OUT_DIR"
echo "=========================================="

mkdir -p "$OUT_DIR"

WANDB_ARGS=()
if [[ -n "$WANDB_PROJECT" ]]; then
    WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT" --wandb_run_name "qv_${CONFIG}_$(date +%m%d_%H%M)")
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

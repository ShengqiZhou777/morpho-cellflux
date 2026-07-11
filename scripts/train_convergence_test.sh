#!/usr/bin/env bash
# Resume from 5-epoch checkpoint to 40 epochs to test convergence hypothesis.
# Tests whether "generated ≈ control" is due to undertraining vs pairing/conditioning.

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

RESUME_CKPT="${1:-outputs/runs/microalgae/timepoint_512_b12_e5_20260706_000411/checkpoint.pth}"
OUT_DIR="${2:-outputs/runs/microalgae/timepoint_512_convergence_e40_$(date +%Y%m%d_%H%M%S)}"

echo "=== PhenoFlux Convergence Test: 5-epoch → 40-epoch ==="
echo "Resume from: $RESUME_CKPT"
echo "Output to:   $OUT_DIR"
echo ""

if [ ! -f "$RESUME_CKPT" ]; then
  echo "ERROR: Checkpoint not found: $RESUME_CKPT"
  echo "Available checkpoints:"
  find outputs/runs/microalgae -name "checkpoint*.pth" -o -name "checkpoint.pth" | head -10
  exit 1
fi

# Training config (matches original 5-epoch run)
CONFIG="microalgae_timepoint_512"
BATCH=12
EPOCHS=40
EVAL_FREQ=5
FID_SAMPLES=512
NPROC=2
USE_INITIAL=1
CFG=0.2

torchrun --standalone --nproc_per_node="$NPROC" -m phenoflux.train \
  --dataset phenoflux \
  --config "$CONFIG" \
  --batch_size "$BATCH" \
  --epochs "$EPOCHS" \
  --use_initial "$USE_INITIAL" \
  --cfg_scale "$CFG" \
  --use_ema \
  --skewed_timesteps \
  --class_drop_prob 0.2 \
  --eval_frequency "$EVAL_FREQ" \
  --fid_samples "$FID_SAMPLES" \
  --compute_fid \
  --resume "$RESUME_CKPT" \
  --output_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/train_stdout.log"

echo ""
echo "✓ Training complete. Output: $OUT_DIR"
echo ""
echo "Next: Generate visual reports at epochs 5/10/20/30/40 to compare generated vs control."
echo "Run: bash scripts/generate_convergence_reports.sh $OUT_DIR"

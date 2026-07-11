#!/usr/bin/env bash
# Stage 2: Train with 62-dimensional omics-enriched condition
# 62 dims = 2 light/dark + 1 time_norm + 1 time_bin_h + 29 RNA PCA + 29 Protein PCA
# Expected to break identity mapping by providing stronger condition signal.

set -e

cd "$(dirname "$0")/.."

OUTPUT_DIR="outputs/runs/microalgae/timepoint_512_62d_e40_$(date +%Y%m%d_%H%M%S)"

echo "=== Stage 2: Training with 62D Omics Condition ==="
echo "Output: $OUTPUT_DIR"
echo "Config: microalgae_timepoint_512_62d"
echo "Epochs: 40 (from scratch, no resume)"
echo ""

torchrun --standalone --nproc_per_node=1 -m phenoflux.train \
  --dataset phenoflux \
  --config microalgae_timepoint_512_62d \
  --batch_size 12 \
  --epochs 40 \
  --use_initial 1 \
  --cfg_scale 0.2 \
  --use_ema \
  --skewed_timesteps \
  --class_drop_prob 0.2 \
  --eval_frequency 5 \
  --fid_samples 512 \
  --compute_fid \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "✓ Training complete: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Generate visual reports:"
echo "     bash scripts/generate_convergence_reports.sh $OUTPUT_DIR"
echo ""
echo "  2. Quantify pixel difference vs Stage 1 (4D baseline):"
echo "     python scripts/compare_stage1_stage2.py"

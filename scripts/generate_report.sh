#!/usr/bin/env bash
# Generate visual evaluation report from a trained checkpoint.
# Uses control-initialized sampling (matches training) to avoid OOD artifacts.

set -e

CHECKPOINT="${1:?checkpoint path required}"
CONFIG="${2:-microalgae_timepoint_512}"
OUTPUT_DIR="${3:-outputs/reports/$(basename $CHECKPOINT .pth)_report_$(date +%Y%m%d_%H%M%S)}"

echo "=== PhenoFlux Visual Report Generator ==="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""

# IMPORTANT: Always use --init-mode control for models trained with use_initial=1
# (which is the default and correct setting). Using --init-mode noise causes
# spatial instability (off-center cells, spurious double-cell artifacts).

python3 scripts/sample_microalgae_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --labels-per-condition 3 \
  --examples-per-label 2 \
  --init-mode control \
  --cfg-scale 0.2 \
  --ode-method midpoint \
  --step-size 0.05 \
  --seed 7

echo ""
echo "✓ Report saved to: $OUTPUT_DIR"
echo "  Grid: $OUTPUT_DIR/grid_control_generated_target.png"

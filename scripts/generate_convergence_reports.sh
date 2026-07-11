#!/usr/bin/env bash
# Generate visual reports from multiple checkpoints to track convergence.
# Usage: bash scripts/generate_convergence_reports.sh <run_dir>

set -e

RUN_DIR="${1:?run directory required}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: Run directory not found: $RUN_DIR"
  exit 1
fi

echo "=== Generating convergence visual reports from $RUN_DIR ==="
echo ""

# Find all checkpoint files
CHECKPOINTS=$(find "$RUN_DIR" -name "checkpoint-*.pth" -o -name "checkpoint-best*.pth" | sort)

for CKPT in $CHECKPOINTS; do
  EPOCH=$(basename "$CKPT" | grep -oP '\d+' | head -1)
  REPORT_DIR="outputs/reports/convergence_$(basename $RUN_DIR)_ep${EPOCH}_$(date +%Y%m%d_%H%M%S)"

  echo "--- Epoch $EPOCH: $CKPT → $REPORT_DIR ---"

  python3 scripts/sample_microalgae_checkpoint.py \
    --checkpoint "$CKPT" \
    --config microalgae_timepoint_512 \
    --output-dir "$REPORT_DIR" \
    --labels-per-condition 3 \
    --examples-per-label 2 \
    --init-mode control \
    --cfg-scale 0.2 \
    --ode-method midpoint \
    --step-size 0.05 \
    --seed 7

  echo ""
done

echo "✓ All reports generated under outputs/reports/"
echo ""
echo "Next: Quantify control vs generated pixel diff across epochs."
echo "Run: python3 scripts/analyze_convergence.py outputs/reports/convergence_*"

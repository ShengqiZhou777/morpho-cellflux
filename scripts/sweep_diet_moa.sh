#!/usr/bin/env bash
# Sweep CFG × USE_INITIAL × checkpoint for Diet MoA.
# Generates images in eval-only mode then runs the condition classifier on each.
#
# Usage:
#   bash scripts/sweep_diet_moa.sh
#
# Set env vars to override defaults:
#   FID_SAMPLES=256  SWEEP_DIR=outputs/sweeps/diet_moa  bash scripts/sweep_diet_moa.sh

set -uo pipefail  # no -e: allow individual combinations to fail without aborting the whole sweep

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_SH="/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH" && conda activate pmf

TORCHRUN=${TORCHRUN:-$(command -v torchrun || true)}
if [[ -z "$TORCHRUN" ]]; then
  echo "torchrun not found. Activate the project environment or set TORCHRUN=/path/to/torchrun." >&2
  exit 127
fi

# --- config ---
FID_SAMPLES=${FID_SAMPLES:-128}
SWEEP_DIR=${SWEEP_DIR:-$PROJECT_DIR/outputs/sweeps/diet_moa}
MOA_CKPT="$PROJECT_DIR/outputs/baselines/moa/diet/condition_classifier.pth"
CONFIG_PATH="$PROJECT_DIR/configs/diet_id.yaml"
# Checkpoints to sweep
CKPT_9="$PROJECT_DIR/outputs/runs/diet/diet_id_v3/checkpoint-9.pth"
CKPT_11="$PROJECT_DIR/outputs/runs/diet/diet_id_v3/checkpoint-11.pth"

# Sweep ranges
CFG_VALS=(0.5 1.0 1.5 2.0 2.5 3.0)
INIT_VALS=(0 1 2)
ALL_CKPT_PATHS=("$CKPT_9" "$CKPT_11")
ALL_CKPT_LABELS=("ep9" "ep11")

# Optional: filter to a specific checkpoint via CKPT_FILTER env var
if [[ -n "${CKPT_FILTER:-}" ]]; then
  CKPT_PATHS=()
  CKPT_LABELS=()
  for i in "${!ALL_CKPT_LABELS[@]}"; do
    if [[ "${ALL_CKPT_LABELS[$i]}" == "$CKPT_FILTER" ]]; then
      CKPT_PATHS+=("${ALL_CKPT_PATHS[$i]}")
      CKPT_LABELS+=("${ALL_CKPT_LABELS[$i]}")
    fi
  done
else
  CKPT_PATHS=("${ALL_CKPT_PATHS[@]}")
  CKPT_LABELS=("${ALL_CKPT_LABELS[@]}")
fi

# --- helpers ---
run_moa() {
  local img_dir="$1"
  local out_json="$2"
    python "$PROJECT_DIR/src/morphoflux/engine/moa/train_moa.py" \
    --config_path "$CONFIG_PATH" \
    --mode eval \
    --img_root_path "$img_dir" \
    --ckpt_path "$MOA_CKPT" \
    --out_json "$out_json"
}

# --- main ---
mkdir -p "$SWEEP_DIR"
SUMMARY="$SWEEP_DIR/summary.csv"
echo "ckpt,cfg,init,moa_acc,macro_f1,fasted_acc,hfd_acc,n_images" > "$SUMMARY"

TOTAL=$(( ${#CKPT_PATHS[@]} * ${#CFG_VALS[@]} * ${#INIT_VALS[@]} ))
echo "=== Diet MoA sweep: $TOTAL combinations, $FID_SAMPLES images each ==="

COUNT=0
for ci in "${!CKPT_PATHS[@]}"; do
  CKPT="${CKPT_PATHS[$ci]}"
  CKPT_LABEL="${CKPT_LABELS[$ci]}"
  for INIT in "${INIT_VALS[@]}"; do
    for CFG in "${CFG_VALS[@]}"; do
      COUNT=$((COUNT + 1))
      NAME="ckpt-${CKPT_LABEL}_init-${INIT}_cfg-${CFG}"
      OUT_DIR="$SWEEP_DIR/$NAME"
      IMG_DIR="$OUT_DIR/fid_samples"

      echo ""
      echo "--- [$COUNT/$TOTAL] $NAME ---"

      # 1) Generate images (eval-only)
      if [[ -d "$IMG_DIR" ]] && [[ -n "$(ls -A "$IMG_DIR" 2>/dev/null)" ]]; then
        echo "  [skip] images already exist at $IMG_DIR"
      else
        mkdir -p "$OUT_DIR"
        echo "  generating $FID_SAMPLES images..."
        cd "$PROJECT_DIR"
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$TORCHRUN" --standalone --nproc_per_node=1 -m morphoflux.engine.train \
          --dataset diet_id --config diet_id --device cuda \
          --batch_size 32 --num_workers 4 --epochs 1000000 \
          --eval_only --resume "$CKPT" \
          --use_initial "$INIT" --noise_level 0.2 \
          --use_ema --skewed_timesteps \
          --class_drop_prob 0.2 --cfg_scale "$CFG" \
          --eval_frequency 1 --fid_samples "$FID_SAMPLES" \
          --compute_fid --save_fid_samples \
          --ode_options '{"step_size": 0.02}' \
          --output_dir "$OUT_DIR" \
          2>&1 | tee "$OUT_DIR/generate.log"
        echo "  done generating"
      fi

      # 2) Run MoA
      MOA_JSON="$OUT_DIR/diet_condition_moa.json"
      if [[ -f "$MOA_JSON" ]]; then
        echo "  [skip] MoA already computed"
      else
        echo "  running MoA classifier..."
        # Find the generated image subdirectory
        GEN_SUBDIR=$(find "$IMG_DIR" -maxdepth 1 -type d -name "epoch-*" 2>/dev/null | head -1)
        if [[ -z "$GEN_SUBDIR" ]]; then
          # Images might be in a flat structure
          GEN_SUBDIR="$IMG_DIR"
        fi
        run_moa "$GEN_SUBDIR" "$MOA_JSON" 2>&1 | tee "$OUT_DIR/moa.log"
        echo "  MoA done"
      fi

      # 3) Parse result into summary
      if [[ -f "$MOA_JSON" ]]; then
        MOA_ACC=$(python3 -c "import json; print(json.load(open('$MOA_JSON'))['moa_acc'])")
        MACRO_F1=$(python3 -c "import json; print(json.load(open('$MOA_JSON'))['macro_f1'])")
        FASTED_ACC=$(python3 -c "import json; print(json.load(open('$MOA_JSON'))['per_class']['fasted']['acc'])")
        HFD_ACC=$(python3 -c "import json; print(json.load(open('$MOA_JSON'))['per_class']['hfd']['acc'])")
        N=$(python3 -c "import json; print(json.load(open('$MOA_JSON'))['n'])")
        echo "$CKPT_LABEL,$CFG,$INIT,$MOA_ACC,$MACRO_F1,$FASTED_ACC,$HFD_ACC,$N" >> "$SUMMARY"
        echo "  => MoA=${MOA_ACC}%  fasted=${FASTED_ACC}%  hfd=${HFD_ACC}%  macro-F1=${MACRO_F1}"
      fi
    done
  done
done

echo ""
echo "=== sweep complete ==="
echo "summary: $SUMMARY"
echo ""
echo "Top 10 by hfd accuracy:"
# Skip header, sort by hfd_acc (column 7), show top 10
tail -n +2 "$SUMMARY" | sort -t, -k7 -nr | head -10 | column -t -s,

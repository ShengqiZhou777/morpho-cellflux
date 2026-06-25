#!/bin/bash
# CFG sweep eval for a trained PhenoFlux/CellFlux checkpoint.
#
# Usage:
#   bash scripts/sweep_cfg.sh \
#     --ckpt outputs/paper/phenoflux_full/checkpoint-19.pth \
#     --config phenoflux_diet \
#     --dataset phenoflux_diet \
#     --run-dir outputs/paper/phenoflux_full \
#     --epoch 19 \
#     --cfg-list "0.5 1.0 1.5 2.0 2.5 3.0"
#
# For each CFG value: generates images, computes FID, gap_closed, and MoA.

set -euo pipefail

# Defaults
NPROC=${NPROC:-2}
FID_SAMPLES=${FID_SAMPLES:-1000}
USE_EMA=${USE_EMA:-1}
DEVICE=${DEVICE:-cuda}
CFG_LIST=""
CKPT=""
CONFIG=""
DATASET=""
RUN_DIR=""
EPOCH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --ckpt) CKPT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --cfg-list) CFG_LIST="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

if [[ -z "$CKPT" || -z "$CONFIG" || -z "$DATASET" || -z "$RUN_DIR" || -z "$EPOCH" || -z "$CFG_LIST" ]]; then
  echo "ERROR: all --ckpt --config --dataset --run-dir --epoch --cfg-list are required"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="$RUN_DIR/cfg_sweep_results.csv"
echo "cfg_scale,fid,gap_closed_fasted_calr,gap_closed_fasted_peri,gap_closed_fasted_tomm20,gap_closed_hfd_calr,gap_closed_hfd_peri,gap_closed_hfd_tomm20,moa_acc,moa_fasted_acc,moa_hfd_acc" > "$RESULTS_FILE"

for CFG in $CFG_LIST; do
  echo ""
  echo "============================================================"
  echo "  CFG = $CFG"
  echo "============================================================"

  EVAL_DIR="$RUN_DIR/eval_cfg${CFG}_ep${EPOCH}"

  # Step 1: Generate images + FID
  if [ ! -f "$EVAL_DIR/done_gen" ]; then
    echo "[1/3] Generating images with CFG=$CFG ..."
    torchrun --standalone --nproc_per_node="$NPROC" \
      -m morphoflux.engine.train \
      --dataset "$DATASET" --config "$CONFIG" --device "$DEVICE" \
      --eval_only --resume "$CKPT" \
      --use_initial 1 --cfg_scale "$CFG" \
      $([ "$USE_EMA" = "1" ] && echo "--use_ema") \
      --fid_samples "$FID_SAMPLES" --compute_fid --save_fid_samples \
      --output_dir "$EVAL_DIR" \
      2>&1 | tee "$EVAL_DIR/gen.log"
    touch "$EVAL_DIR/done_gen"

    # Extract FID from log
    FID_VAL=$(grep -oP 'eval_fid":\s*\K[\d.]+' "$EVAL_DIR/gen.log" | tail -1)
    echo "  FID = $FID_VAL"
  else
    echo "[1/3] Generation already done for CFG=$CFG"
    FID_VAL=$(grep -oP 'eval_fid":\s*\K[\d.]+' "$EVAL_DIR/gen.log" 2>/dev/null | tail -1 || echo "NA")
  fi

  # Find the actual epoch dir
  EPOCH_DIR=$(ls -d "$EVAL_DIR"/fid_samples/epoch-* 2>/dev/null | head -1)
  if [ -z "$EPOCH_DIR" ]; then
    echo "ERROR: no epoch dir found under $EVAL_DIR/fid_samples/"
    continue
  fi
  EVAL_EPOCH=$(basename "$EPOCH_DIR" | sed 's/epoch-//')

  # Step 2: gap_closed
  GAP_PREFIX="${CONFIG}_cfg${CFG}_ep${EVAL_EPOCH}"
  echo "[2/3] Computing gap_closed ..."
  python scripts/diet_marker_distribution_figure.py \
    --run-dir "$EVAL_DIR" \
    --epoch "$EVAL_EPOCH" \
    --out-dir "$EVAL_DIR" \
    --prefix "$GAP_PREFIX" \
    2>&1 | tee "$EVAL_DIR/gap_closed.log"

  GAP_JSON="$EVAL_DIR/${GAP_PREFIX}_marker_distribution_summary.json"
  if [ -f "$GAP_JSON" ]; then
    # Extract gap_closed values per condition/marker
    GAP_FASTED_CALR=$(python3 -c "
import json
d=json.load(open('$GAP_JSON'))
for s in d['summary']:
    if s['condition']=='fasted' and s['marker']=='Calreticulin': print(f\"{s['gap_closed']:.4f}\")
" 2>/dev/null || echo "NA")
    GAP_FASTED_PERI=$(python3 -c "
import json
d=json.load(open('$GAP_JSON'))
for s in d['summary']:
    if s['condition']=='fasted' and s['marker']=='Perilipin': print(f\"{s['gap_closed']:.4f}\")
" 2>/dev/null || echo "NA")
    GAP_FASTED_TOMM=$(python3 -c "
import json
d=json.load(open('$GAP_JSON'))
for s in d['summary']:
    if s['condition']=='fasted' and s['marker']=='TOMM20': print(f\"{s['gap_closed']:.4f}\")
" 2>/dev/null || echo "NA")
    GAP_HFD_CALR=$(python3 -c "
import json
d=json.load(open('$GAP_JSON'))
for s in d['summary']:
    if s['condition']=='hfd' and s['marker']=='Calreticulin': print(f\"{s['gap_closed']:.4f}\")
" 2>/dev/null || echo "NA")
    GAP_HFD_PERI=$(python3 -c "
import json
d=json.load(open('$GAP_JSON'))
for s in d['summary']:
    if s['condition']=='hfd' and s['marker']=='Perilipin': print(f\"{s['gap_closed']:.4f}\")
" 2>/dev/null || echo "NA")
    GAP_HFD_TOMM=$(python3 -c "
import json
d=json.load(open('$GAP_JSON'))
for s in d['summary']:
    if s['condition']=='hfd' and s['marker']=='TOMM20': print(f\"{s['gap_closed']:.4f}\")
" 2>/dev/null || echo "NA")
  else
    GAP_FASTED_CALR="NA"; GAP_FASTED_PERI="NA"; GAP_FASTED_TOMM="NA"
    GAP_HFD_CALR="NA"; GAP_HFD_PERI="NA"; GAP_HFD_TOMM="NA"
    echo "WARNING: gap_closed JSON not found: $GAP_JSON"
  fi

  # Step 3: MoA
  MOA_JSON="$EVAL_DIR/moa_result.json"
  echo "[3/3] Computing MoA ..."
  python src/morphoflux/engine/moa/train_moa.py \
    --config_path "configs/${CONFIG}.yaml" \
    --mode eval \
    --img_root_path "$EPOCH_DIR" \
    --ckpt_path outputs/baselines/moa/diet/condition_classifier.pth \
    --out_json "$MOA_JSON" \
    2>&1 | tee "$EVAL_DIR/moa.log"

  if [ -f "$MOA_JSON" ]; then
    MOA_ACC=$(python3 -c "import json; d=json.load(open('$MOA_JSON')); print(d['moa_acc'])" 2>/dev/null || echo "NA")
    MOA_FASTED=$(python3 -c "import json; d=json.load(open('$MOA_JSON')); print(d['per_class']['fasted']['acc'])" 2>/dev/null || echo "NA")
    MOA_HFD=$(python3 -c "import json; d=json.load(open('$MOA_JSON')); print(d['per_class']['hfd']['acc'])" 2>/dev/null || echo "NA")
  else
    MOA_ACC="NA"; MOA_FASTED="NA"; MOA_HFD="NA"
  fi

  # Write CSV row
  echo "$CFG,$FID_VAL,$GAP_FASTED_CALR,$GAP_FASTED_PERI,$GAP_FASTED_TOMM,$GAP_HFD_CALR,$GAP_HFD_PERI,$GAP_HFD_TOMM,$MOA_ACC,$MOA_FASTED,$MOA_HFD" >> "$RESULTS_FILE"

  echo ""
  echo "  CFG=$CFG done: FID=$FID_VAL MoA=$MOA_ACC"
  echo "  gap_closed fasted: Calr=$GAP_FASTED_CALR Peri=$GAP_FASTED_PERI TOMM=$GAP_FASTED_TOMM"
  echo "  gap_closed hfd:    Calr=$GAP_HFD_CALR Peri=$GAP_HFD_PERI TOMM=$GAP_HFD_TOMM"
done

echo ""
echo "============================================================"
echo "  CFG sweep complete. Results: $RESULTS_FILE"
echo "============================================================"
cat "$RESULTS_FILE" | column -t -s ','

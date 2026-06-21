#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${BENCHMARK:-diet_v3}"
EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-16}"
VAL_BATCH="${VAL_BATCH:-8}"
# Keep one Lightning process. IMPA wraps its submodules in nn.DataParallel, so
# CUDA_VISIBLE_DEVICES=0,1 lets the model use both cards without DDP nesting.
DEVICES="${DEVICES:-1}"

case "$BENCHMARK" in
  diet_v3)
    CONFIG="configs/diet_id_v3.yaml"
    OUT="outputs/baselines/impa/diet_v3"
    ;;
  crispr_v8)
    CONFIG="configs/perturbmulti_train_id.yaml"
    OUT="outputs/baselines/impa/crispr_v8"
    ;;
  *)
    echo "Unknown BENCHMARK=$BENCHMARK; expected diet_v3 or crispr_v8" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/outputs/baselines/_data/$BENCHMARK"

if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  echo "Missing $DATA_DIR/manifest.json; run: bash baselines/export_all_baseline_data.sh" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python "$REPO_ROOT/baselines/impa_train.py" \
  --config "$CONFIG" \
  --data-dir "outputs/baselines/_data/$BENCHMARK" \
  --output "$OUT" \
  --benchmark "$BENCHMARK" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --val-batch-size "$VAL_BATCH" \
  --devices "$DEVICES"

printf -v STEP "%06d" "$EPOCHS"
CHECKPOINT="$REPO_ROOT/$OUT/external_checkpoints/run/checkpoint/${STEP}_nets.ckpt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing IMPA checkpoint $CHECKPOINT" >&2
  exit 2
fi

MAX_SAMPLES_ARG=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARG=(--max-samples "$MAX_SAMPLES")
fi

python "$REPO_ROOT/baselines/impa_export_fid.py" \
  --config "$CONFIG" \
  --data-dir "outputs/baselines/_data/$BENCHMARK" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT" \
  --benchmark "$BENCHMARK" \
  "${MAX_SAMPLES_ARG[@]}"

python "$REPO_ROOT/scripts/aggregate_eval.py" "$REPO_ROOT/$OUT" 5 0

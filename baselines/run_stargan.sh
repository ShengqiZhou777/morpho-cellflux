#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${BENCHMARK:-diet}"
NUM_ITERS="${NUM_ITERS:-50000}"
BATCH="${BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_STEP="${SAVE_STEP:-10000}"
SAMPLE_STEP="${SAMPLE_STEP:-$((NUM_ITERS / 5))}"
RESUME_ITERS="${RESUME_ITERS:-}"

case "$BENCHMARK" in
  diet)
    CONFIG="configs/diet_id.yaml"
    OUT="outputs/baselines/stargan/diet"
    ;;
  crispr_paper)
    CONFIG="configs/crispr_paper_core.yaml"
    OUT="outputs/baselines/stargan/crispr_paper"
    ;;
  *)
    echo "Unknown BENCHMARK=$BENCHMARK; expected diet or crispr_paper" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR_OVERRIDE:-$REPO_ROOT/outputs/baselines/_data/$BENCHMARK}"
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  case "$BENCHMARK" in
    diet) LEGACY_DATA_DIR="$REPO_ROOT/outputs/baselines/_data/diet_v3" ;;
  esac
  if [[ -n "${LEGACY_DATA_DIR:-}" && -f "$LEGACY_DATA_DIR/manifest.json" ]]; then
    echo "Using legacy exported baseline data: $LEGACY_DATA_DIR"
    DATA_DIR="$LEGACY_DATA_DIR"
  fi
fi
STARGAN_ROOT="$REPO_ROOT/baselines/external/stargan"
RUN_ROOT="$REPO_ROOT/$OUT/external_checkpoints"

if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  echo "Missing $DATA_DIR/manifest.json; run: bash baselines/export_all_baseline_data.sh" >&2
  exit 2
fi

C_DIM="$(python - "$DATA_DIR/phendiff_class_to_idx.json" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1]))))
PY
)"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/models" "$RUN_ROOT/samples" "$RUN_ROOT/results"

RESUME_ARGS=()
if [[ -n "$RESUME_ITERS" ]]; then
  RESUME_ARGS=(--resume_iters "$RESUME_ITERS")
fi

(
  cd "$STARGAN_ROOT"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main.py \
    --dataset RaFD \
    --rafd_image_dir "$DATA_DIR/imagefolder/train" \
    --c_dim "$C_DIM" \
    --image_size 128 \
    --rafd_crop_size 128 \
    --batch_size "$BATCH" \
    --num_workers "$NUM_WORKERS" \
    --num_iters "$NUM_ITERS" \
    --num_iters_decay "$((NUM_ITERS / 2))" \
    --model_save_step "$SAVE_STEP" \
    --sample_step "$SAMPLE_STEP" \
    --log_dir "$RUN_ROOT/logs" \
    --model_save_dir "$RUN_ROOT/models" \
    --sample_dir "$RUN_ROOT/samples" \
    --result_dir "$RUN_ROOT/results" \
    --use_tensorboard false \
    "${RESUME_ARGS[@]}"
)

CHECKPOINT="$RUN_ROOT/models/${NUM_ITERS}-G.ckpt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing StarGAN checkpoint $CHECKPOINT" >&2
  exit 2
fi

MAX_SAMPLES_ARG=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARG=(--max-samples "$MAX_SAMPLES")
fi

python "$REPO_ROOT/baselines/stargan_export_fid.py" \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT" \
  --benchmark "$BENCHMARK" \
  "${MAX_SAMPLES_ARG[@]}"

python "$REPO_ROOT/scripts/aggregate_eval.py" "$REPO_ROOT/$OUT" 5 0

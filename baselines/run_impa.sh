#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${BENCHMARK:-diet}"
EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-16}"
VAL_BATCH="${VAL_BATCH:-8}"
# Keep one Lightning process. IMPA wraps its submodules in nn.DataParallel, so
# CUDA_VISIBLE_DEVICES=0,1 lets the model use both cards without DDP nesting.
DEVICES="${DEVICES:-1}"

case "$BENCHMARK" in
  diet)
    CONFIG="configs/diet_id.yaml"
    OUT="outputs/baselines/impa/diet"
    ;;
  crispr_paper)
    CONFIG="configs/crispr_paper_core.yaml"
    OUT="outputs/baselines/impa/crispr_paper"
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

if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  echo "Missing $DATA_DIR/manifest.json; run: bash baselines/export_all_baseline_data.sh" >&2
  exit 2
fi

python - <<'PY'
from pathlib import Path
import os
import torch
from torch.hub import load_state_dict_from_url

url = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"
name = url.rsplit("/", 1)[-1]
torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
cache_dir = torch_home / "hub" / "checkpoints"
cache_dir.mkdir(parents=True, exist_ok=True)
cache_path = cache_dir / name


def valid(path: Path) -> bool:
    try:
        state = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"IMPA FID cache invalid: {path} ({exc})")
        return False
    return isinstance(state, dict) and len(state) > 0


if cache_path.exists() and not valid(cache_path):
    cache_path.unlink()
if not cache_path.exists():
    print(f"Downloading IMPA FID Inception weights to {cache_path}")
    load_state_dict_from_url(url, progress=True)
if not valid(cache_path):
    raise SystemExit(f"IMPA FID cache is still invalid after refresh: {cache_path}")
print(f"IMPA FID cache ok: {cache_path}")
PY

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python "$REPO_ROOT/baselines/impa_train.py" \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
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
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT" \
  --benchmark "$BENCHMARK" \
  "${MAX_SAMPLES_ARG[@]}"

python "$REPO_ROOT/scripts/aggregate_eval.py" "$REPO_ROOT/$OUT" 5 0

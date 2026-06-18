#!/usr/bin/env bash
# Generate interpolation figures from a trained perturbmulti checkpoint. For each cell
# it writes a grid: real control, the ODE trajectory from t=0 to t=1, the generated
# image, and the real perturbed cell alongside for reference.
#
# This is a qualitative figure. It shows generation realism and a smooth morphological
# trajectory, but it is not a correctness proof on its own, because the model's own ODE
# path is always smooth even when it barely moves off the source. Pair it with the
# quantitative Delta-direction from scripts/aggregate_eval.py.
#
# How it works: with --interpolate the solver returns intermediates and writes one grid
# per cell in the first eval batch, then stops (no FID). So:
#   - BATCH is the number of cells visualized.
#   - The genes shown are whatever lands in the first test batch. To feature specific
#     genes, point CONFIG at an index whose treated rows are only those genes, plus the
#     control pool needed for control-initialized pairing.
#   - --edm_schedule is required for a real trajectory. Without it the time grid is just
#     the two endpoints. The number of trajectory frames is set by NFE.
#
# Run on the best checkpoint (peak Delta-direction). Single GPU by default.
#
# Usage:
#   CKPT=outputs/my_run/checkpoint-19.pth OUT=outputs/my_run GPU=0 \
#     bash scripts/interpolate_cellflux.sh
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TORCHRUN=${TORCHRUN:-$(command -v torchrun || echo /home/ubuntu/miniconda3/envs/pmf/bin/torchrun)}

CKPT=${CKPT:?set CKPT=<path to a checkpoint-*.pth>}
OUT=${OUT:?set OUT=<output dir; interpolation/ is written under it>}
CONFIG=${CONFIG:-perturbmulti_stronghits_id}  # config name under configs/.
DATASET=${DATASET:-perturbmulti_id}
USE_INITIAL=${USE_INITIAL:-2}     # 2 starts the trajectory at a real control cell.
NOISE_LEVEL=${NOISE_LEVEL:-0.2}
CFG=${CFG:-1.0}
NFE=${NFE:-10}                    # number of trajectory frames (edm time discretization).
BATCH=${BATCH:-8}                 # number of cells visualized, one grid per cell.
GPU=${GPU:-0}                     # single GPU; leaves the others free for any running job.
EPOCHS=${EPOCHS:-1000000}         # large sentinel: resume sets start_epoch=ckpt_epoch+1 and
                                  # eval_only stops after one iteration, so the range just needs to be non-empty.

mkdir -p "$OUT"
cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TORCHRUN" --standalone --nproc_per_node=1 -m morphoflux.engine.train \
  --dataset "$DATASET" --config "$CONFIG" --device cuda \
  --eval_only --resume "$CKPT" --use_ema \
  --use_initial "$USE_INITIAL" --noise_level "$NOISE_LEVEL" --cfg_scale "$CFG" \
  --interpolate --edm_schedule --ode_options "{\"nfe\": $NFE}" \
  --batch_size "$BATCH" --fid_samples "$BATCH" --epochs "$EPOCHS" \
  --output_dir "$OUT" 2>&1 | tee -a "$OUT/interpolate_stdout.log"
echo "interpolation grids -> $OUT/interpolation/"

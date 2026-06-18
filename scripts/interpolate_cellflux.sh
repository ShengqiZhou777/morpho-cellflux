#!/usr/bin/env bash
# Generate CellFlux-style interpolation figures from a trained perturbmulti checkpoint:
# per cell, a grid [Real Ctrl] -> [ODE trajectory t=0..1] -> [generated], with the real KO
# target shown alongside for reference (CellFlux training/eval_loop.py:save_interpolation_grid).
#
# WHAT THIS FIGURE IS (and is NOT):
#   It is the QUALITATIVE / paper-style figure -- it shows generation realism and the smooth
#   morphological trajectory. It is NOT a correctness proof on its own: the model's own ODE
#   path is always smooth, even when it barely moves off the source. ALWAYS pair it with the
#   quantitative aggregate Delta-direction from scripts/aggregate_eval.py. (This is exactly why
#   the per-cell source|gen|target montage was dropped: comparing one generated cell to one
#   RANDOM real KO cell is ill-posed for unpaired distributions.)
#
# HOW IT WORKS (eval_loop.py): with --interpolate the solver runs return_intermediates=True
# and save_interpolation_grid() writes one grid per cell IN THE FIRST EVAL BATCH, then returns
# (no FID). So:
#   - batch_size = number of cells visualized.
#   - the genes shown = whatever lands in the first test batch. To FEATURE specific strong
#     genes (e.g. Pten / Eif2s1 for Perilipin), point CONFIG at a small index whose treated
#     rows are only those genes (+ the control pool, needed for use_initial=2 ctrl pairing).
#   - --edm_schedule is REQUIRED for a real trajectory: without it time_grid=[0,1] and you get
#     only the two endpoints. Number of trajectory frames = ode_options "nfe".
#
# Run on the BEST checkpoint (peak Delta-direction, before any regression), NOT an early one.
# Single GPU by default so it won't disturb a running DDP training job.
#
# Usage:
#   CKPT=/.../cellflux_pm_geneid_baseline_v6/checkpoint-19.pth \
#   OUT=/.../cellflux_pm_geneid_baseline_v6 GPU=0 bash scripts/interpolate_cellflux.sh
set -euo pipefail

PROJECT_DIR=/home/ubuntu/data/sqzhou/projects/morpho-cellflux
TORCHRUN=/home/ubuntu/miniconda3/envs/pmf/bin/torchrun

CKPT=${CKPT:?set CKPT=<path to a checkpoint-*.pth>}
OUT=${OUT:?set OUT=<output dir; interpolation/ is written under it>}
CONFIG=${CONFIG:-perturbmulti_stronghits_id}  # point at a strong-gene index to feature specific genes
DATASET=${DATASET:-perturbmulti_id}
USE_INITIAL=${USE_INITIAL:-2}     # 2 = trajectory starts at a real control cell (matches v6 training)
NOISE_LEVEL=${NOISE_LEVEL:-0.2}
CFG=${CFG:-1.0}
NFE=${NFE:-10}                    # trajectory frames (edm time discretization)
BATCH=${BATCH:-8}                 # #cells visualized (one grid per cell in the first batch)
GPU=${GPU:-0}                     # single GPU; leave the other free for a running train job
EPOCHS=${EPOCHS:-1000000}         # huge: range(start_epoch, EPOCHS) must be non-empty -- resume sets
                                  # start_epoch=ckpt_epoch+1; eval_only breaks after one iteration.

mkdir -p "$OUT"
cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CELLFLUX_CONFIG_DIR="$PROJECT_DIR/configs/cellflux" \
"$TORCHRUN" --standalone --nproc_per_node=1 -m morphoflux.engine.train \
  --dataset "$DATASET" --config "$CONFIG" --device cuda \
  --eval_only --resume "$CKPT" --use_ema \
  --use_initial "$USE_INITIAL" --noise_level "$NOISE_LEVEL" --cfg_scale "$CFG" \
  --interpolate --edm_schedule --ode_options "{\"nfe\": $NFE}" \
  --batch_size "$BATCH" --fid_samples "$BATCH" --epochs "$EPOCHS" \
  --output_dir "$OUT" 2>&1 | tee -a "$OUT/interpolate_stdout.log"
echo "interpolation grids -> $OUT/interpolation/"

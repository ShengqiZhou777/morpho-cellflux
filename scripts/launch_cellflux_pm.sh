#!/usr/bin/env bash
# Launch the perturbmulti (Perturb-Multi hepatocyte lipid panel) CellFlux training on N GPUs.
# Persists per-step stdout to $OUT/train_stdout.log -- the harness /tmp task log is ephemeral.
#
# Override defaults via env vars, e.g.:
#   OUT=.../outputs/cellflux_pm_lipid_ddp_v2 BATCH=16 ACCUM=2 EPOCHS=60 bash scripts/launch_cellflux_pm.sh
set -euo pipefail

PROJECT_DIR=/home/ubuntu/data/sqzhou/projects/morpho-cellflux
TORCHRUN=/home/ubuntu/miniconda3/envs/pmf/bin/torchrun

OUT=${OUT:-/home/ubuntu/data/sqzhou/projects/morpho-cellflux/outputs/cellflux_pm_lipid_ddp_v1}
BATCH=${BATCH:-16}          # per-GPU batch size (16 -> ~25GB, safe through FID eval on 32GB)
ACCUM=${ACCUM:-1}           # grad accumulation: effective batch = BATCH * ACCUM * NPROC (no extra memory)
EPOCHS=${EPOCHS:-40}
EVAL_FREQ=${EVAL_FREQ:-10}
FID_SAMPLES=${FID_SAMPLES:-1024}
NPROC=${NPROC:-2}
USE_INITIAL=${USE_INITIAL:-1}   # 0=noise->target (generative, cannot copy source), 1=control init, 2=control+noise
NOISE_LEVEL=${NOISE_LEVEL:-0.2} # noise added to control when USE_INITIAL=2
CFG=${CFG:-0.2}                 # classifier-free guidance scale at sampling
CONFIG=${CONFIG:-perturbmulti_stronghits_id}  # CellFlux config (configs/<CONFIG>.yaml) -> data index + embedding
DATASET=${DATASET:-perturbmulti_id}  # model arch: perturbmulti_id (condition_dim 204 = gene identity one-hot)

mkdir -p "$OUT"
cd "$PROJECT_DIR"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CELLFLUX_CONFIG_DIR="$PROJECT_DIR/configs/cellflux" \
"$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m morphoflux.engine.train \
  --dataset "$DATASET" --config "$CONFIG" --device cuda \
  --batch_size "$BATCH" --accum_iter "$ACCUM" --num_workers 10 --epochs "$EPOCHS" \
  --use_initial "$USE_INITIAL" --noise_level "$NOISE_LEVEL" --use_ema --skewed_timesteps \
  --class_drop_prob 0.2 --cfg_scale "$CFG" \
  --eval_frequency "$EVAL_FREQ" --compute_fid --fid_samples "$FID_SAMPLES" \
  --ode_options '{"step_size": 0.02}' --save_fid_samples \
  --output_dir "$OUT" 2>&1 | tee -a "$OUT/train_stdout.log"

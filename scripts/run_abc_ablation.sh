#!/usr/bin/env bash
# Sequential A/B/C collapse-ablation driver (single 4090).
# A: noise start, no GAN | B: control start + GAN | C: noise start + GAN
set -uo pipefail
cd /home/shockley/myproject/PhenoFlux/morpho-cellflux

ODE='{"step_size": 0.05}'

run_arm () {
  local name=$1 ui=$2 gw=$3
  echo "=== ARM $name : use_initial=$ui gan_weight=$gw start=$(date '+%H:%M:%S') ==="
  rm -rf "outputs/runs/microalgae/ablate/$name"
  OUT="$PWD/outputs/runs/microalgae/ablate/$name" \
    USE_INITIAL="$ui" GAN_WEIGHT="$gw" \
    EPOCHS=5 EVAL_FREQ=5 FID_SAMPLES=256 BATCH=8 ODE_OPTIONS="$ODE" \
    conda run -n pmf bash scripts/ablate_collapse.sh
  echo "=== ARM $name DONE end=$(date '+%H:%M:%S') ==="
}

run_arm A 0 0.0
run_arm B 1 0.1
run_arm C 0 0.1
echo "ALL_ARMS_COMPLETE $(date '+%H:%M:%S')"

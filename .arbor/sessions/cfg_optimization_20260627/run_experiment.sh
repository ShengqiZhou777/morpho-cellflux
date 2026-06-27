#!/bin/bash
# Arbor Executor: Run a single PhenoFlux experiment with given CFG scale & class_drop_prob.
# Usage: bash run_experiment.sh <node_id> <cfg_scale> <class_drop_prob>
set -euo pipefail

NODE_ID="${1:?}"
CFG="${2:?}"
DROP="${3:?}"

SESSION_DIR="/home/shockley/myproject/PhenoFlux/morpho-cellflux/.arbor/sessions/cfg_optimization_20260627"
OUT_DIR="${SESSION_DIR}/experiments/${NODE_ID}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/shockley/miniconda3/etc/profile.d/conda.sh
conda activate /data/conda_envs/pmf

echo "=== Experiment ${NODE_ID}: CFG=${CFG}, class_drop_prob=${DROP} ==="
torchrun --standalone --nproc_per_node=1 -m phenoflux.train \
    --dataset phenoflux --config phenoflux_diet_msa_pcd --device cuda \
    --batch_size 2 --num_workers 2 --epochs 2 \
    --use_initial 1 --cfg_scale "${CFG}" \
    --skewed_timesteps --class_drop_prob "${DROP}" \
    --eval_frequency 1 --fid_samples 64 \
    --compute_fid --save_fid_samples \
    --data_index data/processed/diet/index_diet_2k.csv \
    --output_dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/train.log"

echo "=== Training done, running aggregate eval ==="
EPOCH_DIR=$(find "$OUT_DIR/fid_samples" -maxdepth 1 -type d -name "epoch-*" 2>/dev/null | sort | tail -1)
if [[ -n "$EPOCH_DIR" ]]; then
    EPOCH_NUM=$(basename "$EPOCH_DIR" | sed 's/epoch-//')
    cd /home/shockley/myproject/PhenoFlux/morpho-cellflux
    python scripts/aggregate_eval.py "$OUT_DIR" 5 "$EPOCH_NUM" 2>&1 | tee "$OUT_DIR/aggregate_eval.log"
else
    echo "ERROR: No generated images found!"
    exit 1
fi

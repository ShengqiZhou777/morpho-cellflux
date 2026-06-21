#!/usr/bin/env bash
# v9 (cellflux_pm_train_id_v9) = CRISPR with RNA-signature conditioning (concat-413), the
# apples-to-apples counterpart to the one-hot v8: SAME channels [0,14,5] + SAME rna_snr-filtered
# 76-gene index, only the condition differs (embedding_gene_idsig.csv, arch perturbmulti_idsig, condition_dim 413).
#
# Launch AFTER v8 finishes (do not auto-chain into the running run_new_panels.sh). Verify the
# GPUs are free first (`nvidia-smi`), then:  setsid bash scripts/run_idsig.sh >/dev/null 2>&1 &
# Compare aggregate_eval gap_closed: v8 (one-hot) vs this (one-hot + RNA signature).
set -uo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-morpho-cellflux}"
if [[ -z "${CONDA_SH:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Set CONDA_SH=/path/to/conda.sh or activate the environment manually." >&2
    exit 127
  fi
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
cd "$PROJECT"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
OUT=outputs/cellflux_pm_train_id_v9 CONFIG=perturbmulti_train_idsig DATASET=perturbmulti_idsig \
  USE_INITIAL=1 CFG=0.2 EPOCHS=20 EVAL_FREQ=5 NPROC=2 BATCH=16 bash scripts/train.sh

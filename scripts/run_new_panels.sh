#!/usr/bin/env bash
# Sequentially retrain the two NEW-PANEL runs on both GPUs, detached so it survives the
# Claude session. Launch with:   setsid bash scripts/run_new_panels.sh >/dev/null 2>&1 &
# Progress -> outputs/run_new_panels.log (plus each run's own $OUT/train_stdout.log).
#
# Per-dataset panels (evidence: 18-channel effect scan, this session):
#   diet-v3   channels [9,5,8]  = Calreticulin / Perilipin / TOMM20  (diet-responsive)
#   CRISPR-v8 channels [0,14,5] = Alb / Rab7 / Perilipin             (CRISPR-responsive)
#             + rna_snr-filtered index (76 genes with confirmed knockdown)
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
LOG="$PROJECT/outputs/run_new_panels.log"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

echo "[$(date -Is)] START diet-v3 (channels [9,5,8])" >> "$LOG"
OUT=outputs/diet_id_v3 CONFIG=diet_id_v3 DATASET=diet_id USE_INITIAL=1 CFG=0.2 \
  EPOCHS=12 EVAL_FREQ=2 NPROC=2 BATCH=16 bash scripts/train.sh >> "$LOG" 2>&1
echo "[$(date -Is)] diet-v3 exited code $?" >> "$LOG"

echo "[$(date -Is)] START CRISPR-v8 (channels [0,14,5]; rna_snr-filtered index)" >> "$LOG"
OUT=outputs/cellflux_pm_train_id_v8 CONFIG=perturbmulti_train_id DATASET=perturbmulti_id USE_INITIAL=1 CFG=0.2 \
  EPOCHS=20 EVAL_FREQ=5 NPROC=2 BATCH=16 bash scripts/train.sh >> "$LOG" 2>&1
echo "[$(date -Is)] CRISPR-v8 exited code $?" >> "$LOG"
echo "[$(date -Is)] ALL DONE" >> "$LOG"

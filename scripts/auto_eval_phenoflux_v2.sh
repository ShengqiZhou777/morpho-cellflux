#!/bin/bash
# Auto eval script for PhenoFlux v2
# Runs: CFG=1.5 eval → gap_closed (both CFG) → MoA (both CFG)
set -euo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pmf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="outputs/runs/diet/phenoflux_diet_v2"
EPOCHS=(19 20)  # checkpoints to eval
CFG_VALS=(1.0 1.5)
TIMESTAMP=$(TZ='Asia/Shanghai' date '+%Y%m%d_%H%M%S')
LOG_DIR="${RUN_DIR}/eval_logs"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "Auto Eval: PhenoFlux v2"
echo "Start: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

for EPOCH in "${EPOCHS[@]}"; do
    CKPT="${RUN_DIR}/checkpoint-${EPOCH}.pth"
    if [ ! -f "$CKPT" ]; then
        echo "SKIP epoch ${EPOCH}: checkpoint not found at $CKPT"
        continue
    fi

    for CFG in "${CFG_VALS[@]}"; do
        # Skip CFG=1.0 for epoch 19 — already done during training
        if [ "$EPOCH" = "19" ] && [ "$CFG" = "1.0" ]; then
            echo "SKIP CFG=${CFG} epoch=${EPOCH}: already done during training"
            continue
        fi

        EVAL_DIR="${RUN_DIR}/eval_cfg${CFG}_ep${EPOCH}"
        if [ -d "${EVAL_DIR}/fid_samples" ]; then
            echo "SKIP CFG=${CFG} epoch=${EPOCH}: eval dir already exists"
        else
            echo "--- Running eval: CFG=${CFG} epoch=${EPOCH} ---"
            LOG="${LOG_DIR}/eval_cfg${CFG}_ep${EPOCH}_${TIMESTAMP}.log"
            torchrun --standalone --nproc_per_node=2 -m morphoflux.engine.train \
                --dataset phenoflux_diet --config phenoflux_diet --device cuda \
                --eval_only --resume "${CKPT}" --use_initial 1 --cfg_scale "${CFG}" \
                --use_ema \
                --fid_samples 1000 --compute_fid --save_fid_samples \
                --ode_options '{"step_size": 0.02}' \
                --output_dir "${EVAL_DIR}" \
                &> "${LOG}"
            echo "Done: CFG=${CFG} epoch=${EPOCH}"
        fi

        # gap_closed
        GAP_DIR="${EVAL_DIR}/gap_closed"
        if [ -d "${GAP_DIR}" ]; then
            echo "SKIP gap_closed for CFG=${CFG} epoch=${EPOCH}: already exists"
        else
            echo "--- gap_closed: CFG=${CFG} epoch=${EPOCH} ---"
            EPOCH_SUBDIR=$(ls -d "${EVAL_DIR}"/fid_samples/epoch-* 2>/dev/null | head -1)
            if [ -n "${EPOCH_SUBDIR}" ]; then
                EPOCH_NUM=$(basename "${EPOCH_SUBDIR}" | sed 's/epoch-//')
                python scripts/diet_marker_distribution_figure.py \
                    --run-dir "${EVAL_DIR}" \
                    --epoch "${EPOCH_NUM}" \
                    --out-dir "${GAP_DIR}" \
                    --prefix "phenoflux_v2_cfg${CFG}_ep${EPOCH}"
                echo "Done: gap_closed for CFG=${CFG} epoch=${EPOCH}"
            else
                echo "WARN: no epoch dir found in ${EVAL_DIR}/fid_samples/"
            fi
        fi

        # MoA
        MOA_OUT="${EVAL_DIR}/moa_condition.json"
        if [ -f "${MOA_OUT}" ]; then
            echo "SKIP MoA for CFG=${CFG} epoch=${EPOCH}: already exists"
        else
            echo "--- MoA: CFG=${CFG} epoch=${EPOCH} ---"
            EPOCH_SUBDIR=$(ls -d "${EVAL_DIR}"/fid_samples/epoch-* 2>/dev/null | head -1)
            if [ -n "${EPOCH_SUBDIR}" ]; then
                python src/morphoflux/engine/moa/train_moa.py \
                    --config_path configs/diet_id.yaml \
                    --mode eval \
                    --img_root_path "${EPOCH_SUBDIR}" \
                    --ckpt_path outputs/baselines/moa/diet/condition_classifier.pth \
                    --out_json "${MOA_OUT}"
                echo "Done: MoA for CFG=${CFG} epoch=${EPOCH}"
            fi
        fi
    done
done

echo "=============================================="
echo "All evals complete: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# Print summary
echo ""
echo "=== SUMMARY ==="
for EPOCH in "${EPOCHS[@]}"; do
    for CFG in "${CFG_VALS[@]}"; do
        if [ "$EPOCH" = "19" ] && [ "$CFG" = "1.0" ]; then
            EVAL_DIR="${RUN_DIR}"  # training-run eval (CFG=1.0, epoch 19)
        else
            EVAL_DIR="${RUN_DIR}/eval_cfg${CFG}_ep${EPOCH}"
        fi
        echo "--- CFG=${CFG} epoch=${EPOCH} ---"
        # FID
        FID_FILE=$(ls "${EVAL_DIR}"/fid_samples/epoch-*/fid.json 2>/dev/null | head -1)
        if [ -n "${FID_FILE}" ]; then
            echo "  FID: $(cat "${FID_FILE}")"
        fi
        # gap_closed
        GAP_CSV=$(ls "${EVAL_DIR}"/gap_closed/*gap_closed.csv 2>/dev/null | head -1)
        if [ -n "${GAP_CSV}" ]; then
            echo "  gap_closed:"
            cat "${GAP_CSV}"
        fi
        # MoA
        MOA_OUT="${EVAL_DIR}/moa_condition.json"
        if [ -f "${MOA_OUT}" ]; then
            echo "  MoA: $(cat "${MOA_OUT}")"
        fi
    done
done

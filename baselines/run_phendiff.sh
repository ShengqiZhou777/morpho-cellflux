#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${BENCHMARK:-diet}"
EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-16}"
EVAL_BATCH="${EVAL_BATCH:-16}"
NB_GENERATED="${NB_GENERATED:-256}"
NPROC="${NPROC:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GUIDANCE="${GUIDANCE:-1.5}"
FRAC_DIFFUSION_SKIPPED="${FRAC_DIFFUSION_SKIPPED:-0.55}"

case "$BENCHMARK" in
  diet)
    CONFIG="configs/diet_id.yaml"
    OUT="outputs/baselines/phendiff/diet"
    ;;
  crispr_paper)
    CONFIG="configs/crispr_paper_core.yaml"
    OUT="outputs/baselines/phendiff/crispr_paper"
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
RUN_PARENT="$REPO_ROOT/$OUT/external_checkpoints"
PHENDIFF_ROOT="$REPO_ROOT/baselines/external/phendiff"
PIPELINE="$RUN_PARENT/phendiff/$BENCHMARK/full_pipeline_save"

if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  echo "Missing $DATA_DIR/manifest.json; run: bash baselines/export_all_baseline_data.sh" >&2
  exit 2
fi

mkdir -p "$RUN_PARENT"

# train.py forces a per-run torch.hub dir: torch.hub.set_dir("$RUN_PARENT/.torch_hub_cache").
# torch-fidelity then re-downloads its InceptionV3 weights from GitHub at the first
# eval, which has hung (stalled TLS connect, no timeout). Pre-seed that file from the
# warm global cache so NO network download happens.
HUB_CKPT_DIR="$RUN_PARENT/.torch_hub_cache/checkpoints"
mkdir -p "$HUB_CKPT_DIR"
for w in weights-inception-2015-12-05-6726825d.pth inception_v3_google-0cc3c7bd.pth; do
  src="$HOME/.cache/torch/hub/checkpoints/$w"
  [[ -f "$src" ]] && ln -sf "$src" "$HUB_CKPT_DIR/$w"
done

(
  cd "$PHENDIFF_ROOT"
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" accelerate launch --num_processes "$NPROC" --main_process_port "${MAIN_PROCESS_PORT:-29500}" train.py \
    --model_type DDIM \
    --components_to_train denoiser \
    --train_data_dir "$DATA_DIR/imagefolder" \
    --split train \
    --exp_output_dirs_parent_folder "$RUN_PARENT" \
    --wandb_entity local \
    --experiment_name phendiff \
    --run_name "$BENCHMARK" \
    --definition 128 \
    --train_batch_size "$BATCH" \
    --eval_batch_size "$EVAL_BATCH" \
    --dataloader_num_workers 6 \
    --dataloader_prefetch_factor 2 \
    --max_num_epochs "$EPOCHS" \
    --eval_save_model_every_epochs 1 \
    --nb_generated_images "$NB_GENERATED" \
    --kid_subset_size 128 \
    --guidance_factor "$GUIDANCE" \
    --proba_uncond 0.1 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 200 \
    --denoiser_config_path "$DATA_DIR/phendiff_denoiser_config.json" \
    --noise_scheduler_config_path "$PHENDIFF_ROOT/models_configs/noise_scheduler/1k_epsilon_pred.json" \
    --num_train_timesteps 1000 \
    --num_inference_steps 50 \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 2 \
    --mixed_precision "$MIXED_PRECISION"
)

MAX_SAMPLES_ARG=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARG=(--max-samples "$MAX_SAMPLES")
fi

python "$REPO_ROOT/baselines/phendiff_export_fid.py" \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --checkpoint "$PIPELINE" \
  --output "$OUT" \
  --benchmark "$BENCHMARK" \
  --guidance "$GUIDANCE" \
  --frac-diffusion-skipped "$FRAC_DIFFUSION_SKIPPED" \
  "${MAX_SAMPLES_ARG[@]}"

python "$REPO_ROOT/scripts/aggregate_eval.py" "$REPO_ROOT/$OUT" 5 0

# Morpho-CellFlux / PhenoFlux

Marker-aware flow matching for molecular phenotype transport on MERFISH hepatocyte data.

## Project Structure

```
src/morphoflux/engine/
  train.py                    # Entry point
  models/
    unet.py                   # UNetModel with MAC + CCM modules
    mac.py                    # MarkerProfileEncoder, MACAttentionBlock, CCM
    model_configs.py          # MODEL_CONFIGS registry
  training/
    train_loop.py             # Training loop (foreground_loss, marker_profile, CFG)
    eval_loop.py              # Eval loop (FID, image generation, CFGScaledModel)
    dataloader.py             # CellDataset, CellDataLoader
    data_utils.py             # read_files_pert, _load_perturbmulti
configs/                       # YAML configs per experiment
scripts/                       # train.sh, interpolate.sh, sweep_diet_moa.sh, etc.
outputs/runs/diet/             # Training outputs
outputs/sweeps/                # Evaluation outputs
docs/                          # ARCHITECTURE, EVAL_PROTOCOL, REPRODUCING, RESULTS
```

## Key Concepts

### PhenoFlux Architecture (config: `phenoflux_diet`)
- **MAC** (Marker-Aware Conditioning): encodes control cell's 18ch MERFISH profile → tokens; UNet bottleneck attends via cross-attention
- **CCM** (Channel-wise Condition Modulation): per-channel FiLM on output from pooled marker tokens
- Standard UNet body with `condition_dim=3` (diet one-hot: adlib/fasted/hfd)

### Data Flow
1. `data_utils.py:read_files_pert` pairs control+treated cells from same batch
2. `use_initial=1` → ODE starts from control image (not noise)
3. `marker_profile` = `full_ctrl` (18ch control cell profile, for MAC)
4. `concat_conditioning` = 3-dim diet one-hot embedding
5. Flow matching: model learns velocity field from control→target

### Diet Dataset
- 3 conditions: adlib (control), fasted, hfd
- Image panel: channels [9,5,8] = Calreticulin / Perilipin / TOMM20 (3ch, false-color)
- 18ch full profile available for MAC conditioning

## Critical Design Rules (DO NOT BREAK)

1. **marker_profile MUST be `full_ctrl`**, not `full_trt`. MAC conditions on SOURCE molecular state so model learns control→target transport (not target auto-encoding). This was Bug #1 that wasted ~13 GPU-hours.

2. **marker_profile MUST NOT leak into unconditional path**. When `class_drop_prob` triggers CFG dropout (`conditioning={}`), marker_profile must also be absent. Otherwise unconditional model cheats with molecular info, killing CFG. This was Bug #2.

3. **`find_unused_parameters=True` is REQUIRED** for DDP. MAC/CCM params are unused when class_drop triggers (no marker_profile in forward). The DDP warning is expected, not a bug.

4. **EMA unwrapping needed** before checking `use_mac`/`use_ccm`. Model can be `EMA(UNetModel)` — use `getattr(model, 'model', model)` to unwrap.

5. **Every epoch saves checkpoint** (`train.py` modified). Eval frequency is separate. Training can be paused/resumed at any epoch boundary.

## Training Commands

### PhenoFlux (main contribution)
```bash
torchrun --standalone --nproc_per_node=2 -m morphoflux.engine.train \
  --dataset phenoflux_diet --config phenoflux_diet --device cuda \
  --batch_size 20 --epochs 20 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 2 --fid_samples 1000 --compute_fid --save_fid_samples \
  --foreground_loss --foreground_threshold 0.05 --foreground_weight 5.0 --background_weight 0.1 \
  --output_dir outputs/runs/diet/phenoflux_diet_v2
```

### Baseline UNet (diet_id)
```bash
CONFIG=diet_id DATASET=diet_id bash scripts/train.sh
```

## Evaluation

### Primary metric: gap_closed
```bash
# 1. Generate images from checkpoint
torchrun --standalone --nproc_per_node=2 -m morphoflux.engine.train \
  --dataset phenoflux_diet --config phenoflux_diet --device cuda \
  --eval_only --resume <checkpoint.pth> --use_initial 1 --cfg_scale X.X \
  --fid_samples 1000 --compute_fid --save_fid_samples \
  --output_dir <eval_dir>

# 2. Marker distribution (gap_closed)
python scripts/diet_marker_distribution_figure.py \
  --run-dir <eval_dir> --epoch <N> --out-dir outputs/figures/phenoflux

# 3. MoA classifier
python src/morphoflux/engine/moa/train_moa.py \
  --config_path configs/diet_id.yaml --mode eval \
  --img_root_path <eval_dir>/fid_samples/epoch-<N> \
  --ckpt_path outputs/baselines/moa/diet/condition_classifier.pth \
  --out_json <eval_dir>/diet_condition_moa.json
```

### Quick validation protocol
After epoch 1: kill training → generate 64 images → run gap_closed → confirm >0 → resume

### Paper sample sizes
- EVAL_PROTOCOL: 5120 total, cap per-condition for fair comparison
- Quick validation: 64-128 per condition
- Paper result: 500+ per condition

## Time
- **Always use Beijing time (UTC+8 / Asia/Shanghai)**. Never UTC.
- Command: `TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S'`

## Environment
- Conda env: `pmf` (`/home/ubuntu/miniconda3/envs/pmf`)
- Python 3.10, PyTorch 2.11.0+cu128
- GPUs: 2× NVIDIA RTX 5090 (32GB each)
- NEVER set PYTHONNOUSERSITE — it breaks the baseline stack
- pmf env is ubuntu-owned, sqzhou has read-only; pip lands in ~/.local

## Key Files Modified (this session)
- `data_utils.py:150` — `full_trt` → `full_ctrl` (Bug #1 fix)
- `train_loop.py:116-123` — CFG guard for marker_profile (Bug #2 fix)
- `train_loop.py:83` — EMA unwrap for MAC detection
- `eval_loop.py:131` — EMA unwrap for MAC detection
- `load_and_save.py:55` — checkpoint key remap (`cross_attn_blocks` → `mac_blocks`)
- `grad_scaler.py:35` — `torch.cuda.amp` → `torch.amp` deprecation
- `distributed_mode.py:81` — `device_id` for init_process_group
- `eval_loop.py:96,102` — `torch.cuda.amp` → `torch.amp` deprecation
- `train.py:96` — `find_unused_parameters=True` (reverted from False)
- `train.py:159-164` — per-epoch checkpoint saving

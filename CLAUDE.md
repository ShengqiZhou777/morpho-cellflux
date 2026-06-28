# PhenoFlux

Flow matching with pluggable molecular priors for cellular phenotype transport.

## Project Structure

```
phenoflux/                    # Python package
├── train.py                  # Entry point (torchrun -m phenoflux.train)
├── args.py                   # Argument parser
├── models/
│   ├── configs.py            # MODEL_CONFIGS registry (1 entry: phenoflux)
│   ├── unet.py               # UNetModel with internal MSA/PCD
│   ├── msa.py                # Marker Self-Attention (marker co-regulation)
│   ├── pcd.py                # Per-Channel Decoder (per-channel modulation)
│   ├── ema.py                # Exponential Moving Average
│   ├── nn.py                 # NN utilities (SiLU, GroupNorm32, etc.)
│   └── discrete_unet.py      # Discrete UNet variant
├── training/
│   ├── train_loop.py         # Training loop (foreground_loss, CFG, marker_profile)
│   ├── eval_loop.py          # Eval loop (FID, CFGScaledModel, image generation)
│   ├── dataloader.py         # CellDataset, CellDataLoader
│   ├── data_utils.py         # read_files_pert, _load_perturbmulti
│   ├── data_transform.py     # Augmentation transforms
│   ├── distributed.py        # DDP helpers
│   ├── grad_scaler.py        # AMP gradient scaler
│   ├── load_save.py          # Checkpoint load/save
│   └── edm_time.py           # EDM time discretization
└── eval/
    ├── fid.py                # FIDo/c, KIDo/c (matched-N)
    ├── aggregate.py          # PGC (Phenotypic Gap Closure), dir-corr, sign-agreement, Pearson
    ├── moa.py                # MoA classifier (InceptionV3 + MLP)
    └── figures.py            # Marker distribution KDE + bar charts

configs/                      # 6 paper experiment configs
scripts/                      # train.sh, quick_validate.sh, build data scripts
baselines/                    # IMPA, PhenDiff, StarGAN, MorphoDiff adapters
outputs/                      # Training outputs (gitignored)
docs/                         # ARCHITECTURE, EVAL_PROTOCOL, REPRODUCING
```

## Architecture

One UNet body (`phenoflux`), configurable molecular prior via YAML flags:

```
                    ┌─────────────────────────┐
Condition (one-hot) │ base_condition_dim      │  3 (diet) / 40 (crispr)
                    ├─────────────────────────┤
Molecular prior     │ use_msa / use_pcd       │  MSA → PCD (both datasets)
                    │ use_marker_profile      │  Info control: naive 18ch concat
                    ├─────────────────────────┤
condition_dim       │ auto-computed           │  base + 64 (MSA) or +18 (naive)
                    └─────────────────────────┘
```

### Paper Configs (7)

| Config | Dataset | Prior | condition_dim | Proves |
|--------|---------|-------|:---:|--------|
| `phenoflux_diet` | Diet | none | 3 | Baseline |
| `phenoflux_diet_18ch` | Diet | naive 18ch concat | 21 | Raw marker info helps |
| `phenoflux_diet_msa` | Diet | MSA | 67 | Learned attention > naive |
| `phenoflux_diet_msa_pcd` | Diet | MSA+PCD | 67 | Per-channel modulation helps |
| `phenoflux_crispr` | CRISPR | none | 40 | Baseline |
| `phenoflux_crispr_msa` | CRISPR | MSA | 104 | Marker prior generalizes |
| `phenoflux_crispr_msa_pcd` | CRISPR | MSA+PCD | 104 | Per-channel modulation generalizes |

Data size controlled via `--data_index` CLI (not separate configs):
```bash
--data_index data/processed/diet/index_diet_5k.csv    # 5k fast validation
--data_index data/processed/diet/index_diet.csv        # full dataset (default)
```

## Data Flow

1. `data_utils.py:read_files_pert` pairs control+treated cells from same batch
2. `use_initial=1` → ODE starts from control image (not noise)
3. `marker_profile` = population-mean 18ch profile of target condition (broadcast to spatial)
4. MSA processes marker_profile internally in UNet.forward() → 64-dim context concatenated to condition
5. PCD applies per-channel (scale, bias) modulation on UNet output from MSA context
6. Flow matching: model learns velocity field from control→target

## Critical Design Rules

1. **marker_profile MUST NOT leak into unconditional path**. When `class_drop_prob` triggers CFG dropout (`conditioning={}`), marker_profile must also be absent. UNet handles this via zero-padding condition to expected dim.

2. **EMA unwrapping needed** before checking module flags. Use `getattr(model, 'model', model)`.

3. **MSA/PCD are inside UNetModel** (checkpointed). Not externally constructed.

4. **Every epoch saves checkpoint**. Training can be paused/resumed at any epoch boundary.

5. **`find_unused_parameters=True` is REQUIRED** for DDP — MSA/PCD params may be unused during CFG dropout.

## Training Commands

All use `--dataset phenoflux --config <name>`:

```bash
# Quick validation (mini subset, 2 epochs, 64 images)
bash scripts/quick_validate.sh phenoflux_diet_msa_pcd phenoflux

# Diet baseline (5k subset)
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config phenoflux_diet --device cuda \
  --batch_size 32 --epochs 20 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 5 --fid_samples 5120 --compute_fid --save_fid_samples \
  --data_index data/processed/diet/index_diet_5k.csv \
  --output_dir outputs/runs/diet/phenoflux_diet_5k_v1

# Diet + MSA + PCD (full)
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config phenoflux_diet_msa_pcd --device cuda \
  --batch_size 32 --epochs 20 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 5 --fid_samples 5120 --compute_fid --save_fid_samples \
  --output_dir outputs/runs/diet/phenoflux_diet_msa_pcd_v1

# CRISPR baseline
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config phenoflux_crispr --device cuda \
  --batch_size 32 --epochs 40 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 10 --fid_samples 5120 --compute_fid --save_fid_samples \
  --output_dir outputs/runs/crispr/phenoflux_crispr_v1

# CRISPR + MSA + PCD
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config phenoflux_crispr_msa_pcd --device cuda \
  --batch_size 32 --epochs 40 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 10 --fid_samples 5120 --compute_fid --save_fid_samples \
  --output_dir outputs/runs/crispr/phenoflux_crispr_msa_pcd_v1
```

## Evaluation

### Image quality (FIDo/c, KIDo/c)
```bash
python phenoflux/eval/fid.py --real-dir <real_imgs> --gen-dir <gen_imgs> --per-condition-cap 500
```

### Biological metrics (PGC, dir-corr, sign-agreement)
```bash
python phenoflux/eval/aggregate.py <eval_dir> 5 <epoch>
```

### MoA classifier accuracy
```bash
python phenoflux/eval/moa.py \
  --config_path configs/phenoflux_crispr.yaml --mode eval \
  --img_root_path <eval_dir>/fid_samples/epoch-<N> \
  --ckpt_path outputs/baselines/moa/crispr/condition_classifier.pth \
  --out_json <eval_dir>/moa.json
```

### Quick validation protocol
```bash
bash scripts/quick_validate.sh <config> phenoflux [data_index]
# Example: bash scripts/quick_validate.sh phenoflux_diet phenoflux
```

## Environment
- Conda env: `pmf`
- Python 3.10, PyTorch 2.11.0+cu128
- GPUs: 2× NVIDIA RTX 5090 (32GB each)
- NEVER set PYTHONNOUSERSITE — it breaks the baseline stack

## Time
- **Always use Beijing time (UTC+8 / Asia/Shanghai)**. Never UTC.

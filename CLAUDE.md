# PhenoFlux - Microalgae Phenotype Generation

Flow matching for microalgae cellular phenotype transport across time-course conditions.

## Project Overview

**Current Focus**: Microalgae phenotype generation from time-course microscopy data (RGB bright-field imaging).

**Archived**: CRISPR/Diet datasets (18-channel MERFISH data) → `archive_crispr_diet_2026_07_04/`

## Project Structure

```
phenoflux/                    # Python package
├── train.py                  # Entry point (torchrun -m phenoflux.train)
├── args.py                   # Argument parser
├── models/
│   ├── configs.py            # MODEL_CONFIGS registry (simplified)
│   ├── unet.py               # UNetModel (standard backbone)
│   ├── ema.py                # Exponential Moving Average
│   └── nn.py                 # NN utilities
├── training/
│   ├── train_loop.py         # Training loop (flow matching, CFG)
│   ├── eval_loop.py          # Evaluation loop (FID, image generation)
│   ├── dataloader.py         # CellDataset, CellDataLoader
│   ├── data_utils.py         # Microalgae RGB loading
│   ├── data_transform.py     # Augmentation transforms
│   ├── distributed.py        # DDP helpers
│   ├── grad_scaler.py        # AMP gradient scaler
│   ├── load_save.py          # Checkpoint load/save
│   └── edm_time.py           # EDM time discretization
└── eval/
    ├── fid.py                # FIDo/c, KIDo/c
    └── aggregate.py          # Phenotypic metrics

configs/                      # 2 active configs (microalgae)
scripts/                      # Microalgae data building scripts
outputs/                      # Training outputs (gitignored)
archive_crispr_diet_2026_07_04/  # Archived materials (gitignored)
```

## Datasets

### Microalgae (Active)

Two granularity levels for microalgae phenotype generation:

#### 1. Single-Cell Level (`microalgae_base.yaml`)
- **Images**: RGB crops from FusionODE (128×128, 3 channels)
- **Image source**: `../../FusionODE/data/CROPS_RAW_SCALE`
- **Condition**: 61-dimensional embedding
  - Time point (0-72h)
  - Mean RNA PCA components
  - Mean protein PCA components
- **Data index**: `data/processed/index.csv` (~38M, ~214K cell pairs)
- **Use case**: Cell-level phenotype prediction

#### 2. Field-Level (`microalgae_field_base.yaml`)
- **Images**: Raw microscopy fields from FusionODE (variable size, 3 channels)
- **Image source**: `../../FusionODE/data/TIMECOURSE`
- **Condition**: 92-dimensional embedding
  - Field morphology statistics (mean/std of area, circularity, aspect_ratio, etc.)
  - Aligned state-level omics PCs
- **Data index**: `data/processed/field_index.csv` (1.8M, ~5.3K field pairs)
- **Use case**: Field-level phenotype prediction (closer to acquisition unit)

### Data Pairing Strategy

**Mode**: `batch_random` (default for microalgae)
- Control and treated samples randomly paired within the same batch
- Preserves batch-level covariate structure
- Suitable for continuous time-course data

## Architecture

### Standard UNet (No Molecular Priors)

```
Input (3ch RGB) → UNet Encoder → Condition Embedding + Time Embedding
                                           ↓
                                  UNet Decoder with skip connections
                                           ↓
                                  Output (3ch RGB)
```

**Key parameters** (from config YAML):
- `in_channels`: 3 (RGB)
- `out_channels`: 3 (RGB)
- `model_channels`: 128
- `base_condition_dim`: config-dependent
  - **Base configs** (`microalgae_timepoint*`, `microalgae_smoke`): 4 dims — `light/dark/time_norm/time_bin_h`.
  - **Omics-enriched config** (`microalgae_timepoint_512_62d`): 62 dims — 4 base + 29 RNA PCA + 29 Protein PCA (z-scored), built by `scripts/interpolate_omics_to_timepoints.py` → `embedding_62d.csv`. This is the Stage-2 condition that addresses the identity-mapping collapse.
  - The embedding CSV carries a `timegroup_key` index column that the dataloader drops (`index_col=0`), so 63 CSV columns → 62 feature dims.
- `condition_dim`: Same as `base_condition_dim` (no molecular prior concat)

**What we DON'T use** (archived with CRISPR/Diet):
- ❌ MSA (Marker Self-Attention) — requires 18-channel marker panels
- ❌ PCD (Per-Channel Decoder) — marker-specific modulation
- ❌ MGFM/SGLR/AdaIN — marker-gated feature modulation
- ❌ `marker_profile` — population-mean marker statistics

## Training

### Quick Start

```bash
# Single-cell level (small subset for validation)
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config microalgae_base \
  --batch_size 32 --epochs 20 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 5 --fid_samples 512 --compute_fid \
  --output_dir outputs/runs/microalgae/cell_level_v1

# Field-level
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config microalgae_field_base \
  --batch_size 16 --epochs 20 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 5 --fid_samples 512 --compute_fid \
  --output_dir outputs/runs/microalgae/field_level_v1
```

### Key Training Flags

| Flag | Purpose | Recommended |
|------|---------|-------------|
| `--use_initial 1` | ODE starts from control image (not noise) | ✅ Yes |
| `--cfg_scale 0.2` | Classifier-Free Guidance strength | 0.1-0.3 |
| `--class_drop_prob 0.2` | CFG dropout probability during training | 0.1-0.3 |
| `--use_ema` | Exponential Moving Average of weights | ✅ Yes |
| `--skewed_timesteps` | EDM-style log-normal time sampling | ✅ Yes |
| `--compute_fid` | Compute FID during eval epochs | Optional |

## Evaluation

### Image Quality Metrics
```bash
python phenoflux/eval/fid.py \
  --real-dir <real_imgs> \
  --gen-dir <gen_imgs> \
  --per-condition-cap 500
```

### Biological Metrics
```bash
python phenoflux/eval/aggregate.py <eval_dir> 5 <epoch>
```

## Data Building Scripts

```bash
# Build single-cell generation data
python scripts/build_microalgae_generation_data.py

# Build field-level generation data
python scripts/build_microalgae_field_generation_data.py

# Build field metadata (EXIF + morphology summaries)
python scripts/build_microalgae_field_metadata.py
```

## Critical Design Notes

1. **No marker_profile needed**: Microalgae uses continuous condition embeddings (from CSV), not marker-specific profiles.

2. **RGB normalization**: Images loaded as `[-1, 1]` via `(pixel / 127.5) - 1.0`.

3. **Flow matching from control**: `use_initial=1` means ODE integrates from real control image to target, not from noise.

4. **CFG handling**: When `class_drop_prob` triggers dropout, `conditioning={}` is passed (no zero-padding needed for microalgae).

5. **Batch pairing**: `pairing_mode='batch_random'` randomly pairs control/treated within same batch to preserve batch effects.

## Environment
- Conda env: `pmf`
- Python 3.10, PyTorch 2.11.0+cu128
- GPUs: 1× NVIDIA RTX 4090 (24GB)

## Time
- **Always use Beijing time (UTC+8 / Asia/Shanghai)**

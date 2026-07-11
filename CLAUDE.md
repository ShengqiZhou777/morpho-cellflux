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
├── data/                     # Data loading & preprocessing
│   ├── dataloader.py         # CellDataset, CellDataLoader
│   ├── data_utils.py         # Microalgae RGB loading
│   └── data_transform.py     # Augmentation transforms
├── models/
│   ├── configs.py            # MODEL_CONFIGS registry (simplified)
│   ├── unet.py               # UNetModel (standard backbone)
│   ├── ema.py                # Exponential Moving Average
│   └── nn.py                 # NN utilities
├── training/                 # Training orchestration
│   ├── train_loop.py         # Training loop (flow matching, CFG)
│   ├── distributed.py        # DDP helpers
│   ├── grad_scaler.py        # AMP gradient scaler
│   ├── load_save.py          # Checkpoint load/save
│   └── edm_time.py           # EDM time discretization
└── eval/                     # Evaluation (in-loop + post-hoc metrics)
    ├── eval_loop.py          # In-training eval loop (ODE sampling, FID, save images)
    ├── fid.py                # FIDo/c, KIDo/c
    ├── morphology.py         # Morphology feature extraction
    ├── distribution_eval.py  # Distribution-level (population) evaluation
    └── aggregate_microalgae.py  # Phenotypic metrics aggregation

configs/                      # 5 active configs (see configs/README.md)
scripts/                      # Microalgae data building scripts
outputs/                      # Training outputs (gitignored)
archive_crispr_diet_2026_07_04/  # Archived materials (gitignored)
```

## Datasets

### Microalgae (Active)

Two granularity levels for microalgae phenotype generation:

#### 1. Single-Cell Level (`microalgae_timepoint_512_genes.yaml`, primary)
- **Images**: RGB crops (512×512, 3 channels)
- **Image source**: `data/raw/microalgae_v1/single_cell_images`
- **Condition**: 476-dim condition = 2 light/dark + 1 time_norm + 1 time_bin_h + 372 genes + 100 proteins (z-scored)
- **Data index**: `data/processed/microalgae_v1/views/timepoint_512/index.csv`
- **Embedding**: `data/processed/microalgae_v1/views/timepoint_512/embedding_genes.csv`
- **4d baseline**: `microalgae_timepoint_512.yaml` (same view, no omics — ablation counterpart)
- **Use case**: Cell-level phenotype prediction

#### 2. Field-Level (`microalgae_field.yaml`)
- **Images**: Whole microscopy fields (3 channels)
- **Image source**: `data/raw/microalgae_v1/field_images`
- **Condition**: 34-dim embedding (field morphology statistics + aligned omics PCs)
- **Data index**: `data/processed/microalgae_v1/views/field/index.csv`
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
  - **Base configs** (`microalgae_timepoint_512`, `microalgae_smoke`): 4 dims — `light/dark/time_norm/time_bin_h`.
  - **Gene/protein config** (`microalgae_timepoint_512_genes`, primary): 476 dims — 4 base + 372 high-variable genes (log-FPKM std>2) + 100 top-variable proteins, linear-interpolated per feature from the 9 measured timepoints (raw FPKM, **no PCA**), z-scored, built by `scripts/build_gene_condition.py` → `embedding_genes.csv`. Replaces the discarded PCA+cubic approach (empirically unreliable at late timepoints).
  - The embedding CSV carries a `timegroup_key` index column that the dataloader drops (`index_col=0`), so 477 CSV columns → 476 feature dims.
- `condition_dim`: Same as `base_condition_dim` (no molecular prior concat)

**What we DON'T use** (archived with CRISPR/Diet):
- ❌ MSA (Marker Self-Attention) — requires 18-channel marker panels
- ❌ PCD (Per-Channel Decoder) — marker-specific modulation
- ❌ MGFM/SGLR/AdaIN — marker-gated feature modulation
- ❌ `marker_profile` — population-mean marker statistics

## Training

### Quick Start

```bash
# CPU smoke (no external data, validates dataloader+model path)
make smoke

# Build the gene/protein condition embedding for the primary path
make interpolate

# 1-GPU quick sanity run of the primary gene config
make quick

# Full training of the primary gene config (1x RTX 4090)
make train

# Baseline / overrides (scripts/train.sh is env-var parameterized):
CONFIG=microalgae_timepoint_512 BATCH=16 EPOCHS=40 bash scripts/train.sh   # 4d ablation
CONFIG=microalgae_field DATASET=phenoflux bash scripts/train.sh            # field lane
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
python phenoflux/eval/aggregate_microalgae.py <run_dir> 5 <epoch>
```

## Data Building Scripts

```bash
# Build single-cell (timepoint) + field processed views
python scripts/build_microalgae_dataset.py --version microalgae_v1 --views timepoint,field

# Build the gene/protein condition embedding for the primary path
python scripts/build_gene_condition.py

# Build field metadata (EXIF + morphology summaries) only
python scripts/build_field_metadata.py
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

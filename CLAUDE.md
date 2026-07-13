# PhenoFlux (morpho-cellflux) -- Microalgae Phenotype Transport

Flow matching for conditional generation of treated microalgae single-cell crops
from (control_cell, time, omics_condition).

## Core Challenge: Identity Collapse

Deterministic flow matching with MSE loss regresses to population mean. For microalgae,
the population mean of treated cells is close to control cells → model outputs near-zero
velocity → generated image ≈ control image. This is **identity collapse**: the model
learns to copy the input rather than transport the phenotype.

**Detection**: `phenoflux/eval/distribution_eval.py` with identity baseline (E_delta).
Model must beat "just copy the control cell" to pass.

**Current mitigation**: Noise-start (use_initial=0) + GAN adversarial loss (weight=0.1).
MMD distribution-matching loss (weight=0.5) added at commit ab83ec8 but was inert
(.detach() bug) until 10f717d fixed it. GAN is the only proven distribution-matching
gradient signal so far.

## Project Structure

```
phenoflux/
├── train.py                  # Entry point (torchrun -m phenoflux.train)
├── args.py                   # Argument parser
├── data/
│   ├── dataloader.py         # CellDataset, CellDataLoader, pairing strategies
│   ├── data_utils.py         # Microalgae RGB loading, centered_noise
│   └── data_transform.py     # Augmentation transforms
├── models/
│   ├── unet.py               # UNetModel with FiLM per-block condition injection
│   ├── configs.py            # MODEL_CONFIGS registry
│   ├── ema.py                # Exponential Moving Average
│   └── nn.py                 # NN utilities
├── training/
│   ├── train_loop.py         # MSE + GAN + MMD losses, EDM time sampling
│   ├── distributed.py        # DDP helpers
│   ├── grad_scaler.py        # AMP gradient scaler
│   ├── load_save.py          # Checkpoint load/save
│   └── edm_time.py           # EDM time discretization
└── eval/
    ├── eval_loop.py          # In-training eval: ODE sampling + torchmetrics FID
    ├── distribution_eval.py  # PRIMARY: morphology metrics + identity baseline
    ├── morphology.py         # Per-crop 12-D morphology feature extraction
    ├── fid.py                # Stratified FID/KID computation
    └── aggregate_microalgae.py  # DEPRECATED (1-D, blind to morphology)

configs/
  microalgae_timepoint_512_genes.yaml  # PRIMARY: 476-dim gene+protein condition
  microalgae_timepoint_512.yaml        # 4d ablation baseline
  microalgae_field.yaml                # Field-level lane
  microalgae_smoke.yaml                # Smoke test fixture

scripts/
  train.sh                    # Training launcher (env-var parameterized)
  build_gene_condition.py     # 476-dim condition from raw FPKM (no PCA)
  build_microalgae_dataset.py # Dataset builder (timepoint/field views)
  verify_signal_strength.py   # Population-level morphology signal verification
  sample_microalgae_checkpoint.py  # Inference from checkpoint

outputs/runs/microalgae/      # Training artifacts (gitignored)
```

## Architecture

### UNet with FiLM Per-Block Condition Injection (commit 6a19801)

```
Condition (476-D gene+protein) → Linear(476→512) = cond_emb
Time t → time_embed(512) = time_emb

Each ResBlock:
  emb = time_emb + cond_emb     ← re-combined at every block (FiLM)
  h = in_layers(x)
  h = GroupNorm(h) * (1 + scale(emb)) + shift(emb)
  h = out_rest(h)
  return x + h
```

**Note**: FiLM made zero training difference vs single-point injection (identical
logs between balancedt and FiLM runs through epoch 9). The condition signal is
likely still functionally ignored. The root cause is the loss function, not the
injection architecture.

**Key parameters**:
- `in_channels=3, out_channels=3` (RGB)
- `model_channels=128`
- `base_condition_dim=476` (4 base + 372 genes + 100 proteins, z-scored, no PCA)
- `condition_dim=476` (no molecular prior concat)

### What we do NOT use (archived CRISPR/Diet features)
- MSA (Marker Self-Attention), PCD (Per-Channel Decoder)
- MGFM/SGLR/AdaIN marker-gated modulation
- Discrete flow matching, foreground-weighted MSE
- Per-condition p_mean shifts, velocity_bias_proj

## Training

### Quick Start

```bash
make smoke          # CPU smoke test (validates dataloader+model)
make quick          # 1-GPU quick sanity run
make train          # Full training (1x RTX 4090, 24GB)

# Direct launch with current anti-collapse defaults:
bash scripts/train.sh

# Override specific params:
USE_INITIAL=0 GAN_WEIGHT=0.1 MMD_WEIGHT=0.5 CENTER_NOISE_SIGMA=0.4 \
  FID_SAMPLES=2048 EPOCHS=60 BATCH=12 NPROC=1 bash scripts/train.sh
```

### Training Loss Composition

```python
# 1. Flow Matching MSE (always active, dominates)
loss = torch.pow(pred - u_t, 2).mean()          # ~0.005-0.011

# 2. GAN adversarial (weight=0.1, HAS gradient)
g_loss = -PatchDiscriminator(x_pred_target).mean()
loss = loss + 0.1 * g_loss                       # ~0.01-0.1

# 3. MMD distribution-matching (weight=0.5, NOW HAS gradient after 10f717d)
gen_ds = interpolate(x_pred_target, 16x16).flatten(1)  # 768-D pixel space
loss = loss + 0.5 * _mmd_rbf(gen_ds, real_ds)
```

**GAN details**: PatchDiscriminator (4-layer conv, spectral norm), hinge loss,
updated every 8 steps, warmup after epoch 0 or step 500, D reset each epoch.

**MMD details**: RBF kernel (median-heuristic bandwidth, `gamma=1/(2*sigma2)`),
V-statistic (includes diagonals, biased), operates on 16x16 downsampled pixel
patches (768-D). Caveats: d/n=64:1 (severely undersampled), mixes conditions
(51% of batches have all-different conditions), measures pixel-space not
morphology-space distribution.

**EDM time sampling**: P_mean=-0.5, P_std=1.2, log-normal distribution.
ODE solver: dopri5 (adaptive 5th-order, ~44 NFE).

### Key Training Flags

| Flag | Purpose | Current Default | Notes |
|------|---------|-----------------|-------|
| `--use_initial 0` | ODE starts from noise (anti-collapse) | **0** | 1=control (collapses!), 2=control+noise |
| `--center_noise_sigma 0.4` | Cell-centered noise envelope | **0.4** | >0 biases toward one centered cell |
| `--gan_weight 0.1` | PatchGAN adversarial loss | **0.1** | Only proven distribution-matching signal |
| `--mmd_weight 0.5` | MMD distribution-matching loss | **0.5** | Gradient fixed at 10f717d |
| `--cfg_scale 0.2` | Classifier-Free Guidance | 0.2 | Only at inference, not training |
| `--class_drop_prob 0.2` | CFG dropout during training | 0.2 | |
| `--use_ema` | Exponential Moving Average | Yes | |
| `--skewed_timesteps` | EDM log-normal time sampling | Yes | P_mean=-0.5 |
| `--eval_frequency 5` | Epochs between eval | 5 | |
| `--fid_samples 2048` | Samples per eval | **2048** | With shuffle, covers all 105 conditions |
| `--compute_fid` | Compute FID during eval | Yes | |
| `--save_fid_samples` | Save generated PNGs | Yes | Required for post-hoc distribution_eval |
| `--early_stop_patience 5` | Early stop on loss plateau | 5 | 0=disabled |

### train.sh Defaults (as of 10f717d)

```
USE_INITIAL=0            # noise-start (anti-collapse)
CENTER_NOISE_SIGMA=0.4   # cell-centered noise
GAN_WEIGHT=0.1           # PatchGAN on
MMD_WEIGHT=0.5           # MMD on (with gradient)
EVAL_FREQ=5              # eval every 5 epochs
FID_SAMPLES=2048         # covers all conditions with shuffle
BATCH=16                 # per-GPU (12 on 24GB card)
NPROC=2                  # multi-GPU default
EPOCHS=40
```

## Data

### Primary Path: Single-Cell Timepoint (105 conditions, 476-D)

- **Images**: 128x128 RGB crops (centered, unmasked) from bright-field microscopy
- **Source**: `data/raw/microalgae_v1/single_cell_images`
- **Index**: `data/processed/microalgae_v1/views/timepoint_512/index.csv` (128,138 rows)
- **Embedding**: `data/processed/microalgae_v1/views/timepoint_512/embedding_genes.csv` (105 rows × 477 cols)
- **Condition**: 476 dims = 4 base (light/dark, time_norm, time_bin_h) + 372 HV genes (log-FPKM std>2) + 100 HV proteins, z-scored, linear interpolation from 9 measured timepoints, NO PCA

### Dataset Splits

| Split | Pairs | Conditions | Notes |
|-------|-------|-----------|-------|
| Train | 103,162 | 105 | ~982 pairs/condition (min=210, max=1024) |
| Test | 24,976 | 105 | ~238 pairs/condition, used for eval |

### Timepoint Coverage

8 nominal timepoints: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h
2 batches: Dark (48 conditions), Light (57 conditions)
105 total: 5-min EXIF bins, ~1280 cells/bin

### Pairing

`pairing_mode=batch_random`: control and treated randomly paired within same BATCH
(Dark/Light). No true cell-level correspondence exists (no live-cell tracking).

## Evaluation

### Primary: Distribution-Level Morphology Metrics

```bash
python -m phenoflux.eval.distribution_eval <run_dir> [epoch] [--no-fid]
```

Computes 4 metrics on 12-D morphology features, each with **identity baseline**
(control-as-prediction). Model must beat identity to pass.

| Metric | Type | What it measures | Current state |
|--------|------|-----------------|---------------|
| **E_delta** (PRIMARY) | Multivariate energy distance | Joint distribution (shape×intensity×texture) | **+1.29** (PASS) |
| MMD_delta | RBF-kernel MMD | Joint distribution (kernel view) | +0.20 (PASS) |
| KS_delta | Per-feature Kolmogorov-Smirnov | Marginal distributions | **-0.10** (FAIL) |
| WD_delta | Per-feature Wasserstein | Marginal distributions | **-14.0** (FAIL) |

**Critical finding**: KS/WD are NEGATIVE across ALL runs, ALL strata. Model wins on
joint distribution but loses on per-feature marginals -- structural hedging behavior
from deterministic MSE regression. The 12 features are: area, perimeter, circularity,
aspect_ratio, solidity, eccentricity, mean_intensity, std_intensity, texture_contrast,
texture_homogeneity, texture_energy, texture_correlation.

### Morphology Feature Extraction (`phenoflux/eval/morphology.py`)

- Foreground: max(R,G,B) > 0.05, largest connected component
- All 3 populations (gen/target/control) normalized through `_pad_or_shrink_to_canvas_np(canvas_size=128)`
- Raw variable-size crops (28-50px) share pixel scale with generated 128x128 crops

### In-Training FID (`eval_loop.py`)

- torchmetrics FrechetInceptionDistance, n=512 (or fid_samples), all conditions pooled
- Always uses shuffle=True DataLoader (fixed at 10f717d; was gated before)
- In-training FID ~34 (n=512) is NOT comparable to distribution_eval FID ~66 (n=70, small-N bias)

### FID Paradox

Collapsed model (old e40): FID=56.4, E_delta=-2.64 (FAIL) -- realistic images by copying controls.
FiLM model: FID=66.4, E_delta=+1.29 (PASS) -- less realistic images that actually differ from controls.
**Low FID does NOT mean good phenotype transport.** FID measures image quality, not transport quality.

### eval_loop.py FID vs distribution_eval FID

| Property | In-training (eval_loop) | Post-hoc (distribution_eval) |
|----------|------------------------|------------------------------|
| Sample size | 512 (configurable) | 70 pooled (cap=14 per stratum) |
| Sample selection | Shuffled from test dataloader | From saved fid_samples PNGs |
| Canvas normalization | N/A (both 128x128) | _pad_or_shrink_to_canvas_np(128) |
| Per-condition | Pooled all conditions | Per-stratum then pooled |
| Small-N bias | None | +15-30 points |

### Caveat: Eval Sampling Was Blind (fixed at 10f717d)

Before the fix, `eval_batch_size=0` defaulted to `train_bs=12`, the `if eval_bs != train_bs`
gate never opened, and eval used non-shuffled DistributedSampler → only the first 5
alphabetically-sorted Dark conditions (~2h, weakest signal AUC=0.90) were evaluated.
48h/72h conditions (AUC 0.99) were NEVER seen. Now fixed with always-shuffle.

## Known Issues & Limitations

1. **MMD space mismatch**: 768-D pixels with n=12 (d/n=64:1) vs eval 12-D with n=512.
   Even with gradient fixed, MMD measures wrong thing (pixel stats vs morphology).

2. **MMD batch mixing**: 51% of batches have all-different conditions. MMD measures
   cross-condition variance, not within-condition distribution matching.

3. **KS/WD universal failure**: Per-feature marginals distorted by hedging -- model
   outputs "safe" intermediate values. Needs explicit per-feature distribution losses.

4. **Condition signal likely ignored**: FiLM made zero training difference. The 476-D
   condition has only ~17 unique omics anchors (linear interpolation between 9 measured
   timepoints for 105 EXIF bins). Information content per bin may be insufficient.

5. **Eval only covers 5 Dark ~2h conditions** (in saved fid_samples). Need to run eval
   with shuffle+larger fid_samples to see 48h/72h/Light output.

6. **23 commits ahead of origin/main** (unpushed as of 2026-07-13). Push when network available.

## Experiment History (outputs/runs/microalgae/)

| Run | Regime | E_delta | FID (model/identity) | Notes |
|-----|--------|---------|---------------------|-------|
| `genes_noise0_gan0.1_e40` | use_initial=1, no MMD, P_mean=-1.2 | -2.64 (FAIL) | 56.4/36.7 | Collapsed baseline |
| `genes_noise0_gan0.1_mmd0.5_e60` | noise+GAN+MMD, old arch, P_mean=-1.2 | +1.22 (PASS) | N/A | Epoch 4 only |
| `genes_noise0_gan0.1_mmd0.5_balancedt_e60` | noise+GAN+MMD, old arch, P_mean=-0.5 | +1.08 (PASS) | N/A | Epoch 4+9, FID eval pending |
| `genes_noise0_gan0.1_mmd0.5_film_e60` | noise+GAN+MMD, FiLM, P_mean=-0.5 | +1.29 (PASS) | 66.4/55.3 | Epoch 14 (killed SIGTERM) |
| `genes_noise0_gan0.1_e60` | CFG=1.0 misconfiguration | -- | -- | Stopped early |

## Key Memory Files (session context)

Located at `/home/shockley/.claude/projects/-home-shockley-myproject-PhenoFlux-morpho-cellflux/memory/`:

- `phenoflux-fix-direction.md` -- Agreed fix: stochasticity + distribution matching
- `phenoflux-distribution-eval.md` -- Distribution eval design + identity baseline
- `phenoflux-signal-verification.md` -- AUC 0.99 proof signal is strong
- `phenoflux-current-state-20260712.md` -- Pre-session state, running experiments
- `phenoflux-repo-convergence.md` -- Repository cleanup, lost checkpoint warning
- `phenoflux-stage2-omics-condition.md` -- PCA→raw FPKM transition
- `phenoflux-task-definition.md` -- Population distribution transport formulation
- `phenoflux-immediate-next-steps-20260713.md` -- Current task list
- `phenoflux-session-state-20260713.md` -- Session end state (git, tasks, checkpoints)

## Obsidian Knowledge Base

Located at `/home/shockley/ObsidianVault/Projects/PhenoFlux/`:
- `00-Hub.md` -- Project index
- `Knowledge/PhenoFlux-Evaluation-Training-Analysis-20260713.md` -- Complete 15-section analysis (525 lines)

## Commit History (recent, July 11-13)

```
10f717d fix: restore MMD gradient, always-shuffle eval, and anti-collapse defaults
8d7d741 feat(scripts): rebuild unmasked single-cell crops from field images
6a19801 feat(unet): per-block condition injection (CellFlow-style FiLM)
5a16d32 fix(edm): change P_mean from -1.2 to -0.5 for balanced time sampling
ab83ec8 feat(loss): add MMD distribution-matching loss (scDFM 2026)
b170752 feat(train.sh): expose RESUME env for checkpoint continuation
e14d21a feat(omics): replace PCA with raw gene/protein FPKM (476-dim)
0cec442 feat: restore noise-start (use_initial=0/2) + PatchGAN training path
7001d0f refactor(structure): extract phenoflux/data/, move eval_loop into eval/
3427a1a refactor: remove unused-feature/dead code (discrete FM, GAN, foreground, noise-init)
```

Training loop went through delete-restore cycle within 8 hours on July 11 (3427a1a → 0cec442).
GAN implementation came directly from Meta's CellFlux codebase. MMD .detach() bug introduced
at ab83ec8 (copy-paste from discriminator update pattern), fixed at 10f717d.

## Environment

- Conda env: `pmf` (Python 3.10, PyTorch 2.11.0+cu128)
- GPU: 1× NVIDIA RTX 4090 (24GB)
- Timezone: Beijing (UTC+8 / Asia/Shanghai)
- Working tree: clean (all changes committed)

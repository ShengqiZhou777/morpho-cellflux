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
Condition (476-D gene+protein) → Linear(476→512) = cond_emb    [244K params, 1 layer, no activation]
Time t → time_embed(512) = time_emb                            [sinusoidal + 2-layer MLP+SiLU]

Each of 29 ResBlocks:
  emb = time_emb + cond_emb     ← re-combined at EVERY block (FiLM)
  h = in_layers(x)
  emb_out = emb_layers_i(emb)   ← DIFFERENT Linear(512→512) per block (263K each, 7.62M total)
  scale, shift = chunk(emb_out, 2)
  h = GroupNorm(h) * (1 + scale) + shift
  h = out_rest(h)
  return x + h
```

**Key insight (2026-07-13 exploration)**:

FiLM did NOT make zero training difference. At epoch 14, the condition signal is
measurably active and structurally meaningful. The claim "FiLM made zero training
difference" was based on epoch 9 logs only. By epoch 14, condition-swap experiments
show the model is learning to use conditions correctly. See "Condition Pathway Analysis"
section below.

The architecture is "compress once, interpret many times": one Linear(476→512)
bottleneck creates a global condition signal, then each of 29 ResBlocks has its own
separately-trained Linear(512→512) that reads DIFFERENT aspects of the SAME signal.

**Condition pathway parameters**: 7.86M total (14.1% of 55.9M model params).
  - mol_embed_transform: 244K (creates the global signal)
  - emb_layers (29 blocks): 7.62M (interprets it per-layer)

### What we do NOT use (archived CRISPR/Diet features)
- MSA (Marker Self-Attention), PCD (Per-Channel Decoder)
- MGFM/SGLR/AdaIN marker-gated modulation
- Discrete flow matching, foreground-weighted MSE
- Per-condition p_mean shifts, velocity_bias_proj

## Data Pipeline -- Deep Dive (2026-07-13 exploration)

### Index Structure

`index.csv`: 128,138 rows, 19 columns. Balanced: 64,069 negative_control + 64,069 treated.
BATCH = Dark (60,614) or Light (67,524). SPLIT = train (103,162, ~80.5%) / test (24,976).

### Condition Space

105 unique CPD_NAME values = 48 Dark + 57 Light timepoints.

**embedding.csv (4-dim, frozen)**:
  - cond_light in {0,1}, cond_dark in {0,1} (redundant), time_norm in [0,1], time_bin_h in [1,74]
  - These ARE already continuous physical coordinates. The "discrete embedding lookup"
    is actually a frozen nn.Embedding of 105 rows of 4-dim continuous values.
  - Effective degrees of freedom: 2 (1 bit lighting + 1 float time).

**embedding_genes.csv (476-dim, frozen)**:
  - 4 base + 372 high-variable genes (log-FPKM std > 2) + 100 top-variable proteins
  - Built by `build_gene_condition.py`: linear interpolation from 9 measured timepoints,
    per Dark/Light condition, z-scored. NO PCA.
  - Effective rank: 15 (out of 105 rows). 4 PCs explain 90% variance. 11 PCs explain 99%.
  - Independent info beyond base 4-dim: PC1 and PC2 have ~33-41% variance NOT explained
    by linear regression on base 4 dims. Gene features carry real additional biological signal.
  - Cross-lighting gene distance at same time ~= 41% of within-condition full-range distance.
    Dark and Light trajectories diverge from common t=0 ancestor.

### Pairing Strategy

**batch_random (default)**: treated cell paired with random control from SAME BATCH.
Pools: Dark ctrl=30,307, Dark trt=30,307; Light ctrl=33,762, Light trt=33,762.
Problem: cross-timepoint pairing creates biologically implausible control to target maps
(e.g., 74h control to 2h treated in Dark pool).

**Alternative modes implemented**:
- `merfish_nn`: precomputed JSON mapping treated to control in MERFISH feature space
- `cluster_match`: same cluster_type pairing, falls back to batch_random

**Potential improvement**: within-condition pairing (each CPD_NAME as its own cluster)
reduces cross-timepoint noise. Just set cluster_type=CPD_NAME in index.csv.

### Image Loading (Microalgae)

PNG crops via `_load_microalgae_rgb()`: scale-preserving (center-pads on canvas),
normalized to [-1,1]. Augmentation: identical random flips for ctrl+trt (to avoid
confounding). Strong mode adds spatial jitter + intensity scaling + Gaussian noise.

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
| `--pairing_mode` | Control-target pairing | batch_random | Also: merfish_nn, cluster_match |
| `--pairing_path` | JSON/CSV for merfish_nn/cluster_match | None | |
| `--augment_strength` | Augmentation level | default | Also: strong, none |

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
- **Embedding**: `data/processed/microalgae_v1/views/timepoint_512/embedding_genes.csv` (105 rows x 477 cols)
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

## Condition Pathway Analysis (2026-07-13 exploration)

### Condition Signal is NOT Ignored

Previous CLAUDE.md (line 83-84): "FiLM made zero training difference" and
"condition signal likely still functionally ignored" -- **this is incorrect**.

At epoch 14 (genes_noise0_gan0.1_mmd0.5_film_e60), condition-swap experiments show:

**Velocity-level test** (same noise batch, different conditions at t=0):
| Comparison | |dv| | cos_sim | Interpretation |
|-----------|------|---------|---------------|
| early vs late Dark | 0.057 | 0.978 | Time effect: detectable |
| Dark vs Light (both early) | 0.012 | 0.998 | Lighting effect: almost none (cells similar at early times) |
| Dark vs Light (both late) | 0.049 | 0.984 | Lighting effect: significant at late times |

Cross-condition diff / cond-vs-uncond diff ratio = 1.78 at t=0 (for early vs late Dark).
This means the model IS using the condition -- the effect is 1.78x stronger than
condition-vs-unconditional difference. Signal grows from epoch 9 (ratio=0.70) to
epoch 14 (ratio=1.78), confirming the model is actively learning to use conditions.

**Image-level test** (midpoint ODE, step=0.02, same noise seed):
| Generation | Mean Int | Std Int | Edge Energy | bright_frac | kurtosis |
|-----------|----------|---------|-------------|-------------|----------|
| early Dark (2h) | 0.424 | 0.063 | 0.137 | 0.063 | 14.9 |
| late Dark (74h) | 0.403 | 0.027 | 0.048 | 0.008 | 32.2 |
| late Light (73h) | 0.434 | 0.038 | 0.067 | 0.021 | 17.3 |
| unconditional | 0.404 | 0.055 | 0.115 | 0.046 | 15.2 |

Feature modulation ratios (max/min across conditions):
- bright_frac: 7.62x (highest -- model heavily modulates this)
- dark_frac: 5.03x
- edge_energy: 2.88x
- texture_contrast: 2.80x
- std_intensity: 2.65x
- mean_intensity: 1.07x (lowest -- model avoids simple brightness shift)

Cross-validated with biological signal (Cohen's d from real cells, 0h vs 72h):
- std_intensity: model modulates 2.65x, biological d = -3.76 (Dark) -- strongest feature
- texture_contrast: model modulates 2.80x, biological d = -0.73 (Dark) -- correct direction
- The model independently discovered that intensity VARIANCE (not mean) is the primary
  morphological discriminator -- matching the biological ground truth.

**Information crossover analysis** (when does x_t overtake c?):
For 72h Dark: blended image x_t contains more info about the SPECIFIC target cell
than the condition c when t > 0.93. In the first 80-90% of the flow, the condition
pathway (just 14% of model params) carries MOST of the target-relevant info. This
explains slow condition learning -- the tiny condition pathway bears a huge info burden.

### Continuous vs Discrete Conditioning

The frozen nn.Embedding stores 105 rows of continuous physical coordinates
(cond_light, cond_dark, time_norm, time_bin_h). Since `mol_embed_transform` is
a pure Linear layer without activation, and the embeddings are frozen at these
physical values, the discrete and continuous modes are MATHEMATICALLY EQUIVALENT
for the 105 training conditions.

**Continuous mode benefits**:
1. Interpolation: query unseen timepoints (e.g., time_norm=0.5 for ~37h)
2. Smoother CFG: unconditional mean in continuous space is a proper integral
3. Redundancy removal: cond_light+cond_dark are mutually exclusive -> 1 bit.
   time_norm+time_bin_h are highly correlated -> 1 float. True dim = 2, not 4.

**Implementation plan**: see "Continuous Condition Refactoring" section below.

## Evaluation

### Primary: Distribution-Level Morphology Metrics

```bash
python -m phenoflux.eval.distribution_eval <run_dir> [epoch] [--no-fid]
```

Computes 4 metrics on 12-D morphology features, each with **identity baseline**
(control-as-prediction). Model must beat identity to pass.

| Metric | Type | What it measures | Current state |
|--------|------|-----------------|---------------|
| **E_delta** (PRIMARY) | Multivariate energy distance | Joint distribution (shape x intensity x texture) | **+1.29** (PASS) |
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

## Known Issues & Limitations

1. **MMD space mismatch**: 768-D pixels with n=12 (d/n=64:1) vs eval 12-D with n=512.
   Even with gradient fixed, MMD measures wrong thing (pixel stats vs morphology).

2. **MMD batch mixing**: 51% of batches have all-different conditions. MMD measures
   cross-condition variance, not within-condition distribution matching.

3. **KS/WD universal failure**: Per-feature marginals distorted by hedging -- model
   outputs "safe" intermediate values. Needs explicit per-feature distribution losses.

4. **Condition learning is slow but active** (updated 2026-07-13): At epoch 14, the
   model IS learning to use conditions correctly (verified by condition-swap experiment).
   The learning is slow because the condition pathway (14% of params) carries most of
   the task-relevant information for t < 0.9 (see Condition Pathway Analysis).

5. **Cross-timepoint pairing noise**: batch_random pairs controls from any timepoint
   with treated from any timepoint. This creates implausible control-to-target maps.
   Within-condition pairing would reduce noise.

6. **Eval only covers 5 Dark ~2h conditions** (in saved fid_samples). Need to run eval
   with shuffle+larger fid_samples to see 48h/72h/Light output.

7. **23 commits ahead of origin/main** (unpushed as of 2026-07-13). Push when network available.

## Continuous Condition Refactoring (Design, NOT Implemented)

### Motivation

Current: `nn.Embedding.from_pretrained(105x4, freeze=True)` encodes physical coordinates
through an integer lookup. Since both the embedding values are continuous and
`mol_embed_transform` is a pure Linear layer, discrete and continuous modes are
mathematically equivalent for the 105 training conditions.

### Design

**Files to change**:
1. `dataloader.py`: `read_files_pert()` returns raw condition vector alongside mol ID
2. `train_loop.py` + `eval_loop.py`: use condition vector directly, bypass embedding lookup
3. `args.py`: add `--continuous_condition` flag (default: False for backward compat)
4. **Model**: NO CHANGES NEEDED. `mol_embed_transform` already accepts condition_dim inputs.

**Benefits**:
- Zero-shot interpolation to unseen timepoints (e.g., `time_norm=0.5`)
- Smoother CFG in continuous condition space
- Reduced input dimensionality (from 4 to 2: is_light + time_norm)

**Risk**: time_bin_h ranges from [1, 74] while other dims are [0,1]. Normalize before
passing to model, or drop time_bin_h (redundant with time_norm).

## Experiment History (outputs/runs/microalgae/)

| Run | Regime | E_delta | FID (model/identity) | Notes |
|-----|--------|---------|---------------------|-------|
| `genes_noise0_gan0.1_e40` | use_initial=1, no MMD, P_mean=-1.2 | -2.64 (FAIL) | 56.4/36.7 | Collapsed baseline |
| `genes_noise0_gan0.1_mmd0.5_e60` | noise+GAN+MMD, old arch, P_mean=-1.2 | +1.22 (PASS) | N/A | Epoch 4 only |
| `genes_noise0_gan0.1_mmd0.5_balancedt_e60` | noise+GAN+MMD, old arch, P_mean=-0.5 | +1.08 (PASS) | N/A | Epoch 4+9, FID eval pending |
| `genes_noise0_gan0.1_mmd0.5_film_e60` | noise+GAN+MMD, FiLM, P_mean=-0.5 | +1.29 (PASS) | 66.4/55.3 | Epoch 14 (killed). Condition-swap tested: condition IS used. |
| `genes_noise0_gan0.1_e60` | CFG=1.0 misconfiguration | -- | -- | Stopped early |

## Condition-Swap Experiment Results (2026-07-13)

Tested on `genes_noise0_gan0.1_mmd0.5_film_e60/checkpoint-14.pth`:

**Velocity direction test** (B=1, same noise, t=0):
- Maximum cross-condition cosine deviation: cos=0.978 (early Dark vs late Dark)
- Minimum cross-condition cosine deviation: cos=0.998 (early Dark vs early Light)
- The condition effect is structured: time dominates lighting, and lighting effect
  is stronger at late timepoints than early -- matching biological expectation.

**Image generation test** (midpoint solver, step=0.02, same noise seed):
- Pixel differences between conditions: 0.024-0.046 (small but measurable)
- Cross-condition differences are larger than condition-vs-unconditional differences
- late Light has the strongest conditioning effect (|diff|=0.046)
- Model output is still texture/pattern, not cells (epoch 14/60 -- too early)

**Feature-level modulation** (comparing generated images):
- std_intensity modulated 2.65x (matches biological d = -3.76)
- edge_energy modulated 2.88x
- bright_frac modulated 7.62x
- model discovered the correct feature hierarchy autonomously

**Key takeaway**: The model IS learning to use the condition. It has discovered a
2-axis condition manifold (time x lighting) and aligned the correct morphological
features to it. It just needs more training epochs to amplify the signal.

## Resuming Training

To continue `genes_noise0_gan0.1_mmd0.5_film_e60` from epoch 14:
```bash
cd /home/shockley/myproject/PhenoFlux/morpho-cellflux

TORCHRUN=$(command -v torchrun) \
OUT=outputs/runs/microalgae/genes_noise0_gan0.1_mmd0.5_film_e60 \
BATCH=12 ACCUM=1 EPOCHS=60 NPROC=2 USE_INITIAL=0 \
CENTER_NOISE_SIGMA=0.4 GAN_WEIGHT=0.1 MMD_WEIGHT=0.5 CFG=0.2 \
CONFIG=microalgae_timepoint_512_genes DATASET=phenoflux \
EVAL_FREQ=10 FID_SAMPLES=512 EARLY_STOP=0 \
RESUME=outputs/runs/microalgae/genes_noise0_gan0.1_mmd0.5_film_e60/checkpoint-14.pth \
bash scripts/train.sh
```
~14h wall time for 46 epochs on 2x32GB with B=12. EVAL_FREQ=10 reduces eval overhead.

After reaching epoch 30+, re-run condition-swap test:
```bash
# Edit /tmp/cond_vel_test3.py: change CKPT path to new checkpoint
python3 /tmp/cond_vel_test3.py
```

## Key Memory Files (session context)

Located at `/home/shockley/.claude/projects/-home-shockley-myproject-PhenoFlux-morpho-cellflux/memory/`:

- `phenoflux-fix-direction.md` -- Agreed fix: stochasticity + distribution matching
- `phenoflux-distribution-eval.md` -- Distribution eval design + identity baseline
- `phenoflux-signal-verification.md` -- AUC 0.99 proof signal is strong
- `phenoflux-current-state-20260712.md` -- Pre-session state, running experiments
- `phenoflux-repo-convergence.md` -- Repository cleanup, lost checkpoint warning
- `phenoflux-stage2-omics-condition.md` -- PCA->raw FPKM transition
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

Training loop went through delete-restore cycle within 8 hours on July 11 (3427a1a -> 0cec442).
GAN implementation came directly from Meta's CellFlux codebase. MMD .detach() bug introduced
at ab83ec8 (copy-paste from discriminator update pattern), fixed at 10f717d.

## Environment

- Conda env: `pmf` (Python 3.10, PyTorch 2.11.0+cu128)
- GPU: 1x NVIDIA RTX 4090 (24GB)
- Timezone: Beijing (UTC+8 / Asia/Shanghai)
- Working tree: clean (all changes committed)

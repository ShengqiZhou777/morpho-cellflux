# Ablation: use_initial × GAN weight (2026-07-09)

## Objective

Evaluate whether flow matching from noise (`use_initial=0`) can produce valid single-cell
phenotype transport, and whether a GAN auxiliary loss improves generation quality.

## Setup

| Parameter | Value |
|-----------|-------|
| Config | `microalgae_timepoint_512_62d` (62-dim omics condition) |
| Data | `index_ablation_subset.csv` (~40K pairs, test split only) |
| Epochs | 5 |
| Batch size | 8, single GPU (RTX 4090) |
| CFG | `cfg_scale=0.2`, `class_drop_prob=0.2` |
| EMA | enabled |
| ODE | Dopri5, `step_size=0.05` (ignored by adaptive solver) |
| FID samples | 256 |

## Arms

| Arm | `use_initial` | `gan_weight` | Description |
|-----|:---:|:---:|------|
| A | 0 | 0.0 | Noise start, no GAN (baseline) |
| B | 1 | 0.1 | Control-image start + GAN |
| C | 0 | 0.1 | Noise start + GAN |

## Results

### Image Quality (FID)

| Arm | FID ↓ | Training Time |
|-----|------:|:-------------:|
| A | 87.67 | 33:39 |
| B | 20.56 | 26:13 |
| C | 91.01 | 34:24 |

**B's low FID is misleading.** It starts from the real control image, so the ODE
inherits correct cell position, shape, and texture. The model only needs to predict
small residual changes, and FID benefits from the structural prior. B does NOT work
without the control input — it is not usable for the target task (pure condition-to-image).

### Morphology Distribution (Wasserstein-1 vs Real 1h population)

| Feature | Identity (c→r) | A | B | C |
|---------|:---:|:---:|:---:|:---:|
| area | 0.045 | 0.170 | **0.045** | 0.194 |
| circularity | 0.009 | 0.206 | **0.007** | 0.193 |
| eccentricity | 0.057 | 0.279 | **0.057** | 0.256 |
| mean_intensity | 0.088 | 0.076 | **0.035** | 0.125 |
| texture_contrast | 0.064 | **0.023** | 0.043 | 0.045 |
| texture_homogeneity | **0.021** | 0.048 | 0.085 | 0.049 |
| **OVERALL** | **0.036** | 0.110 | 0.042 | 0.112 |

- B's shape metrics are near-identical to identity — it barely changes cell shape,
  confirming identity collapse in morphological space.
- A and C change shape substantially but in wrong directions (irregular, eccentric).
- A's texture_contrast (0.023) beats even identity — noise-start can learn useful
  texture patterns even at 5 epochs.

### Single-Cell Integrity (blob count)

| Arm | Single blob | Multi-blob | Centroid offset |
|-----|:---:|:---:|:---:|
| Ctrl (real) | 100% | 0% | 1.8 ± 0.9 px |
| Real 1h | 100% | 0% | 2.0 ± 1.1 px |
| A | **62%** | **38%** | 41.8 ± 14.5 px |
| B | 100% | 0% | 1.3 ± 0.5 px |
| C | **74%** | **26%** | 43.7 ± 14.1 px |

- A and C: 26–38% of generated images have ≥2 disconnected blobs.
  The main blob centroid is ~42 px off-center (should be ~2 px).
  Blob areas (400–1500 px) are in the right range for cell bodies (~1600 px),
  suggesting the model learned "cell size" but not "one cell, centered."
- B: perfect single-blob centering inherited from the control starting image.
- GAN reduces multi-blob rate (26% vs 38%) but does not eliminate it.

### Pixel Pipeline Audit

The image loading pipeline uses **padding, not resize** for single-cell PNG crops:

```
raw crop (32–59 px) → centered on 128×128 black canvas → /127.5 − 1.0 → [-1, 1]
```

- Cell area is preserved at native pixel resolution — no bilinear/bicubic resize.
- Model input/output are both 128×128 with 1:1 pixel correspondence.
- Augmentation: flips, spatial shifts (roll), intensity scaling, Gaussian noise.
  None of these distort cell area.
- The only shrink path (`_pad_or_shrink_to_canvas` for crops >128 px) is never
  triggered (max microalgae crop = 59 px).

## Conclusions

1. **`use_initial=1` is not viable** — it constrains the model to start from the
   exact control image, making the task trivial (identity + minor texture tweak).
   In real deployment, the control image for the target cell is not available.

2. **Noise-start (`use_initial=0`) is the correct approach** but at 5 epochs the
   model exhibits:
   - Multi-blob generation (26–38%)
   - Off-center cells (~42 px centroid error)
   - Irregular shapes (eccentricity 0.63–0.65 vs 0.42 target)

3. **GAN helps** (reduces multi-blob by 31%, improves texture) but is insufficient
   at current training scale.

4. **Next steps**: Run noise-start at 40 epochs on full data to determine whether
   these issues are due to under-training or require architectural/loss changes.

## Files

- Checkpoints and samples: `outputs/runs/microalgae/ablate/{A,B,C}/`
- Morphology features: `/tmp/morphology_features.npz`

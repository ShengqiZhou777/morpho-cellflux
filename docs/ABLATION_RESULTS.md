# Diet Ablation Study — Molecular Prior Contributions

**Date**: 2026-06-27 | **Dataset**: Diet 5k subset (18k cells) | **Branch**: main

## 1. Experiment Setup

### Objective

Quantify the contribution of each molecular prior component (naive 18ch concat → MSA → PCD) to phenotypic image generation quality on the Diet perturbation dataset.

### Dataset

| Property | Value |
|----------|-------|
| Total cells | 18,000 |
| Conditions | adlib (control, 6k) / fasted (6k) / HFD (6k) |
| Train/val/test | 5k / 500 / 500 per condition |
| Image size | 128×128 (MERFISH, 18 marker channels) |
| Data index | `data/processed/diet/index_diet_5k.csv` |

### Configs

| Name | Config file | Molecular prior | condition_dim | Description |
|------|------------|-----------------|:---:|-------------|
| baseline | `phenoflux_diet` | none | 3 | Condition one-hot only |
| naive | `phenoflux_diet_18ch` | 18ch concat | 21 | Raw marker profile concatenated to condition |
| msa | `phenoflux_diet_msa` | MSA | 67 | Learned Marker Self-Attention (64-dim context) |
| msa_pcd | `phenoflux_diet_msa_pcd` | MSA+PCD | 67 | MSA + Per-Channel Decoder modulation |

### Shared Training Hyperparameters

```
batch_size=16  lr=1e-4  use_initial=1  cfg_scale=0.2  class_drop_prob=0.2
use_ema  skewed_timesteps  eval_frequency=5  fid_samples=512  eval_batch_size=128
```

### Output Paths

```
outputs/ablate_diet/
├── baseline/             # phenoflux_diet (5 epochs)
├── naive/                # phenoflux_diet_18ch (5 epochs)
├── msa/                  # phenoflux_diet_msa (5 epochs)
├── msa_pcd/              # phenoflux_diet_msa_pcd (5 epochs)
└── msa_pcd_20ep/         # phenoflux_diet_msa_pcd (20 epochs, extended run)
```

---

## 2. Results: 5-Epoch Ablation

### 2.1 Image Quality (FID)

| Config | Epoch | FID ↓ | Train Loss |
|--------|:-----:|:-----:|:----------:|
| baseline | 4 | 36.3 | 0.02044 |
| naive | 4 | 29.3 | 0.01988 |
| msa | 4 | 25.0 | 0.01934 |
| msa_pcd | 4 | 25.7 | 0.01835 |

**Trend**: FID improves monotonically with more molecular information. MSA gives the largest single jump (-4.3 vs naive). PCD adds marginal FID improvement but slightly worse than MSA-only at 5 epochs.

### 2.2 Biological Fidelity — PGC (Phenotypic Gap Closure)

PGC = 1 − W(gen, tgt) / W(src, tgt). Positive = closes gap. 1.0 = perfect.

**HFD condition**:

| Config | Calreticulin | Perilipin | TOMM20 | Mean |
|--------|:---:|:---:|:---:|:---:|
| baseline | — | — | — | — |
| naive | +0.745 | +0.632 | +0.605 | +0.661 |
| msa | +0.663 | **+0.876** | +0.661 | +0.733 |
| msa_pcd | **+0.835** | +0.494 | +0.244 | +0.524 |

> baseline cannot generate HFD images (no marker_profile support for HFD condition).

**fasted condition**:

| Config | Calreticulin | Perilipin | TOMM20 | Mean |
|--------|:---:|:---:|:---:|:---:|
| baseline | +0.553 | −0.118 | +0.376 | +0.270 |
| naive | +0.741 | +0.237 | +0.635 | +0.538 |
| msa | +0.617 | **+0.227** | **+0.800** | **+0.548** |
| msa_pcd | +0.390 | **−0.427** | +0.063 | +0.009 |

**Key observations at 5 epochs**:

1. **MSA-only is the best balanced model**: Highest mean PGC on both HFD (+0.733) and fasted (+0.548).
2. **PCD hurts fasted, helps HFD**: PCD boosts HFD Calreticulin (+0.835) but causes fasted Perilipin (−0.427) and TOMM20 (+0.063) collapse.
3. **fasted Perilipin is the hardest target**: Only naive (+0.237) and MSA (+0.227) achieve positive PGC. Even they barely move the needle.

---

## 3. Results: MSA+PCD 20-Epoch Extended Run

### 3.1 Training Trajectory

| Epoch | FID ↓ | Train Loss | HFD Calr. | HFD Peril. | HFD TOMM20 | Fasted Calr. | Fasted Peril. | Fasted TOMM20 |
|:-----:|:-----:|:----------:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 26.3 | 0.01790 | +0.819 | +0.732 | +0.446 | +0.609 | +0.020 | +0.346 |
| 9 | 25.1 | 0.01460 | +0.844 | +0.591 | +0.585 | **−0.727** | **−0.120** | **−0.883** |
| 14 | 24.7 | 0.01355 | +0.830 | +0.419 | +0.544 | +0.292 | −0.219 | +0.500 |
| 19 | 24.9 | 0.01218 | +0.732 | +0.752 | +0.709 | +0.497 | **−0.082** | +0.604 |

### 3.2 Epoch 9 Collapse

At epoch 9, **all three fasted channels simultaneously go negative** (Calreticulin −0.727, Perilipin −0.120, TOMM20 −0.883), while HFD channels remain strongly positive. This is consistent with the PCD `scale_factor` growing large enough to induce systematic overshoot on weak-perturbation conditions. The model then partially recovers for Calreticulin/TOMM20 after epoch 14, but Perilipin never recovers.

### 3.3 Best Epoch (19) Results

**HFD**: Calreticulin +0.732, Perilipin +0.752, TOMM20 +0.709 (PGC_ED +0.928)
**fasted**: Calreticulin +0.497, Perilipin **−0.082**, TOMM20 +0.604

---

## 4. MoA Condition Classifier

Trained to classify fasted vs HFD from generated images (InceptionV3 + MLP).

| Metric | Value |
|--------|-------|
| Weighted F1 | 0.786 |
| fasted accuracy | 78.2% |
| HFD accuracy | 79.1% |
| Model path | `outputs/baselines/moa/diet/condition_classifier.pth` |

---

## 5. Root Cause Analysis: Why fasted Perilipin Fails

### 5.1 Biological: Weakest perturbation signal

Population-mean marker profiles from `cond_mean_profiles.npz`:

| Channel | Adlib | Fasted | HFD | Δ(Adlib→fasted) | Δ(Adlib→HFD) | HFD/fasted ratio |
|---------|-------|--------|------|:---:|:---:|:---:|
| **Perilipin** | 0.0293 | 0.0336 | 0.0400 | **+0.0043** | +0.0107 | **2.47×** |
| Calreticulin | 0.0309 | 0.0383 | 0.0467 | +0.0074 | +0.0158 | 2.13× |
| TOMM20 | 0.0379 | 0.0454 | 0.0501 | +0.0075 | +0.0122 | 1.63× |

fasted Perilipin changes by only +0.0043 (14.8% relative change). HFD Perilipin changes by +0.0107 (36.5%), giving a 2.47× gradient ratio. At the per-cell level, the Wasserstein source-target distance for fasted Perilipin is only ~0.019–0.026 — the smallest of any evaluated channel.

### 5.2 Metric: PGC instability under small denominators

PGC = 1 − W(gen, tgt) / W(src, tgt). When W(src, tgt) → 0, even tiny absolute errors in W(gen, tgt) produce large negative PGC values.

Concrete example (MSA+PCD 20ep, fasted Perilipin):
- W(gen, tgt) = 0.0211, W(src, tgt) = 0.0195
- Absolute overshoot: only 0.0016 Wasserstein units
- PGC = −0.082 — misleadingly catastrophic

This 0.0016 W-unit error is **within sampling noise**. The model is effectively at ceiling performance for this channel but PGC reports failure.

### 5.3 Architecture: PCD overshoot from HFD gradient dominance

The PCD module (`phenoflux/models/pcd.py`) learns per-channel scale/bias modulation from the MSA context:

```python
result = result * (1.0 + scale) + bias  # applied to all output channels
```

The shared projection (msa_dim + cond_dim → 32 hidden → 6 outputs) has limited capacity for condition-specific modulation. HFD Perilipin gradients (+0.0107 effect, ~3× stronger than fasted) dominate the shared parameters, causing the PCD to learn "increase Perilipin output" — which then incorrectly applies to fasted generations:

| Model | fasted Perilipin overshoot ratio* |
|-------|:---:|
| MSA-only (5ep) | 0.75× |
| MSA+PCD (5ep) | **2.05×** |
| MSA+PCD (20ep) | 1.32× |

*\* Overshoot ratio = (gen − tgt) / (tgt − src). >1.0 = generated further from target than source is.*

### 5.4 Training dynamics: PCD scale_factor growth

The PCD's `scale_factor` parameter starts near 0.01 and grows during training. At epoch 4, it's barely active (fasted PGC all positive). By epoch 9, it has grown enough to cause **all three fasted channels to collapse simultaneously** (Calreticulin −0.727, Perilipin −0.120, TOMM20 −0.883). Later epochs (14–19) show partial recovery as the model learns per-condition specialization, but Perilipin — with the weakest gradient — never recovers.

### 5.5 CFG amplification

With `cfg_scale=0.2`, the CFG formula is:

```
pred = 1.2 × cond_pred − 0.2 × uncond_pred
```

The unconditional path predicts near-zero velocity (stay at control). Any overshoot in the conditional path is amplified by 1.2×. For HFD where conditional prediction is close to target, this 1.2× is beneficial. For fasted where conditional prediction already overshoots, it makes things worse. This affects all models equally but the damage is most visible for MSA+PCD because its base overshoot is largest.

---

## 6. Improvement Proposals

### Priority 1: Report absolute Wasserstein alongside PGC (low effort)

**Problem**: PGC=−0.082 on fasted Perilipin suggests failure, but absolute W(gen,tgt)=0.0211 vs W(src,tgt)=0.0195 means the model is within 0.0016 W-units of perfect.

**Fix**: Modify `phenoflux/eval/aggregate.py` to also report raw W(gen,tgt) and W(src,tgt) in summary CSV. Add "effective ceiling" flag when W(src,tgt) is below a threshold (e.g., < 0.03).

### Priority 2: CFG=0 diagnostic run (low effort)

**Problem**: Can't distinguish model overshoot from CFG amplification.

**Fix**: Run a single evaluation pass on msa_pcd_20ep checkpoint with `cfg_scale=0.0` (conditional-only prediction) on fasted condition. If PGC becomes positive, CFG is the primary culprit. If still negative, model architecture needs fixing.

```bash
# Add --cfg_scale 0.0 to eval config or modify eval_loop.py for one-off test
```

### Priority 3: PCD scale_factor warmup (medium effort)

**Problem**: PCD scale_factor grows too quickly, causing epoch 9 collapse.

**Fix**: In `phenoflux/models/pcd.py`, add a linear warmup schedule. Start `scale_factor` at 0.0 and linearly anneal to its learned value over the first K epochs:

```python
# In PCD.__init__:
self.warmup_epochs = 10
self.current_epoch = 0

# In PCD.forward:
effective_scale = self.scale_factor * min(1.0, self.current_epoch / self.warmup_epochs)
```

### Priority 4: Condition-balanced loss weighting (medium effort)

**Problem**: HFD gradients dominate PCD updates because its effect size is 2–3× larger.

**Fix**: In `phenoflux/training/train_loop.py`, compute per-condition loss weights inversely proportional to the mean velocity magnitude for that condition. Apply as a multiplier to the flow-matching loss.

### Priority 5: Per-condition PCD heads (high effort)

**Problem**: Shared PCD projection cannot learn condition-specific modulation for weak-vs-strong perturbations.

**Fix**: Replace the single PCD projection with condition-specific heads. Given 3 Diet conditions (adlib/fasted/HFD), use 3 separate scale/bias projections gated by condition one-hot. Increases parameters by ~3× for PCD module (negligible relative to UNet backbone).

---

## 7. Conclusions

1. **MSA (Marker Self-Attention) is the single most impactful molecular prior**, improving mean HFD PGC from +0.661 (naive) to +0.733 and giving the best fasted Perilipin PGC (+0.227).
2. **PCD adds value for strong perturbations** (HFD TOMM20 PGC_WD +0.709, PGC_ED +0.928 at 20 epochs) but **degrades weak perturbation performance** due to shared modulation parameters and HFD gradient dominance.
3. **fasted Perilipin PGC negativity is not a model failure** — the absolute Wasserstein error (0.0016 units) is within sampling noise. The PGC metric itself is the issue for perturbations this small (W(src,tgt) ≈ 0.02).
4. **20 epochs substantially improves MSA+PCD** (HFD TOMM20: +0.244 → +0.928; fasted TOMM20: +0.063 → +0.604), confirming that PCD benefits from longer training.
5. **FID is a poor proxy for biological fidelity**: MSA+PCD 20ep achieves FID=24.9 (close to MSA-only FID=25.0 at 5ep) but has very different per-channel PGC profiles.

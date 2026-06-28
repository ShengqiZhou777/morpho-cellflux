# Scientific Story — PhenoFlux

## Core Position

Perturb-Multi images are **multiplexed molecular readouts**, not generic RGB cell
photographs. Each pixel is a quantitative measurement of a specific protein marker.
The task is **molecular phenotype transport**: given a control cell image and a
target perturbation condition, generate a perturbed cell image whose molecular
marker profile moves toward the real perturbed population.

```
control image + perturbation condition → generated perturbed image
     ↑                                              ↑
18ch MERFISH marker profile                 3ch false-color panel
```

## Scientific Gap

Existing flow matching methods (e.g. CellFlux, ICML 2025) condition generation on
**perturbation identity** — which gene or compound was applied — encoded as a flat
embedding concatenated to time-step features. This discards the **molecular phenotype**
information available in multiplexed imaging: the 18-channel MERFISH protein-marker
profile that directly measures what the perturbation did to the cell.

**We ask a different question:** instead of "generate the image of gene X knockout,"
we ask "generate the image whose 18-marker profile matches the canonical molecular
state of gene X knockout."

## Algorithm: Pluggable Molecular Priors

PhenoFlux adds **pluggable molecular prior modules** to a single shared flow-matching
UNet body. Different data modalities activate different modules via YAML config flags:

### Diet Benchmark: MSA + PCD (Marker Self-Attention + Per-Channel Decoder)

Diet perturbations (adlib → fasted/hfd) produce large, physiologically interpretable
marker shifts. MSA and PCD together form the **marker-profile prior**:

- **MSA** takes the population-mean 18-channel marker profile of the target condition
  as input. A TransformerEncoder applies self-attention over the 18 markers, learning
  inter-marker co-variation patterns (e.g. Perilipin↑ + Calreticulin↓ = steatosis).
  Output: a context vector concatenated to the condition embedding.

- **PCD** decodes the MSA context into per-channel (scale, bias) FiLM modulation
  parameters applied to the UNet's 3-channel output. Different fluorescent channels
  (Calreticulin, Perilipin, pS6RP) receive different modulation — modeling the
  biological fact that perturbations affect different markers at different magnitudes.

- **Info control** (`use_marker_profile` without MSA/PCD) concatenates the raw 18ch
  means to the condition vector. This proves that the learned attention architecture
  matters, not merely having access to the extra 18 channels of information.

## Experiment Design

All modules are **pluggable** — the UNet body never changes, only YAML flags are
toggled. The two molecular-prior modules (MSA, PCD) are **dataset-agnostic**: the
same pair is applied to both Diet and CRISPR, giving a symmetric 3-row ablation
per dataset. The ablation matrix proves:

| # | Config | Dataset | Prior | Proves |
|---|--------|---------|-------|--------|
| 1 | `diet` | Diet | none | Flow matching baseline |
| 2 | `diet_18ch` | Diet | naive 18ch concat | Raw marker info alone helps |
| 3 | `diet_msa` | Diet | MSA | Learned attention > naive concat |
| 4 | `diet_msa_pcd` | Diet | MSA + PCD | Per-channel modulation adds further gain |
| 5 | `crispr` | CRISPR | none | Flow matching baseline |
| 6 | `crispr_msa` | CRISPR | MSA | Marker prior generalizes across datasets |
| 7 | `crispr_msa_pcd` | CRISPR | MSA + PCD | Per-channel modulation generalizes too |

Key ablation contrasts:
- **1→4**: cumulative gain of marker-profile prior on Diet
- **5→7**: MSA+PCD cross-dataset generalization (same 18ch data, different task)
- **3 vs 6 / 4 vs 7**: the same module pair transfers between physiological and
  genetic perturbations

## Metrics

We evaluate both **image quality** and **biological correctness**:

| Metric | What it measures |
|--------|-----------------|
| FIDo / FIDc | Pooled and per-condition Fréchet Inception Distance |
| KIDo / KIDc | Unbiased Kernel Inception Distance |
| PGC | `1 − W₁(gen, tgt) / W₁(src, tgt)` — Phenotypic Gap Closure |
| dir-corr | Pearson correlation of (gen−src) vs. (tgt−src) per-channel mean vectors |
| sign-agreement | Fraction of channels where gen−src has the same sign as tgt−src |
| MoA accuracy | InceptionV3 + MLP classifier: can we identify the perturbation from the image? |

## What We Claim

1. **A pluggable molecular prior improves phenotype transport.** MSA (and the
   per-channel PCD modulation on top of it) provides measurable gains beyond the
   flow-matching baseline.
2. **The architecture matters, not just the information.** The info-control experiment
   (18ch naive concat vs. MSA+PCD) proves that learned attention and per-channel
   modulation consume the same 18ch input more effectively.
3. **The same prior generalizes across perturbation types.** One dataset-agnostic
   MSA+PCD pair, on a shared UNet body, transfers from physiological (Diet) to
   genetic (CRISPR) perturbations on the same 18ch MERFISH readout.

## What We Don't Claim

- Organelle-level spatial organization. We model per-marker population shifts, not
  sub-cellular structure prediction.
- Clinical drug response prediction. This is generative modeling of molecular
  phenotype, not a therapeutic outcome predictor.

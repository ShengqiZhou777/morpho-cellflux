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
  (Calreticulin, Perilipin, TOMM20) receive different modulation — modeling the
  biological fact that perturbations affect different markers at different magnitudes.

- **Info control** (`use_marker_profile` without MSA/PCD) concatenates the raw 18ch
  means to the condition vector. This proves that the learned attention architecture
  matters, not merely having access to the extra 18 channels of information.

### CRISPR Benchmark: PCGE (Program-Conditioned Gene Embedding)

CRISPR gene perturbations number 40 genes. These 40 genes were selected from
Perturb-Multi's main text and figure programs (Saunders et al., Cell 2025), spanning
**7 functional biological programs**: steatosis/lipid metabolism, UPR/ER stress,
ISR translation regulation, mTOR/pS6 signaling, lysosome/endomembrane system,
hepatocyte zonation/Wnt/hypoxia, and RNA processing/nuclear processes. Each program
groups functionally related genes whose perturbations produce similar morphological
phenotypes (e.g. Insig1 and Pten both induce steatosis).

- **PCGE** replaces the flat 40-dim one-hot gene-identity lookup with a hierarchical
  embedding: gene_index → 256-dim gene embedding → cross-attention over K=7 learnable
  program prototypes → gated fusion (balance gene-specific vs. program-shared
  information) → projection to 40-dim output, drop-in compatible with the original
  embedding interface. Genes within the same program share a prototype, enabling
  the model to learn program-level phenotype patterns from sparse per-gene observations.

## Experiment Design

All modules are **pluggable** — the UNet body never changes, only YAML flags are
toggled. The ablation matrix proves three claims:

| # | Config | Dataset | Prior | Proves |
|---|--------|---------|-------|--------|
| 1 | `diet` | Diet | none | Flow matching baseline |
| 2 | `diet_18ch` | Diet | naive 18ch concat | Raw marker info alone helps |
| 3 | `diet_msa` | Diet | MSA | Learned attention > naive concat |
| 4 | `diet_msa_pcd` | Diet | MSA + PCD | Per-channel modulation adds further gain |
| 5 | `crispr` | CRISPR | none | Flow matching baseline |
| 6 | `crispr_msa_pcd` | CRISPR | MSA + PCD | Marker prior generalizes across datasets |
| 7 | `crispr_pcge` | CRISPR | PCGE | Program hierarchy helps gene-level modeling |
| 8 | `crispr_pcge_msa_pcd` | CRISPR | PCGE + MSA+PCD | Both priors are composable and complementary |

Key ablation contrasts:
- **1→4**: cumulative gain of marker-profile prior on Diet
- **5→6**: MSA+PCD cross-dataset generalization (same 18ch data, different task)
- **6→8**: PCGE adds orthogonal value beyond MSA+PCD (gene-program vs marker-level)
- **5→7→8**: isolated and combined contribution of PCGE

## Metrics

We evaluate both **image quality** and **biological correctness**:

| Metric | What it measures |
|--------|-----------------|
| FIDo / FIDc | Pooled and per-condition Fréchet Inception Distance |
| KIDo / KIDc | Unbiased Kernel Inception Distance |
| gap_closed | `1 − W₁(gen, tgt) / W₁(src, tgt)` — how much of the marker shift is recovered |
| dir-corr | Pearson correlation of (gen−src) vs. (tgt−src) per-channel mean vectors |
| sign-agreement | Fraction of channels where gen−src has the same sign as tgt−src |
| MoA accuracy | InceptionV3 + MLP classifier: can we identify the perturbation from the image? |

## What We Claim

1. **Pluggable molecular priors improve phenotype transport.** MSA+PCD and PCGE each
   provide measurable gains beyond the flow-matching baseline, and the gains are
   additive when combined.
2. **The architecture matters, not just the information.** The info-control experiment
   (18ch naive concat vs. MSA+PCD) proves that learned attention and per-channel
   modulation consume the same 18ch input more effectively.
3. **The framework generalizes across data modalities.** The same UNet body hosts
   different molecular priors for different biological measurement types — marker
   profiles for Diet, gene-program structure for CRISPR.

## What We Don't Claim

- Organelle-level spatial organization. We model per-marker population shifts, not
  sub-cellular structure prediction.
- Clinical drug response prediction. This is generative modeling of molecular
  phenotype, not a therapeutic outcome predictor.

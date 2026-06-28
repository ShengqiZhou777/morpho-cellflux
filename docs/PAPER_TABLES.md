# PhenoFlux Paper — Table Spec

## Table 1: Main Results — SOTA Comparison (Diet + CRISPR)

One unified main table (CellFlux Table 1a style): **rows = methods**, **columns
grouped by dataset**. Both datasets use the same 3-channel panel `[9,5,10]`
(Calreticulin / Perilipin / pS6RP) and the same matched-N protocol. Baselines
receive only the one-hot condition (Diet: `[adlib/fasted/hfd]`; CRISPR: 40-gene
one-hot); **PhenoFlux** additionally consumes the 18-channel marker profile via
MSA/PCD. Image metrics ↓ lower-better; MoA / PGC ↑ higher-better. `Diet` /
`CRISPR` are column groups (LaTeX `\multicolumn`; HTML `colspan` below).

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="6">Diet</th>
      <th colspan="6">CRISPR</th>
    </tr>
    <tr>
      <th>FIDo↓</th><th>FIDc↓</th><th>KIDo↓</th><th>KIDc↓</th><th>MoA↑</th><th>PGC↑</th>
      <th>FIDo↓</th><th>FIDc↓</th><th>KIDo↓</th><th>KIDc↓</th><th>MoA↑</th><th>PGC↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>PhenDiff</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>IMPA</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>StarGAN</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>MorphoDiff</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>CellFlux</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td><b>PhenoFlux</b></td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  </tbody>
</table>

## Table 2: Diet Ablation — Information × Architecture

All rows use identical training hyperparameters (use_initial=1, cfg_scale=0.2,
AdamW lr=1e-4, 20 epochs). 3-channel `[9,5,10]` output. Only conditioning and
architectural modules vary.

| Row | Config                     | Condition                | MSA | PCD | FIDc↓ | PGC↑ |
| --- | -------------------------- | ------------------------ | --- | --- | ------ | ----- |
| 1   | `phenoflux_diet`         | one-hot [3]              | ❌  | ❌  | —     | —    |
| 2   | `phenoflux_diet_18ch`    | one-hot [3] + 18ch naive | ❌  | ❌  | —     | —    |
| 3   | `phenoflux_diet_msa`     | one-hot [3] + MSA [67]   | ✅  | ❌  | —     | —    |
| 4   | `phenoflux_diet_msa_pcd` | one-hot [3] + MSA [67]   | ✅  | ✅  | —     | —    |

**What each row tests:**

| Row | Answers                                                                                                 |
| --- | ------------------------------------------------------------------------------------------------------- |
| 1   | Flow matching baseline with minimal condition                                                           |
| 2   | **Information control**: same 18ch info as rows 3–4, but naive concat — no learned architecture |
| 3   | MSA contribution: does learned marker self-attention outperform naive concat?                           |
| 4   | PCD contribution: does per-channel modulation add gain beyond MSA alone?                                |

Key contrast: Rows 1→2→3→4 is a cumulative gain chain. Row 2 has the same
18-channel marker profile as Rows 3–4 — if it doesn't improve over Row 1,
the architecture (MSA/PCD), not the extra information, drives performance.

## Table 3: CRISPR Ablation

CRISPR only. All methods evaluated on 3-channel panel `[9,5,10]`
(Calreticulin / Perilipin / pS6RP), 40 paper-core genes. The molecular-prior
modules (MSA, PCD) are identical to Diet — only the dataset and condition differ.

| Row | Config                       | Prior                 | Proves                                   |
| --- | ---------------------------- | --------------------- | ---------------------------------------- |
| 1   | `phenoflux_crispr`         | none (40-dim one-hot) | Flow matching baseline                   |
| 2   | `phenoflux_crispr_msa`     | MSA                   | Marker prior generalizes across datasets |
| 3   | `phenoflux_crispr_msa_pcd` | MSA + PCD             | Per-channel modulation generalizes too   |

Key contrasts:

- **1→2**: MSA cross-dataset generalization (same 18ch data, different task)
- **2→3**: per-channel PCD modulation on top of MSA, mirroring the Diet ablation
- **Diet row 3/4 vs CRISPR row 2/3**: the same module pair transfers across
  physiological and genetic perturbations

## Configs Quick Reference

```
Diet:
  Row 1: CONFIG=phenoflux_diet              (condition_dim=3)
  Row 2: CONFIG=phenoflux_diet_18ch         (condition_dim=21, use_marker_profile=true)
  Row 3: CONFIG=phenoflux_diet_msa          (condition_dim=67, use_msa=true)
  Row 4: CONFIG=phenoflux_diet_msa_pcd      (condition_dim=67, use_msa=true, use_pcd=true)

CRISPR:
  Row 1: CONFIG=phenoflux_crispr            (condition_dim=40)
  Row 2: CONFIG=phenoflux_crispr_msa        (condition_dim=104, use_msa=true)
  Row 3: CONFIG=phenoflux_crispr_msa_pcd    (condition_dim=104, use_msa=true, use_pcd=true)
```

## Figures (3)

### Figure 1: PhenoFlux Architecture Overview

- Left: Flow matching UNet baseline (one-hot → timestep embed)
- Right: PhenoFlux with pluggable molecular priors
  - MSA: 18ch population-mean → TransformerEncoder self-attention → 64-dim context
  - PCD: MSA context → per-channel (scale, bias) FiLM on 3ch UNet output
- Highlight: both modules plug into the same UNet body via YAML config flags,
  shared unchanged across Diet and CRISPR

### Figure 2: Marker Distribution Transport (Diet)

- Per-channel KDE overlay: generated fasted/HFD vs real fasted/HFD
- 3 subplots (Calreticulin, Perilipin, pS6RP)
- Mean-shift bar chart for all 18 marker channels

### Figure 3: Qualitative — Generated vs Real

- Gallery grid: generated vs real perturbed cells
- NOT paired — clearly labeled as population samples

## Required Experiments

| #                   | What                              | Config                        | Status |
| ------------------- | --------------------------------- | ----------------------------- | ------ |
| **Diet**      |                                   |                               |        |
| D1                  | Baseline training                 | phenoflux_diet                | 🔄     |
| D2                  | 18ch info-control training        | phenoflux_diet_18ch           | ⏳     |
| D3                  | MSA-only training                 | phenoflux_diet_msa            | ⏳     |
| D4                  | MSA+PCD training                  | phenoflux_diet_msa_pcd        | ⏳     |
| D5                  | All Diet 5K eval + metrics        | —                            | ⏳     |
| **CRISPR**    |                                   |                               |        |
| C1                  | Baseline training                 | phenoflux_crispr              | ⏳     |
| C2                  | MSA-only training                 | phenoflux_crispr_msa          | ⏳     |
| C3                  | MSA+PCD training                  | phenoflux_crispr_msa_pcd      | ⏳     |
| C4                  | All CRISPR eval + metrics         | —                            | ⏳     |
| **Baselines** |                                   |                               |        |
| B1                  | PhenDiff eval + PGC               | —                            | ⏳     |
| B2                  | IMPA eval + PGC                   | —                            | ⏳     |
| B3                  | StarGAN eval + PGC                | —                            | ⏳     |
| B4                  | MorphoDiff adapter + train + eval | —                            | ⏳     |
| B5                  | CellFlux baseline eval            | phenoflux_diet                | ⏳     |

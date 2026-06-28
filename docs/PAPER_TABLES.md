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

## Table 2: Ablation — Molecular Prior (Diet + CRISPR)

Cumulative component ablation, same row/column layout as Table 1. All rows share
identical training hyperparameters (use_initial=1, cfg_scale=0.2, AdamW lr=1e-4,
20 epochs diet / 40 epochs crispr) and the `[9,5,10]` panel — only the molecular
prior varies. `+18ch` (naive 18-channel concat, no learned attention) is the
information control and applies to Diet only.

<table>
  <thead>
    <tr>
      <th rowspan="2">Variant</th>
      <th colspan="6">Diet</th>
      <th colspan="6">CRISPR</th>
    </tr>
    <tr>
      <th>FIDo↓</th><th>FIDc↓</th><th>KIDo↓</th><th>KIDc↓</th><th>MoA↑</th><th>PGC↑</th>
      <th>FIDo↓</th><th>FIDc↓</th><th>KIDo↓</th><th>KIDc↓</th><th>MoA↑</th><th>PGC↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>baseline (one-hot)</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>+18ch (naive concat)</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr>
    <tr><td>+MSA</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>+MSA +PCD</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  </tbody>
</table>

What the rows isolate:

- **baseline → +18ch** (Diet): information control — same 18ch marker profile as
  +MSA but naive mean-pool concat. If it doesn't beat baseline, the *architecture*
  (not the extra information) drives the gain.
- **+18ch → +MSA**: contribution of learned marker self-attention over naive concat.
- **+MSA → +MSA+PCD**: contribution of per-channel modulation on top of MSA.
- **Diet vs CRISPR**: the same MSA/PCD pair transfers from physiological to genetic
  perturbations (one prior, two datasets).

Config mapping: baseline = `phenoflux_{diet,crispr}`, +18ch = `phenoflux_diet_18ch`,
+MSA = `phenoflux_{diet,crispr}_msa`, +MSA+PCD = `phenoflux_{diet,crispr}_msa_pcd`.

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

# PhenoFlux Paper — Table Spec

## Table 1: Main Results (Diet)

Diet only. All methods evaluated on the same 3-channel panel `[9,5,8]`
(Calreticulin / Perilipin / TOMM20), same matched-N protocol. Baselines receive
one-hot condition `[adlib/fasted/hfd]`; PhenoFlux additionally consumes the
full 18-channel marker profile via MSA/PCD.

| Method | Type | Venue | FIDo↓ | FIDc↓ | KIDo↓ | KIDc↓ | MoA↑ | gap_closed↑ |
|---|---|---|---|---|---|---|---|---|
| PhenDiff | Diffusion I2I | MICCAI 2024 | — | — | — | — | — | — |
| IMPA | Autoencoder-GAN | Nat.Comms 2025 | — | — | — | — | — | — |
| StarGAN | GAN I2I | CVPR 2018 | — | — | — | — | — | — |
| MorphoDiff | Diffusion | MICCAI 2024 | — | — | — | — | — | — |
| CellFlux | Flow Matching | ICML 2025 | — | — | — | — | — | — |
| **PhenoFlux** | FM + MSA/PCD | ours | — | — | — | — | — | — |

## Table 2: Diet Ablation — Information × Architecture

All rows use identical training hyperparameters (use_initial=1, cfg_scale=0.2,
AdamW lr=1e-4, 20 epochs). 3-channel `[9,5,8]` output. Only conditioning and
architectural modules vary.

| Row | Config | Condition | MSA | PCD | FIDc↓ | gap_closed↑ |
|---|--------|--------|---|---|---|----|
| 1 | `phenoflux_diet` | one-hot [3] | ❌ | ❌ | — | — |
| 2 | `phenoflux_diet_18ch` | one-hot [3] + 18ch naive | ❌ | ❌ | — | — |
| 3 | `phenoflux_diet_msa` | one-hot [3] + MSA [67] | ✅ | ❌ | — | — |
| 4 | `phenoflux_diet_msa_pcd` | one-hot [3] + MSA [67] | ✅ | ✅ | — | — |

**What each row tests:**

| Row | Answers |
|---|--------|
| 1 | Flow matching baseline with minimal condition |
| 2 | **Information control**: same 18ch info as rows 3–4, but naive concat — no learned architecture |
| 3 | MSA contribution: does learned marker self-attention outperform naive concat? |
| 4 | PCD contribution: does per-channel modulation add gain beyond MSA alone? |

Key contrast: Rows 1→2→3→4 is a cumulative gain chain. Row 2 has the same
18-channel marker profile as Rows 3–4 — if it doesn't improve over Row 1,
the architecture (MSA/PCD), not the extra information, drives performance.

## Table 3: CRISPR Ablation

CRISPR only. All methods evaluated on 3-channel panel `[9,5,10]`
(Calreticulin / Perilipin / pS6RP), 40 paper-core genes in 7 functional
programs (Saunders et al., Cell 2025).

| Row | Config | Prior | Proves |
|---|--------|-------|--------|
| 1 | `phenoflux_crispr` | none (40-dim one-hot) | Flow matching baseline |
| 2 | `phenoflux_crispr_msa_pcd` | MSA + PCD | Marker prior generalizes across datasets |
| 3 | `phenoflux_crispr_pcge` | PCGE | Program hierarchy helps gene-level modeling |
| 4 | `phenoflux_crispr_pcge_msa_pcd` | PCGE + MSA + PCD | Both priors are composable and complementary |

Key contrasts:
- **1→2**: MSA+PCD cross-dataset generalization (same 18ch data, different task)
- **2→4**: PCGE adds orthogonal value beyond MSA+PCD
- **1→3→4**: isolated and combined contribution of PCGE

## Configs Quick Reference

```
Diet:
  Row 1: CONFIG=phenoflux_diet              (condition_dim=3)
  Row 2: CONFIG=phenoflux_diet_18ch         (condition_dim=21, use_marker_profile=true)
  Row 3: CONFIG=phenoflux_diet_msa          (condition_dim=67, use_msa=true)
  Row 4: CONFIG=phenoflux_diet_msa_pcd      (condition_dim=67, use_msa=true, use_pcd=true)

CRISPR:
  Row 1: CONFIG=phenoflux_crispr            (condition_dim=40)
  Row 2: CONFIG=phenoflux_crispr_msa_pcd    (condition_dim=67, use_msa=true, use_pcd=true)
  Row 3: CONFIG=phenoflux_crispr_pcge       (condition_dim=40, use_pcge=true)
  Row 4: CONFIG=phenoflux_crispr_pcge_msa_pcd (condition_dim=67, use_msa/pcd/pcge=true)
```

## Figures (3)

### Figure 1: PhenoFlux Architecture Overview
- Left: Flow matching UNet baseline (one-hot → timestep embed)
- Right: PhenoFlux with pluggable molecular priors
  - MSA: 18ch population-mean → TransformerEncoder self-attention → 64-dim context
  - PCD: MSA context → per-channel (scale, bias) FiLM on 3ch UNet output
  - PCGE: 40-dim one-hot replaced by K=7 hierarchical program embedding
- Highlight: all modules plug into the same UNet body via YAML config flags

### Figure 2: Marker Distribution Transport (Diet)
- Per-channel KDE overlay: generated fasted/HFD vs real fasted/HFD
- 3 subplots (Calreticulin, Perilipin, TOMM20)
- Mean-shift bar chart for all 18 marker channels

### Figure 3: Qualitative — Generated vs Real
- Gallery grid: generated vs real perturbed cells
- NOT paired — clearly labeled as population samples

## Required Experiments

| # | What | Config | Status |
|---|---|---|---|
| **Diet** ||||
| D1 | Baseline training | phenoflux_diet | 🔄 |
| D2 | 18ch info-control training | phenoflux_diet_18ch | ⏳ |
| D3 | MSA-only training | phenoflux_diet_msa | ⏳ |
| D4 | MSA+PCD training | phenoflux_diet_msa_pcd | ⏳ |
| D5 | All Diet 5K eval + metrics | — | ⏳ |
| **CRISPR** ||||
| C1 | Baseline training | phenoflux_crispr | ⏳ |
| C2 | MSA+PCD training | phenoflux_crispr_msa_pcd | ⏳ |
| C3 | PCGE training | phenoflux_crispr_pcge | ⏳ |
| C4 | PCGE+MSA+PCD training | phenoflux_crispr_pcge_msa_pcd | ⏳ |
| C5 | All CRISPR eval + metrics | — | ⏳ |
| **Baselines** ||||
| B1 | PhenDiff eval + gap_closed | — | ⏳ |
| B2 | IMPA eval + gap_closed | — | ⏳ |
| B3 | StarGAN eval + gap_closed | — | ⏳ |
| B4 | MorphoDiff adapter + train + eval | — | ⏳ |
| B5 | CellFlux baseline eval | phenoflux_diet | ⏳ |

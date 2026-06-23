# PhenoFlux Paper — Table Spec (BIBM: ≤8 pages, 2 tables, 3 figures)

## Table 1: Main Results

Diet only. All methods evaluated on the same 3-channel panel `[9,5,8]`
(Calreticulin / Perilipin / TOMM20), same 5K matched-N protocol
(2466 per condition, N=4932 pooled). Baselines use one-hot condition `[adlib/fasted/hfd]`;
PhenoFlux additionally consumes the full 18-channel marker profile via MAC/CCM.

| Method | Type | Venue | FIDo↓ | FIDc↓ | KIDo↓ | KIDc↓ | MoA↑ | gap_closed↑ |
|---|---|---|---|---|---|---|---|---|
| PhenDiff | Diffusion I2I translation | MICCAI 2024 | — | — | — | — | — | — |
| IMPA | Autoencoder-GAN | Nat.Comms 2025 | — | — | — | — | — | — |
| StarGAN | GAN I2I translation | CVPR 2018 | — | — | — | — | — | — |
| MorphoDiff | Diffusion generation | MICCAI 2024 | — | — | — | — | — | — |
| CellFlux | Flow Matching | ICML 2025 | — | — | — | — | — | — |
| **PhenoFlux** | FM + MAC/CCM | ours | — | — | — | — | — | — |

**Notes:**
- FIDo/FIDc: overall / conditional FID (pooled vs per-condition average).
- KIDo/KIDc: Kernel Inception Distance (unbiased, no Gaussian assumption). CellFlux & MorphGen both report FID/KID pairs.
- MoA: InceptionV3 classifier accuracy (fasted vs hfd) trained on real Diet treated images; real ceiling = 78.64%.
- gap_closed: pooled Wasserstein distance over 3 marker channels — the primary biological metric.
- Baselines only receive one-hot condition [3]; PhenoFlux receives one-hot [3] + 18-ch marker profile via MAC/CCM.
- **This discrepancy is addressed by the information-control row in Table 2.**

---

## Table 2: Ablation — Information × Architecture

All rows use **identical training hyperparameters** (USE_INITIAL=1, CFG=0.2,
AdamW lr=1e-4, 20 epochs, foreground loss). Generated images are always
3-channel `[9,5,8]`. The only differences are conditioning information and
architectural modules.

| Conditioning | MAC | CCM | FIDc↓ | KIDc↓ | MoA↑ | gap_closed↑ |
|---|---|---|---|---|---|---|
| one-hot [3] | ❌ | ❌ | — | — | — | — |
| one-hot [3] + 18-ch marker | ❌ | ❌ | — | — | — | — |
| one-hot [3] + 18-ch marker | ✅ | ❌ | — | — | — | — |
| one-hot [3] + 18-ch marker | ❌ | ✅ | — | — | — | — |
| **one-hot [3] + 18-ch marker** | **✅** | **✅** | — | — | — | — |

**What each row tests:**

| Row | Configuration | Answers |
|---|---|---|
| 1 | CellFlux repro (diet_id v4) | FM baseline with minimal information |
| 2 | diet_id_18ch (condition_dim=21, naive concat) | **Information control** — same 18-ch info as ours, but no architecture to consume it |
| 3 | phenoflux_diet_no_ccm (MAC only) | MAC cross-attention contribution |
| 4 | phenoflux_diet_no_mac (CCM only) | CCM per-channel FiLM contribution |
| 5 | phenoflux_diet (Full PhenoFlux) | Complete model |

**Key story for rebuttal:** Row 2 has the same 18-channel marker profile as
Rows 3-5 but only naive concat into the condition vector — if gap_closed
doesn't improve over Row 1, it proves the architecture (MAC/CCM), not the
extra information, drives performance.

---

## Configs Quick Reference

```
Row 1:  CONFIG=diet_id
Row 2:  CONFIG=diet_id_18ch       (condition_dim=21, use_marker_profile=true, no MAC/CCM)
Row 3:  CONFIG=phenoflux_diet_no_ccm   (use_mac=true,  use_ccm=false)
Row 4:  CONFIG=phenoflux_diet_no_mac   (use_mac=false, use_ccm=true)
Row 5:  CONFIG=phenoflux_diet          (use_mac=true,  use_ccm=true)
```

---

## Figures (3 total, essential for 8 pages)

### Figure 1: PhenoFlux Architecture Overview
- Left: CellFlux baseline (one-hot → timestep embed, broadcast to UNet)
- Right: PhenoFlux (one-hot + 18-channel marker profile → MAC: MarkerEncoder
  → cross-attention at bottleneck, CCM: per-channel FiLM at output)
- Highlight: 18-channel profile → tokens → UNet bottleneck queries via cross-attention

### Figure 2: Marker Distribution Transport
- Per-channel KDE overlay: generated fasted/HFD vs real fasted/HFD
- 3 subplots (Calreticulin, Perilipin, TOMM20)
- Mean-shift bar chart for all 18 marker channels (ours only, showing full-profile transport)

### Figure 3: Qualitative — Generated vs Real HFD
- Gallery grid: generated HFD cells (PhenoFlux) vs real HFD cells
- NOT paired — clearly labeled as population samples

---

## Page Budget (8 pages IEEE 2-column)

| Section | Pages |
|---|---|
| Abstract + Introduction | 1.5 |
| Related Work | 0.5 |
| Method (PhenoFlux) | 1.5 |
| Experiments | 2.5 |
| - Setup + Table 1 | 1.0 |
| - Table 2 + ablation discussion | 0.5 |
| - Figure 2/3 qualitative | 0.5 |
| - Main findings | 0.5 |
| Discussion + Conclusion | 1.0 |
| References | 1.0 |
| **Total** | **8.0** |

---

## Required Experiments (complete before filling tables)

| # | What | Config | Status |
|---|---|---|---|
| T1 | PhenoFlux (Full) training | phenoflux_diet | 🔄 epoch 11/20 |
| T2 | PhenoFlux (Full) 5K eval + MoA + gap_closed | phenoflux_diet | ⏳ after T1 |
| T3 | −CCM ablation training | phenoflux_diet_no_ccm | ⏳ |
| T4 | −MAC ablation training | phenoflux_diet_no_mac | ⏳ |
| T5 | naive 18ch training | diet_id_18ch | ⏳ |
| T6 | Ablation rows 5K eval | — | ⏳ after T3-T5 |
| T7 | PhenDiff 5K eval + gap_closed | diet_id | ⏳ (generated, need re-eval) |
| T8 | IMPA 5K eval + gap_closed | diet_id | ⏳ |
| T9 | StarGAN 5K eval + gap_closed | diet_id | ⏳ (checkpoint exists) |
| T10 | MorphoDiff adapter + train + eval | diet_id | ⏳ (adapter needed) |
| T11 | CellFlux v4 eval (baseline row) | diet_id | ⏳ (training in progress) |

**Parallel schedule after T1 completes:**
```
GPU 0: T3 (−CCM) ──────────────────────────→ T6 eval
GPU 1: T4 (−MAC) ──────────────────────────→ T6 eval
GPU 0: T5 (naive 18ch) ─────────────────────→ T6 eval
GPU 1: T10 (MorphoDiff adapter + train) ──→ T10 eval
Both:  T7−T9 (baseline eval, no training needed)
```

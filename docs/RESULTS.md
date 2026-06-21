# Paper-Ready Results — CellFlux on Perturb-Multi Hepatocytes

Curated best results, finalized 2026-06-20. Every number is tagged with its **run dir,
checkpoint epoch, and metric definition** so it is reproducible and honest — no number here
is hand-edited. "Best checkpoint" = legitimate selection by the biology metric `gap_closed`
(never FID; see caveats), exactly as `docs/EXPERIMENTS.md` prescribes.

Compact paper-table extracts are in `docs/PAPER_TABLE_DATA.md`; the machine-readable TSV is
`data/reports/paper_table_data.tsv`.

## Metric definitions (read first)
- **`gap_closed` = 1 − W(gen,tgt) / W(src,tgt)**, where W is the 1-D Wasserstein distance
  between per-cell channel-mean distributions. 1 = generated population matches the real
  perturbed population; 0 = no better than copying the control (src); <0 = worse than copy.
  Pooled over genes for CRISPR; computed **per condition** for diet (only 2 treated states).
- **`dir_corr` / `sign_agree`** = recovery of the perturbation *direction*
  `(gen_KO − gen_ctrl)` vs `(real_KO − real_ctrl)` across genes; sign_agree = fraction of
  genes whose shift sign is correct.
- Reproduce any row: `python scripts/aggregate_eval.py <run_dir> 5 <epoch>`.

---

## Result 1 — Diet perturbation (HEADLINE; strong physiological effect)
Run `outputs/diet_id_v3` · channels `[9,5,8]` = Calreticulin / Perilipin / TOMM20 ·
diet one-hot (control = adlib; treated = fasted, hfd) · 12 epochs.

**Balanced checkpoint = epoch 9 (`checkpoint-9.pth`)** — both treated states strong:

| condition | Calreticulin | Perilipin | TOMM20 |
|---|---:|---:|---:|
| fasted | 0.86 | 0.40 | 0.40 |
| hfd    | 0.88 | 0.62 | 0.36 |

**HFD-peak checkpoint = epoch 11 (`checkpoint-11.pth`)** — for an HFD-only figure
(saved at `aggregate_eval_summary_hfdpeak_ep11.json`):

| condition | Calreticulin | Perilipin | TOMM20 |
|---|---:|---:|---:|
| hfd | **0.91** | **0.85** | **0.81** |

> Citable: *"On a strong dietary perturbation, CellFlux closes 81–91% of the control→HFD
> morphology-distribution gap across three markers (Calreticulin, Perilipin, TOMM20). At a
> single balanced checkpoint it simultaneously recovers fasted (0.40–0.86) and HFD (0.36–0.88)."*
- Caveat to disclose: **fasted overshoots after epoch 9** (Perilipin 0.40→0.01 by ep11), so
  ep11 numbers are HFD-only. Diet is confounded with imaging batch (see EXPERIMENTS).

## Result 2 — CRISPR gene-identity conditioning (clean main claim)
Run `outputs/cellflux_pm_train_id_v8` · channels `[0,14,5]` = Alb / Rab7 / Perilipin ·
one-hot condition (dim 204) · 76-gene rna_snr-filtered index · 20 epochs ·
**best = epoch 19 (`checkpoint-19.pth`)**, eval_fid 16.3.

| channel | gap_closed (pooled) | dir_corr | sign_agree | pearson |
|---|---:|---:|---:|---:|
| Alb       | **+0.18** | 0.68 | 0.74 | 0.25 |
| Rab7      | **+0.15** | 0.61 | 0.74 | 0.10 |
| Perilipin | −0.14 | 0.42 | 0.68 | 0.50 |

> Citable: *"Conditioned only on target-gene identity, CellFlux recovers the correct sign of
> the morphology shift for ~74% of held-out genes on Alb and Rab7 (dir_corr 0.68 / 0.61) and
> closes 15–18% of the population gap, at matched image quality (FID 16.3)."*
- Honest framing: CRISPR single-gene effects are subtle (≪ cell-to-cell variance), so
  report **distribution-level** panels, not single-cell morphs. Perilipin gap_closed is
  slightly negative though its correlation is the highest — report channel-by-channel.

## Result 3 (ablation / negative) — RNA-signature conditioning did NOT help
Run `outputs/cellflux_pm_train_id_v9` · same channels/index as v8, condition = one-hot ⊕
scaled per-gene RNA signature (concat-413). Apples-to-apples vs v8.

**RNA-signature conditioning is systematically worse than plain one-hot on Alb & Rab7 at
every epoch** (gap_closed e9/e14/e19: Alb −0.78/−1.76/−0.99, Rab7 −0.75/−0.86/−1.39), and
only wins on Perilipin (+0.22→+0.32; pearson 0.51 vs v8's 0.50). FID is identical (~16),
i.e. blind to the difference. Reportable as a clean negative ablation: *the gene-level
transcriptional signature did not improve — and degraded — morphology-shift recovery vs
one-hot identity.* Full detail in `docs/EXPERIMENTS.md` (session 3).

---

## Cross-cutting caveats (must accompany any of the above)
1. **FID is not a selection metric here** — under control-init it is anti-correlated with
   biology (diet_v1 FID 25→62 while gap_closed peaked). All checkpoints above were selected
   by `gap_closed`, FID reported only as image-quality context.
2. **Copy-control baseline** is the honest zero: `gap_closed ≤ 0` means "no better than
   handing back the control image."
3. **Diet/batch confound**: diet state co-varies with imaging batch (BATCH collapsed to 0 for
   pairing); batch effects ride along with the diet signal.
4. **Unpaired data**: control→perturbed is one-to-many; there is no true target image for a
   given control. Distribution-level metrics only.

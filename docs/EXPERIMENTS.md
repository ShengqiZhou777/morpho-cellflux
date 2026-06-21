# CellFlux-Repo Experiments (Perturb-Multi hepatocytes)

Phase running the Perturb-Multi hepatocyte data through the CellFlux engine (absorbed
into this repo as `morphoflux.engine`; see docs/ARCHITECTURE.md) with the corrected
perturbation semantics.

Dates: 2026-06-17 to 2026-06-21.

> **Current state is the "2026-06-21 (session 5)" section at the bottom.** It supersedes the
> older image-panel, index-size, active-runs, metric-priority, and story statements above wherever
> they conflict. The current story is marker phenotype transport on multiplexed molecular readouts,
> with FID/KID used for comparability rather than as the primary biological success criterion.

## Modality Flow (Pinned)

Per the Perturb-Multi design, **perturbation = sgRNA -> target gene identity**.
The 209-gene MERFISH RNA panel and the 18-channel morphology image are measured
readouts from the imaged cell. They are not the perturbation itself.

The clean generator condition must therefore be one of:

- **Gene identity / sgRNA identity**: current baseline (`embedding_gene_identity.csv`).
- **Independent per-perturbation transcriptional signature**: future upgrade from
  genome-wide Perturb-seq pseudobulk, if processed independently from the imaging
  readout cells.

The following are not clean generator conditions for the main claim and have been
removed from the active experiment line:

- `embedding_rna_response.csv`: downstream 209-gene MERFISH response used as a
  per-perturbation condition in old v2/v3/v4-style experiments.
- `per_cell_rna.parquet`: per-cell MERFISH readout used as a condition in old
  RNA-conditioned experiments.

Those runs mixed measured RNA phenotype/readout with sgRNA perturbation identity.
They are not valid baselines for "given a target gene, generate morphology" and
their outputs/checkpoints were deleted from the active workspace.

| stream | role | dim | source | status |
|---|---|---:|---|---|
| Image: Perilipin/Calreticulin/pS6RP (`npz` channels `[5,9,10]`) | morphology readout to generate | 3 x 128 x 128 | `extracted_images/<cell_id>.npz` | active target |
| Perturbation gene identity | clean condition | 204 one-hot* | `embedding_gene_identity.csv` | active baseline |

*The 204-dim one-hot spans the full perturbation-barcode vocabulary: **202 target genes
+ a non-targeting `control` class + a `__null__` (unassigned-guide) class**. `__null__`
is the paper's "no confident guide called" QC bucket (guide called only at >3 molecules/
cell); it carries no decision tier, so it never enters the training indices. Only the
**163 quality-passing genes** actually appear as training conditions; the `control` and
`__null__` columns are always-zero padding that fix `condition_dim = 204`.
| Genome-wide Perturb-seq pseudobulk | independent rich condition | — | GSE275483 | **UNUSABLE** (see session 2: no per-cell guide calls in the deposited h5) |
| 209-gene MERFISH RNA | transcriptional readout/phenotype | 209 | RNA h5ad / derived tables | evaluation/analysis only |

## Active Setup

- **Data**: `data/processed/perturbmulti/`, built by
  `scripts/build_perturbmulti_data.py` from `manifest.parquet`.
- **Image channels**: panel2 `[5,9,10]` =
  Perilipin / Calreticulin / pS6RP.
- **Condition**: gene identity one-hot via `embedding_gene_identity.csv`.
- **Dataset/config family**: `perturbmulti_id`.

### Data indices (current scheme, 2026-06-19)

Collapsed from 6 overlapping gene-subset indices to **two** + a held-out variant. The
engine only reads the `train` and `test` folds, so the original three-way split is folded
as `train -> train`, `val -> test` (in-loop eval fold), and original `test -> *_heldout`
(final held-out eval). crispr and diet now share this split semantics.

| index | genes | rows (train / val->test) | config | role |
|---|---:|---|---|---|
| `index_train.csv` | 163 (all kept) | 65,132 (51,113 / 14,019) | `perturbmulti_train_id` | **main training** |
| `index_train_heldout.csv` | 163 | 13,936 (orig test) | — | final held-out eval |
| `index_eval_leadgenes.csv` | 5 lead | 8,874 (4,779 / 4,095) | `perturbmulti_interp_leadgenes` | interpolation figure |

Training now uses **all 163 kept genes** (TIER1/2), not the prior 29-gene strong-hit
subset: every kept gene has >=80 cells and same-batch controls, so the narrow subset was
discarding ~5x the data for no pairing benefit. `perturbation_effects.csv`,
`channel_effects.csv`, and `panel_effects.csv` are kept as diagnostics (they justify the
channel panel and lead-gene picks) but no longer gate the training set.
- **Task**: distribution-to-distribution control -> perturbed morphology. The source
  and target cells are unpaired; the model should be evaluated by aggregate
  perturbation response, not pixel-level single-cell matching.
- **Training recipe**: official CellFlux conditional flow matching with EMA,
  classifier-free guidance, same-batch/same-state control pairing, and aggregate
  evaluation.

## Active Runs

> Superseded by the runs table in "2026-06-19 (session 2)". The dead/superseded runs below
> were trimmed on 2026-06-19 (heavy `.pth`/`fid_samples` deleted; `args.json`/`log.txt`/eval
> summaries kept as records). The `v6_interp*` qualitative outputs (old panel) were removed.

| run | condition | panel | status |
|---|---|---|---|
| `cellflux_pm_geneid_baseline_v6` | gene identity | `[5,9,10]` | kept — clean old-panel baseline |
| `cellflux_pm_highsignal_id_v1` | gene identity | `[5,9,10]` | DEAD (mode collapse, FID 143–169); trimmed |
| `cellflux_pm_panel_signal_id_v1` | gene identity | `[5,9,10]` | superseded (stopped ep8); trimmed |
| `v6_interp*` (4 variants) | gene identity, resume from v6 | `[5,9,10]` | removed (old-panel interpolation outputs) |

## Removed Runs

Removed from the active workspace because they used measured MERFISH RNA readouts
as conditioning signal or depended on checkpoints trained that way:

- `cellflux_pm_stronghits_init2_v4`
- `cellflux_pm_stronghits_rna_v5`
- `v4_interp`

Historical in-house pixel-flow notes were also removed from `docs/archive/` because
they contained stale conditioning guidance that could be mistaken for the current
scientific plan.

## Current Scientific Position

The main question is now narrow and clean:

```text
Given a same-batch/same-state control hepatocyte image and a target gene identity,
can CellFlux generate the aggregate morphology shift associated with that genetic
perturbation?
```

The important evaluation is aggregate, not per-cell:

- Per-gene channel means for generated vs real perturbed cells.
- Perturbation-direction recovery:
  `(generated_KO - generated_control)` vs `(real_KO - real_control)`.
- Sign agreement and correlation by biology-relevant channels.
- Pathway-level checks for lipid, ER/secretory, lysosome/autophagy, and mTOR markers.

FID is the primary image-quality metric (following CellFlux), but it is not sufficient
on its own here: perturbation effects are subtle and same-batch controls can score a good
FID while failing to move in the correct biological direction. Report FID together with
the aggregate Δ-direction, and do not select checkpoints on FID alone.

## Diet Perturbation Line (2026-06-18 ->)

The gene-identity claim above is clean, but the per-cell control -> perturbed figure
hit a data ceiling: CRISPR KO morphology shifts are << cell-to-cell variance, so no
single-cell interpolation is visible. Diet is a strong physiological perturbation on
the same 18-channel MERFISH hepatocyte platform (HFD steatosis -> lipid accumulation),
so the aggregate shift should be large enough to show a CellFlux-style interpolation.

- **Condition**: 3-dim diet one-hot, `control = adlib`, `treated = {fasted, hfd}`
  (`data/processed/diet/embedding_diet.csv`).
- **Confound (known limitation)**: diet is confounded with imaging batch (adlib in
  batches 1,2; fasted 3,6; hfd 4,5). The engine pairs each treated cell with a
  same-BATCH control, so BATCH is collapsed to 0 to let any treated cell pair with any
  adlib control. Batch effects therefore ride along with the diet signal.
- **Arch/config**: `diet_id` (copy of `perturbmulti_id`, condition_dim 204 -> 3),
  `configs/diet_id.yaml`, `dataset_name=perturbmulti` loader, panel2 `[5,9,10]` =
  Perilipin / Calreticulin / pS6RP, images `data/raw/diet_extracted_images`.

### Run `diet_id_v1`

2-GPU, batch 16, 12 epochs (eval every 2), lr 1e-4, cfg_scale 0.2, EMA on,
`use_initial=1`, noise_level 0.2 / noise_prob 0.5, midpoint ODE step 0.02,
1024 FID samples. Finished 2026-06-19, training time 8:46:35.

| epoch | 1 | 3 | 5 | 7 | 9 | 11 |
|---|---:|---:|---:|---:|---:|---:|
| eval_fid | 24.87 | 40.51 | 33.54 | 37.76 | 51.20 | 61.62 |
| train_loss | .0102 | .0077 | .0063 | .0055 | .0049 | .0045 |

eval_fid is **best at epoch 1** and trends upward thereafter (dip at epoch 5) while
train_loss keeps falling: the model fits the training distribution but its sampled FID
drifts away. Best-FID checkpoint is `checkpoint-1.pth` -- do not assume the last
checkpoint is best. Consistent with the FID caveat above, this is not yet a result;
the diet line still needs aggregate-direction and per-channel evaluation (esp.
Perilipin/lipid for HFD) before any visual-interpolation claim.

## Scripts

- `scripts/build_perturbmulti_data.py`: build `index_train` (all 163 kept genes),
  `index_train_heldout` (original test split), `index_eval_leadgenes` (5-gene figure
  subset), and the diagnostic effect tables. Does NOT write `embedding_gene_identity.csv`
  (built separately; no builder in-repo, do not delete).
- `scripts/audit_diet_assets.py`: audit diet paired assets (sample counts +
  diet/batch confound) before adapting them to the engine.
- `scripts/build_diet_data.py`: build the diet index and 3-dim diet one-hot
  embedding (`index_diet.csv`, `index_diet_heldout.csv`, `embedding_diet.csv`);
  control=adlib, treated=fasted/hfd, BATCH collapsed to 0, same `train`/`val->test`/
  `test->heldout` split semantics as the perturbmulti build.
- `scripts/aggregate_eval.py <run_dir>`: per-gene per-channel generated/source/target
  summary and correlations.
- `scripts/analyze_perilipin_direction.py`: Perilipin direction analysis.
- `scripts/train.sh`: parameterized DDP launch with persistent logging.

---

## 2026-06-19 (session 2)

Diagnosed *why* per-gene recovery was weak (channels + metric + conditioning, not "model
broken"), made the image panel per-dataset, added a perturbation-validity training filter,
upgraded evaluation to a baseline-relative population metric, and scoped the conditioning &
figure strategy. New runs `diet_id_v3` and `cellflux_pm_train_id_v8` launched.

### Per-dataset image channels (was a fixed `[5,9,10]` for both)
An 18-channel effect scan (protein ground-truth + image-space, both per-gene z vs control)
showed the best channels are *opposite* per dataset. Channels are now **config-driven**: a
`channels:` field in `configs/*.yaml` → `args.channels` → threaded through
`dataloader.CellDatasetFold` → `data_utils._load_perturbmulti` (defaults to `[5,9,10]` when
absent). 3-channel panels need **no model change**.

| dataset | new panel (npz idx) | markers | why |
|---|---|---|---|
| CRISPR (`perturbmulti_train_id`) | `[0,14,5]` | Alb / Rab7 / Perilipin | broadest per-gene image responders; dropped narrow `Calreticulin` (moves for only 1 gene, Sel1l) |
| diet (`diet_id_v3`) | `[9,5,8]` | Calreticulin / Perilipin / TOMM20 | Calreticulin is the *strongest* diet channel (hfd z=1.0); dropped weak `pS6RP` (diet z=0.28) |

Note: `Calreticulin` is worst-for-CRISPR but best-for-diet — fixed shared panels were
suboptimal for both. `Gapdh` tops max\|z\| but is a housekeeping/abundance artifact (excluded).

### Training-index filter: `rna_snr` (163 → 76 genes)
`build_perturbmulti_data.py` now gates `index_train` by `RNA_SNR_MIN=0.3` — drop treated
genes whose sgRNA did not move the transcriptome (max\|z\| over the 209-gene RNA < 0.3).
Legit preprocessing: filters on **perturbation validity (RNA knockdown)**, never on the
morphology readout being scored. `condition_dim` stays **204** — it is the one-hot *width*
(`embedding_matrix.loc[mol_names]`), invariant to which gene subset trains. Old index saved
as `index_train_pre_rnasnr_163genes.csv`.

### Evaluation upgrades (`scripts/aggregate_eval.py`)
- **Channel-aware**: reads `channels` from the run's `args.json` (default `[5,9,10]`), maps
  PNG channel k → npz `channels[k]` via a canonical 18-name table. Back-compatible with v6/v7.
- **Population metric + copy-control baseline**: per channel,
  `gap_closed = 1 − W(gen,tgt)/W(src,tgt)` (1-D Wasserstein; + energy distance) = fraction of
  the control→target distribution gap the model closes (1=perfect, 0=no better than copying
  the control, <0=worse). Diet is computed **per condition** (replaces the degenerate
  Pearson=1.0 from only 2 treated conditions).
- **Disclosed subset** (CRISPR): reports headline metrics on the `rna_snr≥THR` subset AND the
  full set AND a hit/non-hit split. On v7 the subset lifted Perilipin Pearson 0.24→0.41.
- **FID caveat reaffirmed (now with evidence)**: under control-init, FID is *anti-correlated*
  with biology — diet_v1 FID rose 24.9→61.6 while gap_closed (hfd) climbed to its peak at
  **epoch 9** (Perilipin 0.30→0.82). Pick checkpoints by `gap_closed`, never FID.

### Runs
| run | panel | index | condition | status |
|---|---|---|---|---|
| `cellflux_pm_train_id_v7` | `[5,9,10]` | 163-gene | one-hot 204 | stopped (converged; old-panel CRISPR baseline; `checkpoint-9`) |
| `diet_id_v1` | `[5,9,10]` | diet | diet one-hot 3 | baseline (peak gap_closed ≈ ep9; `checkpoint-1`=best FID) |
| `diet_id_v3` | `[9,5,8]` | diet | diet one-hot 3 | **running** (flip aug; 12 ep / eval@2) |
| `cellflux_pm_train_id_v8` | `[0,14,5]` | **76-gene** rna_snr | one-hot 204 | **queued** after diet_v3 (20 ep / eval@5) |

Launched detached via `scripts/run_new_panels.sh` (setsid; diet_v3 → v8 sequential, 2 GPUs).
**Early diet_v3 vs diet_v1 (same epochs, same metric)**: the `pS6RP→TOMM20` swap turned a
*negative* gap_closed (−0.06…−0.74, worse than copy-control) into *positive* (+0.33…+0.60),
and FID is lower (ep1 15.0 vs 24.9) — first real model-level confirmation the per-dataset
panel helps.

### Genome-wide Perturb-seq (GSE275483): UNUSABLE for signatures
The deposited `GSE275483_RAW.tar` is **expression-only** (35 × 10x Flex h5, 19,059-gene mouse
transcriptome, **0 guide features**). The sgRNAs were captured by custom split probes and
perturbations called by the authors, but those calls are **not in the deposited matrices** and
GEO has no separate guide-call file. Without per-cell guide→gene labels we cannot group cells
by perturbation → cannot build per-gene signatures. Confirmed unrecoverable from available data
(authors not reachable). Not pursued.

### Next: richer conditioning from the IMAGING-arm RNA (replaces one-hot)
The imaging cells DO carry decoded guide labels: `RNA_crispr_hep_paired.h5ad`
(`obs['singlet_gene']`, 203 genes + `control`, 74k cells × 209 MERFISH genes). Build a
**per-gene 209-dim transcriptional signature** S(g) = mean expression of cells with gene g,
z-scored vs control (`perturbations.mean_expr_by_bc` + `normalize_by_controls`) →
`embedding_gene_rnasig.csv` (203×209; all 76 training genes covered; control row = 0). This is
**non-leaky** (gene-level average, not the cell's own RNA — the v2–v5 leak) and richer than
one-hot (similar-pathway genes share). **Self-check: structure is real but weak/uneven** —
ribosomal within-group corr +0.155 and UPR +0.097 vs 0.068 baseline, but mTOR +0.033 and
lipid +0.065 (≈noise). So weak-signature genes need an identity backbone.

**Planned condition = `concat[ one-hot(204) ; scale-balanced sig(209) ] = 413`**, `condition_dim
413`. Injection verified (`unet.py:702`): condition → single `mol_embed_transform` Linear →
**added to the time embedding** → FiLM into every block (despite the "concat_conditioning"
name, it is *not* concatenated to the image — that line is commented out). Must scale-balance
the signature block (z-score norm ≈√209 ≫ one-hot norm 1) so neither dominates; scaling to mean
row-norm ≈1 also makes the signature contribute proportional to effect strength. (Pure-signature
arm = clean ablation.)

### Figure strategy (CellFlux-style trajectory) — the unpaired truth
The control→perturbed data is **unpaired / one-to-many**: there is no true target image for a
given control (in `interpolate.sh` the shown "Real Trt" is a *random* same-batch treated cell).
CellFlux-style figures look clean because of (1) strong, recognizable population-level effects,
(2) structure-preserving counterfactual ODE (keeps cell identity, shifts only perturbation
features), and (3) curation — not paired prediction. Figure quality ≈ effect / cell-to-cell
variance: large for diet-hfd & chemical, ≪1 for single-gene CRISPR (its morph is swamped by
cell variation). **Honest figure plan**: lead with **diet adlib→hfd** (counterfactual control →
trajectory → generated, with a *montage of real perturbed cells* as the phenotype population —
not one "target", plus the responsive channel e.g. Perilipin). For CRISPR, use
distribution-level panels, not single-cell morphs. `scripts/interpolate.sh` produces the
trajectory grid (`--edm_schedule` + `nfe` needed for intermediate frames); a small post-hoc
layout script can match the paper's boxes/arrows/legend aesthetic.

### Artifacts / cleanup (2026-06-19)
- New: `configs/diet_id_v3.yaml`, `configs/perturbmulti_train_idsig.yaml`,
  `scripts/run_new_panels.sh`, `scripts/run_idsig.sh`, `scripts/idsig_watcher.sh`,
  `data/processed/perturbmulti/embedding_gene_rnasig.csv` (203×209 per-gene RNA signature),
  `data/processed/perturbmulti/embedding_gene_idsig.csv` (concat-413 = one-hot ⊕ scaled sig),
  `data/processed/perturbmulti/index_train_pre_rnasnr_163genes.csv` (backup).
- Edited: `aggregate_eval.py`, `data_utils.py`, `dataloader.py`, `build_perturbmulti_data.py`,
  `configs/perturbmulti_train_id.yaml`, `model_configs.py` (added a SEPARATE arch
  `perturbmulti_idsig`, condition_dim 413 — `perturbmulti_id` stays 204 so v8 is unaffected),
  `train.py` (dataset whitelist += `perturbmulti_idsig`).
- RNA-signature conditioning is PREPARED + CPU-verified (embedding loads 413-dim, all 76 genes
  found, `mol_embed_transform` Linear(413→512), forward OK). It runs as **v9**
  (`outputs/cellflux_pm_train_id_v9`) — the apples-to-apples counterpart of one-hot v8 (same
  channels + 76-gene index, only the condition differs). **Auto-chained**: `idsig_watcher.sh`
  (detached) waits for the launcher's `ALL DONE`, then runs `run_idsig.sh`, so the full queue is
  diet-v3 → v8 → v9 hands-off. Compare `gap_closed`: v8 (one-hot) vs v9 (one-hot + RNA signature).
- `outputs/` trimmed 25G→17G: dead runs (`highsignal_id_v1`, `panel_signal_id_v1`) reduced to
  their KB records; `v6_interp*` and stale stray logs / the killed old-channel diet-v2 watcher
  removed. Baselines (`v6`, `v7`, `diet_v1`) and the active `diet_v3` kept.

---

## 2026-06-20 (session 3): pipeline complete, final results, v9 negative, cleanup

The `diet-v3 → v8 → v9` queue finished hands-off overnight (all exit 0; diet-v3 ~21:26 prior,
v8 21:26, v9 23:07). **Paper-ready numbers are curated in `docs/RESULTS.md`** with full
provenance; this section is the experiment-report record (incl. the not-so-good runs) before
artifact cleanup.

### Final selected results (selected by `gap_closed`, not FID)
- **diet-v3** (`[9,5,8]`, diet one-hot, 12 ep): balanced best **epoch 9** — fasted
  {Calret 0.86, Peri 0.40, TOMM20 0.40}, hfd {0.88, 0.62, 0.36}. **HFD continues climbing to
  epoch 11** {Calret 0.91, Peri 0.85, TOMM20 0.81} but **fasted overshoots after ep9**
  (Peri 0.40→0.01). ep11 HFD-peak summary saved as
  `diet_id_v3/aggregate_eval_summary_hfdpeak_ep11.json`; canonical on-disk summary = ep9.
  This is the **headline** result (strong physiological effect; HFD-led trajectory figure).
- **v8** CRISPR one-hot (`[0,14,5]`, 76-gene rna_snr, 20 ep): best **epoch 19**, FID 16.3.
  pooled gap_closed Alb +0.18 / Rab7 +0.15 (beat copy-control) / Peri −0.14; direction
  recovery Alb dir_corr 0.68 sign_agree 0.74, Rab7 0.61/0.74, Peri pearson 0.50. The clean
  "gene identity → morphology" result.

### Negative result — v9 (RNA-signature conditioning, concat-413) did NOT help
Apples-to-apples vs v8 (same channels + 76-gene index; only condition differs: one-hot ⊕
scaled per-gene RNA signature). 20 ep, FID ~16 throughout (identical to v8 — **FID blind**).

| pooled gap_closed | epoch 9 | epoch 14 | epoch 19 |
|---|---:|---:|---:|
| Alb  — v8 / v9 | −0.04 / **−0.78** | −0.12 / **−1.76** | **+0.18** / **−0.99** |
| Rab7 — v8 / v9 | −0.27 / **−0.75** | +0.01 / **−0.86** | **+0.15** / **−1.39** |
| Peri — v8 / v9 | +0.15 / +0.22 | −0.04 / +0.18 | −0.14 / **+0.32** |

v9 is **systematically worse on Alb & Rab7 at every epoch** (W(gen,tgt) 2–4× v8's), only
winning on Perilipin (pearson 0.51 vs 0.50). Conclusion: the gene-level transcriptional
signature did not improve — and degraded — morphology-shift recovery vs plain one-hot.
Reportable as a clean ablation. Caveat: per-run W(src,tgt) baselines differ slightly
(sample subsets not pairwise identical), but the 2–4× absolute gap far exceeds that noise.

### Final runs table
| run | panel | index | condition | best ckpt | status |
|---|---|---|---|---|---|
| `diet_id_v3` | `[9,5,8]` | diet | diet one-hot 3 | ep9 (bal) / ep11 (hfd) | **KEEP — headline** |
| `cellflux_pm_train_id_v8` | `[0,14,5]` | 76-gene | one-hot 204 | ep19 | **KEEP — CRISPR result** |
| `cellflux_pm_train_id_v9` | `[0,14,5]` | 76-gene | idsig 413 | ep19 | negative ablation (trim heavy) |
| `cellflux_pm_train_id_v7` | `[5,9,10]` | 163-gene | one-hot 204 | ep9 | superseded by v8 (trim heavy) |
| `diet_id_v1` | `[5,9,10]` | diet | diet one-hot 3 | ep1 | superseded by diet-v3 (trim heavy) |
| `cellflux_pm_geneid_baseline_v6` | `[5,9,10]` | — | gene identity | — | old-panel baseline (trim heavy) |

### Cleanup (session 3, executed 2026-06-20)
Records kept for every run (`args.json`, `log.txt`, `aggregate_eval_*`, `snapshots/`,
figures, **and `fid_samples/`** so any epoch's `gap_closed` stays re-derivable without
retraining). For the four superseded/negative runs, kept **one** checkpoint each and deleted
the rest of the `.pth`:
- `cellflux_pm_train_id_v9` → keep `checkpoint-19.pth` (negative-ablation reference)
- `cellflux_pm_train_id_v7` → keep `checkpoint-19.pth`
- `diet_id_v1` → keep `checkpoint-9.pth` (biology peak / interpolation ckpt; NOT the best-FID
  ep1, since FID is not the selection metric here)
- `cellflux_pm_geneid_baseline_v6` → keep `checkpoint-19.pth` (delta_scatter figure epoch)

Keepers `diet_id_v3` and `cellflux_pm_train_id_v8` untouched (all checkpoints retained).
`diet_id_v3` also keeps `aggregate_eval_summary_hfdpeak_ep11.json` (HFD-peak) alongside the
ep9 canonical summary. **`outputs/` 28G → 14G.**

## 2026-06-20 (session 4): CellFlux-faithful evaluation + baselines + the FID reality check

**Why this session.** Sessions 1–3 selected/reported on `gap_closed` (a home-grown per-cell
channel-mean Wasserstein metric). To position the paper against the field (it adapts the
CellFlux paradigm to a new in-vivo dataset), evaluation must follow CellFlux
(arXiv:2502.09775) and compare every method under the *same* metrics and *same* sample budget.
Spec pinned in **`docs/EVAL_PROTOCOL.md`** (single source of truth).

### Protocol (from the CellFlux paper)
- Metrics: **FIDo / FIDc** (overall + per-condition-averaged), **KIDo / KIDc**, and **MoA
  classification acc / macro-F1 / weighted-F1** (classifier trained on REAL perturbed, tested
  on GENERATED). CellFlux selects on validation FID; we additionally report `gap_closed`.
- **Matched N is the comparability rule** (CellFlux Table 5: FID/KID are sample-size sensitive).
  N = 5000 (BBBC021 budget); diet ≈ BBBC021 (few strong classes), CRISPR ≈ RxRx1 (many weak).
- All methods read/write the shared `imagefolder/<condition>/*.png` layout.

### Tooling built this session
- **`baselines/compute_image_metrics.py`** — FIDo/FIDc/KIDo/KIDc from imagefolders, matched
  per-condition cap, used identically for every method (errors if a folder lacks the cap).
- **MoA classifier fix** (`src/morphoflux/engine/moa/train_moa.py`) — vendored code classified
  `batch['y_id']` = **ANNOT (treated/control), a trivial 2-way label** (smoke test: all 1s).
  Fixed to `batch['mols']` = perturbation class; `num_classes = len(mol2id)`. Also removed a
  hardcoded `/pasteur2/...` path and rewrote `evaluate_generated_image` to be **imagefolder-
  driven** (was test-loader driven → assumed one generation per test cell, incompatible with
  matched-N subsampling). For diet, `mol2id = {fasted, hfd}` (adlib is control) → MoA = 2-class.
- **`baselines/build_comparison_table.py`** — runs both tools per method at matched N, emits the
  CellFlux-style comparison table (md + tsv).

### MoA real-image ceiling (diet)
Classifier (frozen InceptionV3 + linear head, 3 epochs) on REAL treated images:
**76.5% acc** (fasted 88.8% / hfd 63.5%; chance 50%) → conditions ARE separable on real images
→ **MoA is a usable metric for diet.** CRISPR ceiling expected ≈ chance (77-way, subtle); to be
measured before reporting MoA for CRISPR.

### Diet comparison table (PREVIEW, per-condition cap=500, proposed ep9 @ cfg0.2)
| method | FIDo | FIDc | MoA-Acc | hfd-recall |
|---|---:|---:|---:|---:|
| copy_control | **17.09** | 26.33 | 48.2 (chance) | 7.8% |
| phendiff | 19.41 | 27.63 | **59.2** | 33.6% |
| proposed_ep9 | 35.54 | 45.36 | 55.1 | 22.2% |

**Reality check (verified NOT a normalization artifact — identical [0,1]×255 scale):**
under the CellFlux-standard metrics the proposed method **does not win on diet** — copy-control
has the best FID; PhenDiff the best MoA. Root cause from pixel means: real-treated ≈ **11.3**,
control/copy ≈ 6.7, **proposed (cfg0.2) ≈ 6.6** = sits on the control distribution (still
copy-source) + synthetic artifacts. The home-grown `gap_closed` (0.86) flattered the method;
the standard metrics expose it. This is exactly why the CellFlux protocol was adopted.

### The CFG lever (under-guidance hypothesis)
Diet sampled at **cfg_scale = 0.2** (CellFlux uses 1.2) → under-guided, output glued to the
control init. Resampling ep9 at **cfg = 1.5** moves generated brightness **6.6 → 8.6 (fasted) /
7.7 (hfd)**, i.e. ~40% of the way toward treated (11.3). Promising; full FID/MoA at the higher
cfg pending. Caveat (session-3 note): under use_initial=2, cfg 1.5 once blew FID to 236
(over-guidance / out-of-regime) — watch for a cfg↑ → MoA↑ but FID↑ tradeoff.
> Data-integrity note: `outputs/diet_id_v3/args.json` is STALE (`fid_samples=12, nfe=12` from a
> later short resample). The real diet gen config is in `train_stdout.log`:
> `use_initial=1, cfg_scale=0.2, ode step_size=0.02`. Use that for any regeneration.

### Baseline status
- copy-control: diet ✓ + crispr ✓ (biological null)
- PhenDiff: diet ✓ (17825 samples) · crispr ⏳ training (~7h, per-epoch eval generation)
- IMPA: launching (diet first) — the 2nd control-aware baseline CellFlux pairs with PhenDiff
- StarGAN / MorphoDiff: vendored, not run

### Paper strategy (mirror CellFlux)
Need ≥2 external baselines (currently 1 usable → run IMPA). Lead with **diet** (BBBC021-analog,
strong, MoA works); **CRISPR** as the harder genetic secondary (RxRx1-analog, distribution-level
only). Two viable framings depending on whether cfg-tuned proposed becomes competitive:
**(A)** method paper (proposed beats PhenDiff/IMPA on FIDc/MoA) or **(B)** new in-vivo benchmark
+ honest evaluation study (standard FID misleads on subtle perturbations; copy-control wins FID).

---

## 2026-06-21 (session 5): 5K Diet eval + marker phenotype story

**Why this session.** The generated Diet images look qualitatively promising, but the
standard CellFlux-style metrics and the Perturb-Multi data type force a sharper story.
Perturb-Multi images are not ordinary RGB morphology images; the 18-target panel is a
multiplexed molecular phenotype readout: protein markers of subcellular structures and
signaling pathways plus abundant RNAs. The paper story should be marker distribution
transport, not generic photorealistic cell synthesis.

### CellFlux sample-budget check

CellFlux uses large matched sample budgets because FID/KID are sample-size sensitive:
BBBC021 uses 5120 generated images in the public scripts, RxRx1 uses a much larger
budget, and MoA evaluation is capped at 5120 generated images. Our original
`diet_id_v3` eval only generated 1024 PNGs because `scripts/train.sh` defaulted
`FID_SAMPLES=1024`; Diet has enough held-out cells to run a CellFlux-scale eval.

Diet held-out treated counts are large enough:

| split | adlib | fasted | hfd |
|---|---:|---:|---:|
| train | 105243 | 105544 | 97236 |
| test | 9251 | 9178 | 8647 |

### Proposed Diet 5K eval

Eval-only run:

```text
checkpoint: outputs/runs/diet/diet_id_v3/checkpoint-11.pth
output:     outputs/runs/diet/diet_id_v3_fid5k/fid_samples/epoch-12
config:     configs/diet_id_v3.yaml
panel:      [9,5,8] = Calreticulin / Perilipin / TOMM20
fid_samples: 5120
```

Generated PNG count:

| condition | generated PNGs |
|---|---:|
| fasted | 2654 |
| hfd | 2466 |
| total | 5120 |

Train-loop eval FID for this eval was **30.5982**.

### Matched Diet 5K baseline table

Fair comparison uses the limiting proposed condition count, cap=2466 per treated
condition (`N=4932`). Table path:
`outputs/baselines/_tables/diet_v3_fid5k/comparison_table.md`.

| method | FIDo | FIDc | KIDo | KIDc | MoA-Acc | MoA-MacroF1 | MoA-WeightedF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| copy_control | **7.96** | **12.01** | **0.0039** | **0.0057** | 49.92 | 0.4039 | 0.4039 |
| phendiff | 10.92 | 13.97 | 0.0066 | 0.0075 | 60.69 | 0.5818 | 0.5818 |
| impa | 52.29 | 55.43 | 0.0407 | 0.0424 | **63.97** | **0.6383** | **0.6383** |
| proposed_ep12_5k | 31.26 | 35.43 | 0.0267 | 0.0291 | 54.93 | 0.4859 | 0.4859 |

Readout:

- Copy-control has the best FID/KID, so FID/KID are rewarding same-batch image realism
  and can rank the no-perturbation null highest.
- PhenDiff is the best nontrivial method by FID/KID in this table.
- IMPA is best by MoA.
- The proposed model is not currently the winner under CellFlux-style image metrics.

This rules out an overclaim like "our method is best by FID/MoA." The usable
scientific claim is narrower and stronger: the model can move marker distributions
toward the target perturbation, even when generic image-realism metrics do not reward it.

### Marker distribution evidence

Reusable script added:

```text
python scripts/diet_marker_distribution_figure.py \
  --run-dir outputs/runs/diet/diet_id_v3_fid5k \
  --epoch 12 \
  --out-dir outputs/figures/diet \
  --prefix diet_v3_fid5k
```

Outputs:

```text
outputs/figures/diet/diet_v3_fid5k_marker_distributions.png
outputs/figures/diet/diet_v3_fid5k_mean_shift.png
outputs/figures/diet/diet_v3_fid5k_marker_distribution_summary.csv
outputs/figures/diet/diet_v3_fid5k_marker_distribution_summary.json
```

Generated-vs-target foreground marker means from all 5120 generated PNGs:

| condition | marker | generated | target | generated-target | read |
|---|---|---:|---:|---:|---|
| fasted | Calreticulin | 0.3715 | 0.3528 | +0.0187 | slight overshoot |
| fasted | Perilipin | 0.3208 | 0.3123 | +0.0085 | slight overshoot |
| fasted | TOMM20 | 0.4115 | 0.4122 | -0.0007 | close |
| hfd | Calreticulin | 0.4213 | 0.4173 | +0.0040 | close |
| hfd | Perilipin | 0.3514 | 0.3555 | -0.0041 | close |
| hfd | TOMM20 | 0.4235 | 0.4423 | -0.0189 | under-shift |

This is the current positive result: the HFD Calreticulin and Perilipin
distributions move close to the treated state; TOMM20 moves in the right direction
but remains under-shifted. For fasted, TOMM20 is close, while Calreticulin and
Perilipin slightly overshoot.

### DDP mapping caveat and fix

The 2-GPU 5K eval saved 5120 PNGs, but `trt2ctrl_idx.json` contained only 2560
treated->control mappings because each rank wrote the JSON independently and one
rank overwrote the other. This affects paired control/gen/target analyses and
control-normalized `gap_closed`, but not the generated-vs-target marker-distribution
check above.

Code fix made in `src/morphoflux/engine/training/eval_loop.py`: gather
per-rank `trt2ctrl_idx` dictionaries with `torch.distributed.all_gather_object`,
merge them, and write the global/per-epoch JSON only from the main process. Future
DDP evals should have mapping count equal to PNG count. Rerun Diet 5K after this
fix if fully rigorous paired gap-closure is needed for the figure/table.

### Story decision

Current paper framing:

```text
Morpho-CellFlux adapts CellFlux-style conditional transport to Perturb-Multi
hepatocyte multiplexed marker images. The primary endpoint is whether generated
marker distributions move from control toward real perturbation states.
```

Do not frame the project as ordinary morphology photorealism. Do not select or
claim success by FID alone. The Diet result supports marker distribution migration;
the mean image figure suggests spatial morphology is still imperfect and should
not be overclaimed. CRISPR remains the clean genetic setting but should be shown
with distribution-level panels because single-gene shifts are weak.

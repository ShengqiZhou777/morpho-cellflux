# CellFlux-Repo Experiments (Perturb-Multi hepatocytes)

Phase running the Perturb-Multi hepatocyte data through the CellFlux engine (absorbed
into this repo as `morphoflux.engine`; see docs/ARCHITECTURE.md) with the corrected
perturbation semantics.

Dates: 2026-06-17 to 2026-06-19.

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
| Perturbation gene identity | clean condition | 204 one-hot | `embedding_gene_identity.csv` | active baseline |
| Genome-wide Perturb-seq pseudobulk | independent rich condition | TBD | GSE275483 | pending |
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

> These runs predate the 2026-06-19 index consolidation and were trained on the legacy
> gene-subset indices (`index_stronghits` / `index_highsignal` / `index_panel_signal`,
> now removed). New runs train on `index_train.csv` (all 163 genes). The checkpoints
> remain valid; only the index files were renamed/merged.

| run | condition | index | status |
|---|---|---|---|
| `cellflux_pm_geneid_baseline_v6` | gene identity | `index_stronghits.csv` | active clean baseline |
| `cellflux_pm_highsignal_id_v1` | gene identity | `index_highsignal.csv` | active clean follow-up |
| `cellflux_pm_panel_signal_id_v1` | gene identity | `index_panel_signal.csv` | active clean follow-up |
| `v6_interp` | gene identity, resume from v6 | `index_interp_leadgenes.csv` | active interpolation output |
| `v6_interp_cfg3` | gene identity, resume from v6, CFG 3.0 | `index_interp_leadgenes.csv` | active interpolation output |
| `v6_interp_cfg6` | gene identity, resume from v6, CFG 6.0 | `index_interp_leadgenes.csv` | active interpolation output |
| `v6_interp_noiseinit` | gene identity, resume from v6, noise init | `index_interp_leadgenes.csv` | active interpolation output |

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

FID alone is not a reliable headline metric here because the perturbation effects
are subtle and same-batch controls can score well while failing to move in the
correct biological direction.

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

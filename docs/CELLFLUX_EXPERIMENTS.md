# CellFlux-Repo Experiments (Perturb-Multi hepatocytes)

Phase running the Perturb-Multi hepatocyte data through the CellFlux engine (absorbed
into this repo as `morphoflux.engine`; see docs/ARCHITECTURE.md) with the corrected
perturbation semantics.

Dates: 2026-06-17 to 2026-06-18.

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

- **Data**: `data/processed/cellflux_ext/`, built by
  `scripts/build_cellflux_external.py` from `cellflux_manifest.parquet`.
- **Image channels**: panel2 `[5,9,10]` =
  Perilipin / Calreticulin / pS6RP.
- **Condition**: gene identity one-hot via `embedding_gene_identity.csv`.
- **Dataset/config family**: `perturbmulti_id`.
- **Task**: distribution-to-distribution control -> perturbed morphology. The source
  and target cells are unpaired; the model should be evaluated by aggregate
  perturbation response, not pixel-level single-cell matching.
- **Training recipe**: official CellFlux conditional flow matching with EMA,
  classifier-free guidance, same-batch/same-state control pairing, and aggregate
  evaluation.

## Active Runs

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

## Scripts

- `scripts/build_cellflux_external.py`: build indexes, identity embeddings, and
  effect tables.
- `scripts/aggregate_eval.py <run_dir>`: per-gene per-channel generated/source/target
  summary and correlations.
- `scripts/analyze_perilipin_direction.py`: Perilipin direction analysis.
- `scripts/launch_cellflux_pm.sh`: parameterized DDP launch with persistent logging.

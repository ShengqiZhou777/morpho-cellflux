# Architecture

`morpho-cellflux` is one Python package (`src/morphoflux/`) plus CLI scripts. It models
**perturbation -> morphology** on Perturb-Multi CRISPR hepatocyte images: given a control
cell and a target gene (perturbation identity), generate the perturbed-cell panel.

## Pipeline (data -> engine -> evaluation)

```
raw assets (data/raw/)                        h5ad (RNA/protein) + manifest
        |                                                   |
        v                                                   v
  morphoflux.data.DataFactory  ───────────►  scripts/build_crispr_paper_data.py
  (scripts/materialize_data.py)              - index_paper_programs.csv
  - manifest.parquet                         - index_paper_programs_heldout.csv
  - condition_vocab.json                     - embedding_gene_identity.csv
                                             - program_labels_paper.csv
                                             - paper_panel_effects.csv
        |                                                   |
        └───────────────────────┬───────────────────────────┘
                                 v
            morphoflux.engine  (absorbed CellFlux flow-matching engine)
            torchrun -m morphoflux.engine.train  (scripts/train.sh)
            - conditional flow matching, control->perturbed, CFG, EMA, ODE sampling, FID
            - reads configs/<name>.yaml  (MORPHOFLUX_CONFIG_DIR)
            - writes outputs/<run>/: checkpoints, fid_samples/epoch-<e>/, log.txt
                                 |
                                 v
            evaluation / figures (scripts/, GPU-free, read fid_samples)
            - aggregate_eval.py    : per-gene per-channel Δ-direction (the metric)
            - delta_scatter.py     : Δ-direction scatter (the quantitative figure)
            - population_phenotype.py : control vs generated vs KO distributions (visible effect)
            - interpolate.sh : interpolation trajectory grid (qualitative)
            - plot_flow_figure.py : polished trajectory/population figure from interpolation npz
```

## Package layout
- `src/morphoflux/data/` — `DataFactory`: builds the manifest + condition vocab from raw
  paired assets. The LIVE data layer; everything downstream consumes its outputs.
- `src/morphoflux/engine/` — the absorbed CellFlux engine (model, training loop, ODE
  sampler, FID, CFG). Upstream provenance + our edits in `engine/UPSTREAM.md`. We only
  added a dataset/condition adapter (`perturbmulti_id` arch, `_load_perturbmulti`, the
  split branch, per-epoch trt2ctrl, a torch-2.11 load fix); the generative core is upstream.
- `scripts/` — CLI tools: data build, training/interpolation launchers, evaluation/figures.
- `configs/` — engine run configs (single source of truth).
- `data/` (gitignored), `outputs/` (gitignored), `docs/`.

## Modality semantics (pinned — do not muddle)
- **Perturbation = sgRNA -> target gene IDENTITY** (the condition). The paper
  CRISPR core uses 40 target genes grouped into 7 Perturb-Multi programs.
- **209-gene MERFISH = a transcriptional READOUT** of the imaged cells (NOT the condition).
- **18-channel protein/morphology = the imaging READOUT**. The model generates a
  config-selected 3-channel panel: Diet uses `[9,5,8]` =
  Calreticulin / Perilipin / TOMM20; CRISPR paper core uses `[9,5,10]` =
  Calreticulin / Perilipin / pS6RP.
- Raw sequencing/GEO files are not required for the current pipeline. The
  paired RNA h5ad is a MERFISH readout for diagnostics; the main model
  condition remains gene identity.

## How to run
```bash
pip install -e .            # registers morphoflux + engine (deps in pyproject)

# train (2-GPU DDP), config resolved from configs/
OUT=outputs/runs/crispr/paper_core CONFIG=crispr_paper_core DATASET=perturbmulti_id \
  bash scripts/train.sh

# evaluate a run (per-gene Δ-direction for an epoch)
python scripts/aggregate_eval.py outputs/runs/crispr/paper_core 5 <epoch>
python scripts/delta_scatter.py  outputs/runs/crispr/paper_core 5
python scripts/population_phenotype.py outputs/runs/crispr/paper_core lipid Eif2s1,Pten,Aars,Insig1 <epoch>

# qualitative interpolation grid on the best checkpoint
CKPT=outputs/runs/crispr/paper_core/checkpoint-<e>.pth OUT=outputs/runs/crispr/paper_core \
  CONFIG=figures/crispr_leadgenes_interp GPU=0 bash scripts/interpolate.sh
python scripts/plot_flow_figure.py outputs/runs/crispr/paper_core/interpolation --out outputs/figures/crispr_flow.png
```

## Evaluation philosophy
**FID is the primary image-quality / distribution metric** (following CellFlux). Because
the task is unpaired distribution->distribution and per-cell matching is ill-posed, we
complement FID with **biological-direction metrics**: the **per-gene Δ-direction** (does
generated move from control toward the real KO, across genes) and the **population
phenotype shift** (control vs generated vs KO distributions). For subtle in-vivo CRISPR
effects a good FID does not guarantee the right biological movement, so the two are
reported together — and FID should not be the *sole* checkpoint-selection criterion (it
can keep rising while Δ-direction still improves). The interpolation grid is qualitative
only — its trajectory is always smooth, so it never stands alone as a correctness claim.

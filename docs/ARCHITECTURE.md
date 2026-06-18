# Architecture

`morpho-cellflux` is one Python package (`src/morphoflux/`) plus CLI scripts. It models
**perturbation -> morphology** on Perturb-Multi CRISPR hepatocyte images: given a control
cell and a target gene (perturbation identity), generate the perturbed-cell panel.

## Pipeline (data -> engine -> evaluation)

```
raw assets (morpho-phenotyping)               external h5ad (RNA/protein) + manifest
        |                                                   |
        v                                                   v
  morphoflux.data.DataFactory  ───────────►  scripts/build_perturbmulti_data.py
  (scripts/materialize_data.py)              - index_stronghits.csv (strong-hit subset)
  - manifest.parquet                - embedding_gene_identity.csv (204 one-hot)
  - condition_vocab.json                     - channel_effects.csv, perturbation_effects.csv
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
            - mean_image_figure.py : per-gene spatial mean (documented NEGATIVE result)
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
- **Perturbation = sgRNA -> target gene IDENTITY** (the condition; 204-dim one-hot).
- **209-gene MERFISH = a transcriptional READOUT** of the imaged cells (NOT the condition).
- **18-channel protein/morphology = the imaging READOUT**; panel2 = npz `[5,9,10]` =
  Perilipin / Calreticulin / pS6RP. The model generates this 3-channel panel.
- Genome-wide Perturb-seq (GSE275483) = adjacent sections; a possible future *rich*
  condition. The 209-MERFISH and genome-wide data are different things.

## How to run
```bash
pip install -e .            # registers morphoflux + engine (deps in pyproject)

# train (2-GPU DDP), config resolved from configs/
OUT=outputs/<run> CONFIG=perturbmulti_stronghits_id DATASET=perturbmulti_id \
  bash scripts/train.sh

# evaluate a run (per-gene Δ-direction for an epoch)
python scripts/aggregate_eval.py outputs/<run> 3 <epoch>
python scripts/delta_scatter.py  outputs/<run> 3
python scripts/population_phenotype.py outputs/<run> lipid Eif2s1,Pten,Aars,Insig1 <epoch>

# qualitative interpolation grid on the best checkpoint
CKPT=outputs/<run>/checkpoint-<e>.pth OUT=outputs/<run> \
  CONFIG=perturbmulti_interp_leadgenes GPU=0 bash scripts/interpolate.sh
```

## Evaluation philosophy
Perturbation effects here are subtle (in-vivo CRISPR) and the task is unpaired
distribution->distribution, so **per-pixel / per-cell matching is ill-posed and FID alone
is misleading**. The honest signal is the **per-gene Δ-direction** (does generated move
from control toward the real KO, across genes) and the **population phenotype shift**
(control vs generated vs KO distributions). The interpolation grid is qualitative only —
its trajectory is always smooth, so it never stands alone as a correctness claim.


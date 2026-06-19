# Morpho CellFlux

CellFlux-style perturbation -> morphology modeling on Perturb-Multimodal MERFISH
hepatocytes. Given a control-cell image and a perturbation identity, generate the
perturbed-cell morphology panel via conditional flow matching.

The project is one Python package (`src/morphoflux/`) plus CLI scripts. Large source
assets stay in their original location (`morpho-phenotyping`) and are exposed here through
symlinks under `data/raw`; derived, reproducible tables are written to `data/processed`.
See `docs/ARCHITECTURE.md` for the full pipeline and `docs/EXPERIMENTS.md` for the
running experiment log.

## Two datasets

| dataset | condition | control / treated | images | build script |
|---|---|---|---|---|
| **crispr** (`perturbmulti`) | gene identity (204-dim one-hot) | non-targeting / sgRNA-KO of 163 genes | `data/raw/extracted_images` | `scripts/build_perturbmulti_data.py` |
| **diet** | diet state (3-dim one-hot) | adlib / fasted, hfd | `data/raw/diet_extracted_images` | `scripts/build_diet_data.py` |

Both feed the same engine (`dataset_name: perturbmulti`) and the same 3-channel image
panel: npz channels `[5, 9, 10]` = **Perilipin / Calreticulin / pS6RP**.

## Modality semantics (pinned — do not muddle)

- **Perturbation = the condition.** crispr: sgRNA -> target gene IDENTITY (one-hot).
  diet: diet state (adlib/fasted/hfd one-hot). This is the *only* thing the model is
  conditioned on.
- **18-channel protein/morphology = the imaging READOUT we generate** (the model outputs
  the 3-channel panel above). The per-cell protein-intensity h5ad is the per-channel mean
  of the same image; it is used only upstream to choose channels/genes, never as a
  model input.
- **209-gene MERFISH RNA = a transcriptional READOUT** (z-scored), used only as a
  diagnostic effect-size (SNR) signal. NOT a condition, NOT a model input.

## Data contract

Distribution-level morphology transport, not same-cell trajectories:

```text
control cells within the same batch/state
  -> perturbed cells within the same batch/state
```

Source (control) and target (treated) cells are unpaired. The engine pairs each treated
cell with a random **same-batch** control at load time. Evaluate by aggregate
perturbation response, not pixel-level single-cell matching.

## Layout

```text
configs/                 Engine run configs (single source of truth) + data-factory config.
data/raw/                Symlinks to source manifest/images/RNA/protein assets.
data/processed/          Derived manifest, condition vocab, indices, embeddings.
data/reports/            JSON audit reports from data materialization.
docs/                    ARCHITECTURE, EXPERIMENTS, DATA_FACTORY, SCIENTIFIC_STORY.
scripts/                 CLI entry points (data build, training/interpolation, eval/figures).
src/morphoflux/          data factory + absorbed CellFlux flow-matching engine.
outputs/                 Checkpoints, generated samples, metrics (gitignored).
```

## Quickstart

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pmf
pip install -e .            # registers morphoflux + engine
```

### 1. Build data tables

```bash
# crispr: factory builds manifest.parquet + condition_vocab.json + raw symlinks ...
python scripts/materialize_data.py --config configs/crispr_hep.yaml
# ... then carve the engine indices + diagnostic effect tables
python scripts/build_perturbmulti_data.py

# diet (reads the morpho-phenotyping diet manifest directly)
python scripts/build_diet_data.py
```

`build_perturbmulti_data.py` writes, under `data/processed/perturbmulti/`:

```text
index_train.csv            all 163 kept genes (TIER1/2 + control), SPLIT train->train, val->test
index_train_heldout.csv    same genes, original `test` split (final held-out eval)
index_eval_leadgenes.csv   5 lead genes spanning the 3 panel channels (interpolation figure)
perturbation_effects.csv / channel_effects.csv / panel_effects.csv   diagnostics
```

`embedding_gene_identity.csv` (the 204x204 gene-identity one-hot condition) is **not**
written by any build script and has no in-repo builder — do not delete it.

The engine reads only three things at runtime: the image dir (`image_path`), one index
CSV (`data_index_path`), and one condition embedding (`embedding_path`). Splits are folded
`train -> train`, `val -> test` (in-loop eval fold), `test -> *_heldout`, identically for
both datasets.

### 2. Train (2-GPU DDP)

```bash
# crispr gene-identity baseline (default config)
OUT=outputs/my_run bash scripts/train.sh

# diet
OUT=outputs/diet_run CONFIG=diet_id DATASET=diet_id bash scripts/train.sh
```

`scripts/train.sh` is parameterized by env vars (`OUT`, `BATCH`, `ACCUM`, `EPOCHS`,
`CONFIG`, `DATASET`, `USE_INITIAL`, `CFG`, ...) and tees per-step stdout to
`$OUT/train_stdout.log`. `BATCH` is the per-GPU batch size. Default
`CONFIG=perturbmulti_train_id`, `DATASET=perturbmulti_id` (condition_dim 204).

### 3. Interpolation figure

```bash
CKPT=outputs/my_run/checkpoint-<e>.pth OUT=outputs/my_run GPU=0 \
  bash scripts/interpolate.sh
```

Writes one grid per cell (real control, ODE trajectory t=0->1, generated, real perturbed)
under `$OUT/interpolation/`. Default `CONFIG=perturbmulti_interp_leadgenes` so the eval
batch lands on genes with visible panel signal. Qualitative only — pair with the
quantitative Delta-direction below.

### 4. Evaluate

```bash
python scripts/aggregate_eval.py outputs/my_run 3 <epoch>   # per-gene per-channel Δ-direction
python scripts/delta_scatter.py  outputs/my_run 3           # Δ-direction scatter figure
python scripts/population_phenotype.py outputs/my_run lipid Eif2s1,Pten,Aars,Insig1 <epoch>
```

Perturbation effects are subtle and the task is unpaired distribution->distribution, so
**FID alone is misleading**. The honest signal is the per-gene Delta-direction (does the
generated cell move from control toward the real KO) and the population phenotype shift.

## Makefile shortcuts

```bash
make data               # materialize crispr manifest + factory outputs
make build-perturbmulti # carve crispr indices + effect tables
make build-diet         # build diet indices + embedding
make train              # crispr DDP training (scripts/train.sh)
make smoke              # short sanity run
```

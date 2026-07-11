# Morpho-CellFlux

Flow-matching phenotype transport for FusionODE microalgae microscopy.

Current engineering target: **two clean lanes with separate contracts**.

```text
cell lane:
  data/raw/microalgae_v1/single_cell_images
      -> data/processed/microalgae_v1/views/timepoint/
      -> configs/microalgae_timepoint.yaml

field lane:
  data/raw/microalgae_v1/field_images
      -> data/processed/microalgae_v1/views/field/
      -> configs/microalgae_field.yaml
```

The cell lane is the phenotype-transport base model. The field lane is the
simulator-like / annotation-enhancement lane aligned with whole-field synthetic
microscopy references such as BBBC035 and CytoPacq.

## One Path To Run

### 1. Prepare Raw Data

Copy FusionODE source data into the project-local raw tree:

```bash
FUSIONODE_DATA=/path/to/FusionODE/data bash scripts/prepare_raw_data.sh
```

This prepares:

```text
data/raw/microalgae_v1/single_cell_images
data/raw/microalgae_v1/field_images
```

`single_cell_images` is required for the cell lane. `field_images` is required
for EXIF timing, masks, field summaries, and field-lane builds.

### 2. Build Processed Data

Build the cell lane:

```bash
python scripts/build_microalgae_dataset.py --version microalgae_v1 --views timepoint
```

Expected outputs:

```text
data/processed/microalgae_v1/views/timepoint/index.csv
data/processed/microalgae_v1/views/timepoint/embedding.csv
data/processed/microalgae_v1/views/timepoint/summary.json
```

Build the field lane:

```bash
python scripts/build_microalgae_dataset.py --version microalgae_v1 --views field
```

Expected outputs:

```text
data/processed/microalgae_v1/views/field/index.csv
data/processed/microalgae_v1/views/field/embedding.csv
data/processed/microalgae_v1/views/field/metadata.csv
data/processed/microalgae_v1/views/field/targets.csv
data/processed/microalgae_v1/views/field/prompts.csv
data/processed/microalgae_v1/views/field/summary.json
```

Build both:

```bash
python scripts/build_microalgae_dataset.py --version microalgae_v1 --views all
```

If old flat timepoint files already exist under `data/processed/`, normalize
them once:

```bash
bash scripts/migrate_processed_layout.sh
```

### 3. Quick Validate

```bash
bash scripts/quick_validate.sh microalgae_timepoint
```

For a repo-local smoke check that does not require external FusionODE data:

```bash
python scripts/smoke_validate.py
```

For the real field lane smoke check:

```bash
python scripts/field_smoke_validate.py
```

### 4. Train

```bash
CONFIG=microalgae_timepoint \
DATASET=phenoflux \
OUT=outputs/runs/microalgae/timepoint_v1 \
EPOCHS=40 \
FID_SAMPLES=1024 \
bash scripts/train.sh
```

## Current Public Config

Use these configs:

```text
configs/microalgae_timepoint.yaml
configs/microalgae_field.yaml
```

It points to:

```yaml
image_path: data/raw/microalgae_v1/single_cell_images
data_index_path: data/processed/microalgae_v1/views/timepoint/index.csv
embedding_path: data/processed/microalgae_v1/views/timepoint/embedding.csv
```

The field config points to:

```yaml
image_path: data/raw/microalgae_v1/field_images
data_index_path: data/processed/microalgae_v1/views/field/index.csv
embedding_path: data/processed/microalgae_v1/views/field/embedding.csv
```

## Installation

```bash
conda env create -f environment.yml
conda activate pmf
pip install -e .
```

## Repository Layout

```text
phenoflux/          Training, models, dataloaders, and microalgae evaluation code
configs/            Active timepoint config and generated smoke config
scripts/            Active data builders, smoke check, and training launchers
data/raw/           Project-local raw data root
data/processed/     Versioned processed artifacts
docs/               Data contract, legacy boundary, and cleanup plan
outputs/            Training and evaluation outputs
archive/            Legacy source, configs, docs, and provenance material
```

## Legacy Material

This repository was adapted from a PhenoFlux / CellFlux-style cell phenotype
transport codebase. Diet, CRISPR, Perturb-Multi, MSA/PCD, and external baseline
references are retained for provenance only; they are not the current default
path.

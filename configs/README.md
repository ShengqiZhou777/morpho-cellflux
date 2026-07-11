# Config Index

Current runnable targets:

```text
microalgae_timepoint
microalgae_field
```

Use this first. Do not expand to other views until this path builds, trains,
and evaluates cleanly.

## Active Config

### `microalgae_timepoint.yaml`

Main crop-level acquisition-time view.

```text
image_path:      data/raw/microalgae_v1/single_cell_images
data_index_path: data/processed/microalgae_v1/views/timepoint/index.csv
embedding_path:  data/processed/microalgae_v1/views/timepoint/embedding.csv
dataset:         phenoflux
base_condition_dim: 4
```

Run:

```bash
bash scripts/quick_validate.sh microalgae_timepoint
```

Then:

```bash
CONFIG=microalgae_timepoint DATASET=phenoflux bash scripts/train.sh
```

### `microalgae_smoke.yaml`

Generated local fixture for CPU smoke validation only:

```bash
python scripts/smoke_validate.py
```

### `microalgae_field.yaml`

Whole-field microscopy lane for simulator-like generation and annotation
support.

```text
image_path:      data/raw/microalgae_v1/field_images
data_index_path: data/processed/microalgae_v1/views/field/index.csv
embedding_path:  data/processed/microalgae_v1/views/field/embedding.csv
dataset:         phenoflux
base_condition_dim: 34
```

Build:

```bash
python scripts/build_microalgae_dataset.py --version microalgae_v1 --views field
```

Smoke:

```bash
python scripts/field_smoke_validate.py
```

Legacy and deferred configs are archived under `archive/legacy_20260705/configs/`.

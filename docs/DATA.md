# Data Contract

Current target: **two runnable processed views**.

The active views are:

```text
microalgae_v1/views/timepoint/
microalgae_v1/views/field/
```

The timepoint view is the single-cell crop lane. The field view is the
whole-field simulator-like lane with metadata, masks, targets, and prompts for
annotation-enhancement work.

## Raw Data

The project has one raw root:

```text
data/raw/
```

For the current path, the required raw directory is:

```text
data/raw/microalgae_v1/single_cell_images/
data/raw/microalgae_v1/field_images/
```

Prepare it from an external FusionODE checkout:

```bash
FUSIONODE_DATA=/path/to/FusionODE/data bash scripts/prepare_raw_data.sh
```

`single_cell_images/` backs the single-cell lane. `field_images/` backs EXIF
timing, masks, field summaries, timepoint construction, and the field lane.

## Processed Data

The current processed dataset version is:

```text
data/processed/microalgae_v1/
```

The single-cell crop view is:

```text
data/processed/microalgae_v1/views/timepoint/
├── index.csv
├── embedding.csv
└── summary.json
```

Meaning:

| File | Purpose |
| --- | --- |
| `index.csv` | Train/test source-target image pair index |
| `embedding.csv` | Target-condition vectors looked up by `CPD_NAME` |
| `summary.json` | Row counts and build metadata |

The active config uses exactly these files:

```yaml
image_path: data/raw/microalgae_v1/single_cell_images
data_index_path: data/processed/microalgae_v1/views/timepoint/index.csv
embedding_path: data/processed/microalgae_v1/views/timepoint/embedding.csv
```

The whole-field view is:

```text
data/processed/microalgae_v1/views/field/
├── index.csv
├── embedding.csv
├── metadata.csv
├── summary.csv
├── targets.csv
├── prompts.csv
└── summary.json
```

Meaning:

| File | Purpose |
| --- | --- |
| `index.csv` | Source-target field pairs compatible with the training loader |
| `embedding.csv` | Target-field condition vectors: time, condition, field morphology, omics |
| `metadata.csv` | EXIF, image/mask paths, cell counts, instance-label ranges |
| `summary.csv` | Aggregated per-field morphology statistics |
| `targets.csv` | Target-field labels such as density, size, brightness, texture, phase |
| `prompts.csv` | Text prompts for later text-conditioning experiments |
| `summary.json` | Build summary |

## Build The Current View

```bash
python scripts/build_microalgae_dataset.py --version microalgae_v1 --views timepoint
```

Build the field view:

```bash
python scripts/build_microalgae_dataset.py --version microalgae_v1 --views field
```

Build both:

```bash
python scripts/build_microalgae_dataset.py --version microalgae_v1 --views all
```

## Local Smoke Fixture

For dataloader/model validation without external data:

```bash
python scripts/smoke_validate.py
```

This generates a tiny fixture under `data/smoke/`, which is not publication
data and is ignored by git.

For the real field lane:

```bash
python scripts/field_smoke_validate.py
```

If the repo already has old flat processed files, normalize them once:

```bash
bash scripts/migrate_processed_layout.sh
```

## Training Semantics

Each row belongs to either source/control or target/treated data:

```text
ANNOT == negative_control  -> source/control image
ANNOT == treated           -> target image
CPD_NAME                   -> condition key for embedding lookup
```

Each training batch contains:

```text
X = (source_image, target_image)
mols = target condition id
y_id = annotation id
```

The current recommended training mode is:

```text
--use_initial 1
```

This starts generation from the source image.

## Deferred Views

| View / Split | Path | Status |
| --- | --- | --- |
| `state` | `data/processed/microalgae_v1/views/state/` | Later baseline |
| `leave_time_6h` | `data/processed/microalgae_v1/validations/leave_time_6h/` | Later validation |
| `legacy/continuous` | `data/processed/microalgae_v1/legacy/continuous/` | Compatibility only |

# Config Index

Single active training path points to the **476-dim raw gene/protein
timepoint** config. Everything else is a baseline, a lane, or infra.

```text
Primary → microalgae_timepoint_512_genes   (Stage-2 omics, current focus)
Baseline → microalgae_timepoint_512      (4d, ablation counterpart)
Field    → microalgae_field              (whole-field lane, active 2nd path)
Smoke    → microalgae_smoke              (CPU repo-local validation)
Validation → synthetic_validation        (signal-strength check)
```

Archived 4d variants live under `archive/legacy_configs_2026_07/`
(`microalgae_timepoint` 128px, `microalgae_timepoint_quick`) and
`archive/legacy_20260705/configs/`.

---

## Primary — `microalgae_timepoint_512_genes.yaml`

476-dim gene/protein condition on the 512px single-cell timepoint view. Addresses the
identity-mapping collapse by giving the model a stronger condition signal.

```text
image_path:      data/raw/microalgae_v1/single_cell_images
data_index_path: data/processed/microalgae_v1/views/timepoint_512/index.csv
embedding_path:  data/processed/microalgae_v1/views/timepoint_512/embedding_genes.csv
base_condition_dim: 476     # 2 light/dark + 1 time_norm + 1 time_bin_h + 372 genes + 100 proteins
```

Build the gene embedding, then quick-validate / train:

```bash
python scripts/build_gene_condition.py      # -> embedding_genes.csv
bash scripts/quick_validate.sh microalgae_timepoint_512_genes
CONFIG=microalgae_timepoint_512_genes DATASET=phenoflux bash scripts/train.sh
```

## Baseline — `microalgae_timepoint_512.yaml`

Same 512px view, 4d condition (no omics). Ablation counterpart to the primary —
run it to isolate the effect of the omics condition.

```text
embedding_path:  data/processed/microalgae_v1/views/timepoint_512/embedding.csv
base_condition_dim: 4
```

```bash
bash scripts/quick_validate.sh microalgae_timepoint_512
```

## Field — `microalgae_field.yaml`

Whole-field microscopy lane for simulator-like generation and annotation support.

```text
image_path:      data/raw/microalgae_v1/field_images
data_index_path: data/processed/microalgae_v1/views/field/index.csv
embedding_path:  data/processed/microalgae_v1/views/field/embedding.csv
base_condition_dim: 34
```

```bash
python scripts/build_microalgae_dataset.py --version microalgae_v1 --views field
python scripts/field_smoke_validate.py
```

## Smoke — `microalgae_smoke.yaml`

Generated local fixture for CPU smoke validation only (no external data).

```bash
python scripts/smoke_validate.py
```

## Validation — `synthetic_validation.yaml`

Synthetic dataset for signal-strength verification.

```text
image_path:      data/synthetic_validation
data_index_path: data/synthetic_validation/index.csv
embedding_path:  data/synthetic_validation/embedding.csv
base_condition_dim: 4
```

# Morpho CellFlux

CellFlux-style perturbation modeling scaffold for the Perturb-Multimodal
CRISPR hepatocyte dataset.

This project is intentionally separated from `morpho-phenotyping`. The large
source assets remain in their original location and are exposed here through
symlinks under `data/raw`. Derived, reproducible training tables are written to
`data/processed`.

## Data Contract

The first supported task is distribution-level morphology transport:

```text
control sgRNA hepatocytes within the same batch/state
  -> target-gene perturbation hepatocytes within the same batch/state
```

This is not a same-cell trajectory dataset. Pair tables are stochastic training
pairs between unpaired source and target distributions, matched by metadata.

## Layout

```text
configs/                 Dataset and factory configuration.
data/raw/                Symlinks to source manifest/images/RNA/protein assets.
data/processed/          Derived CellFlux manifest, pair tables, vocab files.
data/reports/            JSON audit reports from data materialization.
docs/                    Notes on design and assumptions.
scripts/                 CLI entry points.
src/morphoflux/          Python package for data factories and future training.
outputs/                 Checkpoints, generated images, and metrics.
logs/                    Runtime logs.
```

## Quickstart

Run from this directory:

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pmf

python scripts/materialize_data.py --config configs/crispr_hep.yaml
```

The command creates:

```text
data/raw/manifest_crispr_hep_paired.parquet -> source manifest
data/raw/extracted_images                   -> source extracted image directory
data/raw/RNA_crispr_hep_paired.h5ad         -> source RNA AnnData
data/raw/protein_crispr_hep_paired.h5ad     -> source protein AnnData
data/processed/cellflux_manifest.parquet
data/processed/condition_vocab.json
data/processed/pairs/train_pairs.parquet
data/processed/pairs/val_pairs.parquet
data/processed/pairs/test_pairs.parquet
data/reports/data_audit.json
```

## Default Pairing Policy

The factory uses control cells with `is_control == True` as sources and targeting
cells with `is_control == False` as targets. By default it samples one source per
target from the same:

```text
split, batch, cluster_type
```

Targets whose stratum has fewer than `min_controls_per_stratum` control cells
are dropped from the pair table. This conservative default avoids learning liver
zonation, tissue-state, or batch differences as if they were gene effects.

## Next Training Step

The pair tables are compatible with a CellFlux-style PyTorch dataset in
`src/morphoflux/data/torch_dataset.py`. The next implementation step is to add a
model/training runner that consumes:

```text
source image: data/raw/extracted_images/<source_image_member>
target image: data/raw/extracted_images/<target_image_member>
condition:   target_gene / condition_id
```


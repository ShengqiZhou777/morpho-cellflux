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
src/morphoflux/          Python package for data factories, models, and training.
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

The pair tables are connected to a CellFlux-style conditional flow matching
training loop in `scripts/train_cellflux.py`. Each training batch consumes:

```text
source image: data/raw/extracted_images/<source_image_member>
target image: data/raw/extracted_images/<target_image_member>
condition:   target_gene / condition_id
```

The model learns a velocity field:

```text
v_theta(x_t, t, condition) ~= target_image - source_image
```

where:

```text
x_t = (1 - t) * source_image + t * target_image
t ~ Uniform(0, 1)
```

Start the current lipid-panel DDP training run with:

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pmf

torchrun --standalone --nproc_per_node=2 \
  scripts/train_cellflux.py --config configs/train_cellflux_lipid_panel.yaml
```

`training.batch_size` is the global batch size under DDP. With two GPUs,
`batch_size: 128` runs 64 examples per rank.

For a quick CPU smoke test:

```bash
python scripts/train_cellflux.py \
  --device cpu \
  --max-steps 2 \
  --batch-size 1 \
  --limit-train-rows 4 \
  --limit-val-rows 2
```

The current lipid-panel run writes metrics and checkpoints under:

```text
outputs/cellflux_lipid_panel_scaffold_ddp_2k/
```

Run it through the Makefile with:

```bash
make train-lipid-panel-ddp
```

Export a preview NPZ to JPG grids and biology-focused RGB composites with:

```bash
python scripts/export_preview_jpg.py \
  outputs/cellflux_long_10k/previews/step_0010000.npz \
  --sample 0 \
  --out-dir outputs/cellflux_long_10k/jpg_previews
```

The default RGB presets are:

```text
lipid_function: Perilipin, Alb, polyT
er_secretory:   Calreticulin, M6PR, Gapdh
mito_autophagy: TOMM20, LC3b, Rab7
```

See `docs/SCIENTIFIC_STORY.md` for the current scientific framing, algorithm
modules, and figure strategy for the 18-channel morphology output.

# Data Factory Notes

## Why symlinks

The source image crops are about 11 GB and the RNA AnnData is about 4.7 GB.
Duplicating them would make the new project harder to maintain. The data factory
therefore links source assets into `data/raw` and writes only derived tables.

## What is materialized

The factory materializes three categories of files:

1. Raw asset symlinks.
2. A normalized CellFlux manifest with image relative paths and condition ids.
3. Per-split source-target pair tables matched by `batch` and `cluster_type`.

## Pair semantics

Rows in `*_pairs.parquet` are unpaired training samples. A source control cell
is sampled from the same metadata stratum as a target perturbation cell. The row
does not mean that the source cell becomes the target cell in reality.

## Default filtering

The default config filters to `cell_type == "Hep"` and drops target cells whose
same-split, same-batch, same-cluster control pool has fewer than 10 cells.

This is deliberate. In liver tissue, hepatocyte state and spatial zonation can
be larger confounders than the perturbation signal. The conservative matching
policy is the first guard against training a model that learns batch or tissue
state instead of gene effects.


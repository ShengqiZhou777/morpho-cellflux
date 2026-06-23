# Reproducing Morpho-CellFlux Experiments

This document describes the artifact contract for reproducing the experiments
from a fresh clone. The repository intentionally does not include raw imaging
data, checkpoints, generated samples, or large baseline exports.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate morpho-cellflux
pip install -e .
```

If your system requires a different CUDA build, install the correct PyTorch /
torchvision packages first, then run `pip install -e .`.

## 2. External Data

Raw data are not committed to this repository. Use the public Perturb-Multi
release and keep the downloaded assets outside git:

- Perturb-Multi paper: https://doi.org/10.1016/j.cell.2025.05.022
- Cell images: https://huggingface.co/datasets/xingjiepan/PerturbMulti/tree/main

This pipeline does not require the GEO raw sequencing release. It uses the
paired image/protein/RNA assets from Perturb-Multi; the RNA h5ad is treated as a
MERFISH readout for filtering/diagnostics and optional ablation, not as the main
generative condition.

Place the downloaded assets under `data/raw/` using the self-contained,
per-dataset layout below. The configs and build scripts read these paths
directly -- no external source tree or `MORPHO_PHENOTYPING_ROOT` variable is
needed:

```text
data/raw/
  crispr/    images/  manifest.parquet  rna.h5ad  protein.h5ad
  diet/      images/  manifest.parquet  rna.h5ad  protein.h5ad
  metadata/  eval_panel.json  decision_table.csv
  Perturb-multimodal.md
```

The CRISPR factory resolves these via the `RAW_ASSETS` map in
`src/morphoflux/data/factory.py` (paired manifest, image npz directory,
RNA/protein h5ad, metadata decision table, evaluation panel, and the
Perturb-Multi paper markdown). The diet build/audit scripts read
`data/raw/diet/` directly.

## 3. Build Derived Tables

CRISPR:

```bash
python scripts/materialize_data.py --config configs/crispr_hep.yaml
python scripts/build_crispr_paper_data.py
```

Diet:

```bash
python scripts/audit_diet_assets.py
python scripts/build_diet_data.py
```

Generated files under `data/raw/` and `data/processed/` are gitignored. The
runtime configs point to these generated paths.

## 4. Train Proposed Models

Diet headline run:

```bash
OUT=outputs/runs/diet/main \
CONFIG=diet_id DATASET=diet_id \
EPOCHS=12 EVAL_FREQ=2 FID_SAMPLES=5120 \
NPROC=2 BATCH=16 USE_INITIAL=1 CFG=0.2 \
bash scripts/train.sh
```

CRISPR paper-core one-hot run:

```bash
OUT=outputs/runs/crispr/paper_core \
CONFIG=crispr_paper_core DATASET=perturbmulti_id \
EPOCHS=20 EVAL_FREQ=5 FID_SAMPLES=5120 \
NPROC=2 BATCH=16 USE_INITIAL=1 CFG=0.2 \
bash scripts/train.sh
```

## 5. Evaluate Proposed Models

Marker gap/direction summaries:

```bash
python scripts/aggregate_eval.py outputs/runs/diet/main 5 9
python scripts/aggregate_eval.py outputs/runs/crispr/paper_core 5 19
```

CRISPR paper-core program classifier:

```bash
python src/morphoflux/engine/moa/train_moa.py \
  --config_path configs/crispr_paper_core.yaml \
  --mode train \
  --ckpt_path outputs/baselines/moa/crispr_paper/program_classifier.pth \
  --label-map-csv data/processed/crispr/program_labels_paper.csv
```

Diet marker distribution figure:

```bash
python scripts/diet_marker_distribution_figure.py \
  --run-dir outputs/runs/diet/fid5k \
  --epoch 12 \
  --out-dir outputs/figures/diet \
  --prefix diet_fid5k
```

The Diet 5K marker-distribution script writes both figures and machine-readable
CSV/JSON summaries.

## 6. Run Baselines

Export shared imagefolder/npy data:

```bash
bash baselines/export_all_baseline_data.sh
```

Run the baseline queue:

```bash
bash baselines/run_paper_baselines.sh
```

Set `CONDA_ENV`, `CONDA_BIN`, or `CONDA_SH` before launching baseline scripts if
your cluster does not expose conda on `PATH`.

Collect method tables:

```bash
python baselines/collect_paper_metrics.py
```

See `baselines/README.md` for per-method output contracts and GPU policy.

## 7. Current Known Caveats

- Diet is confounded with imaging batch; `BATCH` is collapsed so adlib controls
  can pair with fasted/HFD cells.
- The data are unpaired; do not interpret one generated cell as the true future
  of one control cell.
- FID/KID are reported for comparability but are not the primary biological
  metric for Perturb-Multi. A source-control sanity check can score strongly on
  FID/KID because same-batch controls are realistic.
- A pre-fix 2-GPU Diet 5K eval produced 5120 PNGs but only 2560 paired
  treated->control mappings. Future evals include a DDP mapping gather fix.

## 8. Files Intended For Git

Tracked:

```text
source code
configs
small docs
small paper-table TSVs under data/reports/
```

Not tracked:

```text
raw assets
processed image/index exports
checkpoints and model weights
generated PNGs
logs and output folders
external baseline repositories
```

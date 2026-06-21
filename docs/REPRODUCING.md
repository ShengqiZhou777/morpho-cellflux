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

Point the repository to a local Perturb-Multi asset tree:

```bash
export MORPHO_PHENOTYPING_ROOT=/path/to/morpho-phenotyping
```

The CRISPR factory expects the source files listed in
`configs/crispr_hep.yaml`, including the paired manifest, extracted image npz
directory, RNA/protein h5ad files, metadata decision table, evaluation panel,
and the Perturb-Multi paper markdown.

Diet scripts expect:

```text
$MORPHO_PHENOTYPING_ROOT/assets/paired_filtered/diet/
  manifests/manifest_diet_hep_paired.parquet
  protein/protein_diet_hep_paired.h5ad
```

## 3. Build Derived Tables

CRISPR:

```bash
python scripts/materialize_data.py --config configs/crispr_hep.yaml
python scripts/build_perturbmulti_data.py
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
OUT=outputs/runs/diet/diet_id_v3 \
CONFIG=diet_id_v3 DATASET=diet_id \
EPOCHS=12 EVAL_FREQ=2 FID_SAMPLES=5120 \
NPROC=2 BATCH=16 USE_INITIAL=1 CFG=0.2 \
bash scripts/train.sh
```

CRISPR one-hot run:

```bash
OUT=outputs/runs/crispr/cellflux_pm_train_id_v8 \
CONFIG=perturbmulti_train_id DATASET=perturbmulti_id \
EPOCHS=20 EVAL_FREQ=5 FID_SAMPLES=5120 \
NPROC=2 BATCH=16 USE_INITIAL=1 CFG=0.2 \
bash scripts/train.sh
```

Optional CRISPR one-hot + RNA-signature ablation:

```bash
OUT=outputs/runs/crispr/cellflux_pm_train_id_v9 \
CONFIG=perturbmulti_train_idsig DATASET=perturbmulti_idsig \
EPOCHS=20 EVAL_FREQ=5 FID_SAMPLES=5120 \
NPROC=2 BATCH=16 USE_INITIAL=1 CFG=0.2 \
bash scripts/train.sh
```

## 5. Evaluate Proposed Models

Marker gap/direction summaries:

```bash
python scripts/aggregate_eval.py outputs/runs/diet/diet_id_v3 5 9
python scripts/aggregate_eval.py outputs/runs/crispr/cellflux_pm_train_id_v8 5 19
```

Diet marker distribution figure:

```bash
python scripts/diet_marker_distribution_figure.py \
  --run-dir outputs/runs/diet/diet_id_v3_fid5k \
  --epoch 12 \
  --out-dir outputs/figures/diet \
  --prefix diet_v3_fid5k
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

Detached launchers accept `CONDA_ENV`, `CONDA_BIN`, or `CONDA_SH` overrides if
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
  metric for Perturb-Multi. Copy-control can win FID/KID because same-batch
  controls are realistic.
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

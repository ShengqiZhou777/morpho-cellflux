# Morpho-CellFlux

Research code for adapting CellFlux-style conditional flow matching to
Perturb-Multi hepatocyte images.

The core task is **marker phenotype transport**, not generic RGB image synthesis:
given a control hepatocyte marker image and a target perturbation condition,
generate a false-color marker panel whose population distribution moves toward
the real perturbed population.

```text
control marker image + perturbation condition -> generated perturbed marker image
```

The implementation vendors the CellFlux engine under `src/morphoflux/engine/`
and adds Perturb-Multi data adapters, Diet/CRISPR configs, baseline adapters,
and evaluation scripts.

## Status

This repository is the active research/code-release version. Large raw assets,
checkpoints, generated images, and baseline outputs are intentionally not stored
in git. They are reconstructed from the data builders and scripts documented
below.

Current scientific position:

- Perturb-Multi images are multiplexed molecular readouts: protein markers of
  subcellular structures/signaling pathways plus abundant RNAs.
- FID/KID/MoA are reported for CellFlux-style comparability, but FID alone is
  not a reliable biological success criterion on this dataset.
- The primary biological evidence is marker-distribution movement from control
  toward real treated cells.

See [docs/SCIENTIFIC_STORY.md](docs/SCIENTIFIC_STORY.md) and
[docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md) for the full framing.

## Benchmarks

| benchmark | condition | control / treated | active panel |
|---|---|---|---|
| Diet | diet state one-hot | adlib -> fasted / hfd | `[9,5,8]` = Calreticulin / Perilipin / TOMM20 |
| CRISPR | target-gene identity one-hot | non-targeting -> gene perturbation | `[0,14,5]` = Alb / Rab7 / Perilipin |

The RGB PNGs produced by this repo are false-color renderings of selected marker
channels. They are not natural-color microscopy images.

## Repository Layout

```text
configs/                 Dataset/model configs.
baselines/               Copy-control, PhenDiff, IMPA, StarGAN adapters and metric tables.
data/raw/                Local symlinks to raw assets; gitignored.
data/processed/          Derived indices/embeddings; gitignored except placeholders.
data/reports/            Small report tables intended for paper summaries.
docs/                    Architecture, evaluation protocol, experiment log, story.
scripts/                 Data build, train, eval, plotting, and launch scripts.
src/morphoflux/          Data factory plus vendored CellFlux engine.
outputs/                 Checkpoints, generated samples, logs, metrics; gitignored.
```

## Installation

The code is tested with Python 3.10 and PyTorch 2.x. Create an environment from
the included file:

```bash
conda env create -f environment.yml
conda activate morpho-cellflux
pip install -e .
```

If your cluster requires a specific CUDA/PyTorch build, install that PyTorch
build first, then run `pip install -e .`.

## Data Preparation

Raw Perturb-Multi assets are not included. Put or symlink the source asset tree
somewhere locally and point the repo to it:

```bash
export MORPHO_PHENOTYPING_ROOT=/path/to/morpho-phenotyping
```

Public data sources:

- Perturb-Multi paper: https://doi.org/10.1016/j.cell.2025.05.022
- Cell images: https://huggingface.co/datasets/xingjiepan/PerturbMulti/tree/main
- Sequencing data: GEO `GSE275483`
  (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275483)

The expected source layout is documented by
[configs/crispr_hep.yaml](configs/crispr_hep.yaml). The CRISPR data builder
materializes raw symlinks and derived tables:

```bash
python scripts/materialize_data.py --config configs/crispr_hep.yaml
python scripts/build_perturbmulti_data.py
```

Build the Diet index and one-hot condition embedding:

```bash
python scripts/audit_diet_assets.py
python scripts/build_diet_data.py
```

The engine reads three runtime artifacts from each config:

```text
image_path        raw npz crop directory
data_index_path   engine index CSV
embedding_path    condition embedding CSV
```

## Training

Training uses `torchrun` through [scripts/train.sh](scripts/train.sh). The script
is controlled by environment variables and assumes the project environment is
already active.

Diet:

```bash
OUT=outputs/runs/diet/main \
CONFIG=diet_id DATASET=diet_id \
EPOCHS=12 EVAL_FREQ=2 FID_SAMPLES=5120 \
NPROC=2 BATCH=16 USE_INITIAL=1 CFG=0.2 \
bash scripts/train.sh
```

CRISPR:

```bash
OUT=outputs/runs/crispr/main \
CONFIG=perturbmulti_train_id DATASET=perturbmulti_id \
EPOCHS=20 EVAL_FREQ=5 FID_SAMPLES=5120 \
NPROC=2 BATCH=16 USE_INITIAL=1 CFG=0.2 \
bash scripts/train.sh
```

Quick smoke test:

```bash
make smoke
```

## Evaluation

Aggregate marker-distribution and direction metrics:

```bash
python scripts/aggregate_eval.py outputs/runs/diet/main 5 9
python scripts/aggregate_eval.py outputs/runs/crispr/main 5 19
```

Diet marker distribution figure:

```bash
python scripts/diet_marker_distribution_figure.py \
  --run-dir outputs/runs/diet/fid5k \
  --epoch 12 \
  --out-dir outputs/figures/diet \
  --prefix diet_fid5k
```

CellFlux-style method comparison tables use the matched-N tooling under
`baselines/`:

```bash
bash baselines/export_all_baseline_data.sh
bash baselines/run_paper_baselines.sh
python baselines/collect_paper_metrics.py
```

See [baselines/README.md](baselines/README.md) for external baseline setup and
[docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md) for metric definitions.

## Current Result Snapshot

The current Diet 5K comparison shows why FID is not sufficient here:

| method | FIDo | FIDc | KIDo | KIDc | MoA-Acc |
|---|---:|---:|---:|---:|---:|
| copy_control | **7.96** | **12.01** | **0.0039** | **0.0057** | 49.92 |
| PhenDiff | 10.92 | 13.97 | 0.0066 | 0.0075 | 60.69 |
| IMPA | 52.29 | 55.43 | 0.0407 | 0.0424 | **63.97** |
| Morpho-CellFlux | 31.26 | 35.43 | 0.0267 | 0.0291 | 54.93 |

Copy-control wins FID/KID because same-batch control images are realistic, even
though they do not apply the perturbation. The proposed model's positive signal
is the marker-distribution shift, especially HFD Calreticulin/Perilipin moving
close to the treated population.

Full logs and caveats are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Reproducibility Notes

- Raw data, checkpoints, generated PNGs, and baseline outputs are gitignored.
- The DDP eval loop gathers `trt2ctrl_idx.json` mappings across ranks before
  writing, so future multi-GPU evals preserve complete treated->control metadata.
- Older 5K Diet outputs generated before that fix have 5120 PNGs but only 2560
  paired control mappings; gen-vs-target distributions remain valid, paired
  control gap-closure should be rerun for final figures.

Detailed reproduction steps are in
[docs/REPRODUCING.md](docs/REPRODUCING.md).

## Citation

If you use this code, please cite the accompanying Morpho-CellFlux paper when it
is released and cite CellFlux for the vendored flow-matching engine. A provisional
repository citation is provided in [CITATION.cff](CITATION.cff).

## License and Upstream Attribution

This repository is released under the MIT License. The vendored CellFlux engine
is included with its upstream MIT license in
[src/morphoflux/engine/LICENSE](src/morphoflux/engine/LICENSE); adaptation notes
are in [src/morphoflux/engine/UPSTREAM.md](src/morphoflux/engine/UPSTREAM.md).

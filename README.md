# PhenoFlux

**Pluggable molecular priors for flow matching on cellular phenotype transport.**

Given a control cell image and a perturbation condition, PhenoFlux generates the
perturbed cell phenotype. The model injects two dataset-agnostic molecular priors —
marker self-attention (MSA) and per-channel modulation (PCD) — into a shared flow
matching UNet. The same module pair is applied to both Diet hepatocyte and CRISPR
perturbation data.

```text
control cell image + perturbation condition → generated perturbed cell image
         ↑                                              ↑
    18ch MERFISH                                3ch false-color
    marker profile                              marker panel
```

## Benchmarks

| Dataset | Condition | Control → Treated | Image Panel |
|---------|-----------|-------------------|-------------|
| Diet | 3-dim diet one-hot | adlib → fasted / hfd | [9,5,10] = Calreticulin / Perilipin / pS6RP |
| CRISPR | 40-dim gene one-hot | non-targeting → gene KO | [9,5,10] = Calreticulin / Perilipin / pS6RP |

Diet and CRISPR use the identical 3-channel panel (Calreticulin / Perilipin / pS6RP),
so biological readouts are directly comparable across both datasets.

Images are false-color renderings of selected MERFISH marker channels.

## Installation

```bash
conda env create -f environment.yml
conda activate pmf
pip install -e .
```

Python 3.10, PyTorch 2.x, CUDA 12.8. Tested on 2× RTX 5090 (32 GB each).

## Quick Start

```bash
# Quick validation on mini subset (2 epochs, ~5 min)
bash scripts/quick_validate.sh phenoflux_diet phenoflux

# Full training on 5k subset
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config phenoflux_diet_msa_pcd --device cuda \
  --batch_size 32 --epochs 20 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 5 --fid_samples 5120 --compute_fid --save_fid_samples \
  --data_index data/processed/diet/index_diet_5k.csv \
  --output_dir outputs/runs/diet/msa_pcd_5k
```

## Data Preparation

Raw Perturb-Multi assets are not included. Download from
[HuggingFace](https://huggingface.co/datasets/xingjiepan/PerturbMulti) and place
under `data/raw/`:

```text
data/raw/{crispr,diet}/   images/  manifest.parquet  rna.h5ad
```

Build derived indices and embeddings:

```bash
# CRISPR (40 paper-core genes, 41×40 one-hot embedding)
python scripts/build_crispr_paper_data.py

# Diet (335K cells, 3 conditions, 18ch MERFISH profiles)
python scripts/build_diet_data.py
```

Each config YAML points to three runtime artifacts:

```yaml
image_path:        data/raw/diet/images
data_index_path:   data/processed/diet/index_diet.csv
embedding_path:    data/processed/diet/embedding_diet.csv
```

Switch data size without duplicating configs:

```bash
--data_index data/processed/diet/index_diet_mini.csv   # fast dev (300 cells)
--data_index data/processed/diet/index_diet_5k.csv     # ablation (5k)
--data_index data/processed/diet/index_diet.csv        # paper (full, default)
```

## Experiment Configs (6)

All use `--dataset phenoflux`. The `condition_dim` is auto-computed from YAML flags.

| Config | Molecular Prior | `condition_dim` |
|--------|----------------|:---:|
| `phenoflux_diet` | none (baseline) | 3 |
| `phenoflux_diet_18ch` | naive 18ch concat | 21 |
| `phenoflux_diet_msa` | MSA | 67 |
| `phenoflux_diet_msa_pcd` | MSA + PCD | 67 |
| `phenoflux_crispr` | none (baseline) | 40 |
| `phenoflux_crispr_msa` | MSA | 104 |
| `phenoflux_crispr_msa_pcd` | MSA + PCD | 104 |

## Training

```bash
# Diet (all configs)
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config phenoflux_diet_msa_pcd --device cuda \
  --batch_size 32 --epochs 20 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 5 --fid_samples 5120 --compute_fid --save_fid_samples \
  --output_dir outputs/runs/diet/msa_pcd

# CRISPR
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config phenoflux_crispr_msa_pcd --device cuda \
  --batch_size 32 --epochs 40 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 10 --fid_samples 5120 --compute_fid --save_fid_samples \
  --output_dir outputs/runs/crispr/msa_pcd
```

Or use the convenience launcher:

```bash
CONFIG=phenoflux_diet_msa_pcd OUT=outputs/runs/diet/msa_pcd \
  EPOCHS=20 EVAL_FREQ=5 FID_SAMPLES=5120 NPROC=2 \
  bash scripts/train.sh
```

Checkpoints saved every epoch. Resume: `--resume <checkpoint.pth> --eval_only`.

## Evaluation

All metrics in `phenoflux/eval/`:

```bash
# Image quality — FIDo/c, KIDo/c (matched-N)
python phenoflux/eval/fid.py \
  --real-dir <real_imgs> --gen-dir <fid_samples/epoch-N> \
  --per-condition-cap 500

# Biological metrics — PGC (Phenotypic Gap Closure), dir-corr, sign-agreement, Pearson
python phenoflux/eval/aggregate.py <run_dir> 5 <epoch>

# MoA classifier accuracy
python phenoflux/eval/moa.py \
  --config_path configs/phenoflux_crispr.yaml --mode eval \
  --img_root_path <run_dir>/fid_samples/epoch-<N> \
  --ckpt_path outputs/baselines/moa/crispr/condition_classifier.pth \
  --out_json <run_dir>/moa.json

# Marker distribution figures
python phenoflux/eval/figures.py \
  --run-dir <run_dir> --epoch <N> --out-dir outputs/figures
```

### Metrics Summary

| Metric | Script | What it measures |
|--------|--------|-----------------|
| FIDo / FIDc | `eval/fid.py` | Image quality (pooled / per-condition) |
| KIDo / KIDc | `eval/fid.py` | Image quality (unbiased) |
| PGC | `eval/aggregate.py` | `1 − W(gen,tgt) / W(src,tgt)` — Phenotypic Gap Closure |
| dir-corr | `eval/aggregate.py` | Direction consistency of perturbation effect |
| sign-agreement | `eval/aggregate.py` | Sign match of delta direction |
| MoA accuracy | `eval/moa.py` | Condition classification from generated images |

## Baselines

Adapters for IMPA, PhenDiff, StarGAN, and MorphoDiff in `baselines/`.
External method code in `baselines/external/`.

```bash
bash baselines/export_all_baseline_data.sh
bash baselines/run_paper_baselines.sh
python baselines/compare.py
```

## Repository Layout

```
phenoflux/                   # Python package
├── train.py                 # Entry point (torchrun -m phenoflux.train)
├── models/                  # UNetModel, MSA, PCD, EMA
├── training/                # Loops, dataloader, DDP, checkpoint
└── eval/                    # FID, aggregate, MoA, figures
configs/                     # 6 paper experiment YAMLs
scripts/                     # Data build, train launch, quick validate
baselines/                   # External method adapters
docs/                        # ARCHITECTURE, EVAL_PROTOCOL, REPRODUCING
outputs/                     # Training outputs (gitignored)
data/                        # Raw + processed (gitignored)
```

## Citation

If you use this code, please cite the PhenoFlux paper (forthcoming) and CellFlux
for the flow matching engine:

```bibtex
@article{zhang2025cellflux,
  title={CellFlux: Simulating Cellular Morphology Changes via Flow Matching},
  author={Zhang, Yuhui and Su, Yuchang and Wang, Chenyu and others},
  journal={arXiv preprint arXiv:2502.09775},
  year={2025}
}
```

## License

MIT. The codebase is adapted from the CellFlux engine (also MIT).

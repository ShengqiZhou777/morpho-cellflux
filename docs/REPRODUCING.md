# Reproducing PhenoFlux Experiments

This document describes the artifact contract for reproducing the experiments
from a fresh clone. The repository intentionally does not include raw imaging
data, checkpoints, generated samples, or large baseline exports.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate pmf
pip install -e .
```

If your system requires a different CUDA build, install PyTorch/torchvision first,
then run `pip install -e .`.

## 2. External Data

Raw data are not committed. Download from the public Perturb-Multi release:

- Perturb-Multi paper: https://doi.org/10.1016/j.cell.2025.05.022
- Cell images: https://huggingface.co/datasets/xingjiepan/PerturbMulti

Place under `data/raw/`:

```
data/raw/
  crispr/    images/  manifest.parquet  rna.h5ad  protein.h5ad
  diet/      images/  manifest.parquet  rna.h5ad  protein.h5ad
```

## 3. Build Derived Tables

```bash
# CRISPR: 40 genes in 7 functional programs + one-hot embedding
python scripts/build_crispr_paper_data.py

# Diet: 3 conditions, 18ch population-mean profiles + embedding
python scripts/build_diet_data.py

# Diet subsets for fast validation
python scripts/build_diet_subset.py
```

Each config YAML points to three runtime artifacts:

```
image_path        raw npz crop directory
data_index_path   engine index CSV
embedding_path    condition embedding CSV
```

## 4. Training

All training uses `torchrun -m phenoflux.train --dataset phenoflux --config <name>`.

### Diet experiments (4 configs)

```bash
# Baseline
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config phenoflux_diet --device cuda \
  --batch_size 32 --epochs 20 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 5 --fid_samples 5120 --compute_fid --save_fid_samples \
  --output_dir outputs/runs/diet/baseline

# + 18ch naive concat (info control)
torchrun ... --config phenoflux_diet_18ch --output_dir outputs/runs/diet/18ch

# + MSA
torchrun ... --config phenoflux_diet_msa --output_dir outputs/runs/diet/msa

# + MSA + PCD
torchrun ... --config phenoflux_diet_msa_pcd --output_dir outputs/runs/diet/msa_pcd
```

Omit `--data_index` for full dataset. For fast validation:

```bash
--data_index data/processed/diet/index_diet_2k.csv    # 2k cells, very fast
--data_index data/processed/diet/index_diet_5k.csv    # 18k cells, ablations
```

### CRISPR experiments (4 configs)

```bash
# Baseline
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config phenoflux_crispr --device cuda \
  --batch_size 32 --epochs 40 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 10 --fid_samples 5120 --compute_fid --save_fid_samples \
  --data_index data/processed/crispr/index_paper_40.csv \
  --output_dir outputs/runs/crispr/baseline

# + MSA + PCD
torchrun ... --config phenoflux_crispr_msa_pcd --output_dir outputs/runs/crispr/msa_pcd

# + PCGE
torchrun ... --config phenoflux_crispr_pcge --output_dir outputs/runs/crispr/pcge

# + PCGE + MSA + PCD
torchrun ... --config phenoflux_crispr_pcge_msa_pcd --output_dir outputs/runs/crispr/pcge_msa_pcd
```

### Convenience launcher

```bash
CONFIG=phenoflux_diet_msa_pcd OUT=outputs/runs/diet/msa_pcd \
  EPOCHS=20 EVAL_FREQ=5 FID_SAMPLES=5120 NPROC=2 BATCH=32 \
  bash scripts/train.sh
```

### Quick validation

```bash
bash scripts/quick_validate.sh phenoflux_diet phenoflux
```

## 5. Evaluation

### Image quality (FIDo/c, KIDo/c)

```bash
python phenoflux/eval/fid.py \
  --real-dir <real_imgs> --gen-dir <gen_imgs> --per-condition-cap 500
```

### Biological metrics (gap_closed, dir-corr, sign-agreement)

```bash
python phenoflux/eval/aggregate.py <eval_dir> 5 <epoch>
```

### MoA classifier accuracy

```bash
python phenoflux/eval/moa.py \
  --config_path configs/phenoflux_crispr.yaml --mode eval \
  --img_root_path <eval_dir>/fid_samples/epoch-<N> \
  --ckpt_path outputs/baselines/moa/crispr/condition_classifier.pth \
  --out_json <eval_dir>/moa.json
```

### Baseline comparison table

```bash
bash baselines/run_paper_baselines.sh
python baselines/collect_paper_metrics.py
```

## 6. Checkpoint Format

```python
{
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "args": args,
    "scaler": scaler.state_dict(),
    "best_fid": best_fid,
}
```

## 7. Reproducibility Notes

- Raw data, checkpoints, generated PNGs, and baseline outputs are gitignored.
- The DDP eval loop gathers `trt2ctrl_idx.json` mappings across ranks before
  writing, preserving complete treated→control metadata.
- Detailed architecture docs in `docs/ARCHITECTURE.md`.
- See `docs/SCIENTIFIC_STORY.md` for the paper narrative and experiment rationale.

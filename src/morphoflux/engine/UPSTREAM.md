# Vendored CellFlux engine

This directory is a **vendored copy** of the CellFlux generative-model engine, copied into
this repo so the perturbmulti adaptation lives under our own version control (previously the
edits were made in-place in an external clone and were not tracked anywhere).

## Upstream
- Source: https://github.com/yuhui-zh15/CellFlux
- Vendored at upstream commit: `041c4c0` ("Upload MoA Classifier with LFS")
- Copied: Python/YAML/shell source only. EXCLUDED: `.git/`, `__pycache__/`, demo PNGs
  (`data/*.png`), model weights (`*.pth`, `moa/*.pth`), and any `outputs/`.

## Our changes to the upstream code (the perturbmulti adaptation)
The model architecture, flow-matching training loop, ODE sampler, CFG, and FID are upstream
and unchanged. We only added a dataset/condition adapter + small fixes:

- `models/model_configs.py` — added the `perturbmulti_id` arch (in/out=3, condition_dim=204
  gene-identity, attention_resolutions=[4] to fit 32GB, use_checkpoint=False for DDP).
- `training/data_utils.py` — added `_load_perturbmulti()` + `PERTURBMULTI_CHANNELS=[5,9,10]`
  (loads our npz, selects Perilipin/Calreticulin/pS6RP, maps [0,1]->[-1,1]). No CellFlux-
  internal deps.
- `training/dataloader.py` — added the perturbmulti train/test split branch.
- `train.py` — added `perturbmulti_id` to the dataset list.
- `training/eval_loop.py` — write a per-epoch copy of `trt2ctrl_idx.json`
  (`fid_samples/epoch-<e>/`) so each epoch keeps its own treated->control pairing.
- `training/load_and_save.py` — `torch.load(..., weights_only=False)` for torch 2.11 resume.

## How it's used
- This engine is part of the `morphoflux` package: launched as
  `torchrun -m morphoflux.engine.train` (see `scripts/launch_cellflux_pm.sh` and
  `scripts/interpolate_cellflux.sh`).
- Configs live in `configs/*.yaml` (single source of truth). The loader
  resolves a bare `--config NAME` against `MORPHOFLUX_CONFIG_DIR` (default
  `configs/`), or accepts an absolute `.yaml` path.
- Data paths inside those YAMLs point at `data/processed/cellflux_ext/` (this repo).

The old external clone at `/home/ubuntu/data/sqzhou/projects/CellFlux` is DEPRECATED —
do not edit it; this absorbed copy under `src/morphoflux/engine/` is the source of truth.

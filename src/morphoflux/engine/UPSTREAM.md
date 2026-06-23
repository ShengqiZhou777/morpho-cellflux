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

- `models/model_configs.py` — added Perturb-Multi model configs
  (`perturbmulti_id`, `perturbmulti_idsig`, `diet_id`) with 3-channel images and
  task-specific condition dimensions.
- `training/data_utils.py` — added `_load_perturbmulti()` for npz crops with config-driven
  channel panels. Active paper panels are `[9,5,8]` for Diet and `[9,5,10]` for
  CRISPR paper core.
- `training/dataloader.py` — added the Perturb-Multi/Diet train/test split branch.
- `train.py` — added `perturbmulti_id` to the dataset list.
- `training/eval_loop.py` — write a per-epoch copy of `trt2ctrl_idx.json`
  (`fid_samples/epoch-<e>/`) and gather mappings across DDP ranks before writing.
- `training/load_and_save.py` — `torch.load(..., weights_only=False)` for torch 2.11 resume.

## How it's used
- This engine is part of the `morphoflux` package: launched as
  `torchrun -m morphoflux.engine.train` (see `scripts/train.sh` and
  `scripts/interpolate.sh`).
- Configs live in `configs/*.yaml` (single source of truth). The loader
  resolves a bare `--config NAME` against `MORPHOFLUX_CONFIG_DIR` (default
  `configs/`), or accepts an absolute `.yaml` path.
- Data paths inside those YAMLs point at `data/processed/crispr/` and
  `data/processed/diet/` (this repo).

Any external CellFlux clone used during development is deprecated for this project. This
absorbed copy under `src/morphoflux/engine/` is the source of truth.

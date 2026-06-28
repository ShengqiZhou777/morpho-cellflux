# Architecture

PhenoFlux is a Python package (`phenoflux/`) for **cellular phenotype transport** on
Perturb-Multi hepatocyte images: given a control cell image and a target perturbation
condition, generate the corresponding perturbed-cell marker panel.

## Pipeline (data → engine → evaluation)

```
raw assets (data/raw/)                       h5ad (RNA/protein) + manifest
        |                                                  |
        v                                                  v
  phenoflux training dataloader   ──────────►  scripts/build_crispr_paper_data.py
  reads index CSV + embedding CSV              scripts/build_diet_data.py
        |                                       - index CSV + embedding CSV
        |                                                  |
        └──────────────────────┬──────────────────────────┘
                               v
          phenoflux package  (flow-matching engine adapted from CellFlux)
          torchrun -m phenoflux.train  (scripts/train.sh)
          - conditional flow matching, control→perturbed, CFG, EMA, ODE sampling
          - reads configs/<name>.yaml
          - writes outputs/<run>/: checkpoints, fid_samples/epoch-<e>/, log.txt
                               |
                               v
          evaluation (phenoflux/eval/, reads fid_samples)
          - phenoflux/eval/fid.py       : FIDo/c, KIDo/c with matched-N
          - phenoflux/eval/aggregate.py : PGC (Phenotypic Gap Closure), dir-corr, sign-agreement
          - phenoflux/eval/moa.py       : MoA classifier accuracy
          - phenoflux/eval/figures.py   : marker distribution KDE + bar charts
```

## Package Layout

- `phenoflux/` — main Python package. Entry: `torchrun -m phenoflux.train`.
  - `phenoflux/train.py` — entry point, DDP init, main training loop
  - `phenoflux/args.py` — argument parser
  - `phenoflux/models/` — UNetModel, MSA, PCD, EMA, NN utils
  - `phenoflux/training/` — training/eval loops, dataloader, DDP, checkpoint
  - `phenoflux/eval/` — FID/KID, aggregate metrics, MoA classifier, figures
- `configs/` — 8 paper experiment YAMLs (single source of truth)
- `scripts/` — data build, train launch, quick validate, reproduce
- `baselines/` — IMPA, PhenDiff, StarGAN, MorphoDiff adapters
- `data/` (gitignored) — raw assets, processed indices, embeddings
- `outputs/` (gitignored) — checkpoints, generated samples, logs, metrics
- `docs/` — scientific story, architecture, evaluation protocol, reproducing guide

## Molecular Prior Architecture

One UNet body (`phenoflux`), configurable molecular prior via YAML flags:

```
                    ┌─────────────────────────┐
Condition (one-hot) │ base_condition_dim      │  3 (diet) / 40 (crispr)
                    ├─────────────────────────┤
Marker prior        │ use_msa / use_pcd       │  MSA → PCD (cross-dataset)
                    │ use_marker_profile      │  Info control: naive 18ch concat
                    ├─────────────────────────┤
condition_dim       │ auto-computed           │  base + 64 (MSA) or +18 (naive)
                    └─────────────────────────┘
```

### MSA (Marker Self-Attention) — Diet + CRISPR

- Input: population-mean 18-channel MERFISH marker profile of target condition
- TransformerEncoder self-attention over the 18 markers → learns inter-marker
  co-variation patterns (e.g. Perilipin↑ + Calreticulin↓ = steatosis)
- Output: 64-dim context vector concatenated to condition embedding
- ~118K parameters

### PCD (Per-Channel Decoder) — Diet + CRISPR

- Maps MSA context → per-channel (scale, bias) FiLM modulation on UNet's 3ch output
- 3 output channels each receive independent modulation — different markers respond
  at different magnitudes to the same perturbation
- ~2.4K parameters, per-channel only (no spatial dimensions)

### Info Control

- `use_marker_profile` flag: naive 18ch mean-pool + concat (no learned attention)
- Same input information as MSA, but no architecture to consume it
- Answers: does MSA matter, or just having the extra 18ch info?

## Paper Configs (7)

All use `--dataset phenoflux`. `condition_dim` is auto-computed from YAML flags.

| Config | Dataset | Prior | `condition_dim` | Proves |
|--------|---------|-------|:---:|--------|
| `phenoflux_diet` | Diet | none | 3 | Flow matching baseline |
| `phenoflux_diet_18ch` | Diet | naive 18ch concat | 21 | Raw marker info alone helps |
| `phenoflux_diet_msa` | Diet | MSA | 67 | Learned attention > naive |
| `phenoflux_diet_msa_pcd` | Diet | MSA+PCD | 67 | Per-channel modulation adds gain |
| `phenoflux_crispr` | CRISPR | none | 40 | Flow matching baseline |
| `phenoflux_crispr_msa` | CRISPR | MSA | 104 | Marker prior generalizes across datasets |
| `phenoflux_crispr_msa_pcd` | CRISPR | MSA+PCD | 104 | Per-channel modulation generalizes too |

Data size controlled via `--data_index` CLI (not separate configs):
```bash
--data_index data/processed/diet/index_diet_2k.csv     # fast dev
--data_index data/processed/diet/index_diet_5k.csv     # ablation (18k cells)
--data_index data/processed/diet/index_diet.csv         # full dataset (default)
```

## Data Flow

1. `data_utils.py:read_files_pert` pairs control+treated cells from same batch
2. `use_initial=1` → ODE starts from control image (not noise)
3. `marker_profile` = population-mean 18ch profile of target condition (broadcast to spatial)
4. MSA processes marker_profile internally in UNet.forward() → 64-dim context concatenated to condition
5. PCD applies per-channel (scale, bias) modulation on UNet output from MSA context
6. Flow matching: model learns velocity field from control→target

## Critical Design Rules

1. **marker_profile MUST NOT leak into unconditional path.** When `class_drop_prob`
   triggers CFG dropout, marker_profile must also be absent. UNet handles this
   via zero-padding condition to expected dim.

2. **EMA unwrapping needed** before checking module flags. Use `getattr(model, 'model', model)`.

3. **MSA/PCD are inside UNetModel** (checkpointed). Not externally constructed.

4. **Every epoch saves checkpoint.** Training can be paused/resumed at any epoch boundary.

5. **`find_unused_parameters=True` is REQUIRED** for DDP — MSA/PCD params may be
   unused during CFG dropout.

## Evaluation

All metrics in `phenoflux/eval/`:

```bash
# Image quality — FIDo/c, KIDo/c (matched-N)
python phenoflux/eval/fid.py --real-dir <real_imgs> --gen-dir <fid_samples/epoch-N> --per-condition-cap 500

# Biological metrics — PGC (Phenotypic Gap Closure), dir-corr, sign-agreement
python phenoflux/eval/aggregate.py <eval_dir> 5 <epoch>

# MoA classifier accuracy
python phenoflux/eval/moa.py --config_path configs/phenoflux_crispr.yaml --mode eval \
  --img_root_path <eval_dir>/fid_samples/epoch-<N> \
  --ckpt_path outputs/baselines/moa/crispr/condition_classifier.pth \
  --out_json <eval_dir>/moa.json

# Marker distribution figures
python phenoflux/eval/figures.py --run-dir <run_dir> --epoch <N> --out-dir outputs/figures
```

### Metric Suite

| Family | Metric | Definition | Role |
|---|---|---|---|
| Image quality | FIDo/FIDc | overall/conditional FID | CellFlux-style image realism |
| Image quality | KIDo/KIDc | overall/conditional KID | FID robustness check |
| Condition separability | MoA Acc/F1 | classifier on generated images | Auxiliary biological proxy |
| Marker phenotype | PGC | `1 − W₁(gen,tgt)/W₁(ctrl,tgt)` | Primary Diet biological metric |
| Marker phenotype | dir_corr / sign_agree | per-gene perturbation direction recovery | Primary CRISPR biological metric |

## Dataset Mapping

| | Diet | CRISPR |
|---|---|---|
| Perturbation | physiological state (adlib/fasted/hfd) | target gene identity |
| Control | adlib | non-targeting control sgRNA |
| Treated classes | fasted, hfd | 40 genes in 7 functional programs |
| Active panel | `[9,5,10]` Calreticulin / Perilipin / pS6RP | `[9,5,10]` Calreticulin / Perilipin / pS6RP |

## Modality Semantics

- **Perturbation** = condition identity (diet state or target gene). The model condition input.
- **18-channel MERFISH protein** = the imaging readout. The model generates a config-selected 3-channel panel.
- **209-gene MERFISH mRNA** = transcriptional readout of imaged cells (diagnostics only).
- The RGB PNGs are false-color renderings of selected marker channels — not natural-color microscopy.

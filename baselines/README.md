# Baselines

This directory contains adapters for method-comparison baselines. Baselines must
write outputs in the same format as CellFlux eval runs so the existing
`scripts/aggregate_eval.py` computes all paper metrics consistently.

## Proposed method naming

The proposed method is one method family, **Morpho-CellFlux**. Diet and CRISPR
are separate benchmark tasks using the same engine:

- diet: diet-state condition, diet-specific marker panel
  `[9, 5, 8]` = Calreticulin / Perilipin / TOMM20.
- CRISPR: target-gene identity condition, CRISPR-specific marker panel
  `[0, 14, 5]` = Alb / Rab7 / Perilipin.

The task split is for evaluation clarity, not because the paper proposes two
different models.

## Main comparison methods

Following CellFlux and recent cell-morphology generation papers, the primary
method-comparison table should use named perturbation-generation methods:

| Method | Role | Status |
|---|---|---|
| PhenDiff | DDIM inversion + conditional diffusion translation | adapter scripts added |
| IMPA | GAN/AdaIN image perturbation autoencoder | adapter scripts added |
| MorphoDiff | perturbation-encoding diffusion baseline | adapter pending; use after PhenDiff/IMPA |
| StarGAN / CycleGAN | classic image-to-image GAN baseline, supplement if needed | StarGAN adapter scripts added |
| CellFlux baseline | shared-panel/original recipe | existing outputs |
| Morpho-CellFlux | proposed method | existing outputs |

Copy-control is implemented as an internal null/sanity baseline, not as a
headline named method. It can appear in metric-definition text, supplement, or
as a dashed `gap_closed=0` reference line, but it does not need to occupy a row
in the main method-comparison table unless reviewers ask for a null predictor.

| Internal baseline | Role | Status |
|---|---|---|
| Copy-control | biological null; defines `gap_closed = 0` | local adapter |

Optional supplementary baselines include no-control diffusion/FM variants and
MorphGen/MorphDiff-style generation. These should not replace PhenDiff/IMPA in
the main comparison because they do not use control images in the same way.

## Output contract

Each baseline run should create:

```text
outputs/baselines/<method>/<benchmark>/
  args.json
  fid_samples/
    trt2ctrl_idx.json
    epoch-0/
      <condition_or_gene>/
        <target_sample_id>.png
      trt2ctrl_idx.json
```

Then evaluate with:

```bash
python scripts/aggregate_eval.py outputs/baselines/<method>/<benchmark> 5 0
```

## Output directory convention

Keep method-comparison outputs separate from proposed-method runs:

| Method | Diet output | CRISPR output |
|---|---|---|
| Copy-control | `outputs/baselines/copy_control/diet_v3` | `outputs/baselines/copy_control/crispr_v8` |
| PhenDiff | `outputs/baselines/phendiff/diet_v3` | `outputs/baselines/phendiff/crispr_v8` |
| IMPA | `outputs/baselines/impa/diet_v3` | `outputs/baselines/impa/crispr_v8` |
| StarGAN | `outputs/baselines/stargan/diet_v3` | `outputs/baselines/stargan/crispr_v8` |
| MorphoDiff | `outputs/baselines/morphodiff/diet_v3` | `outputs/baselines/morphodiff/crispr_v8` |
| No-control diffusion/FM | `outputs/baselines/no_control_fm/diet_v3` | `outputs/baselines/no_control_fm/crispr_v8` |

Existing proposed-method runs remain:

| Benchmark | Proposed output |
|---|---|
| Diet | `outputs/diet_id_v3` |
| CRISPR | `outputs/cellflux_pm_train_id_v8` |

Existing shared-panel baselines remain where they were trained, but manuscript
tables should refer to them by method name, not by run-directory name:

| Baseline role | Existing output |
|---|---|
| Diet shared-panel CellFlux | `outputs/diet_id_v1` |
| CRISPR shared-panel CellFlux | `outputs/cellflux_pm_train_id_v7` |

## Shared exported data

External methods should train/evaluate from exported panels under:

```text
outputs/baselines/_data/
  diet_v3/
    imagefolder/
    impa_npy/
    metadata.csv
    impa_index.csv
    conditions.json
    manifest.json
  crispr_v8/
    ...
```

Create these exports with:

```bash
bash baselines/export_all_baseline_data.sh
```

Run the full resumable paper-baseline queue with:

```bash
bash baselines/run_paper_baselines_detached.sh

# foreground / debugging:
bash baselines/run_paper_baselines.sh

# optional: increase export parallelism for large image panels
EXPORT_WORKERS=16 bash baselines/run_paper_baselines_detached.sh
```

This runs only non-proposed baselines: shared exports, copy-control, PhenDiff,
IMPA, and StarGAN for each benchmark, then collects table rows. It skips any
method/benchmark that already has `aggregate_eval_summary.json`.

Run the first executable external adapters only when debugging a narrower queue:

```bash
bash baselines/run_primary_baselines_diet_detached.sh

# or foreground:
PHENDIFF_EPOCHS=8 IMPA_EPOCHS=8 bash baselines/run_primary_baselines_diet.sh

# individual methods:
BENCHMARK=diet_v3 bash baselines/run_phendiff.sh
BENCHMARK=diet_v3 bash baselines/run_impa.sh
BENCHMARK=diet_v3 bash baselines/run_stargan.sh
```

Switch `BENCHMARK=crispr_v8` for the CRISPR task after the diet adapters are
validated.

After each external baseline is evaluated with `scripts/aggregate_eval.py`, collect
paper-table rows with:

```bash
python baselines/collect_paper_metrics.py
```

## GPU policy

- Copy-control is CPU/file I/O only; do not allocate GPUs.
- PhenDiff uses two `accelerate` processes by default (`CUDA_VISIBLE_DEVICES=0,1`, `NPROC=2`).
- IMPA uses its internal `nn.DataParallel`; keep `DEVICES=1` and expose both cards with
  `CUDA_VISIBLE_DEVICES=0,1`.
- StarGAN's original trainer is single-process; keep it as an optional/supplemental
  baseline unless/until a DDP/DataParallel training patch is added.
- Do not run two GPU baselines concurrently until memory use is measured. Use one
  baseline training job at a time across both GPUs, then evaluate on CPU/GPU as
  needed.

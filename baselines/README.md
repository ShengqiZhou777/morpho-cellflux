# Baselines

This directory contains adapters for method-comparison baselines. Baselines must
write outputs in the same format as CellFlux eval runs so the existing
`scripts/aggregate_eval.py` computes all paper metrics consistently.

## Proposed method naming

The proposed method is one method family, **Morpho-CellFlux**. Diet and CRISPR
are separate benchmark tasks using the same engine:

- diet: diet-state condition, diet-specific marker panel
  `[9, 5, 8]` = Calreticulin / Perilipin / TOMM20.
- CRISPR paper core: target-gene identity condition, reported by original
  Perturb-Multi programs, marker panel `[9, 5, 10]` =
  Calreticulin / Perilipin / pS6RP.

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
| CellFlux-style baseline | shared-panel CellFlux-engine recipe, not the original public CellFlux benchmark | existing outputs |
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

| Method | Diet output | CRISPR paper output |
|---|---|---|
| PhenDiff | `outputs/baselines/phendiff/diet` | `outputs/baselines/phendiff/crispr_paper` |
| IMPA | `outputs/baselines/impa/diet` | `outputs/baselines/impa/crispr_paper` |
| StarGAN | `outputs/baselines/stargan/diet` | `outputs/baselines/stargan/crispr_paper` |
| MorphoDiff | `outputs/baselines/morphodiff/diet` | `outputs/baselines/morphodiff/crispr_paper` |
| No-control diffusion/FM | `outputs/baselines/no_control_fm/diet` | `outputs/baselines/no_control_fm/crispr_paper` |

Existing proposed-method runs remain:

| Benchmark | Proposed output |
|---|---|
| Diet | `outputs/runs/diet/main` |
| CRISPR paper core | `outputs/runs/crispr/paper_core` |

Shared-panel CellFlux-style baselines, when included, should use the same public
baseline namespace rather than internal training run names. Do not call these
runs "original CellFlux" unless they were produced by the unmodified upstream
repository and original benchmark protocol.

| Baseline role | Output |
|---|---|
| Diet shared-panel CellFlux | `outputs/baselines/cellflux/diet` |
| CRISPR paper shared-panel CellFlux | `outputs/baselines/cellflux/crispr_paper` |

## Shared exported data

External methods should train/evaluate from exported panels under:

```text
outputs/baselines/_data/
  diet/
    imagefolder/
    impa_npy/
    metadata.csv
    impa_index.csv
    conditions.json
    manifest.json
  crispr/
    ...
```

Create these exports with:

```bash
bash baselines/export_all_baseline_data.sh
```

Run the paper-baseline queue with:

```bash
bash baselines/run_paper_baselines.sh

# optional: increase export parallelism for large image panels
EXPORT_WORKERS=16 bash baselines/run_paper_baselines.sh
```

This runs only non-proposed paper baselines: shared exports, PhenDiff, IMPA,
and StarGAN for each benchmark, then collects table rows. It skips any
method/benchmark that already has `aggregate_eval_summary.json`.

Copy-control is an internal sanity check, not a paper method row. Run it only
when explicitly needed:

```bash
INCLUDE_COPY_CONTROL=1 bash baselines/run_paper_baselines.sh
# or, without the full queue:
BENCHMARKS="diet" bash baselines/run_phendiff.sh
```

Run individual external adapters when debugging a narrower queue:

```bash
BENCHMARK=diet bash baselines/run_phendiff.sh
BENCHMARK=diet bash baselines/run_impa.sh
BENCHMARK=diet bash baselines/run_stargan.sh
```

Switch `BENCHMARK=crispr_paper` for the paper CRISPR task after the diet
adapters are validated.

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

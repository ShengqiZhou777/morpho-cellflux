# Baseline Run Queue

The comparison should follow the CellFlux literature line: PhenDiff and IMPA are
the primary control-image-aware baselines; copy-control is the biological null.

## Task definitions

Use the same proposed method name for both tasks: **Morpho-CellFlux**.

| Benchmark | Condition | Config | Proposed run | Primary claim |
|---|---|---|---|---|
| Diet | diet state one-hot | `configs/diet_id_v3.yaml` | `outputs/diet_id_v3` | strong physiological HFD morphology generation |
| CRISPR | target-gene one-hot | `configs/perturbmulti_train_id.yaml` | `outputs/cellflux_pm_train_id_v8` | subtle gene-identity perturbation direction recovery |

The benchmarks are separated because their perturbation semantics and effect
sizes differ. The model family is the same; the input condition and selected
readout panel are task-specific.

## Priority 0: null baseline

Copy-control is implemented locally:

```bash
python baselines/copy_control.py \
  --config configs/diet_id_v3.yaml \
  --output outputs/baselines/copy_control/diet_v3 \
  --split test

python scripts/aggregate_eval.py outputs/baselines/copy_control/diet_v3 5 0
```

```bash
python baselines/copy_control.py \
  --config configs/perturbmulti_train_id.yaml \
  --output outputs/baselines/copy_control/crispr_v8 \
  --split test

python scripts/aggregate_eval.py outputs/baselines/copy_control/crispr_v8 5 0
```

Expected result: `gap_closed` should be approximately zero by construction.
Small deviations can occur because controls are randomly paired and the
aggregate metric recomputes source distributions from the sampled mapping.

## Priority 1: PhenDiff

External method: Bourou et al., MICCAI 2024, conditional DDIM inversion for
cell phenotype translation.

Repository:

```text
https://github.com/WarmongeringBeaver/PhenDiff
```

Run order:

1. Diet first: only three classes (`adlib`, `fasted`, `hfd`), quickest adapter.
2. CRISPR second: 76-gene one-hot task, more classes and likely slower.

Executable adapter:

```bash
BENCHMARK=diet_v3 EPOCHS=8 BATCH=16 bash baselines/run_phendiff.sh
BENCHMARK=crispr_v8 EPOCHS=8 BATCH=16 bash baselines/run_phendiff.sh
```

Adapter behavior:

- Use the same train/test rows from the Morpho-CellFlux config index.
- Train a conditional diffusion model over the selected 3-channel panel.
- At inference, invert/control-start from the paired control image and decode to
  the target condition.
- Save generated samples using the shared output contract in `README.md`.
- Use both GPUs for training when the launcher supports it:
  `CUDA_VISIBLE_DEVICES=0,1`.

## Priority 2: IMPA

External method: Palma et al., Nature Communications 2025, Image Perturbation
Autoencoder using AdaIN-conditioned GAN/style transfer.

Repository:

```text
https://github.com/theislab/IMPA
```

Run order:

1. Diet with 3-d one-hot condition.
2. CRISPR with the same 204-d gene-identity one-hot used by Morpho-CellFlux.

Executable adapter:

```bash
BENCHMARK=diet_v3 EPOCHS=8 BATCH=16 DEVICES=1 bash baselines/run_impa.sh
BENCHMARK=crispr_v8 EPOCHS=8 BATCH=16 DEVICES=1 bash baselines/run_impa.sh
```

Adapter behavior:

- Control image enters the image encoder.
- Target condition enters the perturbation/style branch.
- Generated output is saved as the target sample PNG under the same
  `fid_samples/epoch-0/<condition>/<target_id>.png` layout.
- Use both GPUs for training when the launcher supports it:
  `CUDA_VISIBLE_DEVICES=0,1`. IMPA uses internal `nn.DataParallel`, so keep
  one Lightning process (`DEVICES=1`).

## Priority 3: StarGAN optional baseline

External method: Choi et al., CVPR 2018, multi-domain image-to-image
translation.

Executable adapter:

```bash
BENCHMARK=diet_v3 NUM_ITERS=50000 BATCH=16 bash baselines/run_stargan.sh
BENCHMARK=crispr_v8 NUM_ITERS=50000 BATCH=16 bash baselines/run_stargan.sh
```

Use this as a supplement/main-table candidate after PhenDiff and IMPA are
validated. The original trainer is single-process, so it does not currently use
both GPUs.

## Priority 4: MorphoDiff / no-control supplementary baseline

Add a no-control conditional diffusion or noise-to-data flow-matching baseline
only after PhenDiff and IMPA are complete. This is useful for showing that
control-image initialization/batch calibration matters, but it is not a
replacement for the CellFlux main baselines.

MorphoDiff remains a named-method target, but it should not block the first
paper table because its public code is condition-to-image rather than paired
control-to-target translation. When added, it must still export into the same
`fid_samples` layout and be evaluated only through `scripts/aggregate_eval.py`.

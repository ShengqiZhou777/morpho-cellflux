# Experiment Notes

This is a compact public experiment record. The release keeps only the current
canonical setup and the conclusions needed to interpret the code. Internal queue
logs, obsolete run names, and superseded configs are intentionally omitted.

## Canonical Setups

| benchmark | config | condition | panel | role |
|---|---|---|---|---|
| Diet | `configs/diet_id.yaml` | diet state one-hot | `[9,5,8]` Calreticulin / Perilipin / TOMM20 | strong physiological marker-shift benchmark |
| CRISPR | `configs/perturbmulti_train_id.yaml` | target-gene one-hot | `[0,14,5]` Alb / Rab7 / Perilipin | harder genetic perturbation benchmark |
| Diet figure | `configs/figures/diet_hfd_interp.yaml` | diet state one-hot | `[9,5,8]` | HFD interpolation/figure subset |
| CRISPR figure | `configs/figures/perturbmulti_leadgenes_interp.yaml` | target-gene one-hot | `[0,14,5]` | lead-gene interpolation/figure subset |
| CRISPR ablation | `configs/ablations/perturbmulti_idsig.yaml` | one-hot plus RNA signature | `[0,14,5]` | optional conditioning ablation |

## Data Semantics

Perturb-Multi images are multiplexed marker readouts, not ordinary RGB
microscopy. The 18-channel panel contains protein markers of subcellular
structures/signaling pathways plus abundant RNAs. The active 3-channel panels
are false-color selections from this 18-channel readout.

The task is unpaired distribution transport:

```text
control-cell marker image + perturbation condition -> generated perturbed marker image
```

The generated image should be evaluated by population-level marker movement, not
by pixel-level matching to a single target cell.

## Metric Conclusions

FID/KID and MoA are kept for CellFlux-style comparison, but they are not the only
biological readout. On Diet, copy-control can win FID/KID because same-batch
control images are realistic even though they apply no perturbation. Therefore:

- FID/KID: image-realism/comparability metrics.
- MoA: auxiliary condition-separability metric when real-image ceiling is above chance.
- Marker gap closure: primary Diet biological metric.
- Direction recovery and sign agreement: primary CRISPR biological metrics.

See `docs/EVAL_PROTOCOL.md` for exact definitions.

## Current Diet Result

The Diet 5K marker-distribution check uses the Diet panel
Calreticulin / Perilipin / TOMM20. Generated-vs-target foreground marker means:

| condition | marker | generated | target | read |
|---|---|---:|---:|---|
| fasted | Calreticulin | 0.3715 | 0.3528 | slight overshoot |
| fasted | Perilipin | 0.3208 | 0.3123 | slight overshoot |
| fasted | TOMM20 | 0.4115 | 0.4122 | close |
| hfd | Calreticulin | 0.4213 | 0.4173 | close |
| hfd | Perilipin | 0.3514 | 0.3555 | close |
| hfd | TOMM20 | 0.4235 | 0.4423 | under-shift |

The strongest current positive signal is HFD Calreticulin/Perilipin marker
distribution movement toward the real treated state.

## Matched Diet Baseline Table

Matched per-condition cap 2466 (`N=4932`):

| method | FIDo | FIDc | KIDo | KIDc | MoA-Acc |
|---|---:|---:|---:|---:|---:|
| copy_control | **7.96** | **12.01** | **0.0039** | **0.0057** | 49.92 |
| PhenDiff | 10.92 | 13.97 | 0.0066 | 0.0075 | 60.69 |
| IMPA | 52.29 | 55.43 | 0.0407 | 0.0424 | **63.97** |
| proposed | 31.26 | 35.43 | 0.0267 | 0.0291 | 54.93 |

This table should be interpreted as a metric reality check: the proposed model
does not currently win CellFlux-style image metrics, but it shows marker
distribution migration on the Diet panel.

## Known Caveats

- Diet condition is confounded with imaging batch. The Diet builder collapses
  BATCH to allow adlib controls to pair with fasted/HFD cells.
- CRISPR single-gene effects are subtle relative to hepatocyte heterogeneity; use
  distribution-level plots and gene/channel summaries rather than single-cell
  before/after claims.
- Older Diet 5K outputs generated before the DDP mapping fix have complete PNGs
  but incomplete paired control mapping. Rerun after the DDP mapping fix for
  final paired gap-closure figures.

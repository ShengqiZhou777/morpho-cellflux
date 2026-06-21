# Result Snapshot

This page records the current public-facing result snapshot. It is intentionally
short; detailed run history belongs in internal lab notes, not in the GitHub
release surface.

## Metric Roles

| metric | role |
|---|---|
| FID/KID | CellFlux-style image-realism and comparability metrics |
| MoA | condition-separability proxy when the real-image ceiling is above chance |
| marker gap closure | primary biological metric for Diet marker phenotype transport |
| direction recovery | primary CRISPR metric across genes and marker channels |

See `docs/EVAL_PROTOCOL.md` for exact definitions.

## Diet Marker Transport

Active config: `configs/diet_id.yaml`
Panel: `[9,5,8]` = Calreticulin / Perilipin / TOMM20
Condition: adlib control -> fasted / hfd

The 5K Diet marker-distribution check uses:

```bash
python scripts/diet_marker_distribution_figure.py \
  --run-dir outputs/runs/diet/fid5k \
  --epoch 12 \
  --out-dir outputs/figures/diet \
  --prefix diet_fid5k
```

Generated-vs-target foreground marker means:

| condition | marker | generated | target | read |
|---|---|---:|---:|---|
| fasted | Calreticulin | 0.3715 | 0.3528 | slight overshoot |
| fasted | Perilipin | 0.3208 | 0.3123 | slight overshoot |
| fasted | TOMM20 | 0.4115 | 0.4122 | close |
| hfd | Calreticulin | 0.4213 | 0.4173 | close |
| hfd | Perilipin | 0.3514 | 0.3555 | close |
| hfd | TOMM20 | 0.4235 | 0.4423 | under-shift |

Interpretation: the clearest positive signal is Diet HFD marker-distribution
movement, especially Calreticulin and Perilipin. TOMM20 moves in the right
direction but remains under-shifted for HFD.

## CellFlux-Style Diet Table

Matched Diet 5K table, cap=2466 per treated condition (`N=4932`):

| method | FIDo | FIDc | KIDo | KIDc | MoA-Acc |
|---|---:|---:|---:|---:|---:|
| copy_control | **7.96** | **12.01** | **0.0039** | **0.0057** | 49.92 |
| PhenDiff | 10.92 | 13.97 | 0.0066 | 0.0075 | 60.69 |
| IMPA | 52.29 | 55.43 | 0.0407 | 0.0424 | **63.97** |
| proposed | 31.26 | 35.43 | 0.0267 | 0.0291 | 54.93 |

This table is the main reason FID/KID are not treated as the primary biological
metric here: a no-perturbation copy-control baseline wins FID/KID because
same-batch control images are realistic.

## CRISPR Direction Recovery

Active config: `configs/perturbmulti_train_id.yaml`
Panel: `[0,14,5]` = Alb / Rab7 / Perilipin
Condition: target-gene one-hot

Current public claim: CRISPR effects are subtle, so the result should be shown
with distribution-level metrics and gene/channel summaries, not single-cell
before/after visual claims. The strongest current signal is partial direction
recovery on Alb and Rab7, with weaker Perilipin gap closure.

## Caveats

- Diet state is confounded with imaging batch; this is disclosed in the data
  builder and evaluation protocol.
- Control->treated pairs are unpaired population samples, not same-cell
  trajectories.
- Older Diet 5K outputs generated before the DDP mapping fix have 5120 PNGs but
  only 2560 paired control mappings. Gen-vs-target distributions remain valid;
  paired gap-closure should be rerun for final figures.

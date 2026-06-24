# Result Snapshot

This page records the current public-facing result snapshot. It is intentionally
short; detailed run history belongs in internal lab notes, not in the GitHub
release surface.

## Metric Roles

| metric | role |
|---|---|
| FID/KID | CellFlux-style image-realism and comparability metrics |
| MoA / Program-Acc | condition-separability proxy when the real-image ceiling is above chance; CRISPR paper core uses original-paper program labels |
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
| PhenDiff | 10.92 | 13.97 | 0.0066 | 0.0075 | 60.69 |
| IMPA | 52.29 | 55.43 | 0.0407 | 0.0424 | 48.5 |
| StarGAN | — | — | — | — | 46.3 |
| **CellFlux** | — | — | — | — | **76.66** |
| PhenoFlux | — | — | — | — | — |

An internal no-transport sanity check scores strongly on FID/KID because
same-batch control images are realistic, so it is not included as a paper
baseline row. The paper table should compare named generation methods and use
marker transport as the primary biological readout.

## Diet Condition Classifier Check

Evaluator: Inception-feature classifier trained on real Diet treated images
(`fasted` vs `hfd`) from `configs/diet_id.yaml`.

Real held-out treated-image ceiling after 10 total epochs:

| split | accuracy | macro-F1 | fasted acc | hfd acc |
|---|---:|---:|---:|---:|
| real test | 78.64 | 0.7860 | 80.22 | 76.95 |

Generated target-condition accuracy:

| run | checkpoint | PNGs | accuracy | macro-F1 | fasted acc | hfd acc |
|---|---|---:|---:|---:|---:|---:|
| `diet_id_v3_fid5k` | epoch 12 | 5120 | 58.91 | 0.5631 | 80.33 | 35.85 |
| `diet_id_v3` | epoch 11 | 1024 | 60.35 | 0.5813 | 81.80 | 38.05 |
| `diet_id_v3_cfg15_ep9` | epoch 10 | 1200 | 65.42 | 0.6427 | 80.00 | 49.57 |

Interpretation: Diet gives a much stronger condition-separability signal than
CRISPR with the current three-channel panels. The real-image evaluator ceiling is
near 0.8, and generated fasted images repeatedly reach about 0.8 target-condition
accuracy. Overall generated Diet condition accuracy is not yet a 0.8 paper
number because hfd generation remains weak.

## CRISPR Direction Recovery

Active config: `configs/crispr_paper_core.yaml`
Panel: `[9,5,10]` = Calreticulin / Perilipin / pS6RP
Condition: target-gene one-hot, reported by original-paper program labels

Current public claim: CRISPR effects are subtle, so the result should be shown
with distribution-level metrics, gene/channel summaries, and program-level
classifier metrics, not single-cell before/after visual claims.

Current foreground-weighted run:

```bash
OUT=outputs/runs/crispr/paper_core_masked_loss \
CONFIG=crispr_paper_core DATASET=perturbmulti_id \
EPOCHS=60 EVAL_FREQ=10 FID_SAMPLES=5120 \
NPROC=2 BATCH=16 ACCUM=1 USE_INITIAL=1 CFG=0.2 \
FOREGROUND_LOSS=1 FOREGROUND_THRESHOLD=0.05 \
FOREGROUND_WEIGHT=5.0 BACKGROUND_WEIGHT=0.1 \
bash scripts/train.sh
```

This run changes the CellFlux-style training objective by computing the image
MSE mainly on foreground pixels. The foreground mask is derived online from the
selected raw marker channels after mapping tensors back to `[0,1]`; pixels above
`0.05` in either the source control image or target treated image receive weight
`5.0`, while background pixels receive weight `0.1`. The smoke test completed
with `test_run=True`, `foreground_loss=True`, and first-step loss `0.3169`.

Epoch-9 checkpoint readout:

| metric | value |
|---|---:|
| train_loss | 0.0681 |
| eval_fid | 41.26 |
| generated PNGs / mappings | 2483 |
| full-gene Calreticulin dir-corr / sign-agree | 0.009 / 0.57 |
| full-gene Perilipin dir-corr / sign-agree | 0.016 / 0.38 |
| full-gene pS6RP dir-corr / sign-agree | 0.009 / 0.38 |

Interpretation: epoch 9 is not enough to claim biological transport. Generated
marker means show weak target correlation, but the control-to-treated direction
is not recovered and pooled marker gap closure is negative. Continue only to the
next planned checkpoint, then stop if marker metrics do not improve.

Epoch-19 checkpoint readout:

| metric | value |
|---|---:|
| train_loss | 0.0610 |
| eval_fid | 25.85 |
| generated PNGs / mappings | 2483 |
| full-gene Calreticulin dir-corr / sign-agree | -0.063 / 0.40 |
| full-gene Perilipin dir-corr / sign-agree | -0.016 / 0.40 |
| full-gene pS6RP dir-corr / sign-agree | 0.292 / 0.72 |
| pooled Calreticulin gap_closed | -0.793 |
| pooled Perilipin gap_closed | -0.120 |
| pooled pS6RP gap_closed | 0.449 |

Epoch 19 improves image realism by FID, but the biological marker transport is
still uneven: pS6RP improves, Calreticulin and Perilipin do not close the
source-target gap.

Initial 7-program MoA/Program-Acc check:

| classifier | real test acc | real macro-F1 | epoch 9 acc | epoch 9 macro-F1 | epoch 19 acc | epoch 19 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| unbalanced CE | 35.44 | 0.119 | 31.53 | 0.109 | 33.02 | 0.113 |
| class-balanced CE | 11.20 | 0.097 | 7.69 | 0.064 | 8.34 | 0.069 |

Interpretation: the 7 original Perturb-Multi programs are not yet cleanly
separable with the current three-channel CRISPR panel and Inception-feature MoA
classifier. Program-MoA can be reported as a diagnostic only after establishing
a stronger real-image classifier ceiling; it should not be used as the main
paper metric in this state.

Training policy note: future long runs should not be treated as "must finish all
epochs" jobs. Use early stopping checkpoints based on held-out evaluation rather
than training loss alone. For both Diet and CRISPR, always inspect FID/KID and
condition-classifier metrics at eval checkpoints. For CRISPR, the biology-facing
stop/continue signal should be foreground marker direction recovery, sign
agreement, and original-paper program-level marker summaries. For Diet, use
foreground marker gap closure together with Diet-condition MoA/condition
classifier accuracy and visual inspection of saved samples. Practical checkpoints
are epoch 9/19/39/59 for `EVAL_FREQ=10`; stop after an eval point when FID/KID
degrade substantially or the marker/classifier metrics plateau or regress.
Use smaller quick-eval budgets (`FID_SAMPLES=1024` or `2048`) while selecting
checkpoints, then run a separate final eval with the largest available matched
test budget. Eval logs now record both requested and actually generated sample
counts because CRISPR paper-core has fewer unique held-out treated targets than
some nominal FID budgets.

## Caveats

- Diet state is confounded with imaging batch; this is disclosed in the data
  builder and evaluation protocol.
- Control->treated pairs are unpaired population samples, not same-cell
  trajectories.
- Older Diet 5K outputs generated before the DDP mapping fix have 5120 PNGs but
  only 2560 paired control mappings. Gen-vs-target distributions remain valid;
  paired gap-closure should be rerun for final figures.

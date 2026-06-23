# Scientific Story and Algorithm Plan

## Core Position

This project is not generic RGB cell-image generation. Perturb-Multi images are
**multiplexed molecular phenotype readouts**: protein markers of organelles and
signaling pathways plus abundant RNAs, cropped per segmented hepatocyte and
rendered as false-color panels for modeling.

The paper story should therefore be:

```text
Given a control hepatocyte marker image and a target perturbation condition,
can we transport the cell population toward the marker-intensity distribution
seen under the real perturbation?
```

For Diet, the perturbation condition is physiological state
`adlib -> {fasted, hfd}`. For CRISPR, it is target-gene identity on the fixed
paper-core knockout panel: the 40 main quantitative genes grouped into the
original Perturb-Multi biological programs. In both cases the data are unpaired:
a control image and a treated image are sampled from matched populations, not
from the same cell over time. The defensible claim is distribution-level marker
phenotype transport, not single-cell fate prediction.

## What The 18 Channels Mean

The Perturb-Multi paper states that the imaging panel is an 18-target panel
comprising protein markers of subcellular structures and signaling pathways,
plus four abundant RNAs. The canonical channel order used in this repo is:

| idx | marker |
|---:|---|
| 0 | Alb |
| 1 | polyT |
| 2 | rRNA |
| 3 | M6PR |
| 4 | CathB |
| 5 | Perilipin |
| 6 | Sqstm1 |
| 7 | LC3b |
| 8 | TOMM20 |
| 9 | Calreticulin |
| 10 | pS6RP |
| 11 | Na/K-ATPase |
| 12 | SNAP23 |
| 13 | TOM70 |
| 14 | Rab7 |
| 15 | mtRNA |
| 16 | Vimentin |
| 17 | Gapdh |

The active panels are:

| benchmark | channels | markers | reason |
|---|---|---|---|
| Diet | `[9,5,8]` | Calreticulin / Perilipin / TOMM20 | strong physiological marker shifts, especially HFD lipid/ER/mitochondrial response |
| CRISPR paper core | `[9,5,10]` | Calreticulin / Perilipin / pS6RP | original Perturb-Multi UPR / steatosis / mTOR programs |

So the RGB images in figures are not natural-color microscopy. They are
false-color composites of selected marker channels. This is why a generated
image can look visually plausible while still failing the biological task, and
why distribution metrics on marker intensities are essential.

## Main Claim

The strong version we can currently support is:

```text
Morpho-CellFlux learns conditional transport of multiplexed hepatocyte marker
phenotypes. On strong physiological perturbations, it moves generated marker
distributions close to the real treated distributions. On CRISPR perturbations,
where effects are much weaker than cell-to-cell heterogeneity, it recovers a
partial but measurable gene-conditioned distribution shift.
```

The cautious boundary is equally important:

```text
Current results do not show that the proposed model beats all baselines by
standard image-realism metrics, and they do not prove faithful spatial
morphology reconstruction at single-cell level.
```

That boundary is not a weakness in the story. It makes the story honest: the
dataset is a multiplexed marker-phenotype benchmark, and the correct readout is
whether the conditional distribution moves in the right marker space.

## Evidence So Far

### Diet 5K Marker Shift

Run:

```text
outputs/runs/diet/fid5k
checkpoint: outputs/runs/diet/main/checkpoint-11.pth
generated: outputs/runs/diet/fid5k/fid_samples/epoch-12
N: 5120 generated PNGs
panel: Calreticulin / Perilipin / TOMM20
```

Reproducible figure/script:

```text
python scripts/diet_marker_distribution_figure.py \
  --run-dir outputs/runs/diet/fid5k \
  --epoch 12 \
  --out-dir outputs/figures/diet \
  --prefix diet_fid5k
```

Outputs:

```text
outputs/figures/diet/diet_fid5k_marker_distributions.png
outputs/figures/diet/diet_fid5k_mean_shift.png
outputs/figures/diet/diet_fid5k_marker_distribution_summary.csv
outputs/figures/diet/diet_fid5k_marker_distribution_summary.json
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

The most defensible visual claim is that the model visibly moves the HFD
Calreticulin/Perilipin distributions toward the treated population, with a
weaker TOMM20 shift. For fasted, TOMM20 is strong while Calreticulin and
Perilipin slightly overshoot.

Important caveat: this 5K evaluation was generated before the DDP mapping fix,
so `trt2ctrl_idx.json` contains 2560 paired mappings for 5120 PNGs. Gen-vs-target
distribution plots use all 5120 images and are valid. Paired control
gap-closure values use only mapped rows until the eval is rerun.

### CellFlux-Style Metrics Reality Check

Diet 5K comparison at matched per-condition cap 2466 (`N=4932`):

| method | FIDo | FIDc | KIDo | KIDc | MoA-Acc |
|---|---:|---:|---:|---:|---:|
| PhenDiff | 10.92 | 13.97 | 0.0066 | 0.0075 | 60.69 |
| IMPA | 52.29 | 55.43 | 0.0407 | 0.0424 | **63.97** |
| Morpho-CellFlux | 31.26 | 35.43 | 0.0267 | 0.0291 | 54.93 |

This table prevents overclaiming:

- PhenDiff is the best nontrivial method by FID/KID in this Diet table.
- IMPA is best by MoA accuracy.
- The proposed model is not currently the winner under CellFlux-style image
  metrics, but it shows a clear marker-distribution shift on the Diet panel.

An internal no-transport sanity check can score strongly on FID/KID because it
keeps same-batch control realism. We do not include that check as a paper
baseline row.

Therefore, the proposed method should not be sold as "best image generator" on
the current table. The stronger and cleaner story is that Perturb-Multi exposes
a metric mismatch: image realism and marker phenotype transport are not the same
objective.

### CRISPR Result

CRISPR is the clean genetic setting but the harder visual setting. Single-gene
effects are subtle relative to hepatocyte heterogeneity, so the result should be
reported as distribution-level recovery, not as a single-cell before/after
montage.

Paper-line one-hot CRISPR run:

```text
outputs/runs/crispr/paper_core
panel: Calreticulin / Perilipin / pS6RP
condition: target-gene one-hot
gene set: 40 main quantitative Perturb-Multi paper-program genes
reported labels: original Perturb-Multi programs
```

The 40-gene CRISPR paper-core panel is:

| program | genes |
|---|---|
| steatosis_lipid | Insig1, Pten, Eif2s1, Aars |
| upr_er_stress | Sel1l, Sec61a1, Dnajb9, Atp2a2, Xbp1, Ern1 |
| isr_translation | Nars, Eif2b4 |
| mtor_ps6 | Mtor, Cdc37, Tsc1 |
| lysosome_endomembrane | Npc1, Atp6v0c, Atp6ap1, Lamtor2, Dnm2, Arfrp1, Jtb, Zw10 |
| zonation_wnt_hypoxia | Ctnnb1, Apc, Vhl, Lgr4, Prkar1a, Hs6st1, B4galt7, Zfp830 |
| rna_processing_nuclear | Sf3b6, Sbno1, Polr1a, Kin, Polr2l, Gpn1, Rrn3, Taf1a, Ubtf |

The useful claim is gene-conditioned recovery of original-paper marker programs:
UPR/ER stress through Calreticulin, steatosis through Perilipin, and mTOR
signaling through pS6RP.

## Metric Hierarchy

For Perturb-Multi, the paper should report all CellFlux-style metrics but should
not let FID alone define biological success.

| metric | role in this project |
|---|---|
| FID/KID | comparability and image-realism metrics; useful for detecting gross synthetic artifacts, but not a reliable biological success criterion here |
| MoA / Program-Acc | auxiliary condition-separability metric; for CRISPR paper core, report original-paper program labels rather than raw 76/77 gene identity as the main score |
| marker distribution gap closure | primary biological-fidelity metric for Diet/Perturb-Multi marker transport |
| direction recovery / sign agreement | primary CRISPR biology metric across genes and marker channels |
| qualitative images | explanation and sanity check, not primary evidence |

Why other CellFlux-style papers can use FID more centrally: in conventional
cell-image benchmarks, the visual feature distribution and the biological
phenotype often align better, and a source-control reference does not necessarily win.
Here, the RGB image is a false-color subset of marker channels. A method can be
very realistic as a same-batch control image and still fail to move the relevant
marker distribution.

## Algorithm Modules

### 1. Matched Control-Initialized Transport

Use same-state controls where possible so the model starts from the local
hepatocyte distribution rather than from pure noise. This keeps the task aligned
with CellFlux: control image + target condition -> generated perturbed image.

For Diet, batch is confounded with diet state and BATCH is collapsed to enable
adlib controls for fasted/HFD. This must stay disclosed. For CRISPR, same-batch
and same-cluster control matching remains the cleaner design.

### 2. Conditional Flow Matching Backbone

The engine is CellFlux-style conditional flow matching:

```text
x_t = (1 - t) * source + t * target
loss = MSE(v_theta(x_t, t, condition) - (target - source))
sample = ODE-integrate from control init with EMA and CFG
```

This is a distribution-to-distribution model. The target image in training is an
unpaired treated cell, so the model should be evaluated by population movement,
not by pixel-level matching to a particular target cell.

### 3. Perturbation Condition

Clean active conditions:

- Diet: 3-dim one-hot (`adlib`, `fasted`, `hfd`).
- CRISPR: target-gene one-hot over the perturbation vocabulary.

Do not use per-cell MERFISH RNA readouts as generator inputs for the main claim.
Those RNAs are measured downstream phenotypes from the same imaged cells, not
independent perturbation conditions. They are valid for interpretation and
evaluation, but using them as inputs would leak phenotype information into the
generator.

### 4. Marker Distribution Evaluation

For each generated PNG, compute foreground mean intensity per active marker.
Compare:

```text
generated distribution vs real target distribution
control distribution vs real target distribution
gap_closed = 1 - W(gen, target) / W(control, target)
```

This directly measures the story: did the conditional generator move the marker
population toward the treated marker population?

### 5. Figure Strategy

Lead with figures that match the data:

- Distribution plots for control / generated / target marker intensities.
- Mean-shift bars or density overlays for Diet HFD and fasted.
- Mean images as supporting visual context, with the caveat that spatial
  morphology is not yet the main win.
- For CRISPR, gene/channel heatmaps and direction-recovery plots rather than
  single-cell before/after claims.

Single generated images can be shown, especially if they look good, but the
caption should say what they are: false-color marker-channel renderings from a
multiplexed molecular phenotype panel. The quantitative claim should come from
distribution migration.

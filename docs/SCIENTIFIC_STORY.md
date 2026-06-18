# Scientific Story and Algorithm Plan

## Core Question

Can we learn a perturbation-conditioned virtual hepatocyte that predicts how
genetic perturbations reshape subcellular morphology in intact liver tissue?

The biological emphasis is not generic image synthesis. The Perturb-Multimodal
paper shows that hepatocyte morphology is tied to zonation, metabolic state, and
lipid-droplet biology. In this dataset, the 18-channel morphology panel includes
interpretable markers for liver function, lipid droplets, ER stress, autophagy,
mitochondria, lysosomes, membrane trafficking, and RNA/protein abundance.

The primary scientific question is:

```text
Given a same-batch/same-state control hepatocyte and a CRISPR target gene,
which subcellular morphology programs are predicted to shift, and are those
shifts consistent with known lipid metabolism, ER/secretory, and organelle
stress biology?
```

## Data Contract

Each training pair is an unpaired distribution-level sample:

```text
source:    control sgRNA hepatocyte image, shape [18, 128, 128]
target:    target-gene CRISPR hepatocyte image, shape [18, 128, 128]
condition: target_gene / condition_id
matching:  same split, same batch, same cluster_type
```

The pair is not a real single-cell trajectory. It approximates a transition
from the local control distribution to the perturbation distribution. Source and
target are unpaired (different cells), so the task is **distribution-to-distribution**,
not per-cell prediction.

## Current Approach & Status

Training runs through the **CellFlux** engine, now absorbed into this repo as
`morphoflux.engine` (no longer the original in-house pixel-space flow). The formulation
is conditional flow matching, control→perturbed:

```text
x_t = (1 - t) * source + t * target ;  t ~ Uniform(0,1)
loss = MSE( v_theta(x_t, t, condition) - (target - source) )
sample: ODE-integrate a control (or noise) start through v_theta, with EMA weights + CFG.
```

Key empirical finding: in-vivo CRISPR effects are **subtle** (only 84/406 sgRNAs are
morphologically significant; |z|≈0.5–1) and swamped by hepatocyte heterogeneity, so
naive control→target flow can score well while failing to move in the correct
biological direction. The active recipe is **gene-identity conditioning + matched
control-init + classifier-free guidance**, evaluated in **aggregate** (per-gene
per-channel direction), not per cell.

- Live experiment log + current results: **`docs/EXPERIMENTS.md`**.
- Old experiments that used measured MERFISH RNA readouts as conditioning signal were
  removed from the active workspace. MERFISH RNA is a phenotype/readout, not the
  perturbation condition.

## Algorithm Modules (durable)

### 1. Matched Morphology Transport

Purpose: isolate perturbation effects from animal/batch/state variation.

- Use controls from the same `batch` and `cluster_type`; keep train/val/test split
  separation; drop under-supported strata.
- Prevents the model from explaining zonation or acquisition artifacts as gene effects.

### 2. Conditional Flow Matching Backbone

- Conditional UNet velocity field; time embedding; classifier-free guidance
  (condition dropout); EMA weights for sampling; skewed (EDM) timesteps.
- Provides a generative model of perturbation response rather than a scalar score.

### 3. Perturbation Condition Embedding

The current clean condition is **target gene identity / sgRNA identity**:
`embedding_gene_identity.csv`, a 204-dimensional one-hot embedding for the perturbation.

Do not use the 209-gene MERFISH panel as the generator condition. In this dataset,
MERFISH RNA is a measured downstream phenotype from the imaged cell, like morphology.
It can be used for evaluation or biological interpretation, but not as the clean input
condition for a target-gene-to-morphology generator.

Candidate richer condition:

- Genome-wide Perturb-seq pseudobulk per perturbation, because it is an independent
  per-perturbation transcriptional signature from adjacent sections/different cells.
- Curated pathway or gene-set embeddings derived independently from the imaging
  readout.

These upgrades should test whether the model organizes perturbations by pathway
biology without leaking the measured imaging-panel RNA readout into the generator.

### 4. Channel-Wise Response Quantification

Turn generated images into interpretable biology (`scripts/aggregate_eval.py`):

- Per-channel intensity shift `mean(generated - source)` and its direction vs the real
  `target - control` shift (per-gene correlation + sign agreement).
- Perilipin puncta/level for lipid droplets; Calreticulin/M6PR/CathB for ER-Golgi-lysosome;
  TOMM20/TOM70/mtRNA for mitochondria; LC3b/Sqstm1/Rab7 for autophagy/endolysosome.

### 5. Pathway-Level Perturbation Analysis

Tell the story at the biology level. Priority gene sets:

- Lipid metabolism: `Insig1`, `Pten`, `Eif2s1`, `Aars`, `Cd36`, `Cpt1a`, `Dgat1`,
  `Fasn`, `Scd1`, `Srebf1`, `Ldlr`, `Apoa1`, `Apoc3`.
- ER/secretory UPR: `Sel1l`, `Atp2a2`, `Hspa5`, `Xbp1`, `Dnajb9`.
- Lysosome/autophagy: `Npc1`, `Atp6ap1`, `Atp6v0c`, `Pikfyve` (CathB/LC3b/Rab7 channels).
- mTOR: `Pten`, `Tsc1`, `Tsc2`, `Mtor`, `Cdc37` (pS6RP channel).

## Figure Strategy for 18 Channels

Full 18-channel images are not human-readable as a single RGB image. Use three layers.

### Figure 1: Model Overview

Show the data contract `same-batch control + target gene -> generated perturbed cell`
with three panels: source control crop, generated crop, real target crop (or target
distribution summary).

### Figure 2: Interpretable Channel Composites

Biology-focused RGB composites instead of arbitrary first-three channels, shown as
`source | generated | target`:

- Lipid/function: R Perilipin, G Alb, B polyT/rRNA
- ER/secretory: R Calreticulin, G M6PR, B Gapdh/polyT
- Mito/autophagy: R TOMM20/TOM70, G LC3b, B Rab7/Sqstm1

### Figure 3: Channel-Wise Delta Heatmap

For each gene/pathway compute `delta_channel = mean(generated - source)`; plot genes
(rows) × 18 channels (cols); highlight lipid genes and the Perilipin channel.

### Figure 4: Pathway Case Studies

Lead with the strongest, cleanest hits where the effect is demonstrable:

- Lipid/steatosis → Perilipin: `Insig1`, `Pten`, `Eif2s1`, `Aars`.
- UPR → Calreticulin: `Sel1l`, `Atp2a2`.
- Lysosome → CathB: `Npc1`.
- mTOR → pS6RP: `Pten`, `Tsc1`, `Tsc2`.

Primary visual endpoint: does generated morphology move in the expected direction for
each perturbation while preserving same-batch hepatocyte state?

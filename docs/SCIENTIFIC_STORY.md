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
from the local control distribution to the perturbation distribution.

## Current Algorithm

The current baseline follows the CellFlux formulation:

```text
x_t = (1 - t) * source + t * target
t ~ Uniform(0, 1)
v_target = target - source
loss = MSE(v_theta(x_t, t, condition), v_target)
```

At inference, a control cell is moved through the learned velocity field with
Euler integration:

```text
x_{t + dt} = x_t + dt * v_theta(x_t, t, condition)
```

## Proposed Modules

### 1. Matched Morphology Transport

Purpose: isolate perturbation effects from animal/batch/state variation.

Implementation:

- Use controls from the same `batch` and `cluster_type`.
- Keep train/val/test split separation.
- Track `control_pool_size` and drop under-supported strata.

Scientific role:

- Prevents the model from explaining zonation or acquisition artifacts as gene
  effects.

### 2. Target-Scaffolded Morphology Generation

Purpose: generate perturbation-like intracellular morphology on a specified
cell boundary.

The initial source-to-target CellFlux baseline keeps generated cells too close
to the source morphology, because the source and target are unpaired cells with
different masks. Pixel-level training against an unpaired target encourages a
mixture of source preservation and average velocity, and it does not guarantee
that the generated crop respects the target cell boundary.

The target-scaffolded variant derives a binary cell mask from the target image
and uses it as an additional input channel:

```text
input:  x_t, t, target_gene, target_mask
output: 18-channel morphology inside target_mask
```

During sampling, each integration step is clamped to the target mask:

```text
x_{t + dt} = target_mask * (x_t + dt * v_theta([x_t, target_mask], t, condition))
```

Scientific role:

- Separates cell boundary/segmentation geometry from intracellular marker
  distribution.
- Enables direct visual comparison of generated and observed perturbation
  morphology on the same target-cell scaffold.
- Matches the practical visualization goal for lipid droplets, ER structure,
  mitochondria, and RNA/protein localization.

Limitation:

- This is not a fully de novo prediction when no target boundary is available.
  For a pure virtual perturbation model, the boundary itself must either be
  generated or sampled from a matched target-cell shape distribution.

### 3. Conditional Flow Matching Backbone

Purpose: learn a continuous transformation from control morphology to
perturbed morphology.

Implementation:

- Conditional UNet velocity field.
- Time embedding for `t`.
- Learnable gene condition embedding.
- Condition dropout for classifier-free guidance.
- Source noise augmentation to smooth empirical sample transitions.

Scientific role:

- Provides a generative model of perturbation response rather than a scalar
  phenotype score.

### 4. Biology-Aware Condition Embedding

Purpose: move beyond integer gene IDs.

Next implementation:

- Add curated pathway embeddings from `pert_pathways.csv`.
- Add optional gene sequence / Gene2Vec embeddings later.
- Include pathway heads for lipid metabolism, ER/secretory UPR,
  lysosome/vacuolar, autophagy, mitochondria, and translation.

Scientific role:

- Tests whether the model organizes unseen or weakly sampled genes by pathway
  biology, not only by memorized condition IDs.

### 5. Channel-Wise Response Quantification

Purpose: turn generated images into interpretable biology.

Metrics:

- Per-channel intensity shift: `mean(generated - source)`.
- Per-channel target agreement: MSE / correlation against target distribution.
- Perilipin puncta score for lipid droplets.
- Calreticulin / M6PR / CathB shifts for ER-Golgi-lysosome phenotypes.
- TOMM20/TOM70/mtRNA shifts for mitochondrial state.
- LC3b/Sqstm1/Rab7 shifts for autophagy and endolysosomal state.

Scientific role:

- Converts generated morphology into testable perturbation signatures.

### 6. Pathway-Level Perturbation Analysis

Purpose: tell the story at the biology level rather than single-cell examples.

Priority gene sets:

- Lipid metabolism: `Apoa1`, `Apoa2`, `Apoc1`, `Apoc3`, `Apoe`, `Cd36`,
  `Cpt1a`, `Dgat1`, `Fabp1`, `Fasn`, `Insig1`, `Insig2`, `Ldlr`, `Pcsk9`,
  `Plin2`, `Rxra`, `Scd1`, `Srebf1`, `Srebf2`.
- ER/secretory UPR: `Atp2a2`, `Ckap4`, `Hspa5`, `Xbp1`, `Sec61a1`,
  `Sec23a`, `Rrbp1`.
- Lysosome/vacuolar and autophagy: `Atp6ap1`, `Atp6v0c`, `Npc1`, `Pikfyve`,
  `Sqstm1`-related channels, `LC3b`, `Rab7`.

Scientific role:

- Links predicted morphology shifts to known hepatocyte pathways.

## Figure Strategy for 18 Channels

Full 18-channel images are not human-readable as a single RGB image. Use three
visual layers:

### Figure 1: Model Overview

Show the data contract:

```text
same-batch/state control cell + target gene -> generated perturbed cell
```

Use three panels:

- Source control crop.
- Generated crop.
- Real target crop or target distribution summary.

### Figure 2: Interpretable Channel Composites

Use biology-focused RGB composites instead of arbitrary first-three channels:

- Lipid/function view:
  - Red: Perilipin
  - Green: Alb
  - Blue: polyT or rRNA
- ER/secretory view:
  - Red: Calreticulin
  - Green: M6PR
  - Blue: Gapdh or polyT
- Mito/autophagy view:
  - Red: TOMM20 or TOM70
  - Green: LC3b
  - Blue: Rab7 or Sqstm1

These composites should be shown as:

```text
source | generated | target
```

### Figure 3: Channel-Wise Delta Heatmap

For each target gene or pathway, compute:

```text
delta_channel = mean(generated_channel - source_channel)
```

Plot genes by rows and the 18 channels by columns. Highlight lipid genes and
the Perilipin channel.

### Figure 4: Lipid-Droplet Case Study

Focus on lipid metabolism perturbations:

- `Plin2`: direct lipid-droplet biology.
- `Cd36`: fatty acid uptake.
- `Cpt1a`: fatty acid oxidation.
- `Fasn`, `Scd1`, `Srebf1`: lipogenesis.
- `Apoa1`, `Apoe`, `Pcsk9`, `Ldlr`: lipid transport/clearance.

Primary visual endpoint:

```text
Does generated Perilipin morphology move in the expected direction for lipid
handling perturbations, while preserving same-batch hepatocyte state?
```

## Practical Next Steps

1. Finish a long CellFlux run and export JPG previews.
2. Add channel-aware composite export with named presets.
3. Add a pathway evaluation script that aggregates generated/source/target
   channel deltas by gene and pathway.
4. Add puncta-level Perilipin morphology metrics for lipid-droplet analysis.
5. Add stronger gene/pathway embeddings after the baseline run is stable.

## If Generation Quality Is Poor

Do not blindly increase hidden dimension first. Diagnose the failure mode and
adjust the corresponding module.

### Failure Mode: Generated Images Are Too Smooth

Likely cause:

- MSE / velocity MSE averages over stochastic unpaired target morphology.

Priority fixes:

- Keep foreground-weighted velocity loss so the model spends capacity inside
  the target cell mask.
- Add DoG high-pass loss on biology-relevant channels to penalize smooth
  template-like interiors.
- Add puncta spatial loss on Perilipin/autophagy/mitochondria channels so peak
  locations, top puncta mass, and coarse spatial moments match the target.
- Increase sampling steps from 16 to 32 or 64.

### Failure Mode: Generated Images Still Look Like Source

Likely cause:

- Source image conditioning is too strong.

Priority fixes:

- Use `source_mean` or source channel statistics instead of raw source texture.
- Use a source morphology embedding instead of a full source image.
- Use matched control distribution summaries for the same batch/cluster.
- Keep target scaffold mask as geometry input.

### Failure Mode: Generated Images Do Not Match Perturbation Biology

Likely cause:

- Gene ID embedding is too weak.
- The model lacks multimodal biological context.

Priority fixes:

- Add curated pathway embeddings from `pert_pathways.csv`.
- Compute per-gene RNA delta signatures from the RNA AnnData and use an MLP
  condition encoder.
- Compute per-gene protein delta signatures and use them as condition or
  auxiliary supervision.
- Keep gene ID embedding as a residual condition, not the only condition.

### Failure Mode: Model Capacity Is Too Low

Only after objective and conditioning are reasonable, scale the backbone:

- Increase hidden channels from 96 to 128.
- Use two ResBlocks per scale instead of one.
- Add bottleneck attention and low-resolution attention at 16x16 or 8x8.
- Add EMA weights for sampling.
- Use mixed precision and gradient checkpointing if memory becomes limiting.

### Current Priority Order

```text
1. Objective: foreground + DoG high-pass + puncta spatial losses
2. Conditioning: pathway + RNA delta + protein delta embeddings
3. Backbone: hidden 128, more ResBlocks, attention, EMA
4. Display/evaluation: matched control-target pair selection and pathway heatmaps
```

The current mainline method is:

```text
target scaffold + source_mean initialization + perturbation-conditioned flow
```

This removes most source texture copying while preserving control-state channel
baseline information.

## Experiment Log

### DDP A3 Puncta Baseline

Run:

```text
outputs/cellflux_scaffold_mean_a3_puncta_ddp_2k
```

Configuration prior to the residual-head upgrade:

```text
target scaffold + source_mean initialization
global batch size 128, DDP world size 2, per-GPU batch size 64
foreground_weight 1.0
image_weight 0.5
local variance 1.0
gradient 0.1
raw Perilipin top-k 1.0
```

Training completed:

```text
final step: 2000
best val_loss: 0.019469885 at step 1900
final val_loss: 0.019701252 at step 2000
preview: outputs/cellflux_scaffold_mean_a3_puncta_ddp_2k/jpg_previews_step2000
```

Visual readout:

```text
Target scaffold alignment worked: generated contours follow the target mask.
The interior remained too smooth: Perilipin/Alb/polyT composites still lacked
the target's puncta, local edge heterogeneity, and lipid-droplet-like signal.
```

Decision:

```text
Do not keep tuning the old local-variance/raw-top-k objective. Replace it with
a DoG high-pass and puncta spatial objective that directly targets local
contrast, target peak locations, top puncta mass, and spatial moments.
```

### Puncta DoG Objective

Configuration prior to the residual-head upgrade:

```text
configs/train_cellflux_scaffold_mean_puncta.yaml at the DoG-only revision
```

Run target:

```text
outputs/cellflux_scaffold_mean_puncta_dog_ddp_2k
```

Loss:

```text
foreground_weight: 1.0
image_weight: 0.35
highpass_weight: 1.5
highpass_channels: Alb, Perilipin, LC3b, TOMM20, Rab7
puncta_weight: 1.0
puncta_channels: Perilipin, LC3b, TOMM20, Rab7
```

Rationale:

```text
Alb and Perilipin anchor hepatocyte functional/lipid phenotypes.
LC3b, TOMM20, and Rab7 add autophagy/mitochondrial/endolysosomal structure so
the generated cell is not only lipid-correct but organelle-distribution-aware.
```

Training completed:

```text
final step: 2000
best val_loss: 0.016401427
preview: outputs/cellflux_scaffold_mean_puncta_dog_ddp_2k/jpg_previews_step2000
```

Visual readout:

```text
Scalar validation improved over the DDP A3 baseline, but generated interiors
remained too smooth. The model improved scaffold alignment and some edge/color
variation but still did not synthesize target-like Perilipin/Alb puncta.
```

Decision:

```text
Keep the DoG and puncta distribution terms, but add a dedicated high-frequency
residual velocity head. The main head should carry low-frequency flow; the
residual head is directly supervised to predict target-start DoG velocity.
```

### Active Residual Puncta Objective

Configuration:

```text
configs/train_cellflux_scaffold_mean_puncta.yaml
```

Run target:

```text
outputs/cellflux_scaffold_mean_puncta_residual_ddp_2k
```

Model:

```text
high_frequency_residual: true
output channels: 36 = 18 base velocity + 18 residual velocity
sampling velocity: base + residual
```

Additional loss:

```text
residual_weight: 1.0
residual_channels: Alb, Perilipin, LC3b, TOMM20, Rab7
```

Rationale:

```text
The previous deterministic single-head model minimized risk by predicting a
smooth condition mean. A residual head gives high-frequency morphology its own
capacity and direct supervision without forcing the base velocity head to carry
both cell-scale structure and puncta-scale detail.
```

## Diagnostic Findings

Preview diagnostics:

```text
outputs/diagnostics/preview_diagnostics.md
outputs/diagnostics/preview_diagnostics.json
```

Runs compared:

```text
A3 DDP puncta baseline, step 1000/2000
DoG + puncta objective, step 1000/2000
Residual-head puncta objective, step 1000/2000
256-pair overfit diagnostic, step 300
```

Key findings:

```text
1. Full-run generated images move closer to target than source in low-frequency
   MSE, so the conditional flow is not completely failing.
2. High-frequency and puncta metrics do not improve monotonically from 1000 to
   2000 steps. In the residual-head run, DoG energy, top puncta mass, and std
   ratios all dropped by step 2000.
3. The Perilipin channel is the main failure mode. DoG/puncta/residual runs keep
   generated Perilipin top-mass far below target, around 0.32-0.56 of target in
   the inspected previews.
4. A 256-pair overfit diagnostic, trained for roughly 150 small-data epochs,
   still did not memorize target-like Perilipin/Alb interiors. Train loss
   reached only 0.019 and generated previews remained smooth.
```

Interpretation:

```text
This is not simply a long-training problem. Even when the model sees only 256
pairs repeatedly, the deterministic scaffold + source_mean + condition input
does not recover target puncta. The current objective teaches low-frequency
target-conditioned averages better than stochastic organelle/lipid-droplet
textures.
```

Decision:

```text
Stop adding more pixelwise high-frequency penalties as the primary strategy.
The next model change should be distributional or stochastic: a patch-level
texture discriminator/loss, a stochastic latent high-frequency generator, or a
two-stage low-frequency flow plus texture generator.
```

## Active Lipid-Function Panel

Rationale:

```text
CellFlux's published datasets use 3, 5, or 6 channels, whereas this project was
initially trained jointly on 18 channels. The 18-channel objective may dilute
the main lipid/function phenotype and let low-frequency channels dominate the
optimization. To isolate the core biological question, the active experiment
uses a focused Perilipin/Alb/polyT panel.
```

Configuration:

```text
configs/train_cellflux_lipid_panel.yaml
```

Original-to-panel channel mapping:

```text
original channel 5 Perilipin -> panel channel 0
original channel 0 Alb       -> panel channel 1
original channel 1 polyT     -> panel channel 2
```

Run target:

```text
outputs/cellflux_lipid_panel_scaffold_ddp_2k
```

Question:

```text
Can a scaffolded conditional flow generate target-like lipid/function
distribution when the task is restricted to Perilipin, Alb, and polyT?
If this still fails, the main blocker is not 18-channel dilution; it is the
deterministic/pixelwise generation mechanism. If this improves, the paper can
lead with a focused lipid-function story and treat 18-channel generation as an
extension.
```

Result:

```text
Completed 2026-06-17
global batch size 128, DDP world size 2, per-GPU batch size 64
final step: 2000
best val_loss: 0.0114848018 at step 2000
previews:
  outputs/cellflux_lipid_panel_scaffold_ddp_2k/jpg_previews_step0001000
  outputs/cellflux_lipid_panel_scaffold_ddp_2k/jpg_previews_step0002000
diagnostics:
  outputs/cellflux_lipid_panel_scaffold_ddp_2k/lipid_panel_diagnostics.md
```

Visual readout:

```text
Restricting to Perilipin/Alb/polyT did not solve the core failure. Generated
cells follow the target scaffold, but the interior remains a smooth
low-frequency mixture instead of target-like local Perilipin puncta and
heterogeneous Alb/polyT structure.
```

Metric readout:

```text
step 1000:
  gen-target MSE 0.03080, gen-source MSE 0.04266, target progress 0.466
  Perilipin top-mass ratio generated/target 0.562
step 2000:
  gen-target MSE 0.04628, gen-source MSE 0.04361, target progress 0.225
  Perilipin top-mass ratio generated/target 0.592
```

Decision:

```text
The 18-channel objective may still dilute some phenotypes, but it is not the
main blocker. Even a 3-channel lipid/function panel keeps the deterministic
scaffolded flow in a smooth-average regime. The next model change should add a
distributional or stochastic texture mechanism rather than more pixelwise
high-frequency penalties.
```

## Hardware Target For Next Runs

The completed lipid-panel run used global batch size 128 and was observed around
25 GB/GPU. Future formal DDP experiments should target roughly 30 GB/GPU by
using global batch size 160, i.e. per-GPU batch size 80 on two GPUs. This is an
efficiency target only; it should improve throughput, but it is not expected to
fix the smooth-interior phenotype by itself.

Updated defaults:

```text
configs/train_cellflux_lipid_panel.yaml
configs/train_cellflux_scaffold_mean_puncta.yaml
Makefile train-lipid-panel-ddp
Makefile train-puncta-ddp
```

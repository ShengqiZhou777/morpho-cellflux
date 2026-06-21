# Evaluation Protocol

**Single source of truth for how methods are compared on Perturb-Multi.**

We still report the CellFlux-style metric suite so the work is comparable to
CellFlux and its baselines. But Perturb-Multi is a multiplexed molecular marker
readout, not ordinary RGB morphology. Therefore FID/KID are image-realism and
comparability metrics, while marker-distribution transport is the primary
biological-fidelity metric.

Provenance:

- CellFlux (Zhang et al., arXiv:2502.09775): FIDo/FIDc/KIDo/KIDc, MoA, and
  sample-size sensitivity.
- Perturb-Multi paper: 18-target marker panel of protein markers for
  subcellular structures/signaling pathways plus abundant RNAs.

## 1. Metric Suite

Report all metrics, but keep their roles separate.

| Family | Metric | Definition | Direction | Role here |
|---|---|---|---|---|
| Image quality / comparability | FIDo | overall FID: all generated vs all real perturbed, pooled | lower better | CellFlux-style image realism |
| Image quality / comparability | FIDc | conditional FID: per perturbation class, then averaged | lower better | class-balanced image realism |
| Image quality / comparability | KIDo | overall KID | lower better | FID robustness check |
| Image quality / comparability | KIDc | conditional KID | lower better | class-balanced KID |
| Condition separability | MoA Acc / Macro-F1 / Weighted-F1 | classifier trained on real perturbed images, evaluated on generated images | higher better | auxiliary biological proxy if the real-image ceiling is above chance |
| Marker phenotype | gap_closed | `1 - W(gen,tgt) / W(ctrl,tgt)` on per-cell marker foreground means | higher better | primary Diet biological metric |
| Marker phenotype | dir_corr / sign_agree | recovery of `(gen-ctrl)` vs `(real-ctrl)` perturbation direction across genes | higher better | primary CRISPR biological metric |

FID/KID use Inception features on rendered 3-channel PNGs. Those PNGs are
false-color marker panels, so Inception realism is not guaranteed to align with
marker biology. MoA is more condition-aware but still depends on a classifier
ceiling. Marker metrics directly measure the data's biological readout.

## 2. Why FID Is Not The Primary Biological Metric Here

FID is useful, but it answers a narrower question:

```text
Do generated images look like the real image distribution in Inception feature space?
```

The Perturb-Multi biological question is different:

```text
Did the generated marker population move from control toward the treated marker population?
```

The Diet 5K table proves the distinction. Copy-control has the best FID/KID even
though it applies no perturbation:

| method | FIDo | FIDc | KIDo | KIDc | MoA-Acc |
|---|---:|---:|---:|---:|---:|
| copy_control | **7.96** | **12.01** | **0.0039** | **0.0057** | 49.92 |
| PhenDiff | 10.92 | 13.97 | 0.0066 | 0.0075 | 60.69 |
| IMPA | 52.29 | 55.43 | 0.0407 | 0.0424 | **63.97** |
| Morpho-CellFlux | 31.26 | 35.43 | 0.0267 | 0.0291 | 54.93 |

Conclusion: FID/KID remain required for external comparison, but cannot be used
alone to claim biological success on this dataset. If a no-perturbation
copy-control wins FID, then FID is rewarding same-batch image realism more than
perturbation transport.

Why other papers can lean harder on FID: in many cell-image benchmarks, the
visual feature distribution and the biological condition are more tightly
coupled, and the null copy-control does not trivially win. Here the RGB image is
a selected false-color rendering of marker channels; the marker shift is the
biological endpoint.

## 3. Sample Budget N

Matched sample size is mandatory because CellFlux Table 5 shows FID/KID are
sample-size sensitive.

| benchmark | rule |
|---|---|
| Diet | generate 5120 images when possible; for method comparison, cap every method to the same per-condition minimum |
| CRISPR | use a CellFlux/RxRx1-like larger budget when compute allows; if per-gene counts are low, disclose the cap and avoid overinterpreting FIDc |
| MoA | cap generated-image evaluation at 5120, following CellFlux's MoA script convention |

For the current Diet 5K table, the proposed run produced 5120 images split as
fasted 2654 / hfd 2466. Therefore the fair table uses cap 2466 per condition
for all methods, `N = 4932`.

Absolute FID values are not comparable to published CellFlux BBBC021/RxRx1
numbers because the dataset and channels differ. They are only comparable across
methods inside the same table.

## 4. Dataset Mapping

| | Diet | CRISPR |
|---|---|---|
| Perturbation | physiological state | target gene |
| Control | adlib | non-targeting/control sgRNA |
| Treated classes | fasted, hfd | 76 rna_snr-filtered genes |
| Active panel | `[9,5,8]` Calreticulin / Perilipin / TOMM20 | `[0,14,5]` Alb / Rab7 / Perilipin |
| Main biological metric | per-condition marker gap closure | direction recovery and pooled marker gap closure |
| Current proposed run | `outputs/runs/diet/fid5k` | `outputs/runs/crispr/main` |

Diet is the strong physiological demonstration. CRISPR is the clean gene-identity
benchmark but has weaker effects and should be reported distributionally.

## 5. MoA Ceiling Rule

MoA is only meaningful if a classifier can distinguish the real conditions.
Always report the real-image classifier ceiling before interpreting generated
MoA.

| dataset | interpretation |
|---|---|
| Diet | usable: real treated images are separable above chance |
| CRISPR | report only if real-image ceiling is substantially above 1/76 chance; otherwise say MoA is not informative for CRISPR |

MoA is still not a perfect biological metric. A method can increase condition
classifier accuracy by producing class-specific artifacts. Read it together with
FID/KID and marker metrics.

## 6. Marker Distribution Protocol

For every generated PNG:

1. Read the active marker panel from the run's `args.json` (`channels`).
2. Compute foreground mean intensity per marker.
3. Read the corresponding real treated cell from the raw npz crop.
4. If `trt2ctrl_idx.json` has a mapping, read the paired control cell.
5. For each Diet condition or CRISPR channel/gene group, compute:

```text
W_gen = Wasserstein(generated_marker_means, target_marker_means)
W_ctrl = Wasserstein(control_marker_means, target_marker_means)
gap_closed = 1 - W_gen / W_ctrl
```

Interpretation:

| value | meaning |
|---:|---|
| 1 | generated distribution matches the real treated distribution |
| 0 | no better than copying the control |
| <0 | worse than copy-control |

For CRISPR, also compute `dir_corr` and `sign_agree` across genes:

```text
generated shift = generated_gene_mean - control_gene_mean
real shift = real_treated_gene_mean - control_gene_mean
```

## 7. Current Diet 5K Marker Evidence

Script:

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

Generated-vs-target means:

| condition | marker | generated | target | read |
|---|---|---:|---:|---|
| fasted | Calreticulin | 0.3715 | 0.3528 | slight overshoot |
| fasted | Perilipin | 0.3208 | 0.3123 | slight overshoot |
| fasted | TOMM20 | 0.4115 | 0.4122 | close |
| hfd | Calreticulin | 0.4213 | 0.4173 | close |
| hfd | Perilipin | 0.3514 | 0.3555 | close |
| hfd | TOMM20 | 0.4235 | 0.4423 | under-shift |

This is the strongest current biological evidence for the proposed model:
marker distributions move toward the Diet target, especially HFD
Calreticulin/Perilipin. It is not evidence that the model is best by FID/MoA or
that spatial morphology is fully solved.

## 8. Generation Metadata And DDP Mapping

Evaluation PNGs are saved in:

```text
<run>/fid_samples/epoch-<epoch>/<condition>/<target_id>.png
```

`trt2ctrl_idx.json` maps `target_id -> control_id` and is required for paired
control gap closure. A DDP bug in older evals let each rank overwrite this JSON,
so a 2-GPU 5120-image eval could have only 2560 mappings. The eval loop now
gathers mappings across ranks before the main process writes the JSON.

For outputs produced before this fix:

- gen-vs-target distribution metrics remain usable for all generated images;
- paired control/gap_closed metrics must disclose the mapped-row count;
- rerun eval after the DDP fix for fully rigorous paired gap closure.

## 9. Implementation Status

- [x] Copy-control diet + CRISPR exports.
- [x] PhenDiff diet + CRISPR exports.
- [x] IMPA diet export; CRISPR IMPA was previously blocked by corrupted FID cache and needs rerun.
- [x] FIDo/FIDc/KIDo/KIDc matched-N tooling.
- [x] MoA classifier fixed to classify perturbation class (`mols`), not treated/control annotation.
- [x] Diet 5K proposed eval and matched table.
- [x] Reproducible Diet marker-distribution figure script.
- [x] DDP mapping fix for future evals.
- [ ] Rerun Diet 5K after DDP mapping fix for complete paired control gap closure.
- [ ] Run CRISPR IMPA and CRISPR real-image MoA ceiling before reporting CRISPR MoA.

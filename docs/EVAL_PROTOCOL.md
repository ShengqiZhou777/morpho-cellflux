# Evaluation Protocol — CellFlux-faithful

**Single source of truth for how every method is evaluated and compared in this paper.**
We follow CellFlux (Zhang et al., arXiv:2502.09775) — the method line this work builds on —
and reproduce baselines (PhenDiff, IMPA) under the *identical* protocol. The rule that makes
the comparison valid: **same eval set, same sample budget N, same metric definitions across
every method** (proposed + PhenDiff + IMPA + copy-control).

Provenance: CellFlux §4.2 (metrics), Table 1 (FIDo/FIDc/KIDo/KIDc), Table 2a (MoA),
Table 5 (sample-size sensitivity). PhenDiff (arXiv:2312.08290) reports FID/KID/Precision/Recall.

---

## 1. Metric suite (report all; mirror CellFlux Table 1 + 2a)

| Family | Metric | Definition | Direction |
|---|---|---|---|
| Image quality / distribution | **FIDo** | overall FID: all generated vs all real-perturbed, pooled | lower better |
| | **FIDc** | conditional FID: FID per perturbation class, then averaged across classes | lower better |
| | **KIDo** | overall KID (kernel Inception distance) | lower better |
| | **KIDc** | conditional KID, per-class averaged | lower better |
| Biological fidelity | **MoA Acc / Macro-F1 / Weighted-F1** | classifier trained on REAL perturbed images, evaluated on GENERATED images | higher better |
| Biological (ours, supplementary) | **gap_closed / dir_corr / sign_agree** | per-cell channel-mean Wasserstein gap closure + perturbation-direction recovery | higher better |

FID/KID use the torchmetrics InceptionV3 (2048-dim) features — the same backbone as the
MoA classifier. `gap_closed`/`dir_corr` are this project's existing biologically-motivated
metrics; CellFlux Table 7 (CellProfiler nuclear-size features) is the precedent for reporting
a direct morphological-feature comparison alongside FID/MoA. We keep them as **supplementary**,
not as a replacement for the CellFlux suite (see §5 on selection).

## 2. Sample budget N (the comparability rule)

CellFlux Table 5 explicitly shows FID/KID are sample-size sensitive and reports every method at
matched sizes (BBBC021 5K; JUMP 10–20K). Therefore:

- **N = 5,000 generated images** for the overall metrics (FIDo/KIDo), matched with 5,000 real
  perturbed images. Follows CellFlux's BBBC021 budget.
- **FIDc/KIDc**: computed per condition from the same pool (diet → ~2.5K/condition across 2
  treated states; CRISPR → 5K/76 ≈ 66/gene, the RxRx1 regime CellFlux flagged as low-signal).
- **MoA eval**: capped at **5,120** generated images (CellFlux's cap, `train_moa.py`).
- **Identical N for proposed + PhenDiff + IMPA + copy-control.** Methods that generated more
  (PhenDiff exports one image per treated test cell) are **subsampled** to N at metric time —
  the surplus is wasted compute, not a protocol change. The proposed method's existing 1024-sample
  numbers are **superseded**: its selected checkpoint is **regenerated at N=5,000** for the table.

Absolute FID values are NOT comparable to CellFlux's published BBBC021/RxRx1 numbers (different
dataset); they only need to be internally consistent across the methods compared here.

## 3. Per-dataset mapping

| | Diet | CRISPR |
|---|---|---|
| CellFlux analog | BBBC021 (few classes, strong effect) | RxRx1 (many classes, weak genetic effect) |
| Conditions | 3: adlib (ctrl), fasted, hfd | 76 genes (rna_snr-filtered) + control |
| Treated test rows | fasted 9,178 / hfd 8,647 | per-gene small |
| Channels | [9,5,8] Calreticulin/Perilipin/TOMM20 | [0,14,5] Alb/Rab7/Perilipin |
| Proposed run | `outputs/diet_id_v3` (sel. ep9 balanced / ep11 HFD-peak) | `outputs/cellflux_pm_train_id_v8` (sel. ep19) |

## 4. MoA — real-image ceiling rule

The MoA metric is only meaningful if a classifier can distinguish the conditions on **real**
images. So for each dataset, **first train on real treated images and report the real-image test
accuracy (the ceiling)**; the generated-image accuracy is interpreted relative to it.

- **Diet (3-class):** strong physiological perturbation → real ceiling expected high → MoA is a
  clean, reportable biological-fidelity metric. **Go.**
- **CRISPR (76-way):** single-gene effects ≪ cell-to-cell variance → real ceiling expected near
  chance (1/76). No mechanism/pathway grouping label exists (`cluster_type` = Hep1–6 cell
  subtypes, NOT perturbation mechanism). **Report MoA only if real ceiling ≫ chance;** otherwise
  state it is not informative for CRISPR and rely on FIDc/KIDc + gap_closed/dir_corr.

## 5. Model selection

CellFlux selects checkpoints by **lowest validation FID**. This project previously found FID is
anti-correlated with biology under control-init on this data (see `docs/EXPERIMENTS.md`), and
selected by `gap_closed`. We **disclose this deviation honestly**: report validation FID for
selection transparency, but justify the gap_closed-based selection as a documented finding, not
a silent substitution. The comparison table reports all methods at the same, named checkpoints.

## 6. Generation config (matched where the method family allows)

All methods use **control-image initialization** (the CellFlux premise — control images calibrate
batch effects). Sampler internals differ by method family (proposed: flow-matching ODE; PhenDiff:
DDIM inversion; IMPA: AdaIN GAN) — that is expected and not a confound. Each method uses its own
tuned CFG / inference steps at the value reported in its source. The eval set, the
control→treated pairing, and N are identical.

> **Caveat — proposed diet gen config:** `outputs/diet_id_v3/args.json` is stale (shows
> `fid_samples=12, nfe=12` from a later short resample, contradicting the 1024 PNGs on disk).
> Take the real generation config (use_initial, CFG, ODE steps) from the run's `log.txt` /
> launch command, NOT args.json, when regenerating at N=5,000.

## 7. Deliverable — comparison table (mirrors CellFlux Table 1a / 2a)

One table per benchmark; rows = methods, columns = the full suite:

```
Method        FIDo  FIDc  KIDo  KIDc  MoA-Acc  MoA-MacroF1  MoA-WeightedF1   [gap_closed]
Copy-control   ...   ...   ...   ...    ...        ...           ...            ~0
PhenDiff       ...   ...   ...   ...    ...        ...           ...            ...
IMPA           ...   ...   ...   ...    ...        ...           ...            ...
Morpho-CellFlux(ours) ...  ...  ...    ...        ...           ...            ...
Groundtruth (real perturbed)  0.0  0.0  0.0  0.0  <ceiling>     ...            1.0
```

Copy-control is the biological null (gap_closed ≈ 0); Groundtruth real-perturbed is the upper
bound (FID/KID = 0 by construction, MoA = real-image ceiling).

## 8. Implementation status

- [x] Copy-control (diet + crispr) — done
- [x] PhenDiff training (diet done; crispr training) + sample export (imagefolder/<cond>/<key>.png)
- [ ] Add KIDo/KIDc + FIDc to eval (currently only overall FID) — task #6
- [ ] Wire MoA classifier to diet/crispr + real-image ceiling (`engine/moa/train_moa.py`) — task #7
- [ ] Unified matched-N (=5000) runner emitting the comparison table; regen proposed at N=5000 — task #8
- [ ] Diet end-to-end under protocol, then CRISPR — task #9
- [ ] IMPA, StarGAN runs

#!/usr/bin/env python
"""Verify whether a stable, learnable POPULATION-LEVEL morphology signal exists.

The generation model collapses to identity (see phenoflux.eval.distribution_eval).
Before changing the model or loss, answer the prior question with DATA ONLY:
does the microalgae population actually change morphologically from 0h control to
later timepoints, at the population level, in a way a model could learn?

Task structure (from timepoint_512/index.csv): source is always 0h; targets are
1/2/3/6/12/24/48/72h. So we test 0h-vs-<t> separability per condition (Dark/Light).

Four complementary diagnostics per (condition, target_time):
  1. Separability AUC: 5-fold CV logistic regression on morphology features
     (0h=class0 vs target=class1). AUC~0.5 => no signal; high => signal exists.
  2. Effect size: Cohen's d and SNR (|mean drift| / pooled std) per feature.
  3. Energy-distance ratio: d(0h, target) / d(0h, 0h-null-split). >1 => the real
     shift exceeds the within-control noise floor.
  4. Temporal trend: does multivariate 0h-vs-t energy distance grow with t?

Cross-check: also compute separability AUC on Inception features, so a negative
morphology verdict cannot be blamed on hand-crafted features missing the signal.

Usage:
  python scripts/verify_signal_strength.py \
      [--crops-root <dir>] [--per-group 500] [--conditions Dark,Light] [--seed 0] \
      [--no-inception] [--out signal_report.json]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from phenoflux.eval.morphology import FEATURES, extract_population

logger = logging.getLogger(__name__)

DEFAULT_CROPS_ROOT = Path("/home/shockley/myproject/FusionODE/data/CROPS_RAW_SCALE")
TARGET_TIMES = [1, 2, 3, 6, 12, 24, 48, 72]
CANVAS = 128


def _sample_paths(
    crops_root: Path, time_label: str, condition: str, n: int, rng: np.random.Generator
) -> list[Path]:
    """Randomly sample up to n crop paths for a (time, condition) group."""
    d = crops_root / time_label / condition
    files = sorted(d.glob("*.png"))
    if not files:
        return []
    idx = rng.permutation(len(files))[:n]
    return [files[i] for i in sorted(idx)]


def _cv_auc(x0: np.ndarray, x1: np.ndarray, seed: int) -> float:
    """5-fold CV AUC of logistic regression separating two feature populations."""
    if len(x0) < 10 or len(x1) < 10:
        return float("nan")
    x = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))])
    x = StandardScaler().fit_transform(x)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, x, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


def _cohens_d(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Per-feature Cohen's d (target vs control)."""
    m0, m1 = x0.mean(0), x1.mean(0)
    s0, s1 = x0.std(0, ddof=1), x1.std(0, ddof=1)
    pooled = np.sqrt((s0**2 + s1**2) / 2.0)
    pooled = np.where(pooled < 1e-8, 1.0, pooled)
    return (m1 - m0) / pooled


def _energy_mv(a: np.ndarray, b: np.ndarray) -> float:
    """Multivariate energy distance on standardized features (subsample for cost)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")

    def pmean(u: np.ndarray, v: np.ndarray) -> float:
        d = np.sqrt(np.maximum(((u[:, None, :] - v[None, :, :]) ** 2).sum(-1), 0.0))
        return float(d.mean())

    return max(2.0 * pmean(a, b) - pmean(a, a) - pmean(b, b), 0.0)


def _inception_features(paths: list[Path], device, canvas: int = CANVAS) -> np.ndarray:
    """Inception pool3 (2048-d) features for a list of crops, canvas-normalized."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    from phenoflux.eval.morphology import _load_rgb

    fid = FrechetInceptionDistance(normalize=True).to(device)
    feats = []
    bs = 64
    imgs = []
    for p in paths:
        img = _load_rgb(Path(p), canvas_size=canvas)
        if img is None:
            continue
        imgs.append(np.clip(img, 0, 1).transpose(2, 0, 1))
    if not imgs:
        return np.empty((0, 2048))
    # fid.inception expects uint8 [0,255] (the normalize=True path only applies
    # inside .update()); convert here since we call the feature extractor directly.
    t = torch.from_numpy((np.stack(imgs) * 255.0).clip(0, 255).astype(np.uint8))
    with torch.no_grad():
        for i in range(0, len(t), bs):
            batch = t[i : i + bs].to(device)
            feats.append(fid.inception(batch).detach().cpu().numpy())
    return np.concatenate(feats, 0)


def run(
    crops_root: Path,
    per_group: int,
    conditions: list[str],
    seed: int,
    use_inception: bool,
    out_path: Path,
) -> dict:
    rng = np.random.default_rng(seed)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report: dict = {"crops_root": str(crops_root), "per_group": per_group, "by_condition": {}}
    for cond in conditions:
        logger.info("Condition %s: extracting 0h control morphology", cond)
        ctrl_paths = _sample_paths(crops_root, "0h", cond, per_group, rng)
        ctrl_feat = extract_population(ctrl_paths, canvas_size=CANVAS)
        # Null split: control vs control -> noise floor for energy distance.
        half = len(ctrl_feat) // 2
        null_energy = _energy_mv(
            StandardScaler().fit_transform(ctrl_feat[:half]),
            StandardScaler().fit_transform(ctrl_feat[half:]),
        )
        ctrl_inc = _inception_features(ctrl_paths, device) if use_inception else None

        cond_out: dict = {"null_energy_morph": null_energy, "targets": {}}
        for t in TARGET_TIMES:
            tpaths = _sample_paths(crops_root, f"{t}h", cond, per_group, rng)
            if not tpaths:
                continue
            tfeat = extract_population(tpaths, canvas_size=CANVAS)
            scaler = StandardScaler().fit(np.vstack([ctrl_feat, tfeat]))
            energy = _energy_mv(scaler.transform(ctrl_feat), scaler.transform(tfeat))
            d = _cohens_d(ctrl_feat, tfeat)
            entry = {
                "n_ctrl": len(ctrl_feat),
                "n_target": len(tfeat),
                "auc_morph": _cv_auc(ctrl_feat, tfeat, seed),
                "energy_morph": energy,
                "energy_ratio": energy / null_energy if null_energy > 1e-8 else float("nan"),
                "max_abs_cohens_d": float(np.nanmax(np.abs(d))),
                "cohens_d": {f: float(v) for f, v in zip(FEATURES, d)},
            }
            if use_inception and ctrl_inc is not None and len(ctrl_inc):
                tinc = _inception_features(tpaths, device)
                entry["auc_inception"] = _cv_auc(ctrl_inc, tinc, seed)
            cond_out["targets"][t] = entry
            logger.info(
                "  %s 0h->%dh: AUC_morph=%.3f e_ratio=%.2f maxd=%.2f%s",
                cond, t, entry["auc_morph"], entry["energy_ratio"],
                entry["max_abs_cohens_d"],
                f" AUC_inc={entry.get('auc_inception', float('nan')):.3f}" if use_inception else "",
            )
        report["by_condition"][cond] = cond_out

    _verdict(report)
    out_path.write_text(json.dumps(report, indent=2, default=float))
    logger.info("Saved %s", out_path)
    return report


def _verdict(report: dict) -> None:
    """Attach a top-level verdict + print a compact table."""
    aucs, ratios, inc_aucs = [], [], []
    print(f"\n{'Cond':<6}{'t(h)':>5}{'AUC_morph':>11}{'e_ratio':>9}{'max|d|':>8}{'AUC_inc':>9}")
    print("-" * 48)
    for cond, cd in report["by_condition"].items():
        for t, e in cd["targets"].items():
            aucs.append(e["auc_morph"])
            ratios.append(e["energy_ratio"])
            if "auc_inception" in e:
                inc_aucs.append(e["auc_inception"])
            print(
                f"{cond:<6}{t:>5}{e['auc_morph']:>11.3f}{e['energy_ratio']:>9.2f}"
                f"{e['max_abs_cohens_d']:>8.2f}{e.get('auc_inception', float('nan')):>9.3f}"
            )
    max_auc = float(np.nanmax(aucs)) if aucs else float("nan")
    max_ratio = float(np.nanmax(ratios)) if ratios else float("nan")
    max_inc = float(np.nanmax(inc_aucs)) if inc_aucs else float("nan")
    # Heuristic thresholds: AUC>0.65 or energy ratio>1.5 => learnable signal exists.
    signal = bool((max_auc > 0.65) or (max_ratio > 1.5) or (max_inc > 0.65))
    report["verdict"] = {
        "signal_exists": signal,
        "max_auc_morph": max_auc,
        "max_energy_ratio": max_ratio,
        "max_auc_inception": max_inc,
    }
    tag = (
        "SIGNAL EXISTS -> model/loss redesign is justified"
        if signal
        else "SIGNAL TOO WEAK -> task/pairing redesign needed (model changes likely futile)"
    )
    print(
        f"\n>>> VERDICT: {tag}\n"
        f"    max AUC_morph={max_auc:.3f}  max energy_ratio={max_ratio:.2f}  "
        f"max AUC_inception={max_inc:.3f}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops-root", type=Path, default=DEFAULT_CROPS_ROOT)
    ap.add_argument("--per-group", type=int, default=500)
    ap.add_argument("--conditions", default="Dark,Light")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-inception", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("signal_report.json"))
    args = ap.parse_args()
    run(
        crops_root=args.crops_root,
        per_group=args.per_group,
        conditions=[c.strip() for c in args.conditions.split(",")],
        seed=args.seed,
        use_inception=not args.no_inception,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Distribution-level evaluation for microalgae phenotype transport.

Motivation
----------
Per-pair pixel distance is misleading here: deterministic flow matching regresses
to the population mean, and for microalgae the population mean is close to the
control (identity). The right question is whether the GENERATED POPULATION
reproduces the REAL-TREATED POPULATION, not whether each generated cell matches a
specific target cell.

Every metric is reported next to an IDENTITY BASELINE: the same metric computed
with the paired CONTROL crop as the prediction. The model only "passes" a metric
if it beats the identity baseline. A collapse-to-identity model therefore cannot
silently pass -- the failure is visible in the delta column.

Metrics (all: distance from a prediction population to the target population)
  Primary (interpretable morphology space, features from phenoflux.eval.morphology):
    - energy distance (multivariate, z-scored features)
    - MMD with an RBF kernel (median-heuristic bandwidth)
    - per-dimension 1-D Wasserstein / KS (mean across dims; per-dim in pooled row)
  Secondary (comparable to CellFlux baselines):
    - stratified FID/KID (gen vs target and control vs target), via fid.py

Usage
-----
  python -m phenoflux.eval.distribution_eval <run_dir> [epoch] \
      [--crops-root <dir>] [--min-n 5] [--margin 0.0] [--fid-cap N] [--no-fid]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import energy_distance, ks_2samp, wasserstein_distance

from phenoflux.eval.morphology import FEATURES, extract_population

logger = logging.getLogger(__name__)

DEFAULT_CROPS_ROOT = Path("/home/shockley/myproject/FusionODE/data/CROPS_RAW_SCALE")
_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Distribution distances                                                      #
# --------------------------------------------------------------------------- #
def _zscore(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score both populations using the TARGET mean/std (per feature dim).

    Standardising by the target keeps every metric on the same scale for the
    model and the identity baseline, and stops large-magnitude features (area)
    from dominating the multivariate distances.
    """
    mu = target.mean(axis=0)
    sd = target.std(axis=0)
    sd = np.where(sd < _EPS, 1.0, sd)
    return (pred - mu) / sd, (target - mu) / sd


def energy_distance_mv(pred: np.ndarray, target: np.ndarray) -> float:
    """Multivariate energy distance on Euclidean norms of z-scored vectors.

    E = 2*E||X-Y|| - E||X-X'|| - E||Y-Y'||, with X~pred, Y~target.
    """
    if len(pred) < 2 or len(target) < 2:
        return float("nan")

    def _pairwise_mean(a: np.ndarray, b: np.ndarray) -> float:
        # mean Euclidean distance between rows of a and rows of b
        d = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0))
        return float(d.mean())

    xy = _pairwise_mean(pred, target)
    xx = _pairwise_mean(pred, pred)
    yy = _pairwise_mean(target, target)
    return max(2.0 * xy - xx - yy, 0.0)


def mmd_rbf(pred: np.ndarray, target: np.ndarray) -> float:
    """Squared MMD with an RBF kernel; bandwidth via the median heuristic."""
    if len(pred) < 2 or len(target) < 2:
        return float("nan")
    z = np.vstack([pred, target])
    sq = np.maximum(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1), 0.0)
    med = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
    gamma = 1.0 / (med + _EPS)
    n, m = len(pred), len(target)
    kxx = np.exp(-gamma * sq[:n, :n])
    kyy = np.exp(-gamma * sq[n:, n:])
    kxy = np.exp(-gamma * sq[:n, n:])
    # unbiased-ish: exclude diagonal for the within-sample terms
    kxx_off = (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) if n > 1 else 0.0
    kyy_off = (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) if m > 1 else 0.0
    return max(float(kxx_off + kyy_off - 2.0 * kxy.mean()), 0.0)


def per_dim_distances(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Per-feature 1-D Wasserstein and KS statistics (raw, un-z-scored feats)."""
    wd = np.full(len(FEATURES), np.nan)
    ks = np.full(len(FEATURES), np.nan)
    if len(pred) < 2 or len(target) < 2:
        return {"wasserstein": wd, "ks": ks}
    for j in range(len(FEATURES)):
        wd[j] = wasserstein_distance(pred[:, j], target[:, j])
        ks[j] = ks_2samp(pred[:, j], target[:, j]).statistic
    return {"wasserstein": wd, "ks": ks}


def morphology_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """All morphology-space distances from `pred` population to `target` population."""
    if len(pred) < 2 or len(target) < 2:
        return {"energy": np.nan, "mmd": np.nan, "ks_mean": np.nan, "wd_mean": np.nan}
    pz, tz = _zscore(pred, target)
    per_dim = per_dim_distances(pred, target)
    # per-dim energy on z-scored features, summed, as a scale-free 1-D companion
    return {
        "energy": energy_distance_mv(pz, tz),
        "mmd": mmd_rbf(pz, tz),
        "ks_mean": float(np.nanmean(per_dim["ks"])),
        "wd_mean": float(np.nanmean(per_dim["wasserstein"])),
    }


# --------------------------------------------------------------------------- #
# Loading / pairing                                                           #
# --------------------------------------------------------------------------- #
def _resolve_epoch_dir(run_dir: Path, epoch: str | None) -> Path:
    """Resolve fid_samples/epoch-<N> (latest if epoch is None)."""
    fs = run_dir / "fid_samples"
    if epoch is not None:
        d = fs / f"epoch-{epoch}"
        if not d.exists():
            raise SystemExit(f"epoch dir not found: {d}")
        return d
    epoch_dirs = sorted(fs.glob("epoch-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not epoch_dirs:
        raise SystemExit(f"no epoch-* dirs under {fs}")
    return epoch_dirs[-1]


def _load_pairing(run_dir: Path, epoch_dir: Path) -> dict[str, str]:
    """treated-relpath -> control-relpath (per-epoch file preferred)."""
    for cand in (epoch_dir / "trt2ctrl_idx.json", run_dir / "fid_samples" / "trt2ctrl_idx.json"):
        if cand.exists():
            return json.load(open(cand))
    logger.warning("no trt2ctrl_idx.json found; control/identity baseline unavailable")
    return {}


def _collect_stratum_paths(
    epoch_dir: Path, crops_root: Path, trt2ctrl: dict[str, str]
) -> dict[str, dict[str, list[Path]]]:
    """Per stratum (condition dir name) collect gen / target / control crop paths.

    gen: the generated PNG. target: real-treated crop (crops_root / trt-relpath,
    matched to the gen PNG by basename). control: crops_root / ctrl-relpath.
    Only cells with all three resolvable are kept, so populations are comparable.
    """
    # basename -> (treated_relpath, control_relpath)
    by_base = {Path(k).name: (k, v) for k, v in trt2ctrl.items()}
    strata: dict[str, dict[str, list[Path]]] = {}
    for cond_dir in sorted(p for p in epoch_dir.iterdir() if p.is_dir()):
        gen_paths, tgt_paths, ctrl_paths = [], [], []
        for png in sorted(cond_dir.glob("*.png")):
            key = by_base.get(png.name)
            if key is None:
                continue
            trt_rel, ctrl_rel = key
            tgt = crops_root / trt_rel
            ctrl = crops_root / ctrl_rel
            if not tgt.exists() or not ctrl.exists():
                continue
            gen_paths.append(png)
            tgt_paths.append(tgt)
            ctrl_paths.append(ctrl)
        if gen_paths:
            strata[cond_dir.name] = {
                "gen": gen_paths,
                "target": tgt_paths,
                "control": ctrl_paths,
            }
    return strata


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _score_stratum(paths: dict[str, list[Path]], canvas_size: int = 128) -> dict[str, float]:
    """Morphology metrics for model (gen) and identity (control) vs target.

    All three populations pass through the same canvas normalization so raw
    variable-size crops share the pixel scale of the generated 128x128 crops.
    """
    gen_f = extract_population(paths["gen"], canvas_size=canvas_size)
    tgt_f = extract_population(paths["target"], canvas_size=canvas_size)
    ctrl_f = extract_population(paths["control"], canvas_size=canvas_size)
    model = morphology_metrics(gen_f, tgt_f)
    identity = morphology_metrics(ctrl_f, tgt_f)
    row: dict[str, float] = {"n_gen": len(gen_f), "n_target": len(tgt_f), "n_control": len(ctrl_f)}
    for metric in ("energy", "mmd", "ks_mean", "wd_mean"):
        m, i = model[metric], identity[metric]
        row[f"{metric}_model"] = m
        row[f"{metric}_identity"] = i
        row[f"{metric}_delta"] = i - m  # lower distance is better -> positive = model wins
        row[f"{metric}_pass"] = bool(np.isfinite(m) and np.isfinite(i) and (i - m) > 0)
    return row


def run_distribution_eval(
    run_dir: Path,
    epoch: str | None = None,
    crops_root: Path | None = None,
    min_n: int = 5,
    margin: float = 0.0,
    compute_fid: bool = True,
    fid_cap: int | None = None,
) -> dict:
    """Run the full distribution-level evaluation; write CSV + JSON; return summary."""
    run_dir = Path(run_dir)
    if crops_root is None:
        args_json = run_dir / "args.json"
        if args_json.exists():
            crops_root = Path(json.load(open(args_json)).get("image_path", DEFAULT_CROPS_ROOT))
        else:
            crops_root = DEFAULT_CROPS_ROOT
    epoch_dir = _resolve_epoch_dir(run_dir, epoch)
    trt2ctrl = _load_pairing(run_dir, epoch_dir)
    strata = _collect_stratum_paths(epoch_dir, Path(crops_root), trt2ctrl)
    if not strata:
        raise SystemExit(f"no usable strata in {epoch_dir} (check crops_root={crops_root})")

    # Morphology metrics per stratum (+ pooled from concatenated paths).
    rows: dict[str, dict[str, float]] = {}
    for name, paths in strata.items():
        if len(paths["gen"]) < min_n:
            continue
        rows[name] = _score_stratum(paths)
    pooled_paths = {
        k: [p for s in strata.values() for p in s[k]] for k in ("gen", "target", "control")
    }
    rows["pooled"] = _score_stratum(pooled_paths)

    # Secondary: stratified FID (gen vs target, control vs target) reusing fid.py.
    fid_block: dict = {}
    if compute_fid:
        fid_block = _compute_fid_block(strata, fid_cap)

    verdict = bool(rows["pooled"].get("energy_delta", float("nan")) > margin)
    df = pd.DataFrame(rows).T
    out_csv = run_dir / "distribution_eval_by_stratum.csv"
    out_json = run_dir / "distribution_eval_summary.json"
    df.to_csv(out_csv)
    summary = {
        "run_dir": str(run_dir),
        "epoch_dir": str(epoch_dir),
        "crops_root": str(crops_root),
        "primary_metric": "energy_distance_morph",
        "margin": margin,
        "verdict_model_beats_identity": verdict,
        "by_stratum": rows,
        "fid": fid_block,
        "features": FEATURES,
    }
    json.dump(summary, open(out_json, "w"), indent=2, default=float)
    _print_report(rows, fid_block, verdict, out_csv, out_json)
    return summary


def _compute_fid_block(
    strata: dict[str, dict[str, list[Path]]], fid_cap: int | None, canvas_size: int = 128
) -> dict:
    """Stratified FID for gen-vs-target and control-vs-target.

    Raw crops are variable-size, so we materialize canvas-normalized 128x128
    PNGs into a temp tree (not symlinks) so fid.load_pngs can np.stack them and
    every population shares the pixel scale the model saw.
    """
    import tempfile

    import torch
    from PIL import Image

    from phenoflux.eval.fid import fid_kid_stratified
    from phenoflux.eval.morphology import _load_rgb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = fid_cap or min(len(s["gen"]) for s in strata.values())

    def _materialize(key: str, root: Path) -> dict[str, Path]:
        mapping = {}
        for name, paths in strata.items():
            d = root / key / name
            d.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(paths[key]):
                img = _load_rgb(Path(p), canvas_size=canvas_size)
                if img is None:
                    continue
                arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(d / f"{i:06d}.png")
            mapping[name] = d
        return mapping

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gen_dirs = _materialize("gen", root)
        tgt_dirs = _materialize("target", root)
        ctrl_dirs = _materialize("control", root)
        model_fid = fid_kid_stratified(gen_dirs, tgt_dirs, device, cap)
        identity_fid = fid_kid_stratified(ctrl_dirs, tgt_dirs, device, cap)
    return {"model": model_fid, "identity": identity_fid, "cap": cap}


def _print_report(rows: dict, fid_block: dict, verdict: bool, csv: Path, js: Path) -> None:
    """Console table with the MODEL vs IDENTITY verdict."""
    print(f"\n{'Stratum':<18}{'E_model':>10}{'E_ident':>10}{'E_delta':>10}{'pass':>6}{'N_gen':>7}")
    print("-" * 61)
    for name, r in rows.items():
        print(
            f"{name:<18}{r.get('energy_model', float('nan')):>10.4f}"
            f"{r.get('energy_identity', float('nan')):>10.4f}"
            f"{r.get('energy_delta', float('nan')):>10.4f}"
            f"{str(r.get('energy_pass', False)):>6}{int(r.get('n_gen', 0)):>7}"
        )
    if fid_block:
        mp = fid_block["model"].get("pooled", {})
        ip = fid_block["identity"].get("pooled", {})
        print(
            f"\nFID pooled  model={mp.get('fid', float('nan')):.2f}  "
            f"identity={ip.get('fid', float('nan')):.2f}  (cap={fid_block['cap']})"
        )
    tag = "PASS: model beats identity" if verdict else "FAIL: model does NOT beat identity (metric reveals collapse)"
    print(f"\n>>> VERDICT (primary=morphology energy distance, pooled): {tag}")
    print(f"Saved: {csv}\n       {js}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("epoch", nargs="?", default=None, help="epoch number (default: latest)")
    ap.add_argument("--crops-root", type=Path, default=None)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--fid-cap", type=int, default=None)
    ap.add_argument("--no-fid", action="store_true")
    args = ap.parse_args()
    run_distribution_eval(
        run_dir=args.run_dir,
        epoch=args.epoch,
        crops_root=args.crops_root,
        min_n=args.min_n,
        margin=args.margin,
        compute_fid=not args.no_fid,
        fid_cap=args.fid_cap,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "energy_distance_mv",
    "mmd_rbf",
    "per_dim_distances",
    "morphology_metrics",
    "run_distribution_eval",
]

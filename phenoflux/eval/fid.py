#!/usr/bin/env python
"""Compute the CellFlux image-quality metric suite (FIDo/FIDc/KIDo/KIDc) from imagefolders.

Protocol: docs/EVAL_PROTOCOL.md. The same module is used for every reported
method so the comparison is apples-to-apples: identical InceptionV3 features,
identical sample budget N, identical condition split
sanity check can use this module too, but it is not a default paper-table row.

Layout expected (shared across all methods):
    <real-dir>/<condition>/*.png      real perturbed images (reference)
    <gen-dir>/<condition>/*.png       generated images for that condition

Definitions (CellFlux arXiv:2502.09775 §4.2, Table 1):
  FIDo / KIDo  overall: all generated vs all real-perturbed, pooled across conditions.
  FIDc / KIDc  conditional: metric per condition, then averaged across conditions.

Sample budget: per condition we use exactly `--per-condition-cap` images from BOTH
real and generated (default = n // n_conditions). FID/KID are sample-size sensitive
(CellFlux Table 5), so every method MUST be run at the same cap; the module errors if a
folder has fewer than the cap so mismatches are caught rather than silently biasing FID.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def list_conditions(real_dir: Path, gen_dir: Path) -> list[str]:
    """Conditions = subdirs present in BOTH real and gen (so control-only dirs drop out)."""
    real_c = {p.name for p in real_dir.iterdir() if p.is_dir()}
    gen_c = {p.name for p in gen_dir.iterdir() if p.is_dir()}
    return sorted(real_c & gen_c)


def load_pngs(folder: Path, cap: int, rng: np.random.Generator) -> torch.Tensor:
    """Load up to `cap` PNGs as a float tensor (B,3,H,W) in [0,1], deterministically sampled."""
    files = sorted(folder.glob("*.png"))
    if len(files) < cap:
        raise ValueError(
            f"{folder} has {len(files)} pngs < per-condition cap {cap}; "
            f"all methods must reach the cap for matched-N comparability."
        )
    idx = rng.permutation(len(files))[:cap]
    chosen = [files[i] for i in sorted(idx)]
    arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in chosen])  # (B,H,W,3) uint8
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float() / 255.0
    return t


def fid_kid_for_pair(
    real: torch.Tensor, gen: torch.Tensor, device: torch.device, kid_subset: int
) -> tuple[float, float, float]:
    """Return (fid, kid_mean, kid_std) for one matched real/gen tensor pair."""
    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=kid_subset, normalize=True).to(device)
    bs = 64
    for metric in (fid, kid):
        for i in range(0, real.shape[0], bs):
            metric.update(real[i : i + bs].to(device), real=True)
        for i in range(0, gen.shape[0], bs):
            metric.update(gen[i : i + bs].to(device), real=False)
    fid_v = float(fid.compute().detach().cpu())
    kid_m, kid_s = kid.compute()
    return fid_v, float(kid_m.detach().cpu()), float(kid_s.detach().cpu())


def fid_kid_stratified(
    pred_dir_by_strata: dict[str, Path],
    real_dir_by_strata: dict[str, Path],
    device: torch.device,
    cap: int,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Compute FID/KID per stratum (+ a pooled row) for a prediction vs real split.

    Used by distribution_eval to score both `gen vs target` and the identity
    baseline `control vs target` with matched-N, reusing load_pngs and
    fid_kid_for_pair. Strata are keyed by directory name (already = timepoint x
    condition). A stratum is skipped if either folder has fewer than `cap` PNGs.

    Args:
        pred_dir_by_strata: {stratum: folder of prediction PNGs}.
        real_dir_by_strata: {stratum: folder of real target PNGs}.
        device: torch device for the InceptionV3 features.
        cap: per-stratum sample count (both sides sampled to this many).
        seed: base RNG seed (real uses seed, pred uses seed+1, as in main()).

    Returns:
        {stratum: {"fid", "kid", "kid_std", "n"}} plus a "pooled" entry.
    """
    out: dict[str, dict[str, float]] = {}
    real_pool: list[torch.Tensor] = []
    pred_pool: list[torch.Tensor] = []
    for stratum in sorted(set(pred_dir_by_strata) & set(real_dir_by_strata)):
        real_dir, pred_dir = real_dir_by_strata[stratum], pred_dir_by_strata[stratum]
        n_real = len(sorted(Path(real_dir).glob("*.png")))
        n_pred = len(sorted(Path(pred_dir).glob("*.png")))
        stratum_cap = min(cap, n_real, n_pred)
        if stratum_cap < 2:
            continue
        real_t = load_pngs(real_dir, stratum_cap, np.random.default_rng(seed))
        pred_t = load_pngs(pred_dir, stratum_cap, np.random.default_rng(seed + 1))
        kid_subset = min(1000, stratum_cap)
        f, km, ks = fid_kid_for_pair(real_t, pred_t, device, kid_subset)
        out[stratum] = {"fid": f, "kid": km, "kid_std": ks, "n": stratum_cap}
        real_pool.append(real_t)
        pred_pool.append(pred_t)

    if real_pool:
        real_all = torch.cat(real_pool, 0)
        pred_all = torch.cat(pred_pool, 0)
        f, km, ks = fid_kid_for_pair(
            real_all, pred_all, device, min(1000, real_all.shape[0])
        )
        out["pooled"] = {"fid": f, "kid": km, "kid_std": ks, "n": int(real_all.shape[0])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True, help="imagefolder of REAL perturbed images: <cond>/*.png")
    ap.add_argument("--gen-dir", required=True, help="generated images: <cond>/*.png (e.g. fid_samples/epoch-K)")
    ap.add_argument("--n", type=int, default=5000, help="overall sample budget (FIDo/KIDo)")
    ap.add_argument("--per-condition-cap", type=int, default=None, help="override per-condition images (default n//n_cond)")
    ap.add_argument("--conditions", default=None, help="comma-separated; default = intersection of real & gen dirs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output json path (default <gen-dir>/../image_metrics.json)")
    ap.add_argument("--label", default=None, help="method label for the printout")
    args = ap.parse_args()

    real_dir, gen_dir = Path(args.real_dir), Path(args.gen_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    conditions = (
        [c.strip() for c in args.conditions.split(",")]
        if args.conditions
        else list_conditions(real_dir, gen_dir)
    )
    if not conditions:
        raise SystemExit(f"no shared conditions between {real_dir} and {gen_dir}")
    cap = args.per_condition_cap or (args.n // len(conditions))

    # Per-condition tensors (matched real/gen counts), reused for both FIDc and pooled FIDo.
    per_cond, real_pool, gen_pool = {}, [], []
    for cond in conditions:
        real_t = load_pngs(real_dir / cond, cap, np.random.default_rng(args.seed))
        gen_t = load_pngs(gen_dir / cond, cap, np.random.default_rng(args.seed + 1))
        per_cond[cond] = (real_t, gen_t)
        real_pool.append(real_t)
        gen_pool.append(gen_t)

    kid_subset = min(1000, cap)

    # Conditional: FIDc/KIDc = per-condition metric averaged across conditions.
    per_condition_out = {}
    for cond, (real_t, gen_t) in per_cond.items():
        f, km, ks = fid_kid_for_pair(real_t, gen_t, device, kid_subset)
        per_condition_out[cond] = {"fid": f, "kid": km, "kid_std": ks, "n": cap}
        print(f"  [{cond:8}] FID={f:8.2f}  KID={km:.4f}±{ks:.4f}  (n={cap})")

    fid_c = float(np.mean([v["fid"] for v in per_condition_out.values()]))
    kid_c = float(np.mean([v["kid"] for v in per_condition_out.values()]))

    # Overall: pool all conditions.
    real_all = torch.cat(real_pool, 0)
    gen_all = torch.cat(gen_pool, 0)
    fid_o, kid_o, kid_o_std = fid_kid_for_pair(real_all, gen_all, device, min(1000, real_all.shape[0]))

    result = {
        "label": args.label,
        "real_dir": str(real_dir),
        "gen_dir": str(gen_dir),
        "n_overall": int(real_all.shape[0]),
        "per_condition_cap": cap,
        "conditions": conditions,
        "fid_o": fid_o,
        "kid_o": kid_o,
        "kid_o_std": kid_o_std,
        "fid_c": fid_c,
        "kid_c": kid_c,
        "per_condition": per_condition_out,
        "seed": args.seed,
    }
    out = Path(args.out) if args.out else gen_dir.parent / "image_metrics.json"
    out.write_text(json.dumps(result, indent=2))
    print(
        f"[{args.label or gen_dir}]  FIDo={fid_o:.2f} KIDo={kid_o:.4f}  "
        f"FIDc={fid_c:.2f} KIDc={kid_c:.4f}  (N={real_all.shape[0]}, cap={cap}) -> {out}"
    )


if __name__ == "__main__":
    main()

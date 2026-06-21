"""Aggregate, per-perturbation evaluation of a CellFlux perturbmulti eval run.

Publication-oriented metrics (carried by aggregates, not cherry-picked images).

CHANNEL-AWARE: the panel is read from the run's args.json ("channels"; defaults to
[5,9,10] for older runs that predate per-config channels). PNG channel k corresponds to
npz channel channels[k], so generated and real cells are always compared on the SAME
markers the run trained on.

For each panel channel and each perturbation (CRISPR gene / diet condition) we take the
foreground-mean intensity of generated / real-target / source(control) cells and report:
  - Pearson/Spearman( per-gene generated mean , per-gene real-target mean )
        -> cross-gene morphology ranking (meaningful only when many perturbations exist).
  - (control-init) dir-corr & sign-agreement of (gen-src) vs (tgt-src)   -> direction.
  - DISTRIBUTION metrics vs a COPY-CONTROL baseline: 1D Wasserstein & energy distance
    between the generated-cell and real-target-cell foreground-mean distributions, and the
    same for the trivial source(control) predictor. The headline is
        gap_closed = 1 - d(gen, tgt) / d(src, tgt)
    = fraction of the control->target population gap the model closes (1=perfect, 0=no
    better than copying the control, <0=worse). This is the honest metric for diet, where
    only 2 treated conditions exist and cross-gene Pearson is degenerate (==1.0).
  - CRISPR only: headline metrics ALSO on the rna_snr>=THR subset -- a perturbation-VALIDITY
    filter (did the sgRNA knock the transcript down), measured in RNA space, NOT the
    morphology readout being scored -- plus the full set and a hit/non-hit split. Filtering
    on rna_snr is legitimate preprocessing; filtering on morph_*/lipid_hit would be circular.

Usage:  python scripts/aggregate_eval.py <eval_run_dir> [min_n] [epoch] [--snr THR]
The run dir must contain fid_samples/<epoch>/<gene>/<target_id>.png and
fid_samples/trt2ctrl_idx.json (produced by train.py --save_fid_samples).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, energy_distance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default image dir; overridden per-run from the run's args.json image_path in main()
# so eval reads the SAME npz cells the run trained on (e.g. diet vs CRISPR).
IMG = os.path.join(REPO_ROOT, "data/raw/extracted_images")

# Canonical 18-channel npz var order (index -> marker name).
CH_NAMES = ["Alb", "polyT", "rRNA", "M6PR", "CathB", "Perilipin", "Sqstm1", "LC3b",
            "TOMM20", "Calreticulin", "pS6RP", "Na/K-ATPase", "SNAP23", "TOM70", "Rab7",
            "mtRNA", "Vimentin", "Gapdh"]
DEFAULT_CHANNELS = [5, 9, 10]            # legacy panel (Perilipin/Calreticulin/pS6RP)
RNA_SNR_DEFAULT = 0.3                    # perturbation-validity threshold for the subset

# Known lead genes per marker (for the qualitative lead-gene table); channels not present
# are simply skipped.
LEAD = {"Perilipin": ["Insig1", "Pten", "Eif2s1", "Aars"],   # lipid/steatosis
        "Calreticulin": ["Sel1l", "Atp2a2", "Dnajb9"],        # UPR
        "pS6RP": ["Pten", "Tsc1", "Tsc2", "Mtor", "Cdc37"],   # mTOR
        "Alb": ["Alb", "Apc", "Hnf4a", "Cyp3a11"],            # hepatocyte identity/function
        "Rab7": ["Prkar1a", "Npc1"],                          # endolysosomal
        "TOMM20": ["Npc1"]}                                   # mitochondria


def fg_means(flat_pixels):
    """Per-channel foreground means; fg = pixels where max-over-channels > 0.05.
    flat_pixels: (P, C) array of pixel intensities in [0,1]."""
    fg = flat_pixels.max(axis=1) > 0.05
    sel = flat_pixels[fg] if fg.sum() > 20 else flat_pixels
    return sel.mean(axis=0)              # [C]


def npz_panel_means(cid, channels):
    p = f"{IMG}/{cid}.npz"
    if not os.path.exists(p):
        return None
    panel = np.load(p)["x"][channels]    # (C,H,W) in the panel's channel order
    return fg_means(panel.reshape(len(channels), -1).T)


def metric_table(gg, names, use_initial, label):
    """Cross-gene Pearson/Spearman/dir-corr/sign-agree per channel on aggregated means."""
    print(f"\n[{label}]  genes used: {len(gg)}")
    print(f"{'channel':12} {'Pearson(gen,tgt)':>17} {'Spearman':>9} "
          f"{'dir-corr(Δ)':>12} {'sign-agree':>10}")
    out = {}
    for name in names:
        if len(gg) >= 3:
            pe = pearsonr(gg[f"gen_{name}"], gg[f"tgt_{name}"])[0]
            sp = spearmanr(gg[f"gen_{name}"], gg[f"tgt_{name}"])[0]
        else:
            pe = sp = np.nan
        dcorr, sign = np.nan, np.nan
        if use_initial in (1, 2):
            sub = gg[[f"gen_{name}", f"tgt_{name}", f"src_{name}"]].dropna()
            if len(sub) > 2:
                dg = sub[f"gen_{name}"] - sub[f"src_{name}"]
                dr = sub[f"tgt_{name}"] - sub[f"src_{name}"]
                dcorr = pearsonr(dg, dr)[0]
                sign = float((np.sign(dg) == np.sign(dr)).mean())
        out[name] = dict(pearson=pe, spearman=sp, dir_corr=dcorr, sign_agree=sign)
        print(f"{name:12} {pe:17.3f} {sp:9.3f} {dcorr:12.3f} {sign:10.2f}")
    return out


def dist_metrics(sub_df, names):
    """Population distribution metrics per channel on per-cell foreground means.
    Returns {name: {wd_gen, wd_src, gap_closed_wd, ed_gen, ed_src, gap_closed_ed}}."""
    out = {}
    for name in names:
        gen = sub_df[f"gen_{name}"].dropna().to_numpy()
        tgt = sub_df[f"tgt_{name}"].dropna().to_numpy()
        src = sub_df[f"src_{name}"].dropna().to_numpy()
        if len(gen) < 5 or len(tgt) < 5:
            continue
        wd_gen = wasserstein_distance(gen, tgt)
        ed_gen = energy_distance(gen, tgt)
        if len(src) >= 5:
            wd_src = wasserstein_distance(src, tgt)
            ed_src = energy_distance(src, tgt)
            gc_wd = 1 - wd_gen / wd_src if wd_src > 0 else np.nan
            gc_ed = 1 - ed_gen / ed_src if ed_src > 0 else np.nan
        else:
            wd_src = ed_src = gc_wd = gc_ed = np.nan
        out[name] = dict(wd_gen=wd_gen, wd_src=wd_src, gap_closed_wd=gc_wd,
                         ed_gen=ed_gen, ed_src=ed_src, gap_closed_ed=gc_ed)
    return out


def print_dist(dist, label):
    print(f"\n[{label}] population distance to real target (lower=closer) + copy-control baseline")
    print(f"{'channel':12} {'W(gen,tgt)':>11} {'W(src,tgt)':>11} {'gap_closed':>11}  read")
    for name, d in dist.items():
        gc = d["gap_closed_wd"]
        read = ("model beats copy-control" if gc > 0.05 else
                "≈ copy-control" if gc > -0.05 else "WORSE than copy-control")
        print(f"{name:12} {d['wd_gen']:11.4f} {d['wd_src']:11.4f} {gc:11.3f}  {read}")


def main():
    global IMG
    argv = [a for a in sys.argv[1:]]
    snr_thr = RNA_SNR_DEFAULT
    if "--snr" in argv:
        i = argv.index("--snr")
        snr_thr = float(argv[i + 1])
        del argv[i:i + 2]
    run = argv[0].rstrip("/")
    min_n = int(argv[1]) if len(argv) > 1 else 5
    epoch_arg = argv[2] if len(argv) > 2 else None

    use_initial, channels = None, DEFAULT_CHANNELS
    is_diet = False
    aj = f"{run}/args.json"
    if os.path.exists(aj):
        _a = json.load(open(aj))
        use_initial = _a.get("use_initial")
        channels = _a.get("channels") or DEFAULT_CHANNELS
        _ip = _a.get("image_path", "")
        is_diet = "diet" in (_ip or "")
        if _ip:
            IMG = _ip if os.path.isabs(_ip) else os.path.join(REPO_ROOT, _ip)
    names = [CH_NAMES[c] for c in channels]          # PNG channel k -> CH_NAMES[channels[k]]

    if epoch_arg is not None:
        epoch_dir = f"{run}/fid_samples/epoch-{epoch_arg}"
    else:
        epoch_dir = sorted(glob.glob(f"{run}/fid_samples/epoch-*"),
                           key=lambda p: int(p.split("-")[-1]))[-1]
    # Prefer per-epoch pairing so old epochs keep their OWN treated->control pairing.
    tp_epoch = f"{epoch_dir}/trt2ctrl_idx.json"
    tp_global = f"{run}/fid_samples/trt2ctrl_idx.json"
    tp = tp_epoch if os.path.exists(tp_epoch) else tp_global
    trt2ctrl = json.load(open(tp)) if os.path.exists(tp) else {}

    rows = []
    for gdir in sorted(glob.glob(f"{epoch_dir}/*")):
        gene = os.path.basename(gdir)
        if not os.path.isdir(gdir):
            continue
        for png in glob.glob(f"{gdir}/*.png"):
            tid = os.path.splitext(os.path.basename(png))[0]
            cid = trt2ctrl.get(tid)
            arr = np.asarray(Image.open(png).convert("RGB")) / 255.0   # (H,W,k) k=panel order
            gen = fg_means(arr.reshape(-1, len(channels)))
            tgt = npz_panel_means(tid, channels)
            src = npz_panel_means(cid, channels) if cid else None
            if tgt is None:
                continue
            rec = {"gene": gene}
            for k, name in enumerate(names):
                rec[f"gen_{name}"] = gen[k]
                rec[f"tgt_{name}"] = tgt[k]
                rec[f"src_{name}"] = src[k] if src is not None else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    g = df.groupby("gene").mean(numeric_only=True)
    g["n"] = df.groupby("gene").size()
    gg = g[g["n"] >= min_n]

    print(f"run: {run}")
    print(f"use_initial: {use_initial}   epoch: {os.path.basename(epoch_dir)}   "
          f"panel(channels {channels}): {names}")
    print(f"images: {len(df)}   perturbations(n>={min_n}): {len(gg)}   diet={is_diet}")

    summary = {"channels": channels, "names": names, "epoch": os.path.basename(epoch_dir)}

    if is_diet:
        # Cross-gene ranking is degenerate with 2 conditions -> distribution metrics per condition.
        summary["per_condition_dist"] = {}
        for cond, sub in df.groupby("gene"):
            print(f"\n=== diet condition: {cond}  (cells={len(sub)}) ===")
            d = dist_metrics(sub, names)
            print_dist(d, f"diet:{cond}")
            summary["per_condition_dist"][cond] = d
    else:
        # CRISPR: cross-gene ranking + direction, on full set and the rna_snr>=THR subset.
        summary["full"] = metric_table(gg, names, use_initial, "FULL gene set")
        snr = _load_rna_snr()
        if snr is not None:
            valid = [gene for gene in gg.index if snr.get(gene, 0.0) >= snr_thr]
            ggv = gg.loc[gg.index.isin(valid)]
            summary["rna_snr_subset"] = {"thr": snr_thr, "n_genes": len(ggv),
                                         "metrics": metric_table(ggv, names, use_initial,
                                         f"rna_snr>={snr_thr} subset (perturbation-validity, disclosed)")}
            # stratified: hit (valid) vs non-hit
            nonhit = gg.loc[~gg.index.isin(valid)]
            if len(nonhit) >= 3:
                metric_table(nonhit, names, use_initial, f"rna_snr<{snr_thr} (non-hit, for transparency)")
        else:
            print("\n(rna_snr table not found -> skipping subset; showing full set only)")
        # pooled population metrics + copy-control baseline (per channel, all cells)
        summary["dist_pooled"] = dist_metrics(df, names)
        print_dist(summary["dist_pooled"], "pooled (all genes)")

    # lead-gene qualitative table
    for name in names:
        genes = LEAD.get(name, [])
        cols = [c for c in ["n", f"src_{name}", f"gen_{name}", f"tgt_{name}"] if c in g.columns]
        sub = g.loc[g.index.isin(genes), cols]
        if len(sub):
            print(f"\nlead genes for {name}:")
            print(sub.to_string())

    g.to_csv(f"{run}/aggregate_eval_by_gene.csv")
    json.dump(_jsonable(summary), open(f"{run}/aggregate_eval_summary.json", "w"), indent=2)
    print(f"\nsaved: {run}/aggregate_eval_by_gene.csv + aggregate_eval_summary.json")


def _load_rna_snr():
    """Per-gene rna_snr (perturbation validity) from the data-build diagnostic table."""
    p = os.path.join(REPO_ROOT, "data/processed/perturbmulti/perturbation_effects.csv")
    if not os.path.exists(p):
        return None
    t = pd.read_csv(p)
    if "rna_snr" not in t.columns or "target_gene" not in t.columns:
        return None
    return dict(zip(t["target_gene"], t["rna_snr"]))


def _jsonable(obj):
    """Recursively cast numpy scalars to plain floats so json.dump succeeds."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


if __name__ == "__main__":
    main()

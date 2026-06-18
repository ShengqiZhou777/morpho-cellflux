"""GPU-free Perilipin direction analysis from saved CellFlux fid_samples.

NOTE: original single-channel prototype, SUPERSEDED by scripts/aggregate_eval.py
(which computes the same Delta-direction metric for all three current panel
channels Perilipin/Calreticulin/pS6RP and is run-agnostic). Kept for provenance;
RUN below still points at the old v2 lipid panel.

For each saved generated image (fid_samples/<epoch>/<gene>/<target_id>.png) we have:
  generated = the model's output (R channel = Perilipin)
  target    = real perturbed cell <target_id> (npz channel 5 = Perilipin)
  source    = real control cell trt2ctrl_idx[<target_id>] (npz channel 5 = Perilipin)

Direction test: does generated Perilipin move from source toward target?
  delta_gen  = peri(generated) - peri(source)
  delta_real = peri(target)    - peri(source)
A model that captures perturbation direction has sign(delta_gen)==sign(delta_real)
and delta_gen correlated with delta_real across genes (esp. steatosis genes up).
"""
import json
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(REPO_ROOT, "outputs/cellflux_pm_lipid_ddp_v2")
IMG = os.path.join(REPO_ROOT, "data/raw/extracted_images")
PERI = 5  # Perilipin channel index in the 18-ch npz
EPOCH_DIR = sorted(glob.glob(f"{RUN}/fid_samples/epoch-*"),
                   key=lambda p: int(p.split("-")[-1]))[-1]
STEATOSIS = ["Insig1", "Pten", "Eif2s1", "Aars"]


def fg_mean(arr2d):
    """Mean over foreground pixels (>0.05); falls back to global mean."""
    m = arr2d > 0.05
    return float(arr2d[m].mean()) if m.sum() > 20 else float(arr2d.mean())


def peri_from_npz(cell_id):
    p = f"{IMG}/{cell_id}.npz"
    if not os.path.exists(p):
        return None
    return fg_mean(np.load(p)["x"][PERI])


def main():
    trt2ctrl = json.load(open(f"{RUN}/fid_samples/trt2ctrl_idx.json"))
    rows = []
    for gdir in sorted(glob.glob(f"{EPOCH_DIR}/*")):
        gene = os.path.basename(gdir)
        for png in glob.glob(f"{gdir}/*.png"):
            tid = os.path.splitext(os.path.basename(png))[0]
            cid = trt2ctrl.get(tid)
            if cid is None:
                continue
            gen_peri = fg_mean(np.asarray(Image.open(png).convert("RGB"))[..., 0] / 255.0)
            src_peri = peri_from_npz(cid)
            tgt_peri = peri_from_npz(tid)
            if src_peri is None or tgt_peri is None:
                continue
            rows.append(dict(gene=gene, tid=tid, cid=cid, png=png,
                             gen=gen_peri, src=src_peri, tgt=tgt_peri))
    df = pd.DataFrame(rows)
    print(f"epoch dir: {EPOCH_DIR}  generated images analyzed: {len(df)}")

    # per-gene aggregate
    g = df.groupby("gene").agg(n=("gen", "size"), gen=("gen", "mean"),
                               src=("src", "mean"), tgt=("tgt", "mean")).reset_index()
    g["delta_gen"] = g["gen"] - g["src"]
    g["delta_real"] = g["tgt"] - g["src"]
    gg = g[g["n"] >= 5].copy()

    # direction agreement across well-sampled genes
    sign_agree = float((np.sign(gg["delta_gen"]) == np.sign(gg["delta_real"])).mean())
    corr = float(np.corrcoef(gg["delta_gen"], gg["delta_real"])[0, 1]) if len(gg) > 2 else float("nan")
    print(f"\ngenes with n>=5: {len(gg)}")
    print(f"sign agreement (delta_gen vs delta_real): {sign_agree:.2f}")
    print(f"Pearson corr (delta_gen vs delta_real):   {corr:.3f}")
    # does the model reproduce the real Perilipin gene-ranking?
    if len(gg) > 2:
        rank_corr = gg[["gen", "tgt"]].corr(method="spearman").iloc[0, 1]
        print(f"Spearman corr gen-Perilipin vs real-target-Perilipin (gene means): {rank_corr:.3f}")

    print("\n=== steatosis genes (expect delta_gen > 0, toward target) ===")
    s = g[g["gene"].isin(STEATOSIS)].sort_values("delta_real", ascending=False)
    print(s[["gene", "n", "src", "gen", "tgt", "delta_gen", "delta_real"]].to_string(index=False))

    print("\n=== top 10 real Perilipin increasers (delta_real) with n>=5 ===")
    top = gg.sort_values("delta_real", ascending=False).head(10)
    print(top[["gene", "n", "src", "gen", "tgt", "delta_gen", "delta_real"]].to_string(index=False))

    # scatter plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.scatter(gg["delta_real"], gg["delta_gen"], s=14, alpha=0.6)
    for _, r in gg[gg.gene.isin(STEATOSIS)].iterrows():
        ax.scatter(r["delta_real"], r["delta_gen"], color="red", s=40)
        ax.annotate(r["gene"], (r["delta_real"], r["delta_gen"]), color="red", fontsize=8)
    ax.set_xlabel("real Perilipin shift (target - source)")
    ax.set_ylabel("generated Perilipin shift (gen - source)")
    ax.set_title(f"Perilipin direction  (corr={corr:.2f}, sign-agree={sign_agree:.0%})")
    fig.tight_layout()
    out = f"{RUN}/perilipin_direction_{os.path.basename(EPOCH_DIR)}.png"
    fig.savefig(out, dpi=120)
    print(f"\nscatter saved: {out}")
    g.sort_values("delta_real", ascending=False).to_csv(
        f"{RUN}/perilipin_direction_by_gene.csv", index=False)

    # ---- source | generated | target montage (lipid RGB: Perilipin=R Alb=G polyT=B) ----
    def rgb_npz(cid):
        a = np.load(f"{IMG}/{cid}.npz")["x"][[5, 0, 1]]
        return np.clip(np.transpose(a, (1, 2, 0)), 0, 1)

    want = [x for x in STEATOSIS if (df.gene == x).any()]
    want += [x for x in g.sort_values("delta_real", ascending=False)["gene"]
             if x not in want][: max(0, 4 - len(want))]
    panels = []
    for gene in want:
        for _, r in df[df.gene == gene].head(3).iterrows():
            panels.append((gene, rgb_npz(r["cid"]),
                           np.asarray(Image.open(r["png"]).convert("RGB")) / 255.0,
                           rgb_npz(r["tid"])))
    if panels:
        n = len(panels)
        fig, axes = plt.subplots(n, 3, figsize=(6, 2.0 * n))
        axes = np.atleast_2d(axes)
        for i, (gene, src, gen, tgt) in enumerate(panels):
            for j, (im, t) in enumerate([(src, "source ctrl"), (gen, "generated"), (tgt, "target KO")]):
                axes[i, j].imshow(im); axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(t, fontsize=10)
            axes[i, 0].text(-0.15, 0.5, gene, transform=axes[i, 0].transAxes,
                            va="center", ha="right", fontsize=9)
        fig.suptitle("Perilipin=R  Alb=G  polyT=B", fontsize=9)
        fig.tight_layout()
        mout = f"{RUN}/steatosis_triptych_{os.path.basename(EPOCH_DIR)}.jpg"
        fig.savefig(mout, dpi=110)
        print(f"montage saved: {mout}  (rows: {[p[0] for p in panels]})")


if __name__ == "__main__":
    main()

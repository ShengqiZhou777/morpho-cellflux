#!/usr/bin/env python3
"""Build a gene/protein-based condition embedding for the microalgae timepoint view.

Replaces the old PCA + cubic-spline omics interpolation
(`interpolate_omics_to_timepoints.py`), which was empirically shown to be
unreliable: PCA compression distorts temporal interpolation (48h leave-one-out
error ~8-10x the replicate-noise baseline), and cubic spline oscillates on the
sparse log-spaced timepoints.

This pipeline instead:
  1. Reads RAW transcriptome (log2 FPKM) + proteome (log2 abundance) from the
     source xlsx.
  2. Selects high-variable features (genes with log-FPKM std > threshold,
     top-N variable proteins) -- the biologically-changing signal.
  3. Linear-interpolates each feature across the 9 measured timepoints, per
     Dark/Light condition, to the fine EXIF time bins (leave-one-out validated:
     raw-space linear interp error ~1.6-2.4x noise at 48h, vs 8-10x for PCA).
  4. z-scores each feature and assembles:
         [cond_light, cond_dark, time_norm, time_bin_h, <genes>, <proteins>]

No PCA anywhere. Output dim = 4 + n_genes + n_proteins.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

XLSX = "data/raw/TIMECOURSE对齐数据汇总.xlsx"
VIEW = "data/processed/microalgae_v1/views/timepoint_512"
BASE_EMB = f"{VIEW}/embedding.csv"                 # 4 base dims per timegroup bin
OUT_EMB = f"{VIEW}/embedding_genes.csv"
OUT_STATS = f"{VIEW}/embedding_genes_stats.csv"
OUT_FEATURES = f"{VIEW}/embedding_genes_features.csv"

GENE_STD_THRESHOLD = 2.0   # log-FPKM std cutoff -> ~372 strongly-varying genes
N_PROTEINS = 100           # top-N most variable proteins
TIMES = [0, 1, 2, 3, 6, 12, 24, 48, 72]

GENE_PAT = re.compile(r"^(?:(D|L)-)?(\d+)-([ABC]) \(FPKM\)$")
PROT_PAT = re.compile(r"^([DL])(\d+)([ABC])$")


def _parse(columns, pat):
    """Return list of (col_name, cond|None, time:int, rep) for matching columns."""
    out = []
    for c in columns:
        m = pat.match(str(c))
        if m:
            out.append((c, m.group(1), int(m.group(2)), m.group(3)))
    return out


def _cond_time_means(mat, samples, cond):
    """mat: (n_features, n_samples). Return (len(TIMES), n_features) means.

    time==0 pools ALL 0h samples (shared ancestor before the Dark/Light split).
    """
    rows = []
    for t in TIMES:
        if t == 0:
            idx = [i for i, s in enumerate(samples) if s[2] == 0]
        else:
            idx = [i for i, s in enumerate(samples) if s[2] == t and s[1] == cond]
        if not idx:
            raise ValueError(f"no samples for cond={cond} t={t}")
        rows.append(mat[:, idx].mean(axis=1))
    return np.array(rows)  # (9, n_features)


def _load_matrix(sheet, pat):
    df = pd.read_excel(XLSX, sheet_name=sheet)
    samples = _parse(df.columns, pat)
    vals = df[[s[0] for s in samples]].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    logm = np.log2(vals.astype(float) + 1.0)  # (n_features, n_samples)
    return df, samples, logm


def build():
    # ---- transcriptome: select high-variable genes ----
    tx, g_samples, G = _load_matrix("Transcriptome", GENE_PAT)
    g_std = G.std(axis=1)
    g_sel = np.where(g_std > GENE_STD_THRESHOLD)[0]
    g_names = (tx["Symbol"].astype(str) + "|" + tx["GeneID"].astype(str)).values[g_sel]
    G = G[g_sel]                                  # (n_genes, n_samples)
    print(f"genes selected (std>{GENE_STD_THRESHOLD}): {len(g_sel)}")

    # ---- proteome: top-N variable proteins ----
    pr, p_samples, P = _load_matrix("Proteome", PROT_PAT)
    p_std = P.std(axis=1)
    p_sel = np.argsort(p_std)[-N_PROTEINS:]
    p_names = ("PROT|" + pr["Gene"].astype(str) + "|" + pr["Accession"].astype(str)).values[p_sel]
    P = P[p_sel]                                  # (n_proteins, n_samples)
    print(f"proteins selected (top {N_PROTEINS} by std): {len(p_sel)}")

    # ---- base timegroup rows ----
    base = pd.read_csv(BASE_EMB)
    feat_cols = list(g_names) + list(p_names)
    n_feat = len(feat_cols)

    # ---- interpolate each feature per condition to each row's time_bin_h ----
    inter = {}  # cond -> interp1d over (9, n_feat)
    for cond in ("D", "L"):
        gm = _cond_time_means(G, g_samples, cond)   # (9, n_genes)
        pm = _cond_time_means(P, p_samples, cond)   # (9, n_proteins)
        merged = np.concatenate([gm, pm], axis=1)   # (9, n_feat)
        inter[cond] = interp1d(TIMES, merged, kind="linear", axis=0, fill_value="extrapolate")

    out = np.zeros((len(base), n_feat), dtype=float)
    for i, row in base.iterrows():
        cond = "L" if row["cond_light"] == 1.0 else "D"
        out[i] = inter[cond](float(row["time_bin_h"]))

    # ---- z-score each feature (persist stats for inference reuse) ----
    mean = out.mean(axis=0)
    std = out.std(axis=0)
    std[std == 0.0] = 1.0
    out_z = (out - mean) / std

    feat_df = pd.DataFrame(out_z, columns=feat_cols)
    emb = pd.concat([base.reset_index(drop=True), feat_df], axis=1)
    emb.to_csv(OUT_EMB, index=False)
    pd.DataFrame({"feature": feat_cols, "mean": mean, "std": std}).to_csv(OUT_STATS, index=False)
    pd.DataFrame({"feature": feat_cols}).to_csv(OUT_FEATURES, index=False)

    base_dim = 4  # cond_light, cond_dark, time_norm, time_bin_h
    total = base_dim + n_feat
    print(f"rows: {len(emb)}  base_dim: {base_dim}  genes: {len(g_sel)}  proteins: {len(p_sel)}")
    print(f"embedding total dim (base_condition_dim): {total}")
    print(f"saved: {OUT_EMB}  (shape {emb.shape})")
    print(f"z-scored feature range: [{out_z.min():.2f}, {out_z.max():.2f}]")
    return total


if __name__ == "__main__":
    build()

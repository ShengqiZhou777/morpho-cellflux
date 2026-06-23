#!/usr/bin/env python
"""Audit diet paired assets before adapting them to CellFlux.

The diet data are much larger than the CRISPR subset, but the experimental
condition is diet state (adlib/fasted/hfd), not sgRNA. This script checks both
sample counts and whether condition is confounded with batch before any training
code treats adlib as a control distribution.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIET_ROOT = PROJECT_ROOT / "data/raw/diet"
MANIFEST = DIET_ROOT / "manifest.parquet"
PROTEIN_H5AD = DIET_ROOT / "protein.h5ad"
PANEL = ["Perilipin", "Calreticulin", "pS6RP"]


def _dense(x, rows: np.ndarray) -> np.ndarray:
    sub = x[np.asarray(rows, dtype=int)]
    return sub.toarray() if sp.issparse(sub) else np.asarray(sub)


def main() -> None:
    manifest = pd.read_parquet(MANIFEST)
    print(f"manifest: {MANIFEST}")
    print(f"rows: {len(manifest):,}")
    print("\ncounts by split/condition:")
    print(manifest.groupby(["split", "cond"], observed=True).size().to_string())

    print("\ncounts by batch/condition:")
    batch_cond = manifest.groupby(["batch", "cond"], observed=True).size().unstack(fill_value=0)
    print(batch_cond.to_string())
    mixed_batches = (batch_cond.gt(0).sum(axis=1) > 1).sum()
    if mixed_batches == 0:
        print("\nWARNING: no batch contains multiple diet conditions; condition and batch are confounded.")
        print("Do not claim same-batch control calibration for adlib -> fasted/hfd CellFlux runs.")

    protein = ad.read_h5ad(PROTEIN_H5AD, backed="r")
    channels = list(map(str, protein.var_names))
    panel_idx = [channels.index(name) for name in PANEL]

    train = manifest[manifest["split"].eq("train")]
    adlib_rows = train.loc[train["cond"].eq("adlib"), "protein_index"].to_numpy()
    adlib = _dense(protein.X, adlib_rows)
    mu = np.nanmean(adlib, axis=0)
    sd = np.nanstd(adlib, axis=0) + 1e-6

    rows = []
    for cond in ["fasted", "hfd"]:
        for split in ["train", "val", "test"]:
            idx = manifest.loc[
                manifest["split"].eq(split) & manifest["cond"].eq(cond),
                "protein_index",
            ].to_numpy()
            vals = _dense(protein.X, idx)
            z = (np.nanmean(vals, axis=0) - mu) / sd
            rec = {
                "split": split,
                "cond": cond,
                "n": int(len(idx)),
                "max_absz": float(np.nanmax(np.abs(z))),
                "lead_channel": channels[int(np.nanargmax(np.abs(z)))],
            }
            for name, chan_idx in zip(PANEL, panel_idx):
                rec[f"{name}_z"] = float(z[chan_idx])
                rec[f"{name}_mean"] = float(np.nanmean(vals[:, chan_idx]))
            rows.append(rec)

    print("\nprotein effect vs train adlib baseline:")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\ntop train-channel effects:")
    for cond in ["fasted", "hfd"]:
        idx = manifest.loc[
            manifest["split"].eq("train") & manifest["cond"].eq(cond),
            "protein_index",
        ].to_numpy()
        vals = _dense(protein.X, idx)
        z = (np.nanmean(vals, axis=0) - mu) / sd
        order = np.argsort(np.nan_to_num(np.abs(z), nan=-1.0))[::-1][:10]
        print(f"\n{cond}:")
        for chan_idx in order:
            print(f"  {channels[chan_idx]:14s} z={z[chan_idx]:+.3f}")


if __name__ == "__main__":
    main()

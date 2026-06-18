"""Build CellFlux-format external artifacts for the Perturb-Multi hepatocyte data.

Produces, under data/processed/cellflux_ext/:
  index_lipid_panel.csv / _heldout.csv   CellFlux data index (Hep, TIER1/2 + control)
  perturbation_effects.csv               per-gene morphological |z| (18ch protein) + RNA SNR diagnostic + flags
  channel_effects.csv                    per-channel (18) effect ranking -> the evidence behind the image panel choice
  index_stronghits.csv                   strong-hit subset = morph top-TOP_K & n_cells>=MIN_CELLS_STRONGHIT (+ full control pool)
  (the gene-identity embedding is built separately.)

Modality semantics (see data/processed/cellflux_ext/README.md):
- Perturbation = sgRNA -> target gene (IDENTITY). The model condition is gene identity
  (embedding_gene_identity.csv, one-hot), NOT any RNA readout.
- The 209-gene MERFISH RNA and the 18-ch morphology are measured READOUTS of the imaged
  cell. Here the 209 RNA only yields a per-gene effect-size (SNR) diagnostic; the 18-ch
  protein gives the morphological effect size used for strong-hit selection.
- Image channel panel: panel2 = Perilipin/Calreticulin/pS6RP (npz channels [5,9,10]).
"""

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = "/home/ubuntu/data/sqzhou/projects/morpho-cellflux"
OUT = os.path.join(ROOT, "data/processed/cellflux_ext")
MANIFEST = os.path.join(ROOT, "data/processed/cellflux_manifest.parquet")
VOCAB = os.path.join(ROOT, "data/processed/condition_vocab.json")
RNA_H5AD = os.path.join(ROOT, "data/raw/RNA_crispr_hep_paired.h5ad")
PROT_H5AD = os.path.join(ROOT, "data/raw/protein_crispr_hep_paired.h5ad")

CONTROL_LABEL = "control"
KEEP_DECISIONS = ["TRAIN_EVAL_TIER1", "TRAIN_EVAL_TIER2", "CONTROL"]
MIN_CELLS = 20  # min train cells for a stable per-gene signature
TOP_K = 50      # top morphological-effect genes to flag as the focus set (cf. eval_panel n=50)
LIPID_Z = 0.5   # |Perilipin z| to flag a lipid-droplet hit
MIN_CELLS_STRONGHIT = 80  # min train cells for a gene to enter the strong-hit subset (cell gap: kept>=82, dropped<=70)


def _dense_cols(X, rows, cols=None):
    """Return a dense [len(rows), n_cols] array for given row indices."""
    rows = np.asarray(sorted(set(int(r) for r in rows)))
    sub = X[rows]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return sub if cols is None else sub[:, cols]


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_parquet(MANIFEST)
    vocab = json.load(open(VOCAB))
    pert_genes = [g for g in vocab if g != CONTROL_LABEL]
    print(f"manifest rows={len(m)}  perturbation genes={len(pert_genes)}")

    # ---- 1. index CSVs -------------------------------------------------------
    sel = (m["cell_type"] == "Hep") & (m["decision"].isin(KEEP_DECISIONS))
    s = m.loc[sel].copy()
    s["SAMPLE_KEY"] = s["cell_id"].astype(str)
    s["CPD_NAME"] = s["target_gene"].astype(str)
    s["ANNOT"] = np.where(s["is_control"], "negative_control", "treated")
    s["BATCH"] = s["batch"].astype(str)

    def write_index(df, split_map, path, genes=None):
        # Treated rows follow the split map. Controls are the shared SOURCE
        # distribution used for same-batch pairing (never an eval target), so the
        # FULL control pool is added to every output split -- otherwise a batch
        # whose few controls all landed in one split crashes eval pairing
        # ("No control samples found in the same batch").
        # genes (optional): restrict treated rows to this gene subset (controls
        # are never filtered) -- used to carve the strong-hit index out of the
        # same manifest so it stays consistent with index_lipid_panel.
        cols = ["SAMPLE_KEY", "CPD_NAME", "ANNOT", "BATCH", "SPLIT",
                "sgRNA", "cluster_type", "condition_id"]
        treated = df[(~df["is_control"]) & (df["split"].isin(split_map))].copy()
        if genes is not None:
            treated = treated[treated["CPD_NAME"].isin(set(genes))].copy()
        treated["SPLIT"] = treated["split"].map(split_map)
        controls_all = df[df["is_control"]].copy()
        frames = [treated]
        for out_split in sorted(set(split_map.values())):
            c = controls_all.copy()
            c["SPLIT"] = out_split
            frames.append(c)
        out = pd.concat(frames, ignore_index=True)[cols].reset_index(drop=True)
        out.to_csv(path)  # index_col=0 on read
        return out

    main_idx = write_index(s, {"train": "train", "val": "test"},
                           os.path.join(OUT, "index_lipid_panel.csv"))
    held_idx = write_index(s, {"test": "test"},
                           os.path.join(OUT, "index_lipid_panel_heldout.csv"))
    print(f"index_lipid_panel.csv: {len(main_idx)} rows "
          f"(train={int((main_idx.SPLIT=='train').sum())}, "
          f"test/val={int((main_idx.SPLIT=='test').sum())})")
    print(f"index_lipid_panel_heldout.csv: {len(held_idx)} rows")

    # ---- 2. per-gene RNA effect-size (SNR) for the effects table -------------
    #        Diagnostic only. The 209-gene MERFISH is a READOUT, never a condition;
    #        the perturbation condition is gene IDENTITY (embedding_gene_identity.csv).
    rna = ad.read_h5ad(RNA_H5AD, backed="r")
    Xr = rna.X
    train_m = m[m["split"] == "train"]
    ctrl_rows = train_m.loc[train_m["is_control"], "rna_index"].to_numpy()
    ctrl = _dense_cols(Xr, ctrl_rows)
    ctrl_mean = ctrl.mean(axis=0)
    ctrl_std = ctrl.std(axis=0) + 1e-6
    print(f"RNA: {rna.shape}  control train cells={len(ctrl_rows)}")

    snr = {}
    for g in pert_genes:
        idx = train_m.loc[train_m["target_gene"] == g, "rna_index"].to_numpy()
        if len(idx) < MIN_CELLS:
            idx = m.loc[m["target_gene"] == g, "rna_index"].to_numpy()
        if len(idx) == 0:
            snr[g] = 0.0
            continue
        z = (_dense_cols(Xr, idx).mean(axis=0) - ctrl_mean) / ctrl_std
        snr[g] = float(np.abs(z).max())

    # ---- 3. morphological effect size from 18ch protein ---------------------
    prot = ad.read_h5ad(PROT_H5AD, backed="r")
    prot_ch = list(map(str, prot.var_names))
    Xp = prot.X
    pc_mean = _dense_cols(Xp, train_m.loc[train_m["is_control"], "protein_index"]).mean(axis=0)
    pc_std = _dense_cols(Xp, train_m.loc[train_m["is_control"], "protein_index"]).std(axis=0) + 1e-6
    peri = prot_ch.index("Perilipin") if "Perilipin" in prot_ch else 0

    recs = []
    for g in pert_genes:
        idx = train_m.loc[train_m["target_gene"] == g, "protein_index"].to_numpy()
        if len(idx) == 0:
            idx = m.loc[m["target_gene"] == g, "protein_index"].to_numpy()
        n = len(idx)
        zc = (_dense_cols(Xp, idx).mean(axis=0) - pc_mean) / pc_std if n else np.zeros(len(prot_ch))
        recs.append({
            "target_gene": g,
            "n_cells": int(n),
            "morph_maxabs_z": float(np.abs(zc).max()),
            "morph_l2_z": float(np.linalg.norm(zc)),
            "Perilipin_z": float(zc[peri]),
            "rna_snr": snr.get(g, 0.0),
        })
    eff = pd.DataFrame(recs).sort_values("morph_maxabs_z", ascending=False).reset_index(drop=True)
    eff["morph_rank"] = np.arange(1, len(eff) + 1)
    eff["morph_significant"] = eff["morph_rank"] <= TOP_K  # ranked focus set
    eff["lipid_hit"] = eff["Perilipin_z"].abs() >= LIPID_Z
    eff.to_csv(os.path.join(OUT, "perturbation_effects.csv"), index=False)
    print(f"perturbation_effects.csv: {len(eff)} genes; focus set (top {TOP_K} by "
          f"morph effect): {int(eff['morph_significant'].sum())}; "
          f"lipid hits (|Perilipin z|>={LIPID_Z}): {int(eff['lipid_hit'].sum())}")
    print("\ntop 15 morphological hits:")
    print(eff.head(15)[["target_gene", "n_cells", "morph_maxabs_z",
                         "Perilipin_z", "rna_snr"]].to_string(index=False))
    # lipid genes of interest from the paper
    lipid = ["Insig1", "Pten", "Eif2s1", "Aars", "Srebf1", "Ldlr", "Scd1", "Fasn"]
    print("\npaper lipid genes:")
    print(eff[eff.target_gene.isin(lipid)][["target_gene", "n_cells",
          "morph_maxabs_z", "Perilipin_z", "rna_snr"]].to_string(index=False))

    # ---- 4. per-channel effect ranking (evidence for the image panel choice) -
    #        For each of the 18 protein channels, how strongly is it moved by
    #        perturbation across genes (train cells, >=50/gene)? max|z| can be
    #        inflated by trivial on-target self-knockdown (e.g. Gapdh->Gapdh), so
    #        also report top3-mean and #genes with |z|>0.5 (breadth of response).
    #        npz_ch = channel index into the saved <cell>.npz['x'] (== var order).
    zrows = {}
    for g in pert_genes:
        idx = train_m.loc[train_m["target_gene"] == g, "protein_index"].to_numpy()
        if len(idx) < 50:
            continue
        zrows[g] = (_dense_cols(Xp, idx).mean(axis=0) - pc_mean) / pc_std
    Z = pd.DataFrame(zrows, index=prot_ch).T            # genes x 18
    absZ = Z.abs()
    chan = pd.DataFrame({
        "npz_ch": np.arange(len(prot_ch)),
        "max_absz": absZ.max(axis=0),
        "top3_mean_absz": absZ.apply(lambda c: c.nlargest(3).mean(), axis=0),
        "n_genes_absz_gt_0p5": (absZ > 0.5).sum(axis=0),
        "lead_gene": absZ.idxmax(axis=0),
    }).sort_values("max_absz", ascending=False)
    chan.index.name = "channel"
    chan.to_csv(os.path.join(OUT, "channel_effects.csv"))
    print(f"\nchannel_effects.csv: 18 channels ranked by max|z| over "
          f"{len(zrows)} genes (>=50 train cells). current panel npz[5,9,10]:")
    print(chan.to_string(float_format=lambda x: f"{x:.2f}"))

    # ---- 5. strong-hit subset index ----------------------------------------
    #        SELECTION CRITERION (our heuristic, NOT a list from the paper): genes
    #        flagged morph_significant (top-TOP_K by morph_maxabs_z over all 18
    #        channels) AND with >= MIN_CELLS_STRONGHIT train cells. The paper only
    #        states ~84/406 sgRNAs are morphologically significant (Fig S7H); the
    #        exact gene set here is ours. Carved from the same `s` manifest as
    #        index_lipid_panel (treated rows restricted to the subset; the full
    #        control pool is kept) so the two indices stay consistent.
    stronghit_genes = sorted(eff.loc[eff["morph_significant"]
                             & (eff["n_cells"] >= MIN_CELLS_STRONGHIT),
                             "target_gene"].tolist())
    sh_idx = write_index(s, {"train": "train", "val": "test"},
                         os.path.join(OUT, "index_stronghits.csv"),
                         genes=stronghit_genes)
    print(f"\nindex_stronghits.csv: {len(sh_idx)} rows; "
          f"{len(stronghit_genes)} strong-hit genes "
          f"(morph top-{TOP_K} & n_cells>={MIN_CELLS_STRONGHIT}) + full control pool")
    print(f"  genes: {stronghit_genes}")


if __name__ == "__main__":
    main()

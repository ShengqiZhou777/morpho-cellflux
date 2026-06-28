"""Build CRISPR paper-line artifacts from Perturb-Multi-supported programs.

NOTE: The program-label outputs (index_paper_programs*.csv, program_labels_paper.csv)
were originally consumed by the PCGE module, which has been removed. They are kept
here only as a record of the per-gene functional-program grouping (used for
descriptive program-level reporting); the live training/eval path uses
index_paper_40.csv + a flat gene one-hot, with MSA/PCD as the molecular prior.

This script creates the CRISPR paper-core subset under `data/processed/crispr/`:

  index_paper_programs.csv
      Train + in-loop eval index. Original manifest split `train` stays train;
      original `val` becomes engine split `test`. Treated rows are genes from
      the Perturb-Multi paper programs below. The full control pool is copied
      into every output split for same-batch source pairing.

  index_paper_programs_heldout.csv
      Original manifest split `test` only, plus the full control pool.

  program_labels_paper.csv
      Gene-to-program labels used for paper-level Program-Acc/F1 reporting.

  paper_panel_effects.csv
      Per-gene z-scored marker shifts on the core paper panel:
      Calreticulin / Perilipin / pS6RP = npz channels [9, 5, 10].

  paper_program_summary.csv/json
      Counts and provenance summary for auditing.

The program definitions are taken from the local Perturb-Multi manuscript notes
(`data/raw/Perturb-multimodal.md`) and are deliberately fixed in this script so
the paper story is reproducible.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data/processed/crispr"
MANIFEST = REPO_ROOT / "data/processed/manifest.parquet"
PROT_H5AD = REPO_ROOT / "data/raw/crispr/protein.h5ad"
PAPER_MD = REPO_ROOT / "data/raw/Perturb-multimodal.md"

CONTROL_LABEL = "control"
CORE_PANEL_CHANNELS = [9, 5, 10]
CORE_PANEL_NAMES = ["Calreticulin", "Perilipin", "pS6RP"]
TRAIN_SPLIT = {"train": "train", "val": "test"}
HELDOUT_SPLIT = {"test": "test"}

MAIN_PROGRAMS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        ("steatosis_lipid", ["Insig1", "Pten", "Eif2s1", "Aars"]),
        ("upr_er_stress", ["Sel1l", "Sec61a1", "Dnajb9", "Atp2a2", "Xbp1", "Ern1"]),
        ("isr_translation", ["Nars", "Eif2b4"]),
        ("mtor_ps6", ["Mtor", "Cdc37", "Tsc1"]),
        (
            "lysosome_endomembrane",
            ["Npc1", "Atp6v0c", "Atp6ap1", "Lamtor2", "Dnm2", "Arfrp1", "Jtb", "Zw10"],
        ),
        (
            "zonation_wnt_hypoxia",
            ["Ctnnb1", "Apc", "Vhl", "Lgr4", "Prkar1a", "Hs6st1", "B4galt7", "Zfp830"],
        ),
        (
            "rna_processing_nuclear",
            ["Sf3b6", "Sbno1", "Polr1a", "Kin", "Polr2l", "Gpn1", "Rrn3", "Taf1a", "Ubtf"],
        ),
    ]
)

VALIDATION_PROGRAMS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        ("on_target_validation", ["Alb", "Gapdh"]),
    ]
)

EXCLUDED_LOW_N: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        ("excluded_low_n_optional_mtor_ps6", ["Tsc2"]),
    ]
)


def _dense_cols(X, rows, cols=None):
    rows = np.asarray(sorted(set(int(r) for r in rows)))
    sub = X[rows]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return sub if cols is None else sub[:, cols]


def build_label_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    program_id = {program: i for i, program in enumerate(MAIN_PROGRAMS)}
    for program, genes in MAIN_PROGRAMS.items():
        for gene in genes:
            rows.append(
                {
                    "target_gene": gene,
                    "program": program,
                    "program_id": program_id[program],
                    "source": "Perturb-Multi main text / figure programs",
                    "role": "main_quantitative",
                    "in_manifest": bool((manifest["target_gene"] == gene).any()),
                }
            )
    for program, genes in VALIDATION_PROGRAMS.items():
        for gene in genes:
            rows.append(
                {
                    "target_gene": gene,
                    "program": program,
                    "program_id": -1,
                    "source": "Perturb-Multi on-target validation",
                    "role": "validation_only",
                    "in_manifest": bool((manifest["target_gene"] == gene).any()),
                }
            )
    for program, genes in EXCLUDED_LOW_N.items():
        for gene in genes:
            rows.append(
                {
                    "target_gene": gene,
                    "program": program,
                    "program_id": -1,
                    "source": "Perturb-Multi mTOR program; too few cells locally",
                    "role": "excluded_low_n",
                    "in_manifest": bool((manifest["target_gene"] == gene).any()),
                }
            )
    return pd.DataFrame(rows)


def annotate_index_rows(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_cols = labels[["target_gene", "program", "program_id", "role"]].rename(
        columns={"role": "paper_role"}
    )
    out = df.merge(label_cols, on="target_gene", how="left")
    out["program"] = np.where(out["is_control"], CONTROL_LABEL, out["program"])
    out["program_id"] = np.where(out["is_control"], -1, out["program_id"]).astype(int)
    out["paper_role"] = np.where(out["is_control"], "control_source", out["paper_role"])
    out["SAMPLE_KEY"] = out["cell_id"].astype(str)
    out["CPD_NAME"] = out["target_gene"].astype(str)
    out["ANNOT"] = np.where(out["is_control"], "negative_control", "treated")
    out["BATCH"] = out["batch"].astype(str)
    out["PROGRAM"] = out["program"].astype(str)
    out["PROGRAM_ID"] = out["program_id"].astype(int)
    out["PAPER_ROLE"] = out["paper_role"].astype(str)
    out["TARGET_GENE"] = out["target_gene"].astype(str)
    return out


def write_index(df: pd.DataFrame, split_map: dict[str, str], path: Path) -> pd.DataFrame:
    cols = [
        "SAMPLE_KEY",
        "CPD_NAME",
        "ANNOT",
        "BATCH",
        "SPLIT",
        "PROGRAM",
        "PROGRAM_ID",
        "TARGET_GENE",
        "PAPER_ROLE",
        "sgRNA",
        "cluster_type",
        "condition_id",
    ]
    treated = df[(~df["is_control"]) & (df["split"].isin(split_map))].copy()
    treated["SPLIT"] = treated["split"].map(split_map)

    controls_all = df[df["is_control"]].copy()
    frames = [treated]
    for out_split in sorted(set(split_map.values())):
        controls = controls_all.copy()
        controls["SPLIT"] = out_split
        frames.append(controls)

    out = pd.concat(frames, ignore_index=True)[cols].reset_index(drop=True)
    out.to_csv(path)
    return out


def write_program_summaries(manifest: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_rows = []
    for rec in labels.to_dict("records"):
        gene = rec["target_gene"]
        g = manifest[manifest["target_gene"] == gene]
        split_counts = g["split"].value_counts()
        gene_rows.append(
            {
                "target_gene": gene,
                "program": rec["program"],
                "program_id": rec["program_id"],
                "role": rec["role"],
                "train": int(split_counts.get("train", 0)),
                "val": int(split_counts.get("val", 0)),
                "heldout_test": int(split_counts.get("test", 0)),
                "total": int(len(g)),
                "decision": ";".join(sorted(g["decision"].dropna().astype(str).unique())),
            }
        )
    gene_counts = pd.DataFrame(gene_rows)
    gene_counts.to_csv(OUT / "paper_program_gene_counts.csv", index=False)

    program_rows = []
    for (program, role), g in gene_counts.groupby(["program", "role"], sort=False):
        program_rows.append(
            {
                "program": program,
                "role": role,
                "n_genes": int(g["target_gene"].nunique()),
                "train": int(g["train"].sum()),
                "val": int(g["val"].sum()),
                "heldout_test": int(g["heldout_test"].sum()),
                "total": int(g["total"].sum()),
            }
        )
    program_summary = pd.DataFrame(program_rows)
    program_summary.to_csv(OUT / "paper_program_summary.csv", index=False)
    return gene_counts, program_summary


def write_panel_effects(manifest: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    prot = ad.read_h5ad(PROT_H5AD, backed="r")
    prot_ch = list(map(str, prot.var_names))
    panel_names = [prot_ch[i] for i in CORE_PANEL_CHANNELS]
    if panel_names != CORE_PANEL_NAMES:
        raise ValueError(f"Unexpected panel names for {CORE_PANEL_CHANNELS}: {panel_names}")

    train_m = manifest[manifest["split"] == "train"]
    ctrl_rows = train_m.loc[train_m["is_control"], "protein_index"].to_numpy()
    Xp = prot.X
    ctrl = _dense_cols(Xp, ctrl_rows, CORE_PANEL_CHANNELS)
    ctrl_mean = ctrl.mean(axis=0)
    ctrl_std = ctrl.std(axis=0) + 1e-6

    rows = []
    label_lookup = labels.set_index("target_gene")
    for gene in labels["target_gene"]:
        gtrain = train_m[train_m["target_gene"] == gene]
        idx = gtrain["protein_index"].to_numpy()
        if len(idx) == 0:
            idx = manifest.loc[manifest["target_gene"] == gene, "protein_index"].to_numpy()
        z = np.zeros(len(CORE_PANEL_CHANNELS))
        if len(idx):
            z = (_dense_cols(Xp, idx, CORE_PANEL_CHANNELS).mean(axis=0) - ctrl_mean) / ctrl_std
        meta = label_lookup.loc[gene]
        rec = {
            "target_gene": gene,
            "program": meta["program"],
            "role": meta["role"],
            "n_train_cells": int(len(gtrain)),
            "n_total_cells": int((manifest["target_gene"] == gene).sum()),
            "panel_maxabs_z": float(np.abs(z).max()),
            "panel_lead_channel": panel_names[int(np.abs(z).argmax())],
        }
        for name, zval in zip(panel_names, z):
            rec[f"{name}_z"] = float(zval)
        rows.append(rec)

    effects = pd.DataFrame(rows).sort_values("panel_maxabs_z", ascending=False)
    effects.to_csv(OUT / "paper_panel_effects.csv", index=False)
    return effects


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(MANIFEST)
    if "cell_type" in manifest:
        manifest = manifest[manifest["cell_type"].astype(str) == "Hep"].copy()

    labels = build_label_table(manifest)
    missing = labels.loc[~labels["in_manifest"], "target_gene"].tolist()
    if missing:
        raise ValueError(f"Paper genes missing from manifest: {missing}")

    labels.to_csv(OUT / "program_labels_paper.csv", index=False)

    main_genes = set(labels.loc[labels["role"] == "main_quantitative", "target_gene"])
    selected = manifest[manifest["is_control"] | manifest["target_gene"].isin(main_genes)].copy()
    selected = annotate_index_rows(selected, labels)

    train_idx = write_index(selected, TRAIN_SPLIT, OUT / "index_paper_programs.csv")
    heldout_idx = write_index(selected, HELDOUT_SPLIT, OUT / "index_paper_programs_heldout.csv")
    gene_counts, program_summary = write_program_summaries(manifest, labels)
    panel_effects = write_panel_effects(manifest, labels)

    summary = {
        "source_paper": str(PAPER_MD.relative_to(REPO_ROOT)),
        "core_panel_channels": CORE_PANEL_CHANNELS,
        "core_panel_names": CORE_PANEL_NAMES,
        "train_index": "data/processed/crispr/index_paper_programs.csv",
        "heldout_index": "data/processed/crispr/index_paper_programs_heldout.csv",
        "main_programs": MAIN_PROGRAMS,
        "validation_programs": VALIDATION_PROGRAMS,
        "excluded_low_n": EXCLUDED_LOW_N,
        "counts": {
            "main_genes": int(len(main_genes)),
            "train_index_rows": int(len(train_idx)),
            "train_index_treated_rows": int((train_idx["ANNOT"] == "treated").sum()),
            "heldout_index_rows": int(len(heldout_idx)),
            "heldout_index_treated_rows": int((heldout_idx["ANNOT"] == "treated").sum()),
            "controls_total": int(manifest["is_control"].sum()),
        },
        "files": {
            "labels": "data/processed/crispr/program_labels_paper.csv",
            "program_summary": "data/processed/crispr/paper_program_summary.csv",
            "gene_counts": "data/processed/crispr/paper_program_gene_counts.csv",
            "panel_effects": "data/processed/crispr/paper_panel_effects.csv",
        },
    }
    (OUT / "paper_program_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("Wrote CRISPR paper-line artifacts")
    print(f"main genes: {len(main_genes)}")
    print(
        "index_paper_programs.csv: "
        f"{len(train_idx)} rows, treated={(train_idx['ANNOT'] == 'treated').sum()}"
    )
    print(
        "index_paper_programs_heldout.csv: "
        f"{len(heldout_idx)} rows, treated={(heldout_idx['ANNOT'] == 'treated').sum()}"
    )
    print("\nprogram summary:")
    print(program_summary.to_string(index=False))
    print("\ntop panel effects:")
    cols = ["target_gene", "program", "role", "panel_maxabs_z"] + [f"{name}_z" for name in CORE_PANEL_NAMES]
    print(panel_effects[cols].head(15).to_string(index=False))
    print("\nlow-count/excluded genes:")
    print(gene_counts[gene_counts["role"] != "main_quantitative"].to_string(index=False))


if __name__ == "__main__":
    main()

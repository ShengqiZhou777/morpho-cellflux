"""Build the engine-format data artifacts for the diet (Perturb-Multi diet) hepatocytes.

Diet is a STRONG physiological perturbation (unlike the subtle single-gene CRISPR KOs),
so control->perturbed morphology shifts are large enough to be visible per cell -- the
setting where the interpolation figure can actually look like the CellFlux paper.

Conditions (manifest `cond`): adlib = control, fasted / hfd = treated.

IMPORTANT batch confound: diet condition is fully confounded with imaging batch
(adlib in batches 1,2; fasted in 3,6; hfd in 4,5). The engine pairs each treated cell
with a same-BATCH control, so we COLLAPSE BATCH to a single value (0) -- every treated
cell can then pair with any adlib control. This means the diet effect and the batch
effect are not separable here; that is inherent to the experiment, acceptable for a
generative demo of the morphology shift, and noted in the README.

Produces, under data/processed/diet/:
  index_diet.csv        engine data index (adlib control + fasted/hfd treated, BATCH=0)
  embedding_diet.csv    3x3 one-hot over {adlib, fasted, hfd} (the condition)
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data/processed/diet")
MANIFEST = ("/home/ubuntu/data/sqzhou/projects/morpho-phenotyping/assets/"
            "paired_filtered/diet/manifests/manifest_diet_hep_paired.parquet")

CONTROL = "adlib"
CONDITIONS = ["adlib", "fasted", "hfd"]  # embedding row order


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_parquet(MANIFEST)
    m = m[m["cell_type"] == "Hep"].copy()

    # cell_id is the npz stem (image_member = "<cell_id>.npz")
    s = pd.DataFrame({
        "SAMPLE_KEY": m["cell_id"].astype(str),
        "CPD_NAME": m["cond"].astype(str),
        "ANNOT": np.where(m["cond"] == CONTROL, "negative_control", "treated"),
        "BATCH": 0,  # collapsed: lets treated pair with any adlib control (see header)
        "SPLIT": m["split"].astype(str),
        "sgRNA": m["cond"].astype(str),     # no guides here; mirror schema with cond
        "cluster_type": m["cluster_type"].astype(str),
        "condition_id": m["cond"].map({c: i for i, c in enumerate(CONDITIONS)}),
    })
    print(f"diet manifest Hep rows: {len(s)}")
    print("cond x split:")
    print(pd.crosstab(s["CPD_NAME"], s["SPLIT"]).to_string())

    out_index = os.path.join(OUT, "index_diet.csv")
    s.reset_index(drop=True).to_csv(out_index)
    print(f"\nwrote {out_index}: {len(s)} rows "
          f"({(s.ANNOT=='treated').sum()} treated, {(s.ANNOT=='negative_control').sum()} control)")

    emb = pd.DataFrame(np.eye(len(CONDITIONS)), index=CONDITIONS,
                       columns=[f"id_{c}" for c in CONDITIONS])
    out_emb = os.path.join(OUT, "embedding_diet.csv")
    emb.to_csv(out_emb)
    print(f"wrote {out_emb}: {emb.shape} one-hot over {CONDITIONS}")


if __name__ == "__main__":
    main()

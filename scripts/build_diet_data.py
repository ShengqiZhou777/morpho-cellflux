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
  index_diet.csv          engine training index (adlib control + fasted/hfd treated, BATCH=0).
                          SPLIT = {train->train, val->test}: the engine only reads the
                          train/test folds, so original `val` becomes the in-loop eval fold.
  index_diet_heldout.csv  same cells, original `test` split only -- final held-out eval.
  embedding_diet.csv      3x3 one-hot over {adlib, fasted, hfd} (the condition)
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

# The engine only iterates the 'train' and 'test' folds. Fold the original three-way
# split as: train -> training fold, val -> in-loop eval fold, test -> a separate
# held-out index. This mirrors scripts/build_perturbmulti_data.py so diet and crispr
# share the same split semantics (without this remap the ~27k `val` cells were silently
# dropped, since the loader never reads a 'val' fold).
TRAIN_SPLIT = {"train": "train", "val": "test"}
HELDOUT_SPLIT = {"test": "test"}


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
        "split_orig": m["split"].astype(str),
        "sgRNA": m["cond"].astype(str),     # no guides here; mirror schema with cond
        "cluster_type": m["cluster_type"].astype(str),
        "condition_id": m["cond"].map({c: i for i, c in enumerate(CONDITIONS)}),
    })
    print(f"diet manifest Hep rows: {len(s)}")
    print("cond x split:")
    print(pd.crosstab(s["CPD_NAME"], s["split_orig"]).to_string())

    cols = ["SAMPLE_KEY", "CPD_NAME", "ANNOT", "BATCH", "SPLIT",
            "sgRNA", "cluster_type", "condition_id"]

    def write_index(split_map, path):
        # Apply split_map to every row (treated and adlib control alike). BATCH is
        # collapsed to 0, so within each fold any treated cell can pair with any adlib
        # control present in that fold; each original split already contains adlib.
        d = s[s["split_orig"].isin(split_map)].copy()
        d["SPLIT"] = d["split_orig"].map(split_map)
        out = d[cols].reset_index(drop=True)
        out.to_csv(path)  # index_col=0 on read
        return out

    train_idx = write_index(TRAIN_SPLIT, os.path.join(OUT, "index_diet.csv"))
    held_idx = write_index(HELDOUT_SPLIT, os.path.join(OUT, "index_diet_heldout.csv"))
    print(f"\nindex_diet.csv: {len(train_idx)} rows "
          f"({(train_idx.ANNOT=='treated').sum()} treated, "
          f"{(train_idx.ANNOT=='negative_control').sum()} control); "
          f"train={int((train_idx.SPLIT=='train').sum())}, "
          f"val->test={int((train_idx.SPLIT=='test').sum())}")
    print(f"index_diet_heldout.csv: {len(held_idx)} rows (original test split)")

    emb = pd.DataFrame(np.eye(len(CONDITIONS)), index=CONDITIONS,
                       columns=[f"id_{c}" for c in CONDITIONS])
    out_emb = os.path.join(OUT, "embedding_diet.csv")
    emb.to_csv(out_emb)
    print(f"wrote {out_emb}: {emb.shape} one-hot over {CONDITIONS}")


if __name__ == "__main__":
    main()

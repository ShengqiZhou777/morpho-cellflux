#!/usr/bin/env python
"""Build the CellFlux-style comparison table for one benchmark (docs/EVAL_PROTOCOL.md).

For every method it runs the SAME two tools at the SAME per-condition cap:
  1. baselines/compute_image_metrics.py  -> FIDo / FIDc / KIDo / KIDc
  2. engine/moa/train_moa.py --mode eval  -> MoA acc / macro-F1 / weighted-F1 (vs real ceiling)
then collects the per-method JSONs into a markdown table + a TSV.

Matched-N is enforced by compute_image_metrics (errors if a folder lacks `cap` images), so
the table is apples-to-apples by construction.

Example:
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python baselines/build_comparison_table.py \
      --benchmark diet_v3 --config configs/diet_id_v3.yaml \
      --real-dir outputs/baselines/_data/diet_v3/imagefolder/test \
      --classifier outputs/baselines/moa/diet_v3/classifier.pth \
      --cap 500 --out-dir outputs/baselines/_tables/diet_v3 \
      --methods "copy_control:outputs/baselines/copy_control/diet_v3/fid_samples/epoch-0,phendiff:outputs/baselines/phendiff/diet_v3/fid_samples/epoch-0,proposed_ep9:outputs/diet_id_v3/fid_samples/epoch-9"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--config", required=True, help="config the MoA dataloader/mol2id is built from")
    ap.add_argument("--real-dir", required=True, help="real perturbed imagefolder: <cond>/*.png")
    ap.add_argument("--classifier", required=True, help="trained MoA classifier .pth")
    ap.add_argument("--cap", type=int, default=2500, help="per-condition cap (N = cap * n_conditions)")
    ap.add_argument("--methods", required=True, help="comma list of name:gen_dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-moa", action="store_true", help="image metrics only (e.g. if MoA ceiling ~ chance)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [(m.split(":", 1)[0], m.split(":", 1)[1]) for m in args.methods.split(",")]
    py = sys.executable

    rows = []
    for name, gen_dir in methods:
        print(f"\n=== {name}  ({gen_dir}) ===", flush=True)
        img_json = out_dir / f"{name}.image_metrics.json"
        run([py, str(HERE / "compute_image_metrics.py"),
             "--real-dir", args.real_dir, "--gen-dir", gen_dir,
             "--per-condition-cap", str(args.cap), "--label", name, "--out", str(img_json)])
        row = {"method": name, **json.loads(img_json.read_text())}

        if not args.skip_moa:
            moa_json = out_dir / f"{name}.moa.json"
            run([py, str(REPO / "src/morphoflux/engine/moa/train_moa.py"),
                 "--config_path", args.config, "--mode", "eval",
                 "--ckpt_path", args.classifier, "--img_root_path", gen_dir,
                 "--gen-cap", str(args.cap), "--out_json", str(moa_json)])
            row["moa"] = json.loads(moa_json.read_text())
        rows.append(row)

    # ---- assemble table ----
    def g(r, k):
        return r.get(k, float("nan"))

    hdr = ["method", "FIDo", "FIDc", "KIDo", "KIDc"]
    if not args.skip_moa:
        hdr += ["MoA-Acc", "MoA-MacroF1", "MoA-WeightedF1"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    tsv = ["\t".join(hdr)]
    for r in rows:
        cells = [r["method"], f"{g(r,'fid_o'):.2f}", f"{g(r,'fid_c'):.2f}",
                 f"{g(r,'kid_o'):.4f}", f"{g(r,'kid_c'):.4f}"]
        if not args.skip_moa:
            m = r.get("moa", {})
            cells += [f"{m.get('moa_acc', float('nan')):.2f}",
                      f"{m.get('macro_f1', float('nan')):.4f}",
                      f"{m.get('weighted_f1', float('nan')):.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
        tsv.append("\t".join(cells))

    n_cond = len(rows[0]["conditions"])
    table_md = (
        f"# {args.benchmark} comparison (per-condition cap={args.cap}, N={args.cap * n_cond})\n\n"
        + "\n".join(lines) + "\n"
    )
    (out_dir / "comparison_table.md").write_text(table_md)
    (out_dir / "comparison_table.tsv").write_text("\n".join(tsv) + "\n")
    print("\n" + table_md)
    print(f"-> {out_dir}/comparison_table.md (+ .tsv)")


if __name__ == "__main__":
    main()

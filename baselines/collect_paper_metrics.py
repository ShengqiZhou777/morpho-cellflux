"""Collect aggregate-eval summaries into paper-table CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_RUNS = {
    "diet": {
        "CellFlux-shared": "outputs/diet_id_v1",
        "Morpho-CellFlux": "outputs/diet_id_v3",
        "PhenDiff": "outputs/baselines/phendiff/diet_v3",
        "IMPA": "outputs/baselines/impa/diet_v3",
        "MorphoDiff": "outputs/baselines/morphodiff/diet_v3",
        "StarGAN": "outputs/baselines/stargan/diet_v3",
    },
    "crispr": {
        "CellFlux-shared": "outputs/cellflux_pm_train_id_v7",
        "Morpho-CellFlux": "outputs/cellflux_pm_train_id_v8",
        "PhenDiff": "outputs/baselines/phendiff/crispr_v8",
        "IMPA": "outputs/baselines/impa/crispr_v8",
        "MorphoDiff": "outputs/baselines/morphodiff/crispr_v8",
        "StarGAN": "outputs/baselines/stargan/crispr_v8",
    },
}


def read_summary(run_dir: Path) -> dict | None:
    path = run_dir / "aggregate_eval_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def diet_rows(runs: dict[str, str]) -> list[dict]:
    rows = []
    for method, rel in runs.items():
        summary = read_summary(REPO_ROOT / rel)
        if summary is None or "per_condition_dist" not in summary:
            rows.append({"method": method, "run": rel, "status": "missing"})
            continue
        rec = {"method": method, "run": rel, "status": "ok", "epoch": summary.get("epoch")}
        hfd_vals = []
        for condition, by_channel in summary["per_condition_dist"].items():
            vals = []
            for channel, metrics in by_channel.items():
                key = f"{condition}_{channel}_gap_wd"
                rec[key] = metrics.get("gap_closed_wd")
                vals.append(metrics.get("gap_closed_wd"))
                if condition == "hfd":
                    hfd_vals.append(metrics.get("gap_closed_wd"))
            rec[f"{condition}_mean_gap_wd"] = _mean(vals)
        rec["hfd_mean_gap_wd"] = _mean(hfd_vals)
        rows.append(rec)
    return rows


def crispr_rows(runs: dict[str, str]) -> list[dict]:
    rows = []
    for method, rel in runs.items():
        summary = read_summary(REPO_ROOT / rel)
        if summary is None or "dist_pooled" not in summary:
            rows.append({"method": method, "run": rel, "status": "missing"})
            continue
        rec = {"method": method, "run": rel, "status": "ok", "epoch": summary.get("epoch")}
        for channel, metrics in summary.get("dist_pooled", {}).items():
            rec[f"{channel}_gap_wd"] = metrics.get("gap_closed_wd")
        for channel, metrics in summary.get("full", {}).items():
            rec[f"{channel}_dir_corr"] = metrics.get("dir_corr")
            rec[f"{channel}_sign_agree"] = metrics.get("sign_agree")
            rec[f"{channel}_pearson"] = metrics.get("pearson")
        rows.append(rec)
    return rows


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/baselines/paper_tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = REPO_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    diet = pd.DataFrame(diet_rows(DEFAULT_RUNS["diet"]))
    crispr = pd.DataFrame(crispr_rows(DEFAULT_RUNS["crispr"]))
    diet.to_csv(out / "diet_method_comparison.csv", index=False)
    crispr.to_csv(out / "crispr_method_comparison.csv", index=False)
    print(f"wrote {out / 'diet_method_comparison.csv'}")
    print(f"wrote {out / 'crispr_method_comparison.csv'}")


if __name__ == "__main__":
    main()

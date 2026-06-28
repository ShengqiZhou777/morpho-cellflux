"""Collect aggregate-eval summaries into paper-table CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_RUNS = {
    "diet": {
        "StarGAN": "outputs/baselines/stargan/diet",
        "PhenDiff": [
            "outputs/baselines/phendiff/diet",
            "outputs/baselines/phendiff/diet_v3",
        ],
        "IMPA": [
            "outputs/baselines/impa/diet",
            "outputs/baselines/impa/diet_v3",
        ],
        "CellFlux": "outputs/runs/diet/cellflux_diet_5k_v1",
        "MorphoDiff": "outputs/baselines/morphodiff/diet",
        "MSA+PCD": "outputs/runs/diet/diet_id_msa_pcd_5k_v1",
    },
    "crispr_paper": {
        "StarGAN": "outputs/baselines/stargan/crispr_paper",
        "PhenDiff": "outputs/baselines/phendiff/crispr_paper",
        "IMPA": "outputs/baselines/impa/crispr_paper",
        "CellFlux": "outputs/runs/crispr/perturbmulti_id_v1",
        "MorphoDiff": "outputs/baselines/morphodiff/crispr_paper",
        "PhenoFlux": "outputs/runs/crispr/phenoflux_crispr_v2",
    },
}


def resolve_run(spec: str | list[str]) -> tuple[Path, str]:
    candidates = [spec] if isinstance(spec, str) else spec
    for rel in candidates:
        run_dir = REPO_ROOT / rel
        if (run_dir / "aggregate_eval_summary.json").exists():
            return run_dir, rel
    rel = candidates[0]
    return REPO_ROOT / rel, rel


def read_summary(run_dir: Path) -> dict | None:
    path = run_dir / "aggregate_eval_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def diet_rows(runs: dict[str, str]) -> list[dict]:
    rows = []
    for method, spec in runs.items():
        run_dir, rel = resolve_run(spec)
        summary = read_summary(run_dir)
        if summary is None or "per_condition_dist" not in summary:
            rows.append({"method": method, "run": rel, "status": "missing"})
            continue
        rec = {"method": method, "run": rel, "status": "ok", "epoch": summary.get("epoch")}
        hfd_vals = []
        for condition, by_channel in summary["per_condition_dist"].items():
            vals = []
            for channel, metrics in by_channel.items():
                key = f"{condition}_{channel}_gap_wd"
                rec[key] = metrics.get("pgc_wd")
                vals.append(metrics.get("pgc_wd"))
                if condition == "hfd":
                    hfd_vals.append(metrics.get("pgc_wd"))
            rec[f"{condition}_mean_gap_wd"] = _mean(vals)
        rec["hfd_mean_gap_wd"] = _mean(hfd_vals)
        rows.append(rec)
    return rows


def crispr_rows(runs: dict[str, str]) -> list[dict]:
    rows = []
    for method, spec in runs.items():
        run_dir, rel = resolve_run(spec)
        summary = read_summary(run_dir)
        if summary is None or "dist_pooled" not in summary:
            rows.append({"method": method, "run": rel, "status": "missing"})
            continue
        rec = {"method": method, "run": rel, "status": "ok", "epoch": summary.get("epoch")}
        for channel, metrics in summary.get("dist_pooled", {}).items():
            rec[f"{channel}_gap_wd"] = metrics.get("pgc_wd")
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
    crispr_paper = pd.DataFrame(crispr_rows(DEFAULT_RUNS["crispr_paper"]))
    diet.to_csv(out / "diet_method_comparison.csv", index=False)
    crispr_paper.to_csv(out / "crispr_paper_method_comparison.csv", index=False)
    print(f"wrote {out / 'diet_method_comparison.csv'}")
    print(f"wrote {out / 'crispr_paper_method_comparison.csv'}")


if __name__ == "__main__":
    main()

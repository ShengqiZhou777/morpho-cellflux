from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


RAW_LINK_NAMES = {
    "manifest": "manifest_crispr_hep_paired.parquet",
    "extracted_images": "extracted_images",
    "rna_h5ad": "RNA_crispr_hep_paired.h5ad",
    "protein_h5ad": "protein_crispr_hep_paired.h5ad",
    "decision_table": "v6_decision_table.csv",
    "eval_panel": "eval_panel.json",
    "paper": "Perturb-multimodal.md",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _expand_path(path: str | Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if "$" in expanded:
        raise OSError(
            f"Unresolved environment variable in path {path!r}. "
            "Set the variable or replace the path in the config."
        )
    return Path(expanded)


def _resolve(project_root: Path, path: str | Path) -> Path:
    p = _expand_path(path)
    return p if p.is_absolute() else project_root / p


@dataclass(frozen=True)
class FactoryPaths:
    project_root: Path
    source_root: Path
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path


class DataFactory:
    """Materialize CellFlux-ready tables from Perturb-Multimodal assets."""

    def __init__(self, config: dict[str, Any], project_root: str | Path):
        self.config = config
        root = Path(project_root).resolve()
        paths = config["paths"]
        self.paths = FactoryPaths(
            project_root=root,
            source_root=_expand_path(paths["source_root"]).resolve(),
            raw_dir=_resolve(root, paths["raw_dir"]),
            processed_dir=_resolve(root, paths["processed_dir"]),
            reports_dir=_resolve(root, paths["reports_dir"]),
        )

    def source_asset_path(self, key: str) -> Path:
        rel = self.config["source_assets"][key]
        return self.paths.source_root / rel

    def output_path(self, key: str) -> Path:
        return _resolve(self.paths.project_root, self.config["outputs"][key])

    def ensure_dirs(self) -> None:
        self.paths.raw_dir.mkdir(parents=True, exist_ok=True)
        self.paths.processed_dir.mkdir(parents=True, exist_ok=True)
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        self.output_path("pairs_dir").mkdir(parents=True, exist_ok=True)

    def ensure_raw_links(self) -> dict[str, dict[str, str]]:
        self.ensure_dirs()
        report: dict[str, dict[str, str]] = {}
        for key, link_name in RAW_LINK_NAMES.items():
            src = self.source_asset_path(key)
            dst = self.paths.raw_dir / link_name
            if not src.exists():
                raise FileNotFoundError(f"Source asset does not exist: {src}")

            status = "created"
            if dst.exists() or dst.is_symlink():
                if dst.is_symlink() and dst.resolve() == src.resolve():
                    status = "exists"
                else:
                    raise FileExistsError(
                        f"Refusing to overwrite existing raw asset path: {dst}"
                    )
            else:
                os.symlink(src, dst, target_is_directory=src.is_dir())

            report[key] = {"source": str(src), "link": str(dst), "status": status}
        return report

    def load_manifest(self) -> pd.DataFrame:
        return pd.read_parquet(self.source_asset_path("manifest"))

    def load_decision_table(self) -> pd.DataFrame:
        path = self.source_asset_path("decision_table")
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def build_manifest(self, raw: pd.DataFrame) -> pd.DataFrame:
        data_cfg = self.config["data"]
        df = raw.copy()

        cell_type = data_cfg.get("cell_type_filter")
        if cell_type:
            df = df[df["cell_type"].astype(str) == str(cell_type)].copy()

        keep_cols = [
            "cache_index",
            "cell_id",
            "image_member",
            "rna_index",
            "protein_index",
            "condition",
            "is_control",
            "guide_type",
            "sgRNA",
            "target_gene",
            "batch",
            "cell_type",
            "cluster_type",
            "split",
            "fov",
            "x",
            "y",
            "global_x",
            "global_y",
            "area",
            "bc1",
            "bc3",
            "n_thresh1",
            "n_thresh3",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols].copy()

        df["cell_id"] = df["cell_id"].astype(str)
        df["batch"] = df["batch"].astype(str)
        df["cluster_type"] = df["cluster_type"].astype(str)
        df["image_relpath"] = "data/raw/extracted_images/" + df["image_member"].astype(str)
        df["role"] = np.where(df["is_control"], "control_source", "perturbed_target")
        condition_col = data_cfg["condition_column"]
        df["condition_key"] = np.where(df["is_control"], "control", df[condition_col])

        decision = self.load_decision_table()
        if not decision.empty and "target_gene" in decision.columns:
            decision_cols = [
                c
                for c in [
                    "target_gene",
                    "n_cells_hep",
                    "n_sgRNAs",
                    "cells_per_sgRNA_min",
                    "cells_per_sgRNA_median",
                    "cells_per_sgRNA_max",
                    "decision",
                ]
                if c in decision.columns
            ]
            df = df.merge(decision[decision_cols], on="target_gene", how="left")

        vocab = self.build_condition_vocab(df)
        df["condition_id"] = df["condition_key"].map(vocab).fillna(-1).astype(int)
        return df

    def build_condition_vocab(self, manifest: pd.DataFrame) -> dict[str, int]:
        targets = sorted(
            str(v)
            for v in manifest.loc[~manifest["is_control"], "condition_key"].dropna().unique()
        )
        vocab = {"__null__": 0}
        vocab.update({name: i + 1 for i, name in enumerate(targets)})
        return vocab

    def write_condition_vocab(self, manifest: pd.DataFrame) -> dict[str, int]:
        vocab = self.build_condition_vocab(manifest)
        out = self.output_path("condition_vocab")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(vocab, f, indent=2, sort_keys=True)
        return vocab

    def build_pairs(self, manifest: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
        data_cfg = self.config["data"]
        split_col = data_cfg["split_column"]
        match_cols = list(data_cfg["match_columns"])
        min_controls = int(data_cfg["min_controls_per_stratum"])
        pairs_per_target = int(data_cfg.get("pairs_per_target", 1))
        same_split = data_cfg.get("control_split_policy", "same_split") == "same_split"
        rng = np.random.default_rng(int(data_cfg.get("random_seed", 17)))

        controls_all = manifest[manifest["is_control"]].copy()
        targets_all = manifest[~manifest["is_control"]].copy()
        vocab = self.build_condition_vocab(manifest)

        pair_tables: dict[str, pd.DataFrame] = {}
        audit: dict[str, Any] = {
            "pair_strategy": {
                "match_columns": match_cols,
                "control_split_policy": data_cfg.get("control_split_policy", "same_split"),
                "min_controls_per_stratum": min_controls,
                "pairs_per_target": pairs_per_target,
            },
            "splits": {},
        }

        for split, targets in targets_all.groupby(split_col, sort=True):
            controls = (
                controls_all[controls_all[split_col] == split].copy()
                if same_split
                else controls_all.copy()
            )
            controls = controls.reset_index(drop=True)
            targets = targets.reset_index(drop=True)

            group_indices: dict[tuple[Any, ...], np.ndarray] = {}
            for key, idx in controls.groupby(match_cols, sort=False).indices.items():
                if not isinstance(key, tuple):
                    key = (key,)
                group_indices[key] = np.asarray(idx)

            rows: list[dict[str, Any]] = []
            dropped = 0
            dropped_by_reason = {"too_few_controls": 0}
            for target in targets.to_dict("records"):
                key = tuple(target[c] for c in match_cols)
                pool = group_indices.get(key)
                pool_size = 0 if pool is None else int(len(pool))
                if pool_size < min_controls:
                    dropped += 1
                    dropped_by_reason["too_few_controls"] += 1
                    continue

                for repeat_idx in range(pairs_per_target):
                    source = controls.iloc[int(rng.choice(pool))]
                    rows.append(
                        {
                            "pair_id": f"{split}:{len(rows)}",
                            "split": split,
                            "pair_repeat": repeat_idx,
                            "pair_strategy": "same_" + "_".join(match_cols),
                            "condition_key": target["condition_key"],
                            "condition_id": vocab.get(str(target["condition_key"]), -1),
                            "target_gene": target["target_gene"],
                            "sgRNA": target["sgRNA"],
                            "batch": target["batch"],
                            "cluster_type": target["cluster_type"],
                            "source_cell_id": source["cell_id"],
                            "target_cell_id": target["cell_id"],
                            "source_cache_index": int(source["cache_index"]),
                            "target_cache_index": int(target["cache_index"]),
                            "source_rna_index": int(source["rna_index"]),
                            "target_rna_index": int(target["rna_index"]),
                            "source_protein_index": int(source["protein_index"]),
                            "target_protein_index": int(target["protein_index"]),
                            "source_image_member": source["image_member"],
                            "target_image_member": target["image_member"],
                            "source_image_relpath": source["image_relpath"],
                            "target_image_relpath": target["image_relpath"],
                            "source_fov": int(source["fov"]),
                            "target_fov": int(target["fov"]),
                            "source_global_x": float(source["global_x"]),
                            "source_global_y": float(source["global_y"]),
                            "target_global_x": float(target["global_x"]),
                            "target_global_y": float(target["global_y"]),
                            "source_area": float(source["area"]),
                            "target_area": float(target["area"]),
                            "control_pool_size": pool_size,
                        }
                    )

            pairs = pd.DataFrame(rows)
            pair_tables[str(split)] = pairs
            audit["splits"][str(split)] = {
                "n_targets": int(len(targets)),
                "n_controls": int(len(controls)),
                "n_pairs": int(len(pairs)),
                "n_dropped_targets": int(dropped),
                "drop_rate": float(dropped / max(len(targets), 1)),
                "dropped_by_reason": dropped_by_reason,
            }

        return pair_tables, audit

    def build_audit(
        self,
        raw: pd.DataFrame,
        manifest: pd.DataFrame,
        pair_audit: dict[str, Any],
        link_report: dict[str, Any],
    ) -> dict[str, Any]:
        controls = manifest[manifest["is_control"]]
        targets = manifest[~manifest["is_control"]]
        match_cols = list(self.config["data"]["match_columns"])

        stratum_counts = (
            controls.groupby(match_cols).size().rename("n_controls").reset_index()
            if len(controls)
            else pd.DataFrame(columns=match_cols + ["n_controls"])
        )
        target_counts = targets["target_gene"].value_counts()

        audit = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(self.paths.project_root),
            "source_root": str(self.paths.source_root),
            "raw_links": link_report,
            "raw_manifest_rows": int(len(raw)),
            "manifest_rows": int(len(manifest)),
            "counts": {
                "controls": int(len(controls)),
                "targets": int(len(targets)),
                "target_genes": int(targets["target_gene"].nunique()),
                "sgRNAs": int(manifest["sgRNA"].nunique()),
                "batches": int(manifest["batch"].nunique()),
                "clusters": int(manifest["cluster_type"].nunique()),
            },
            "split_x_control": pd.crosstab(manifest["split"], manifest["is_control"]).to_dict(),
            "batch_x_control": pd.crosstab(manifest["batch"], manifest["is_control"]).to_dict(),
            "control_strata": {
                "n_strata": int(len(stratum_counts)),
                "min": int(stratum_counts["n_controls"].min()) if len(stratum_counts) else 0,
                "median": float(stratum_counts["n_controls"].median()) if len(stratum_counts) else 0.0,
                "max": int(stratum_counts["n_controls"].max()) if len(stratum_counts) else 0,
            },
            "target_gene_count_summary": {
                "min": int(target_counts.min()) if len(target_counts) else 0,
                "median": float(target_counts.median()) if len(target_counts) else 0.0,
                "max": int(target_counts.max()) if len(target_counts) else 0,
                "genes_lt_100_cells": int((target_counts < 100).sum()),
                "genes_ge_200_cells": int((target_counts >= 200).sum()),
                "genes_ge_500_cells": int((target_counts >= 500).sum()),
            },
            "pairs": pair_audit,
        }
        return _json_safe(audit)

    def materialize(self, make_links: bool = True) -> dict[str, Any]:
        self.ensure_dirs()
        link_report = self.ensure_raw_links() if make_links else {}

        raw = self.load_manifest()
        manifest = self.build_manifest(raw)
        manifest_path = self.output_path("manifest")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_parquet(manifest_path, index=False)

        vocab = self.write_condition_vocab(manifest)

        pair_tables, pair_audit = self.build_pairs(manifest)
        pairs_dir = self.output_path("pairs_dir")
        pair_paths: dict[str, str] = {}
        for split, pairs in pair_tables.items():
            out = pairs_dir / f"{split}_pairs.parquet"
            pairs.to_parquet(out, index=False)
            pair_paths[split] = str(out)

        audit = self.build_audit(raw, manifest, pair_audit, link_report)
        report_path = self.output_path("audit_report")
        audit["outputs"] = {
            "manifest": str(manifest_path),
            "condition_vocab": str(self.output_path("condition_vocab")),
            "pairs": pair_paths,
            "audit_report": str(report_path),
            "n_conditions": len(vocab),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w") as f:
            json.dump(audit, f, indent=2, sort_keys=True)

        return audit

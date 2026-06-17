from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CellFluxPairDataset(Dataset):
    """PyTorch dataset backed by materialized CellFlux pair tables."""

    def __init__(
        self,
        pairs_path: str | Path,
        project_root: str | Path | None = None,
        image_key: str = "x",
        return_onehot: bool = False,
        num_conditions: int | None = None,
    ):
        self.pairs_path = Path(pairs_path)
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else self.pairs_path.resolve().parents[3]
        )
        self.image_key = image_key
        self.return_onehot = return_onehot
        self.num_conditions = num_conditions
        self.pairs = pd.read_parquet(self.pairs_path)

        if return_onehot and num_conditions is None:
            max_id = int(self.pairs["condition_id"].max())
            self.num_conditions = max_id + 1

    def __len__(self) -> int:
        return int(len(self.pairs))

    def _load_image(self, relpath: str) -> torch.Tensor:
        path = self.project_root / relpath
        arr = np.load(path)[self.image_key].astype(np.float32, copy=False)
        return torch.from_numpy(arr)

    def _condition(self, condition_id: int) -> torch.Tensor:
        if not self.return_onehot:
            return torch.tensor(condition_id, dtype=torch.long)
        vec = torch.zeros(int(self.num_conditions), dtype=torch.float32)
        vec[int(condition_id)] = 1.0
        return vec

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.pairs.iloc[int(idx)]
        source = self._load_image(row["source_image_relpath"])
        target = self._load_image(row["target_image_relpath"])
        condition = self._condition(int(row["condition_id"]))
        meta = {
            "pair_id": row["pair_id"],
            "condition_key": row["condition_key"],
            "target_gene": row["target_gene"],
            "sgRNA": row["sgRNA"],
            "batch": row["batch"],
            "cluster_type": row["cluster_type"],
            "source_cell_id": row["source_cell_id"],
            "target_cell_id": row["target_cell_id"],
        }
        return source, target, condition, meta

import json
from pathlib import Path

import numpy as np

from tsfm.data.synthetic import series, standardize


class RealData:
    """GiftEvalPretrain subset stored by scripts/prepare_real.py, sampled with dataset rebalancing."""

    def __init__(self, root, split="train", temperature=0.5):
        root = Path(root)
        self.datasets = json.loads((root / "datasets.json").read_text())
        index = np.load(root / "index.npy")
        self.index = index[index["val"] == (split == "val")]
        self.values = np.memmap(root / "values.f32", dtype=np.float32, mode="r")
        self.by_dataset = [np.flatnonzero(self.index["dataset"] == d) for d in range(len(self.datasets))]
        multi = (self.index["n_target"] > 1) | (self.index["n_past"] > 0)
        self.multi_by_dataset = [ids[multi[ids]] for ids in self.by_dataset]
        self.weights = self._weights(self.by_dataset, temperature)
        self.multi_weights = self._weights(self.multi_by_dataset, temperature)

    @staticmethod
    def _weights(groups, temperature):
        w = np.array([len(g) for g in groups], dtype=np.float64) ** temperature
        return w / w.sum()

    def item(self, i):
        row = self.index[i]
        n_rows = row["n_target"] + row["n_past"]
        data = np.asarray(self.values[row["offset"] : row["offset"] + n_rows * row["length"]]).reshape(n_rows, -1)
        return series(data[: row["n_target"]], data[row["n_target"] :])

    def sample(self, rng, multivariate=False):
        groups, weights = (self.multi_by_dataset, self.multi_weights) if multivariate else (self.by_dataset, self.weights)
        ids = groups[rng.choice(len(groups), p=weights)]
        return self.item(ids[rng.integers(len(ids))])

    def mixup(self, rng, length, max_components=3, alpha=1.5):
        """TSMixup (Chronos): convex combination of standardized univariate windows from different series."""
        parts, n_parts = [], rng.integers(1, max_components + 1)
        while len(parts) < n_parts:
            target = self.sample(rng)["target"]
            row = target[rng.integers(len(target))]
            start = rng.integers(max(1, len(row) - length + 1))
            window = row[start : start + length]
            if (~np.isnan(window)).sum() > 1:
                parts.append(np.where(np.isnan(window), np.nanmean(window), window))
        n = min(len(p) for p in parts)
        parts = standardize(np.stack([p[-n:] for p in parts]))
        return series(rng.dirichlet(alpha * np.ones(len(parts))) @ parts)

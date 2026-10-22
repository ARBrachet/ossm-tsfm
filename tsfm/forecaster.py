import math

import numpy as np
import torch
import yaml

from tsfm.model.heads import FUTURE, PAST, TARGET
from tsfm.model.model import TSFM


class Forecaster:
    """Zero-shot inference: items are dicts with `target` [n_t, T], optional `past` [n_p, T], `future` [n_f, T+H]."""

    def __init__(self, model, max_context=2048, batch_tokens=16384, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.patch = model.patch_size
        self.max_context = max_context
        self.batch_tokens = batch_tokens
        self.levels = model.quantiles.cpu().numpy()

    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        config = torch.load(path, map_location="cpu")["config"]
        kwargs.setdefault("max_context", max(p["window"] for p in config["curriculum"]))
        return cls(TSFM.from_checkpoint(path), **kwargs)

    def predict(self, items, horizon, quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
        """Returns one array [n_target, horizon, len(quantile_levels)] per item."""
        h_patches = math.ceil(horizon / self.patch)
        lengths = [min(np.asarray(it["target"]).shape[-1], self.max_context) for it in items]
        chunks, chunk, rows = [], [], 0
        for i in np.argsort(lengths):
            n, cost = self._n_rows(items[i]), math.ceil(lengths[i] / self.patch) + h_patches
            if chunk and (rows + n) * cost > self.batch_tokens:
                chunks, chunk, rows = chunks + [chunk], [], 0
            chunk, rows = chunk + [i], rows + n
        chunks.append(chunk)

        out = [None] * len(items)
        for chunk in chunks:
            ctx_patches = math.ceil(max(lengths[i] for i in chunk) / self.patch)
            for i, q in zip(chunk, self._predict_batch([items[i] for i in chunk], horizon, ctx_patches, h_patches)):
                out[i] = self._interpolate(q, quantile_levels)
        return out

    @staticmethod
    def _n_rows(item):
        return sum(len(np.atleast_2d(item[k])) for k in ("target", "past", "future") if k in item and len(item[k]))

    def _predict_batch(self, items, horizon, ctx_patches, h_patches):
        p, ctx = self.patch, ctx_patches * self.patch
        window = ctx + h_patches * p
        rows, roles, n_targets = [], [], []
        for it in items:
            target = np.atleast_2d(np.asarray(it["target"], dtype=np.float32))
            past = np.atleast_2d(np.asarray(it.get("past", np.zeros((0, target.shape[1]))), dtype=np.float32))
            future = np.atleast_2d(np.asarray(it.get("future", np.zeros((0, target.shape[1] + horizon))), dtype=np.float32))
            t = target.shape[1]
            block = np.full((len(target) + len(past) + len(future), window), np.nan, dtype=np.float32)
            keep = min(t, ctx)
            block[: len(target), ctx - keep : ctx] = target[:, t - keep :]
            block[len(target) : len(target) + len(past), ctx - keep : ctx] = past[:, t - keep :]
            if len(future):
                f = future[:, t - keep : t + horizon]
                block[len(target) + len(past) :, ctx - keep : ctx - keep + f.shape[1]] = f
            rows.append(block)
            roles.append([TARGET] * len(target) + [PAST] * len(past) + [FUTURE] * len(future))
            n_targets.append(len(target))
        v_max = max(len(r) for r in roles)
        values = np.full((len(items), v_max, window), np.nan, dtype=np.float32)
        role = np.zeros((len(items), v_max), dtype=np.int64)
        for i, (block, r) in enumerate(zip(rows, roles)):
            values[i, : len(r)], role[i, : len(r)] = block, r
        observed = ~np.isnan(values)
        to = lambda a: torch.from_numpy(a).to(self.device)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            q = self.model.forecast(to(np.nan_to_num(values)), to(observed), to(role), ctx_patches, h_patches)
        q = q.float().cpu().numpy()[:, :, :horizon]
        return [q[i, : n_targets[i]] for i in range(len(items))]

    def _interpolate(self, q, levels):
        levels = np.clip(np.asarray(levels), self.levels[0], self.levels[-1])
        idx = np.clip(np.searchsorted(self.levels, levels) - 1, 0, len(self.levels) - 2)
        lo, hi = self.levels[idx], self.levels[idx + 1]
        w = (levels - lo) / (hi - lo)
        return q[..., idx] * (1 - w) + q[..., idx + 1] * w


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

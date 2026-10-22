from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from tsfm.data import synthetic as syn
from tsfm.data.real import RealData
from tsfm.model.heads import FUTURE, PAST, TARGET


@dataclass
class Phase:
    until_step: int
    window: int
    max_horizon_patches: int
    p_multivariate: float


class Curriculum:
    """Training gets harder over time: longer windows, longer blanked horizons, more multivariate tasks."""

    def __init__(self, phases):
        self.phases = [Phase(**p) for p in phases]

    def __call__(self, step):
        return next((p for p in self.phases if step < p.until_step), self.phases[-1])


def build_task(rng, s, window, patch, max_horizon_patches, max_variates, cpm_prob, cpm_max_patches, univariate):
    """Crop / left-pad a series to `window`, mask the trailing horizon and random context spans.

    Returns values, observed, visible [V, window] and role [V], or None when the series is too short.
    """
    target, past, future = s["target"], s["past"], s["future"]
    if univariate:
        target, past, future = target[rng.integers(len(target)) : None][:1], past[:0], future[:0]
    n_t = min(len(target), max_variates)
    n_cov = min(len(past) + len(future), max_variates - n_t)
    cov_ids = rng.permutation(len(past) + len(future))[:n_cov]
    rows = np.concatenate([target[rng.permutation(len(target))[:n_t]], np.concatenate([past, future])[cov_ids]])
    role = np.array([TARGET] * n_t + [PAST if i < len(past) else FUTURE for i in cov_ids], dtype=np.int64)

    length = rows.shape[1]
    if length > window:
        start = rng.integers(length - window + 1)
        rows, length = rows[:, start : start + window], window
    n_patches = length // patch
    if n_patches < 2:
        return None
    values = np.full((len(rows), window), np.nan, dtype=np.float32)
    values[:, window - n_patches * patch :] = rows[:, length - n_patches * patch :]

    if rng.random() < 0.1:
        for v in np.flatnonzero(role == TARGET):
            gap_start, gap = rng.integers(window), rng.integers(1, 4 * patch)
            values[v, gap_start : gap_start + gap] = np.nan
    observed = ~np.isnan(values)
    if not observed[role == TARGET].any():
        return None

    visible = observed.copy()
    horizon = rng.integers(1, min(max_horizon_patches, n_patches - 1) + 1)
    context_end = window - horizon * patch
    visible[role != FUTURE, context_end:] = False

    first_patch = (window - n_patches * patch) // patch
    context_patches = context_end // patch
    for v in np.flatnonzero(role == TARGET):
        if rng.random() < cpm_prob and context_patches - first_patch > 2:
            span = rng.integers(1, min(cpm_max_patches, context_patches - first_patch - 1) + 1)
            start = rng.integers(first_patch + 1, context_patches - span + 1)
            visible[v, start * patch : (start + span) * patch] = False
    return np.nan_to_num(values), observed, visible, role


def collate(tasks):
    v_max = max(len(t[3]) for t in tasks)
    b, window = len(tasks), tasks[0][0].shape[1]
    values = np.zeros((b, v_max, window), np.float32)
    observed = np.zeros((b, v_max, window), bool)
    visible = np.zeros((b, v_max, window), bool)
    role = np.zeros((b, v_max), np.int64)
    for i, (x, o, vis, r) in enumerate(tasks):
        values[i, : len(r)], observed[i, : len(r)], visible[i, : len(r)], role[i, : len(r)] = x, o, vis, r
    return {k: torch.from_numpy(a) for k, a in
            dict(values=values, observed=observed, visible=visible, role=role).items()}


class TaskStream(IterableDataset):
    """Infinite stream of batches. A batch is either univariate or multivariate (curriculum-controlled)."""

    def __init__(self, data_cfg, curriculum, patch_size, tokens_per_batch, seed=0, start_step=0):
        self.cfg = data_cfg
        self.curriculum = Curriculum(curriculum)
        self.patch = patch_size
        self.tokens_per_batch = tokens_per_batch
        self.seed, self.start_step = seed, start_step

    def _setup(self):
        info = get_worker_info()
        self.worker_id, self.n_workers = (info.id, info.num_workers) if info else (0, 1)
        self.rng = np.random.default_rng([self.seed, self.worker_id, self.start_step])
        self.real = RealData(self.cfg["real_dir"])
        self.bank = np.load(self.cfg["kernel_bank"], mmap_mode="r")

    def sample_series(self, source, length):
        rng, bank, m = self.rng, self.bank, self.cfg["max_variates"]
        if source == "real":
            return self.real.sample(rng)
        if source == "real_multi":
            return self.real.sample(rng, multivariate=True)
        if source == "mixup":
            return self.real.mixup(rng, length)
        if source == "kernel_synth":
            return syn.series(syn.kernel_sample(rng, bank, min(length, bank.shape[1])))
        if source in syn.UNIVARIATE_GENERATORS:
            return syn.series(syn.UNIVARIATE_GENERATORS[source](rng, length))
        if source == "ssm_multi":
            return syn.ssm_multi(rng, length, m)
        if source == "multivariatizer":
            return syn.multivariatizer(rng, length, bank, m)
        return syn.covariate_effects(rng, length, bank, m)

    def make_batch(self, step):
        phase = self.curriculum(step)
        multivariate = self.rng.random() < phase.p_multivariate
        sources = self.cfg["multivariate_sources" if multivariate else "univariate_sources"]
        names, weights = list(sources), np.array(list(sources.values()), dtype=np.float64)
        rows_per_task = self.cfg["max_variates"] if multivariate else 1
        n_tasks = max(1, self.tokens_per_batch // (phase.window // self.patch) // rows_per_task)
        tasks = []
        while len(tasks) < n_tasks:
            source = names[self.rng.choice(len(names), p=weights / weights.sum())]
            s = self.sample_series(source, phase.window)
            task = build_task(self.rng, s, phase.window, self.patch, phase.max_horizon_patches, self.cfg["max_variates"],
                              self.cfg["cpm_prob"], self.cfg["cpm_max_patches"], univariate=not multivariate)
            if task is not None:
                tasks.append(task)
        batch = collate(tasks)
        batch["phase_window"] = phase.window
        return batch

    def __iter__(self):
        self._setup()
        local_step = 0
        while True:
            yield self.make_batch(self.start_step + local_step * self.n_workers + self.worker_id)
            local_step += 1

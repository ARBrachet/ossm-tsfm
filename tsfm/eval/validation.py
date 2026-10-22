"""Quick zero-shot check on held-out GiftEvalPretrain items (last window), relative to seasonal naive."""
import numpy as np
from gluonts.time_feature import get_seasonality

from tsfm.data.real import RealData
from tsfm.eval.baselines import Naive, SeasonalNaive, fill_missing

LEVELS = np.arange(1, 10) / 10
HORIZONS = {"A": 6, "Y": 6, "Q": 8, "M": 18, "W": 13, "D": 30, "H": 48, "T": 48, "S": 60}


def horizon_for(freq):
    key = "".join(c for c in freq if c.isalpha()).upper()[:1]
    return HORIZONS.get(key, 48)


def build_tasks(real_dir, max_items=50, max_context=2048, seed=0, horizon_multiplier=1):
    real = RealData(real_dir, split="val")
    rng = np.random.default_rng(seed)
    tasks = []
    for d, info in enumerate(real.datasets):
        ids = real.by_dataset[d]
        if len(ids) == 0:
            continue
        h = horizon_for(info["freq"]) * horizon_multiplier
        items, futures = [], []
        for i in rng.permutation(ids)[:max_items]:
            s = real.item(i)
            if s["target"].shape[1] < 2 * h:
                continue
            ctx = slice(max(0, s["target"].shape[1] - h - max_context), -h)
            items.append({"target": s["target"][:, ctx], "past": s["past"][:, ctx]})
            futures.append(s["target"][:, -h:])
        if items:
            tasks.append({"name": info["name"], "freq": info["freq"], "horizon": h, "items": items, "futures": futures,
                          "season": get_seasonality(info["freq"])})
    return tasks


def scores(task, forecasts):
    mase, ql_sum, abs_sum = [], 0.0, 0.0
    for item, future, q in zip(task["items"], task["futures"], forecasts):
        y = fill_missing(item["target"])
        s = task["season"] if y.shape[1] > task["season"] else 1
        scale = np.mean(np.abs(y[:, s:] - y[:, :-s]), axis=1) if y.shape[1] > s else np.ones(len(y))
        valid = ~np.isnan(future)
        err = np.where(valid, np.abs(future - q[..., 4]), 0.0).sum(1) / valid.sum(1).clip(1)
        mase.extend((err / np.where(scale > 0, scale, 1.0))[valid.any(1)])
        diff = np.nan_to_num(future)[..., None] - q
        pinball = np.maximum(LEVELS * diff, (LEVELS - 1) * diff) * valid[..., None]
        ql_sum += 2 * pinball.mean(-1).sum()
        abs_sum += np.abs(np.nan_to_num(future)).sum()
    return {"MASE": float(np.mean(mase)), "WQL": ql_sum / max(abs_sum, 1e-8)}


def evaluate(predict_fn, tasks):
    """predict_fn(items, horizon, levels) -> list of [n_target, horizon, 9] arrays. Returns per-dataset scores."""
    return {t["name"]: scores(t, predict_fn(t["items"], t["horizon"], LEVELS)) for t in tasks}


def score(record):
    """Geometric mean of the relative metrics present in a validation record (short and long horizons)."""
    keys = [k for k in ("rel_MASE", "rel_WQL", "long_rel_MASE", "long_rel_WQL") if k in record]
    return float(np.exp(np.mean([np.log(record[k]) for k in keys])))


def relative_summary(results, baseline):
    out = {}
    for metric in ("MASE", "WQL"):
        ratios = [results[k][metric] / baseline[k][metric] for k in results
                  if np.isfinite(results[k][metric]) and baseline[k][metric] > 0]
        out[f"rel_{metric}"] = float(np.exp(np.mean(np.log(np.clip(ratios, 1e-6, None)))))
    return out


def baseline_results(tasks):
    snaive = {t["name"]: scores(t, SeasonalNaive(t["season"]).predict(t["items"], t["horizon"], LEVELS)) for t in tasks}
    naive = evaluate(Naive().predict, tasks)
    return snaive, naive

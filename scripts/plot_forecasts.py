"""Forecast fans of one or more checkpoints on held-out GiftEvalPretrain series, and validation curves."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsfm.eval import validation
from tsfm.eval.baselines import SeasonalNaive
from tsfm.forecaster import Forecaster


def plot_forecasts(checkpoints, out, n=8, context=256, seed=0):
    tasks = validation.build_tasks("data/real", max_items=5, seed=seed)
    rng = np.random.default_rng(seed)
    picks = [(t, rng.integers(len(t["items"]))) for t in rng.permutation(np.array(tasks, dtype=object))[:n]]
    models = {Path(c).parent.name: Forecaster.from_checkpoint(c) for c in checkpoints}
    fig, axes = plt.subplots(n, len(models) + 1, figsize=(5 * (len(models) + 1), 2.2 * n), squeeze=False)
    for row, (task, i) in enumerate(picks):
        item, future, h = task["items"][i], task["futures"][i], task["horizon"]
        y = item["target"][0, -context:]
        forecasters = {**{k: m.predict for k, m in models.items()}, "seasonal_naive": SeasonalNaive(task["season"]).predict}
        for col, (name, predict) in enumerate(forecasters.items()):
            q = predict([item], h, [0.1, 0.5, 0.9])[0][0]
            ax = axes[row, col]
            t_ctx, t_fut = np.arange(-len(y), 0), np.arange(h)
            ax.plot(t_ctx, y, color="0.3", lw=0.8)
            ax.plot(t_fut, future[0], color="0.3", lw=0.8, ls=":")
            ax.fill_between(t_fut, q[:, 0], q[:, 2], alpha=0.3, color="C0")
            ax.plot(t_fut, q[:, 1], color="C0", lw=1)
            ax.set_title(f"{name} | {task['name']} ({task['freq']})", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out / "forecasts.png", dpi=110)


def plot_curves(checkpoints, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for c in checkpoints:
        records = [json.loads(l) for l in open(Path(c).parent / "log.jsonl")]
        train = [r for r in records if "loss" in r]
        val = [r for r in records if "rel_MASE" in r]
        name = Path(c).parent.name
        axes[0].plot([r["step"] for r in train], [r["loss"] for r in train], label=name, lw=0.8)
        axes[1].plot([r["step"] for r in val], [r["rel_MASE"] for r in val], marker="o", label=name)
        axes[2].plot([r["step"] for r in val], [r["rel_WQL"] for r in val], marker="o", label=name)
    for ax, title in zip(axes, ["train quantile loss", "held-out MASE / seasonal naive", "held-out WQL / seasonal naive"]):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    for ax in axes[1:]:
        ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_curves(args.checkpoints, out)
    plot_forecasts(args.checkpoints, out)


if __name__ == "__main__":
    main()

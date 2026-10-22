"""fev-bench (https://github.com/autogluon/fev): targets, past-only and known-future covariates are all passed to the model."""
import argparse
from pathlib import Path

import fev
import numpy as np
import pandas as pd

from tsfm.eval.baselines import Naive, SeasonalNaive

RESOURCES = Path(__file__).parent / "resources"
RESULTS_URL = "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/results/{}.csv"


def numeric(columns):
    """Columns of one series (past + future parts concatenated) -> float rows; categoricals become ordinal codes."""
    rows = []
    for col in columns:
        col = np.asarray(col)
        if col.dtype.kind in "fiub":
            rows.append(col.astype(np.float32))
        else:
            rows.append(pd.factorize(col)[0].astype(np.float32))
    return np.stack(rows) if rows else None


def to_items(task, past_data, future_data):
    items = []
    for past, future in zip(past_data, future_data):
        item = {"target": np.stack([np.asarray(past[c], dtype=np.float32) for c in task.target_columns])}
        if task.past_dynamic_columns:
            item["past"] = numeric([past[c] for c in task.past_dynamic_columns])
        if task.known_dynamic_columns:
            item["future"] = numeric([np.concatenate([past[c], future[c]]) for c in task.known_dynamic_columns])
        items.append(item)
    return items


def predict_task(task, predict_fn):
    levels = sorted(set(task.quantile_levels) | {0.5})
    predictions_per_window = []
    for window in task.iter_windows(num_proc=1):
        past_data, future_data = window.get_input_data()
        forecasts = predict_fn(to_items(task, past_data, future_data), task.horizon, levels)
        predictions = {col: [] for col in task.target_columns}
        for q in forecasts:
            for j, col in enumerate(task.target_columns):
                pred = {"predictions": q[j, :, levels.index(0.5)]}
                pred.update({str(l): q[j, :, levels.index(l)] for l in task.quantile_levels})
                predictions[col].append(pred)
        predictions_per_window.append(predictions)
    return predictions_per_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="checkpoint path, or 'naive' / 'seasonal_naive'")
    parser.add_argument("--name", default=None)
    parser.add_argument("--tasks", default=str(RESOURCES / "fev_bench_tasks_mini.yaml"))
    parser.add_argument("--out", default="results/fev_bench")
    parser.add_argument("--compare", nargs="*", default=["seasonal_naive", "naive", "chronos-2", "t0-beta", "tirex"])
    args = parser.parse_args()

    name = args.name or (Path(args.model).parent.name if args.model.endswith(".pt") else args.model)
    forecaster = None
    if args.model.endswith(".pt"):
        from tsfm.forecaster import Forecaster

        forecaster = Forecaster.from_checkpoint(args.model)
    summaries = []
    for task in fev.Benchmark.from_yaml(args.tasks).tasks:
        if forecaster is not None:
            predict_fn = forecaster.predict
        else:
            predict_fn = (SeasonalNaive(task.seasonality) if args.model == "seasonal_naive" else Naive()).predict
        summary = task.evaluation_summary(predict_task(task, predict_fn), model_name=name)
        summaries.append(summary)
        print(f"{task.dataset_config:30s} {task.eval_metric}={summary['test_error']:.4f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out / f"{name}.csv", index=False)
    tasks = {s["task_name"] for s in summaries}
    reference = [pd.read_csv(RESULTS_URL.format(m)) for m in args.compare]
    reference = [r[r["task_name"].isin(tasks)] for r in reference]
    print(fev.leaderboard([pd.DataFrame(summaries)] + reference, baseline_model="Seasonal Naive").to_string())


if __name__ == "__main__":
    main()

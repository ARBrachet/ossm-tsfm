"""GIFT-Eval aggregate: per configuration MASE / CRPS divided by Seasonal Naive, geometric mean over configurations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {"MASE": "eval_metrics/MASE[0.5]", "CRPS": "eval_metrics/mean_weighted_sum_quantile_loss"}


def summarize(csv_paths, baseline_path):
    baseline = pd.read_csv(baseline_path).set_index("dataset")
    rows = []
    for path in csv_paths:
        df = pd.read_csv(path).set_index("dataset")
        common = df.index.intersection(baseline.index)
        row = {"model": df["model"].iloc[0], "configs": len(common)}
        for name, col in METRICS.items():
            ratio = df.loc[common, col] / baseline.loc[common, col]
            row[name] = float(np.exp(np.log(ratio.clip(lower=1e-6)).mean()))
        row["term"] = "all"
        rows.append(row)
        for term in ("short", "medium", "long"):
            idx = [c for c in common if c.endswith(f"/{term}")]
            if idx:
                rows.append({"model": row["model"], "configs": len(idx), "term": term,
                             **{n: float(np.exp(np.log((df.loc[idx, c] / baseline.loc[idx, c]).clip(lower=1e-6)).mean()))
                                for n, c in METRICS.items()}})
    return pd.DataFrame(rows).sort_values(["term", "CRPS"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--baseline", default="results/gift_eval_reference/Seasonal_Naive.csv")
    args = parser.parse_args()
    paths = [p for pattern in args.csv for p in sorted(Path().glob(pattern))] or args.csv
    print(summarize(paths, args.baseline).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()

"""GIFT-Eval benchmark (https://github.com/SalesforceAIResearch/gift-eval), leaderboard-compatible CSV output.

Data: hf download Salesforce/GiftEval --repo-type dataset --local-dir data/gift_eval
The dataset splitting below reproduces gift_eval/data.py (Apache-2.0, Salesforce) without its heavy dependencies.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import datasets
import numpy as np
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import split
from gluonts.ev.metrics import MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE, MeanWeightedSumQuantileLoss
from gluonts.model import evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality, norm_freq_str
from pandas.tseries.frequencies import to_offset

from tsfm.eval.baselines import Naive, SeasonalNaive
from tsfm.forecaster import Forecaster

SHORT = ("m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D "
         "electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W "
         "saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H "
         "kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W "
         "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H "
         "ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D "
         "bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application "
         "bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H").split()
MED_LONG = ("electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T "
            "LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
            "bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T "
            "bizitobs_l2c/H").split()
PRETTY = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
          "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}
M4_PRED = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
PRED = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
OLD_FREQ = {"Y": "A", "YE": "A", "QE": "Q", "ME": "M", "h": "H", "min": "T", "s": "S", "us": "U"}
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METRICS = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(), RMSE(), NRMSE(),
           ND(), MeanWeightedSumQuantileLoss(quantile_levels=LEVELS)]
COLUMNS = ["MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]", "MSIS", "RMSE[mean]",
           "NRMSE[mean]", "ND[0.5]", "mean_weighted_sum_quantile_loss"]


class GiftEvalDataset:
    def __init__(self, root, name, term):
        self.hf = datasets.load_from_disk(str(Path(root) / name)).with_format("numpy")
        self.freq = self.hf[0]["freq"]
        target = self.hf[0]["target"]
        multivariate = target.ndim > 1
        process = ProcessDataEntry(self.freq, one_dim_target=not multivariate)
        entries = (process({**e, "start": e["start"].item()}) for e in self.hf)
        self.entries = list(self._univariate(entries) if multivariate else entries)
        freq = OLD_FREQ.get(norm_freq_str(to_offset(self.freq).name), norm_freq_str(to_offset(self.freq).name))
        base = M4_PRED[freq] if "m4" in name else PRED[freq]
        self.prediction_length = {"short": 1, "medium": 10, "long": 15}[term] * base
        min_length = min(len(e["target"]) for e in self.entries)
        self.windows = 1 if "m4" in name else min(max(1, math.ceil(0.1 * min_length / self.prediction_length)), 20)

    @staticmethod
    def _univariate(entries):
        for e in entries:
            for i, row in enumerate(e["target"]):
                yield {**e, "target": row, "item_id": f"{e['item_id']}_dim{i}"}

    @property
    def test_data(self):
        _, template = split(self.entries, offset=-self.prediction_length * self.windows)
        return template.generate_instances(prediction_length=self.prediction_length, windows=self.windows,
                                           distance=self.prediction_length)


class Predictor:
    def __init__(self, predict_fn, prediction_length):
        self.predict_fn, self.prediction_length = predict_fn, prediction_length

    def predict(self, inputs, **_):
        inputs = list(inputs)
        quantiles = self.predict_fn([{"target": e["target"]} for e in inputs], self.prediction_length, LEVELS)
        for e, q in zip(inputs, quantiles):
            arrays = np.concatenate([q[0].T, q[0][:, 4][None]])
            yield QuantileForecast(arrays, start_date=e["start"] + len(e["target"]),
                                   forecast_keys=[str(l) for l in LEVELS] + ["mean"], item_id=e.get("item_id"))


def configs(selection=None):
    for name in SHORT:
        for term in ("short", "medium", "long"):
            if term != "short" and name not in MED_LONG:
                continue
            if selection is None or name in selection:
                yield name, term


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="checkpoint path, or 'naive' / 'seasonal_naive'")
    parser.add_argument("--name", default=None)
    parser.add_argument("--data", default="data/gift_eval")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out", default="results/gift_eval")
    args = parser.parse_args()

    name = args.name or (Path(args.model).parent.name if args.model.endswith(".pt") else args.model)
    forecaster = Forecaster.from_checkpoint(args.model) if args.model.endswith(".pt") else None
    properties = json.loads((Path(__file__).parent / "resources" / "gift_eval_dataset_properties.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model"] + [f"eval_metrics/{c}" for c in COLUMNS] + ["domain", "num_variates"])
        for ds_name, term in configs(args.datasets):
            ds = GiftEvalDataset(args.data, ds_name, term)
            key = PRETTY.get(ds_name.split("/")[0].lower(), ds_name.split("/")[0].lower())
            freq = ds_name.split("/")[1] if "/" in ds_name else properties[key]["frequency"]
            season = get_seasonality(ds.freq)
            if forecaster is not None:
                predict_fn = forecaster.predict
            else:
                predict_fn = (SeasonalNaive(season) if args.model == "seasonal_naive" else Naive()).predict
            res = evaluate_model(Predictor(predict_fn, ds.prediction_length), test_data=ds.test_data, metrics=METRICS,
                                 batch_size=1024, axis=None, mask_invalid_label=True, allow_nan_forecast=False,
                                 seasonality=season)
            row = [f"{key}/{freq}/{term}", name] + [res[c][0] for c in COLUMNS]
            writer.writerow(row + [properties[key]["domain"], properties[key]["num_variates"]])
            f.flush()
            print(f"{key}/{freq}/{term:6s} MASE {res['MASE[0.5]'][0]:.3f}  CRPS {res[COLUMNS[-1]][0]:.3f}", flush=True)


if __name__ == "__main__":
    main()

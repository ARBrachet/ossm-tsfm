import numpy as np
from scipy.stats import norm


def fill_missing(y):
    y = np.array(y, dtype=np.float64)
    for row in y:
        valid = np.flatnonzero(~np.isnan(row))
        if len(valid) == 0:
            row[:] = 0.0
            continue
        idx = np.maximum.accumulate(np.where(np.isnan(row), 0, np.arange(len(row))))
        idx[: valid[0]] = valid[0]
        row[:] = row[idx]
    return y


class SeasonalNaive:
    """Repeats the last season; Gaussian intervals growing with the number of seasons ahead (statsforecast-style)."""

    def __init__(self, season=1):
        self.season = season

    def predict(self, items, horizon, quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
        z = norm.ppf(np.asarray(quantile_levels))
        out = []
        for item in items:
            y = fill_missing(np.atleast_2d(item["target"]))
            s = self.season if y.shape[1] > self.season else 1
            k = np.arange(horizon)
            point = y[:, y.shape[1] - s + (k % s)]
            resid = y[:, s:] - y[:, :-s] if y.shape[1] > s else np.zeros((len(y), 1))
            sigma = np.sqrt(np.mean(resid**2, axis=1))
            spread = sigma[:, None, None] * np.sqrt(k // s + 1)[None, :, None] * z
            out.append(point[..., None] + spread)
        return out


class Naive(SeasonalNaive):
    def __init__(self, season=1):
        super().__init__(season=1)

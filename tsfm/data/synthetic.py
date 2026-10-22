"""Synthetic generators. Each returns a dict with 2-D float arrays `target`, `past`, `future` of a common length.

- KernelSynth (Chronos): Gaussian-process samples with randomly composed kernels (precomputed bank).
- TSI (Chronos-2): trend + seasonality + irregularity.
- ETS / AR: classical stochastic processes.
- SSM: random linear state-space systems (modal form), univariate or multi-output with exogenous inputs.
- Multivariatizers (Chronos-2): cotemporaneous / lead-lag / cointegration mixing of base series.
- Covariate effects (t0): events, spikes, drivers, steps and decoys acting on a target.
"""
import numpy as np
from scipy.signal import lfilter

SEASONAL_PERIODS = np.array([4, 7, 12, 24, 48, 52, 96, 144, 168, 288, 336, 365.25])
KERNEL_PERIODS = (24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 6, 12)


def series(target, past=None, future=None):
    target = np.atleast_2d(np.asarray(target, dtype=np.float32))
    empty = np.zeros((0, target.shape[1]), np.float32)
    rows = lambda x: empty if x is None or len(x) == 0 else np.atleast_2d(np.asarray(x, dtype=np.float32))
    return {"target": target, "past": rows(past), "future": rows(future)}


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)


def loguniform(rng, low, high, size=None):
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def shift(x, lag):
    return x if lag == 0 else np.concatenate([np.full(lag, x[0]), x[:-lag]])


# ---------------------------------------------------------------- KernelSynth

def _kernel_bank(length):
    return ([("linear", c) for c in (0.0, 1.0, 10.0)] + [("rbf", l) for l in (0.1, 1.0, 10.0)]
            + [("periodic", p / length) for p in KERNEL_PERIODS] + [("rq", a) for a in (0.1, 1.0, 10.0)]
            + [("white", s) for s in (0.1, 1.0)] + [("constant", 1.0)])


def _kernel(kind, param, x):
    d = x[:, None] - x[None, :]
    if kind == "linear":
        return x[:, None] * x[None, :] + param
    if kind == "rbf":
        return np.exp(-0.5 * d**2 / param**2)
    if kind == "periodic":
        return np.exp(-2.0 * np.sin(np.pi * np.abs(d) / param) ** 2)
    if kind == "rq":
        return (1.0 + d**2 / (2.0 * param)) ** -param
    if kind == "white":
        return param * np.eye(len(x))
    return np.full_like(d, param)


def kernel_synth(rng, length=1024, max_kernels=5):
    x = np.linspace(0.0, 1.0, length)
    bank = _kernel_bank(length)
    picks = rng.integers(len(bank), size=rng.integers(1, max_kernels + 1))
    cov = _kernel(*bank[picks[0]], x)
    for i in picks[1:]:
        k = _kernel(*bank[i], x)
        cov = cov + k if rng.random() < 0.5 else cov * k
    scale = np.abs(np.diag(cov)).mean() + 1e-12
    for jitter in (1e-6, 1e-4, 1e-2):
        try:
            chol = np.linalg.cholesky(cov + jitter * scale * np.eye(length))
            return chol @ rng.standard_normal(length)
        except np.linalg.LinAlgError:
            continue
    return rng.standard_normal(length)


def kernel_sample(rng, bank, length):
    row = bank[rng.integers(len(bank))]
    if len(row) >= length:
        start = rng.integers(len(row) - length + 1)
        return row[start : start + length].astype(np.float64)
    return np.interp(np.linspace(0, len(row) - 1, length), np.arange(len(row)), row)


# ---------------------------------------------------------------- TSI

def _trend(rng, u):
    kind = rng.choice(["none", "linear", "quadratic", "exp", "piecewise", "logistic"])
    if kind == "linear":
        return rng.normal(0, 2) * u
    if kind == "quadratic":
        return rng.normal(0, 2) * u + rng.normal(0, 2) * u**2
    if kind == "exp":
        r = rng.uniform(-4, 4)
        return np.sign(rng.normal()) * (np.exp(r * u) - 1) / max(abs(np.exp(r) - 1), 1e-3)
    if kind == "piecewise":
        knots = np.sort(rng.uniform(0, 1, rng.integers(1, 5)))
        slopes = rng.normal(0, 3, len(knots) + 1)
        return np.cumsum(slopes[np.searchsorted(knots, u)]) / len(u)
    if kind == "logistic":
        return rng.normal(0, 2) / (1 + np.exp(-rng.uniform(5, 30) * (u - rng.uniform(0.2, 0.8))))
    return np.zeros_like(u)


def _period(rng, length):
    valid = SEASONAL_PERIODS[SEASONAL_PERIODS * 2.5 <= length]
    if len(valid) == 0 or rng.random() < 0.25:
        return loguniform(rng, 2.5, max(3.0, length / 3))
    return rng.choice(valid)


def _seasonal(rng, t, period):
    kind = rng.choice(["fourier", "fourier", "square", "sawtooth"])
    phase = 2 * np.pi * t / period + rng.uniform(0, 2 * np.pi)
    if kind == "square":
        return np.sign(np.sin(phase))
    if kind == "sawtooth":
        return 2 * ((phase / (2 * np.pi)) % 1.0) - 1
    n_harmonics = rng.integers(1, max(2, min(6, int(period // 2))))
    coefs = rng.normal(0, 1, n_harmonics) / np.arange(1, n_harmonics + 1)
    shifts = rng.uniform(0, 2 * np.pi, n_harmonics)
    return sum(c * np.sin(k * phase + s) for k, (c, s) in enumerate(zip(coefs, shifts), start=1))


def _noise(rng, length, scale):
    kind = rng.choice(["gaussian", "student", "ar1", "hetero"])
    if kind == "student":
        e = rng.standard_t(3, length) / 1.7
    elif kind == "ar1":
        phi = rng.uniform(0.3, 0.95)
        e = lfilter([1.0], [1.0, -phi], rng.standard_normal(length)) * np.sqrt(1 - phi**2)
    elif kind == "hetero":
        e = rng.standard_normal(length) * np.exp(lfilter([1.0], [1.0, -0.99], 0.1 * rng.standard_normal(length)))
    else:
        e = rng.standard_normal(length)
    return scale * e


def tsi(rng, length):
    t = np.arange(length, dtype=np.float64)
    trend = _trend(rng, t / length)
    season = np.zeros(length)
    for _ in range(rng.choice([0, 1, 2, 3], p=[0.15, 0.45, 0.3, 0.1])):
        component = _seasonal(rng, t, _period(rng, length)) * loguniform(rng, 0.2, 2.0)
        if rng.random() < 0.3:
            component *= 1 + rng.uniform(-0.8, 2.0) * t / length
        season += component
    signal = trend + season
    y = signal + _noise(rng, length, (signal.std() + 0.1) * loguniform(rng, 0.02, 1.0))
    if rng.random() < 0.15:
        spikes = rng.random(length) < loguniform(rng, 0.001, 0.02)
        y += spikes * rng.normal(0, 5 * (y.std() + 1e-3), length)
    return _positivity(rng, y)


def _positivity(rng, y):
    u = rng.random()
    if u < 0.15:
        return np.exp(0.5 * standardize(y))
    if u < 0.25:
        rate = loguniform(rng, 0.3, 50) * np.exp(0.5 * standardize(y))
        if rng.random() < 0.5:
            rate *= rng.random(len(y)) < rng.uniform(0.1, 0.7)
        return rng.poisson(rate).astype(np.float64)
    return y


# ---------------------------------------------------------------- ETS / AR

def ar(rng, length, burn_in=200):
    p = rng.integers(1, 6)
    roots = []
    while len(roots) < p:
        radius = rng.uniform(0.3, 0.99)
        if p - len(roots) >= 2 and rng.random() < 0.6:
            angle = rng.uniform(0.05, np.pi)
            roots += [radius * np.exp(1j * angle), radius * np.exp(-1j * angle)]
        else:
            roots.append(radius * rng.choice([-1.0, 1.0]))
    y = lfilter([1.0], np.poly(roots).real, rng.standard_normal(length + burn_in))[burn_in:]
    return np.cumsum(y) if rng.random() < 0.3 else y


def ets(rng, length):
    alpha, beta, gamma = rng.uniform(0.01, 0.5), rng.uniform(0.0, 0.2) * (rng.random() < 0.5), rng.uniform(0.0, 0.3)
    phi = rng.uniform(0.8, 1.0)
    period = int(round(_period(rng, length))) if rng.random() < 0.7 else 1
    level, slope = rng.normal(), rng.normal(0, 0.05)
    season = rng.normal(0, 1, max(period, 1)) if period > 1 else np.zeros(1)
    season -= season.mean()
    sigma = loguniform(rng, 0.05, 0.5)
    e = sigma * rng.standard_normal(length)
    y = np.empty(length)
    for i in range(length):
        s = season[i % len(season)]
        y[i] = level + phi * slope + s + e[i]
        level, slope = level + phi * slope + alpha * e[i], phi * slope + beta * e[i]
        season[i % len(season)] = s + gamma * e[i]
    return y


def ets_ar(rng, length):
    return ar(rng, length) if rng.random() < 0.5 else ets(rng, length)


# ---------------------------------------------------------------- linear state-space systems

def _modes(rng, n_modes, length):
    lam = np.empty(n_modes, dtype=np.complex128)
    for k in range(n_modes):
        u = rng.random()
        if u < 0.55:
            lam[k] = (1 - loguniform(rng, 1e-4, 0.1)) * np.exp(2j * np.pi / _period(rng, length))
        elif u < 0.85:
            lam[k] = 1 - loguniform(rng, 1e-4, 0.3)
        else:
            lam[k] = rng.uniform(-0.9, 0.7)
    return lam


def _simulate(rng, lam, drive):
    z0 = rng.normal(size=len(lam)) + 1j * rng.normal(size=len(lam))
    return np.stack([lfilter([1.0], [1.0, -l], d, zi=[l * z])[0] for l, d, z in zip(lam, drive, z0)])


def ssm_system(rng, length, n_modes, n_out, inputs=None):
    """x_{t+1} = A x_t + B u_t + w_t, y_t = C x_t + v_t with A diagonal in a complex modal basis."""
    lam = _modes(rng, n_modes, length)
    noise_scale = loguniform(rng, 0.01, 1.0) * np.maximum(np.sqrt(1 - np.abs(lam) ** 2), 0.02)
    drive = noise_scale[:, None] * (rng.standard_normal((n_modes, length)) + 1j * rng.standard_normal((n_modes, length)))
    if inputs is not None and len(inputs):
        b = (rng.normal(size=(n_modes, len(inputs))) + 1j * rng.normal(size=(n_modes, len(inputs))))
        drive = drive + (np.abs(1 - lam)[:, None] * b) @ standardize(inputs)
    states = _simulate(rng, lam, drive)
    c = rng.normal(size=(n_out, n_modes)) + 1j * rng.normal(size=(n_out, n_modes))
    y = (c @ states).real
    return y + loguniform(rng, 0.005, 0.3) * y.std(-1, keepdims=True) * rng.standard_normal(y.shape)


def ssm(rng, length):
    return ssm_system(rng, length, rng.integers(1, 7), 1)[0]


def exogenous(rng, length):
    kind = rng.choice(["tsi", "events", "steps", "ets_ar"])
    if kind == "events":
        return _events(rng, length)
    if kind == "steps":
        return _steps(rng, length)
    return tsi(rng, length) if kind == "tsi" else ets_ar(rng, length)


def ssm_multi(rng, length, max_variates=8):
    n_in = rng.choice([0, 1, 2, 3], p=[0.25, 0.4, 0.25, 0.1])
    n_out = rng.integers(2, max(3, max_variates - n_in + 1))
    inputs = np.stack([exogenous(rng, length) for _ in range(n_in)]) if n_in else None
    outputs = ssm_system(rng, length, rng.integers(2, 11), n_out, inputs)
    return assign_roles(rng, outputs, inputs)


def assign_roles(rng, outputs, inputs=None):
    n_targets = rng.integers(1, len(outputs) + 1)
    target, past, future = list(outputs[:n_targets]), [], []
    for row in outputs[n_targets:]:
        past.append(row)
    for row in [] if inputs is None else inputs:
        (future if rng.random() < 0.6 else past).append(row)
    return series(target, past, future)


# ---------------------------------------------------------------- multivariatizers

def base_univariate(rng, length, kernel_bank=None):
    kinds = ["tsi", "ets_ar", "ssm"] + (["kernel"] if kernel_bank is not None else [])
    kind = rng.choice(kinds)
    if kind == "kernel":
        return kernel_sample(rng, kernel_bank, length)
    return {"tsi": tsi, "ets_ar": ets_ar, "ssm": ssm}[kind](rng, length)


def multivariatizer(rng, length, kernel_bank=None, max_variates=8):
    k = rng.integers(2, max_variates + 1)
    base = standardize(np.stack([base_univariate(rng, length, kernel_bank) for _ in range(rng.integers(2, k + 1))]))
    kind = rng.choice(["linear", "nonlinear", "lead_lag", "cointegration"])
    w = rng.normal(size=(k, len(base))) * (rng.random((k, len(base))) < 0.7)
    if kind == "linear":
        y = w @ base
    elif kind == "nonlinear":
        f = rng.choice([np.tanh, lambda x: np.log1p(np.exp(x)), lambda x: x * np.abs(x), np.sin])
        y = f(w @ base)
    elif kind == "lead_lag":
        max_lag = max(1, min(30, length // 10))
        y = np.stack([sum(w[j, i] * shift(base[i], rng.integers(0, max_lag)) for i in range(len(base))) for j in range(k)])
    else:
        common = np.cumsum(rng.standard_normal(length)) / np.sqrt(length) * 5
        y = rng.normal(size=(k, 1)) * common + 0.3 * (w @ base)
    y = y + 0.05 * rng.standard_normal(y.shape)
    roles = rng.permutation(k)
    n_targets = rng.integers(1, k + 1)
    is_future = rng.random(k) < 0.5
    target = y[roles[:n_targets]]
    past = [y[i] for i in roles[n_targets:] if not is_future[i]]
    future = [y[i] for i in roles[n_targets:] if is_future[i]]
    return series(target, past, future)


# ---------------------------------------------------------------- covariate effects

def _events(rng, length):
    c = np.zeros(length)
    for _ in range(rng.poisson(loguniform(rng, 2, max(3, length / 20)))):
        start = rng.integers(length)
        c[start : start + rng.integers(1, 15)] = 1.0
    return c


def _steps(rng, length):
    block = rng.integers(4, 64)
    return np.repeat(rng.normal(size=length // block + 1), block)[:length]


def covariate_effects(rng, length, kernel_bank=None, max_variates=8):
    y = standardize(base_univariate(rng, length, kernel_bank))
    covariates, effect = [], np.zeros(length)
    for kind in rng.choice(["event", "spike", "driver", "steps", "decoy"], size=rng.integers(1, min(4, max_variates))):
        beta = rng.choice([-1, 1]) * loguniform(rng, 0.3, 3.0)
        if kind == "event":
            c = _events(rng, length)
            lead, lag = rng.integers(0, 4), rng.integers(1, 8)
            response = np.concatenate([np.linspace(0.3, 0.8, lead), np.ones(1), -rng.uniform(0, 0.5) * np.exp(-np.arange(lag) / 2)])
            e = np.convolve(c, response, mode="full")[lead : lead + length]
        elif kind == "spike":
            c = (rng.random(length) < loguniform(rng, 0.002, 0.05)).astype(float)
            e = lfilter([1.0], [1.0, -rng.uniform(0.0, 0.9)], c)
        elif kind == "driver":
            c = standardize(tsi(rng, length))
            g = rng.choice([lambda x: x, lambda x: np.maximum(x - rng.uniform(-1, 1), 0), lambda x: x**2 - 1, np.tanh])
            e = shift(g(c), rng.integers(0, 5))
        elif kind == "steps":
            c = _steps(rng, length)
            e = c
        else:
            c, e = rng.standard_normal(length), np.zeros(length)
        covariates.append(c)
        effect += beta * e
    future = [c for c in covariates if rng.random() < 0.75]
    past = [c for c in covariates if not any(c is f for f in future)]
    return series(y + effect, past, future)


UNIVARIATE_GENERATORS = {"tsi": tsi, "ets_ar": ets_ar, "ssm": ssm}

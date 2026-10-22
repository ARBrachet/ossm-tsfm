import numpy as np

from tsfm.data import synthetic as syn
from tsfm.data.loader import build_task
from tsfm.eval.baselines import SeasonalNaive
from tsfm.model.heads import FUTURE, PAST, TARGET

P = 16


def test_build_task_masks_horizon_but_not_known_future():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = syn.covariate_effects(rng, 300)
        task = build_task(rng, s, 512, P, 4, 8, 1.0, 4, univariate=False)
        values, observed, visible, role = task
        assert values.shape[1] == 512 and np.isfinite(values).all()
        assert not visible[:, : 512 - 18 * P].any()
        last = visible[:, -P:]
        assert not last[role != FUTURE].any()
        assert (last[role == FUTURE] == observed[role == FUTURE][:, -P:]).all()
        assert (visible <= observed).all()


def test_generators_are_finite():
    rng = np.random.default_rng(1)
    for fn in (syn.tsi, syn.ets_ar, syn.ssm):
        assert np.isfinite(fn(rng, 1000)).all()
    for fn in (syn.ssm_multi, syn.multivariatizer, syn.covariate_effects):
        s = fn(rng, 1000)
        assert len(s["target"]) >= 1
        assert all(np.isfinite(s[k]).all() for k in s)


def test_seasonal_naive_repeats_last_season():
    y = np.tile(np.arange(7.0), 10)
    q = SeasonalNaive(7).predict([{"target": y}], 14, [0.5])[0]
    assert np.allclose(q[0, :, 0], np.tile(np.arange(7.0), 2))

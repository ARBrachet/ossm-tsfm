"""Examples of every training source: synthetic generators and real GiftEvalPretrain series."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsfm.data import synthetic as syn
from tsfm.data.real import RealData

COLORS = {"target": "C0", "past": "C2", "future": "C3"}


def draw(ax, s, title):
    offset = 0.0
    for role, color in COLORS.items():
        for i, row in enumerate(s[role][:4]):
            row = np.asarray(row, dtype=np.float64)
            ax.plot((row - np.nanmean(row)) / (np.nanstd(row) + 1e-8) - offset, color=color, lw=0.7,
                    label=role if i == 0 else None)
            offset += 4.0
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/figures/synthetic.png")
    parser.add_argument("--length", type=int, default=1024)
    args = parser.parse_args()
    rng = np.random.default_rng(3)
    bank = np.load("data/synthetic/kernel_synth.npy", mmap_mode="r")
    real = RealData("data/real")
    n = args.length
    rows = [
        ("KernelSynth", lambda: syn.series(syn.kernel_sample(rng, bank, n))),
        ("TSI", lambda: syn.series(syn.tsi(rng, n))),
        ("ETS / AR", lambda: syn.series(syn.ets_ar(rng, n))),
        ("SSM (univariate)", lambda: syn.series(syn.ssm(rng, n))),
        ("SSM (multi-output + inputs)", lambda: syn.ssm_multi(rng, n)),
        ("multivariatizer", lambda: syn.multivariatizer(rng, n, bank)),
        ("covariate effects", lambda: syn.covariate_effects(rng, n, bank)),
        ("TSMixup (real)", lambda: real.mixup(rng, n)),
        ("real (GiftEvalPretrain)", lambda: real.sample(rng, multivariate=True)),
    ]
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 1.9 * len(rows)))
    for r, (name, make) in enumerate(rows):
        for c in range(3):
            draw(axes[r, c], make(), name)
    axes[4, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)


if __name__ == "__main__":
    main()

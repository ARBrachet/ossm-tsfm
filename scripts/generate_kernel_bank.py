"""Precompute KernelSynth series (GP sampling is cubic in the length, too slow to run inside the data loader)."""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tsfm.data.synthetic import kernel_synth


def work(args):
    seed, n, length = args
    rng = np.random.default_rng(seed)
    return np.stack([kernel_synth(rng, length) for _ in range(n)]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic/kernel_synth.npy")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    chunk = 250
    jobs = [(seed, chunk, args.length) for seed in range(args.n // chunk)]
    with Pool(args.workers) as pool:
        bank = np.concatenate(pool.map(work, jobs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, bank)
    print(bank.shape)


if __name__ == "__main__":
    main()

"""Share of each training source: nominal (token budget from the config) and effective (non-padding patches)."""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from tsfm.data.loader import TaskStream, build_task
from tsfm.model.heads import TARGET

GROUPS = {"real": "real", "real_multi": "real", "mixup": "real (TSMixup)", "ssm": "SSM", "ssm_multi": "SSM",
          "kernel_synth": "other synthetic", "tsi": "other synthetic", "ets_ar": "other synthetic",
          "multivariatizer": "other synthetic", "covariate_effects": "other synthetic"}


def useful_patches(task, patch):
    observed, role = task[1], task[3]
    patch_obs = observed.reshape(len(role), -1, patch).any(-1)
    started = np.cumsum(patch_obs, axis=1) > 0
    return started.sum(), started[role == TARGET].sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--tasks-per-source", type=int, default=200)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    data, phases, steps = cfg["data"], cfg["curriculum"], cfg["train"]["steps"]
    stream = TaskStream(data, phases, cfg["model"]["patch_size"], cfg["train"]["tokens_per_batch"])
    stream._setup()
    patch = cfg["model"]["patch_size"]

    rows, start = [], 0
    for phase in phases:
        duration = min(phase["until_step"], steps) - start
        start = phase["until_step"]
        for pool, p_pool in [("univariate_sources", 1 - phase["p_multivariate"]), ("multivariate_sources", phase["p_multivariate"])]:
            weights = data[pool]
            total = sum(weights.values())
            for source, w in weights.items():
                used = all_rows = target_rows = 0
                for _ in range(args.tasks_per_source):
                    s = stream.sample_series(source, phase["window"])
                    task = build_task(stream.rng, s, phase["window"], patch, phase["max_horizon_patches"],
                                      data["max_variates"], data["cpm_prob"], data["cpm_max_patches"],
                                      univariate=pool == "univariate_sources")
                    capacity = (data["max_variates"] if pool == "multivariate_sources" else 1) * phase["window"] // patch
                    all_rows += capacity
                    if task is not None:
                        u, t = useful_patches(task, patch)
                        used, target_rows = used + u, target_rows + t
                rows.append({"window": phase["window"], "w": duration / steps, "source": source, "group": GROUPS[source],
                             "nominal": p_pool * w / total, "fill": used / all_rows, "target_fill": target_rows / all_rows})
    df = pd.DataFrame(rows)
    df["effective"] = df.nominal * df.fill
    df["effective_target"] = df.nominal * df.target_fill
    run = df.assign(nominal=df.nominal * df.w, effective=df.effective * df.w, effective_target=df.effective_target * df.w)

    def shares(frame, key):
        out = frame.groupby(key)[["nominal", "effective", "effective_target"]].sum()
        return (out / out.sum()).sort_values("nominal", ascending=False)

    pd.set_option("display.float_format", "{:.1%}".format)
    for window, frame in df.groupby("window"):
        print(f"\nphase window {window}:\n", shares(frame, "group").to_string())
    print("\nwhole run, by group:\n", shares(run, "group").to_string())
    print("\nwhole run, by source:\n", shares(run, "source").to_string())


if __name__ == "__main__":
    main()

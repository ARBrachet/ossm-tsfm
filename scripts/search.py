"""Sequential search over the OSSM transition (dense / capped / modal / selective), its damping and the learning rate.

Each run uses a compressed 8k-step curriculum and is validated every 1k steps on held-out series at the short
horizon and at 10x that horizon. A run is stopped as soon as its score (geometric mean of the four relative
metrics) is more than `margin` worse than the best finished run at the same step. Finished runs are reused.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

STEPS = 8000
SCHEDULE = [
    f"train.steps={STEPS}", "train.warmup_steps=500", "train.eval_every=1000", "train.log_every=100",
    "train.long_validation=true", "train.prune_after=3000",
    'curriculum=[{"until_step": 2000, "window": 512, "max_horizon_patches": 8, "p_multivariate": 0.1},'
    ' {"until_step": 5000, "window": 1024, "max_horizon_patches": 16, "p_multivariate": 0.2},'
    ' {"until_step": 8000, "window": 2048, "max_horizon_patches": 32, "p_multivariate": 0.4}]',
]
BASE_LR = 3e-4
VARIANTS = {"dense": [], "capped": ["model.a_cap=1.0"], "modal": ["model.transition=modal"],
            "selective": ["model.transition=selective"],
            "selective_ctrl": ["model.transition=selective", "model.observer=controlled"],
            "selective_bc": ["model.transition=selective", "model.selective_bc=true"],
            "selective_ctrl_bc": ["model.transition=selective", "model.observer=controlled", "model.selective_bc=true"],
            "selective_bc_relu": ["model.transition=selective", "model.selective_bc=true", "model.readout_activation=relu"],
            "selective_bc_lin": ["model.transition=selective", "model.selective_bc=true", "model.decoder=linear"],
            "selective_bc_lin_relu": ["model.transition=selective", "model.selective_bc=true", "model.decoder=linear",
                                      "model.readout_activation=relu"],
            "selective_ctrl_bc_lin": ["model.transition=selective", "model.observer=controlled", "model.selective_bc=true",
                                      "model.decoder=linear"],
            "dense_lin": ["model.decoder=linear"],
            "mlp": ["model.head=mlp"]}
# (variant, rho(A), rho(A - LC)); rho 0.999 / 0.9 was stopped at 5k steps, level with 0.99 / 0.9, and is not rerun
DAMPINGS = [("dense", 0.9, 0.91), ("dense", 0.97, 0.9), ("dense", 0.99, 0.9),
            ("capped", 0.99, 0.9), ("modal", 0.99, 0.9), ("selective", 0.99, 0.9),
            ("selective", 0.99, 0.5), ("selective_bc", 0.99, 0.9),
            ("selective_ctrl", 0.99, 0.9), ("selective_ctrl_bc", 0.99, 0.9),
            ("selective_ctrl", 0.99, 0.7), ("selective_ctrl", 0.99, 0.5),
            ("mlp", 0.99, 0.9),  # mlp: reference head, the damping does not apply
            # OSSM readout of Eq. 4: the quantile decoder D is a single matrix
            ("selective_bc_lin", 0.99, 0.9), ("selective_bc_lin_relu", 0.99, 0.9),
            ("selective_ctrl_bc_lin", 0.99, 0.9), ("dense_lin", 0.99, 0.9)]
LEARNING_RATES = [6e-4, 1e-3]


def run_name(variant, rho_a, rho_obs, lr):
    if variant == "mlp":
        return f"mlp_lr{lr}"
    prefix = "" if variant == "dense" else f"{variant}_"
    return f"{prefix}rhoA{rho_a}_rhoO{rho_obs}_lr{lr}"


def run(variant, rho_a, rho_obs, lr, reference, margin, ckpt_dir):
    name = run_name(variant, rho_a, rho_obs, lr)
    log = Path(ckpt_dir) / name / "log.jsonl"
    records = [json.loads(line) for line in open(log) if "rel_MASE" in line] if log.exists() else []
    if not records or not (records[-1]["pruned"] or records[-1]["step"] == STEPS):
        shutil.rmtree(log.parent, ignore_errors=True)
        overrides = SCHEDULE + ["model.head=ossm"] + VARIANTS[variant] + [
            f"run_name={name}", f"train.ckpt_dir={ckpt_dir}", f"model.rho_a={rho_a}",
            f"model.rho_observer={rho_obs}", f"train.lr={lr}", f"train.prune_margin={margin}"]
        if reference:
            overrides.append(f"train.prune_against={reference}")
        subprocess.run([sys.executable, "-W", "ignore", "-m", "tsfm.train", "configs/small.yaml", *overrides], check=True)
        records = [json.loads(line) for line in open(log) if "rel_MASE" in line]
    last = records[-1]
    return {"run": name, "variant": variant, "rho_a": rho_a, "rho_observer": rho_obs, "lr": lr, "steps": last["step"],
            "pruned": last["pruned"], "score": last["score"],
            **{k: last[k] for k in ("rel_MASE", "rel_WQL", "long_rel_MASE", "long_rel_WQL")}, "log": str(log)}


def best_finished(rows, ossm_only=False):
    finished = [r for r in rows if not r["pruned"] and not (ossm_only and r["variant"] == "mlp")]
    return min(finished, key=lambda r: r["score"]) if finished else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--ckpt-dir", default="checkpoints/search")
    parser.add_argument("--out", default="results/search.csv")
    parser.add_argument("--skip-lr", action="store_true")
    parser.add_argument("--no-prune", action="store_true", help="run every new run to the end")
    args = parser.parse_args()

    rows = []

    def record(row):
        rows.append(row)
        pd.DataFrame(rows).drop(columns="log").to_csv(args.out, index=False)
        print(json.dumps({k: v for k, v in row.items() if k != "log"}), flush=True)

    for variant, rho_a, rho_obs in DAMPINGS:
        best = best_finished(rows, ossm_only=True)
        reference = best["log"] if best and variant != "mlp" and not args.no_prune else None
        record(run(variant, rho_a, rho_obs, BASE_LR, reference, args.margin, args.ckpt_dir))
    best = best_finished(rows, ossm_only=True)
    for lr in [] if args.skip_lr else LEARNING_RATES:
        record(run(best["variant"], best["rho_a"], best["rho_observer"], lr, best_finished(rows)["log"],
                   args.margin, args.ckpt_dir))
    print(pd.DataFrame(rows).drop(columns="log").sort_values("score").to_string(index=False))


if __name__ == "__main__":
    main()

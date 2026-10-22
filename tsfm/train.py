import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tsfm.data.loader import TaskStream
from tsfm.eval import validation
from tsfm.forecaster import Forecaster, load_config
from tsfm.model.model import TSFM


def lr_at(step, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["steps"] - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * min(progress, 1.0)))


def apply_overrides(cfg, overrides):
    for item in overrides:
        key, value = item.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for k in path:
            node = node[k]
        node[last] = json.loads(value) if value[:1] in "[{0123456789-tf" else value
    return cfg


def run_validation(model, suites, device, max_context):
    """suites: {prefix: (tasks, (seasonal naive results, naive results))}, e.g. short and 10x longer horizons."""
    model.eval()
    forecaster = Forecaster(model, max_context=max_context, device=device)
    record = {}
    for prefix, (tasks, (snaive, naive)) in suites.items():
        results = validation.evaluate(forecaster.predict, tasks)
        record.update({prefix + k: v for k, v in validation.relative_summary(results, snaive).items()})
        record.update({prefix + "naive_" + k: v for k, v in validation.relative_summary(naive, snaive).items()})
    model.train()
    return record


def load_reference(path):
    """Validation scores per step of a finished run, used to stop a run that falls clearly behind it."""
    if not path:
        return {}
    records = [json.loads(line) for line in open(path)]
    return {r["step"]: validation.score(r) for r in records if "rel_MASE" in r}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    tcfg = cfg["train"]
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TSFM(**cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    no_decay_names = ("A", "C", "L", "G", "null_key", "radius_logit", "theta", "kappa")
    decays = lambda n, p: p.dim() >= 2 and n.split(".")[-1] not in no_decay_names and "role_embed" not in n
    named = list(model.named_parameters())
    decay = [p for n, p in named if decays(n, p)]
    no_decay = [p for n, p in named if not decays(n, p)]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=tcfg["lr"], betas=(0.9, 0.95))

    stream = TaskStream(cfg["data"], cfg["curriculum"], cfg["model"]["patch_size"], tcfg["tokens_per_batch"], cfg["seed"])
    loader = DataLoader(stream, batch_size=None, num_workers=tcfg["num_workers"], pin_memory=True,
                        prefetch_factor=4 if tcfg["num_workers"] else None, persistent_workers=tcfg["num_workers"] > 0)

    out_dir = Path(tcfg["ckpt_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out_dir / "log.jsonl", "a")
    print(f"{n_params / 1e6:.2f}M parameters, logging to {out_dir}", flush=True)

    suites = {}
    for prefix, multiplier in [("", 1)] + ([("long_", 10)] if tcfg.get("long_validation") else []):
        tasks = validation.build_tasks(cfg["data"]["real_dir"], horizon_multiplier=multiplier)
        suites[prefix] = (tasks, validation.baseline_results(tasks))
    reference = load_reference(tcfg.get("prune_against"))

    final_context = max(p["window"] for p in cfg["curriculum"])
    best_wql = float("inf")

    def save(step, name="model.pt", **extra):
        torch.save({"model": model.state_dict(), "config": cfg, "step": step, **extra}, out_dir / name)

    start, losses = time.time(), []
    for step, batch in enumerate(loader):
        if step >= tcfg["steps"]:
            break
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, tcfg)
        tensors = {k: batch[k].to(device, non_blocking=True) for k in ("values", "observed", "visible", "role")}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = model.loss(**tensors)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()
        model.constrain()
        losses.append(loss.item())

        if (step + 1) % tcfg["log_every"] == 0:
            record = {"step": step + 1, "loss": sum(losses) / len(losses), "lr": lr_at(step, tcfg),
                      "grad_norm": grad_norm.item(), "window": batch["phase_window"],
                      "rows": tensors["role"].numel(), "elapsed_min": (time.time() - start) / 60}
            losses = []
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
        if (step + 1) % tcfg["eval_every"] == 0 or step + 1 == tcfg["steps"]:
            record = {"step": step + 1, "context": batch["phase_window"],
                      **run_validation(model, suites, device, batch["phase_window"])}
            record["score"] = validation.score(record)
            behind = record["score"] > reference.get(step + 1, float("inf")) * (1 + tcfg.get("prune_margin", 0.03))
            record["pruned"] = bool(behind and step + 1 >= tcfg.get("prune_after", 0))
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            save(step + 1)
            if record["context"] == final_context and record["rel_WQL"] < best_wql:
                best_wql = record["rel_WQL"]
                save(step + 1, "best.pt", validation=record)
            if record["pruned"]:
                return
    save(tcfg["steps"])


if __name__ == "__main__":
    main()

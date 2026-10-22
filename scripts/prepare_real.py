"""Download a subset of GiftEvalPretrain and store it as one flat float32 memmap + an index.

Each item keeps its targets and past covariates (rows), truncated to the last `max_length` steps.
A small fraction of items per dataset is held out for validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "theforecastingcompany/GiftEvalPretrain"

EXCLUDED_PREFIXES = ("era5_", "cmip6_", "buildings_900k", "azure_vm_traces_2017", "Q-TRAFFIC")
# overlap with fev-bench tasks: kept out to keep the fev-bench evaluation zero-shot
FEV_BENCH_OVERLAP = ("favorita_sales", "favorita_transactions", "m5", "gfc12_load", "gfc14_load", "gfc17_load",
                     "kdd2022", "fred_md")
# heavy, near-duplicate or redundant collections: one shard of one representative is enough locally
SKIPPED = ("largest_2018", "largest_2019", "largest_2020", "largest_2021", "residential_load_power", "subseasonal",
           "extended_web_traffic_with_missing")

INDEX_DTYPE = np.dtype([("dataset", "i4"), ("offset", "i8"), ("n_target", "i4"), ("n_past", "i4"),
                        ("length", "i4"), ("val", "?")])


def list_shards():
    shards = {}
    for f in HfApi().list_repo_files(REPO, repo_type="dataset"):
        if f.endswith(".parquet"):
            name = f.split("/")[0]
            shards.setdefault(name, []).append(f)
    return {k: sorted(v) for k, v in sorted(shards.items())}


def selected(name):
    return not (name.startswith(EXCLUDED_PREFIXES) or name in FEV_BENCH_OVERLAP or name in SKIPPED)


def read_items(path, max_items, rng):
    table = pq.read_table(path, columns=["freq", "target", "past_feat_dynamic_real"])
    rows = np.arange(table.num_rows)
    if len(rows) > max_items:
        rows = np.sort(rng.choice(rows, max_items, replace=False))
    freq = table.column("freq")[0].as_py()
    for i in rows:
        target = table.column("target")[int(i)].as_py()
        past = table.column("past_feat_dynamic_real")[int(i)].as_py() or []
        yield freq, np.asarray(target, dtype=np.float32), np.asarray(past, dtype=np.float32).reshape(-1, len(target[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/real")
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    datasets, index = [], []
    offset = 0
    with open(out / "values.f32", "wb") as f:
        for name, files in list_shards().items():
            if not selected(name):
                continue
            n_before = len(index)
            per_shard = max(1, args.max_items // min(len(files), args.max_shards))
            for file in files[: args.max_shards]:
                path = hf_hub_download(REPO, file, repo_type="dataset")
                for freq, target, past in read_items(path, per_shard, rng):
                    series = np.concatenate([target, past])[:, -args.max_length :]
                    if series.shape[1] < args.min_length or np.isnan(target[:, -args.max_length :]).all():
                        continue
                    f.write(np.ascontiguousarray(series).tobytes())
                    index.append((len(datasets), offset, len(target), len(past), series.shape[1], False))
                    offset += series.size
            n_items = len(index) - n_before
            if n_items:
                n_val = max(1, int(args.val_fraction * n_items)) if n_items >= 10 else 0
                for j in rng.choice(np.arange(n_before, len(index)), n_val, replace=False):
                    index[j] = index[j][:5] + (True,)
                datasets.append({"name": name, "freq": freq, "n_items": n_items})
                print(f"{name:40s} {freq:6s} {n_items:6d} items  total {offset * 4 / 1e9:.2f} GB", flush=True)

    np.save(out / "index.npy", np.array(index, dtype=INDEX_DTYPE))
    (out / "datasets.json").write_text(json.dumps(datasets, indent=1))


if __name__ == "__main__":
    main()

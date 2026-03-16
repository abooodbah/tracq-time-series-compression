#!/usr/bin/env python
"""
Minor experiment: effect of variable ordering on PNG compression ratio.

Compares three orderings for Enhanced TRACQ 8-bit PNG output:
  1. Original dataset column order
  2. Correlation-sorted order (built-in reorder_variables heuristic)
  3. Five random permutations

Outputs:
  - variable_ordering_summary.json
  - variable_ordering_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracq.codec import ImageCodec
from tracq.core_enhanced import EnhancedTimeSeriesGrid


DATASETS = {
    "uci_air_quality": PROJECT_ROOT / "data" / "processed" / "uci_air_quality.csv",
    "uci_appliances_energy": PROJECT_ROOT / "data" / "processed" / "uci_appliances_energy.csv",
    "uci_metro_traffic": PROJECT_ROOT / "data" / "processed" / "uci_metro_traffic.csv",
}


def png_size(arr: np.ndarray, *, reorder_variables: bool) -> int:
    grid = EnhancedTimeSeriesGrid(
        arr,
        adaptive_clamp=True,
        use_mu_law=True,
        auto_offset=True,
        reorder_variables=reorder_variables,
    )
    q, meta = grid.quantize_8bit()
    fd, tmp_png = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        ImageCodec.save_png(tmp_png, q, meta, compress_level=6)
        return int(os.path.getsize(tmp_png))
    finally:
        if os.path.exists(tmp_png):
            os.remove(tmp_png)


def run_dataset(path: Path, *, num_random_orders: int, seed: int) -> Dict[str, float | List[int]]:
    df = pd.read_csv(path, header=None)
    data = df.values.T.astype(np.float64)[:, :5000]

    baseline_png_bytes = png_size(data, reorder_variables=False)
    corr_png_bytes = png_size(data, reorder_variables=True)

    rng = np.random.default_rng(seed)
    random_png_bytes: List[int] = []
    for _ in range(int(num_random_orders)):
        perm = rng.permutation(data.shape[0])
        random_png_bytes.append(png_size(data[perm], reorder_variables=False))

    random_mean = float(np.mean(random_png_bytes))
    return {
        "baseline_png_bytes": baseline_png_bytes,
        "corr_png_bytes": corr_png_bytes,
        "corr_vs_baseline_pct": float(100.0 * (baseline_png_bytes - corr_png_bytes) / baseline_png_bytes),
        "random_png_bytes": random_png_bytes,
        "random_mean_bytes": random_mean,
        "random_std_bytes": float(np.std(random_png_bytes)),
        "corr_vs_random_mean_pct": float(100.0 * (random_mean - corr_png_bytes) / random_mean),
        "baseline_vs_random_mean_pct": float(100.0 * (random_mean - baseline_png_bytes) / random_mean),
    }


def save_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    fieldnames = [
        "dataset",
        "baseline_png_bytes",
        "corr_png_bytes",
        "corr_vs_baseline_pct",
        "random_mean_bytes",
        "random_std_bytes",
        "corr_vs_random_mean_pct",
        "baseline_vs_random_mean_pct",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate PNG compression sensitivity to variable ordering.")
    ap.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "paper_results" / "ordering")
    ap.add_argument("--num-random-orders", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    results = {}
    rows = []
    for dataset_name, path in DATASETS.items():
        if not path.exists():
            print(f"Skipping {dataset_name}: {path} not found")
            continue
        print(f"Running variable-ordering experiment on {dataset_name} ...")
        res = run_dataset(path, num_random_orders=args.num_random_orders, seed=args.seed)
        results[dataset_name] = res
        rows.append({"dataset": dataset_name, **{k: v for k, v in res.items() if k != "random_png_bytes"}})

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "variable_ordering_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_csv(args.outdir / "variable_ordering_summary.csv", rows)
    print(f"Saved ordering results to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

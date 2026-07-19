#!/usr/bin/env python3
"""Terabyte-scale windowed streaming encode, node-parallel.

Scales the paper's streaming experiment (16-variable synthetic sensor stream,
100,000-step windows, generated on the fly, fast entropy setting) to 1 TB on a
single node with a process pool. Each worker reports its own peak RSS so the
constant-memory claim is verified per worker at scale.

Usage: python3 terabyte_stream.py <total_gb> <n_workers> <out_json>
"""

import json
import os
import resource
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracq import lattice

N_VARS = 16
WINDOW = 100_000
BYTES_PER_WINDOW = N_VARS * WINDOW * 8


def one_window(idx):
    rng = np.random.default_rng(idx)
    base = 100 + 10 * rng.standard_normal((N_VARS, 1))
    data = base + np.cumsum(rng.normal(0, 0.05, (N_VARS, WINDOW)), axis=1)
    t0 = time.perf_counter()
    blob, _, _ = lattice.encode(data, eps=1e-3, mode="abs", predictors="p1", zstd_level=1)
    enc_s = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return len(blob), enc_s, rss_mb


def main():
    total_gb = float(sys.argv[1])
    n_workers = int(sys.argv[2])
    out_path = sys.argv[3]
    n_windows = int(total_gb * 1e9 / BYTES_PER_WINDOW)
    total_bytes = n_windows * BYTES_PER_WINDOW

    t0 = time.perf_counter()
    with Pool(n_workers) as pool:
        results = pool.map(one_window, range(n_windows), chunksize=8)
    wall = time.perf_counter() - t0

    out_bytes = sum(r[0] for r in results)
    enc_core_s = sum(r[1] for r in results)
    rss = [r[2] for r in results]
    report = {
        "total_gb": total_bytes / 1e9,
        "n_windows": n_windows,
        "n_workers": n_workers,
        "wall_s": wall,
        "aggregate_mbs": total_bytes / 1e6 / wall,
        "per_core_mbs": total_bytes / 1e6 / enc_core_s,
        "ratio": out_bytes / total_bytes,
        "worker_rss_mb_max": max(rss),
        "worker_rss_mb_median": sorted(rss)[len(rss) // 2],
        "hostname": os.uname().nodename,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

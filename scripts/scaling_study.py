#!/usr/bin/env python3
"""Strong- and weak-scaling study of the node-parallel windowed encoder.

Strong scaling: fixed 50 GB synthetic stream encoded at 1..112 workers.
Weak scaling: 8 GB per worker at the same worker counts.
Three repetitions of each point; the JSON stores every run so medians and
spread can be computed downstream. lscpu output is captured for the record.

Usage: python3 scaling_study.py <out_json>
"""

import json
import os
import resource
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracq import lattice

N_VARS = 16
WINDOW = 100_000
BYTES_PER_WINDOW = N_VARS * WINDOW * 8
COUNTS = [1, 2, 4, 8, 16, 32, 56, 112]
REPS = 3
STRONG_GB = 50.0
WEAK_GB_PER_WORKER = 8.0


def one_window(idx):
    rng = np.random.default_rng(idx)
    base = 100 + 10 * rng.standard_normal((N_VARS, 1))
    data = base + np.cumsum(rng.normal(0, 0.05, (N_VARS, WINDOW)), axis=1)
    t0 = time.perf_counter()
    blob, _, _ = lattice.encode(data, eps=1e-3, mode="abs", predictors="p1", zstd_level=1)
    enc_s = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return len(blob), enc_s, rss_mb


def run(total_gb, n_workers):
    n_windows = int(total_gb * 1e9 / BYTES_PER_WINDOW)
    total_bytes = n_windows * BYTES_PER_WINDOW
    t0 = time.perf_counter()
    with Pool(n_workers) as pool:
        results = pool.map(one_window, range(n_windows), chunksize=8)
    wall = time.perf_counter() - t0
    enc_core_s = sum(r[1] for r in results)
    rss = sorted(r[2] for r in results)
    return {
        "workers": n_workers,
        "total_gb": total_bytes / 1e9,
        "n_windows": n_windows,
        "wall_s": wall,
        "agg_mbs": total_bytes / 1e6 / wall,
        "enc_core_mbs": total_bytes / 1e6 / enc_core_s,
        "ratio": sum(r[0] for r in results) / total_bytes,
        "rss_mb_max": rss[-1],
        "rss_mb_median": rss[len(rss) // 2],
    }


def main():
    out_path = sys.argv[1]
    report = {
        "hostname": os.uname().nodename,
        "lscpu": subprocess.run(["lscpu"], capture_output=True, text=True).stdout,
        "strong_gb": STRONG_GB,
        "weak_gb_per_worker": WEAK_GB_PER_WORKER,
        "strong": [],
        "weak": [],
    }

    def flush():
        with open(out_path, "w") as f:
            json.dump(report, f, indent=1)

    for rep in range(REPS):
        for w in COUNTS:
            r = run(STRONG_GB, w)
            r["rep"] = rep
            report["strong"].append(r)
            print("strong rep%d w=%3d wall=%7.1fs agg=%8.1f MB/s enc/core=%6.1f" %
                  (rep, w, r["wall_s"], r["agg_mbs"], r["enc_core_mbs"]), flush=True)
            flush()
        for w in COUNTS:
            r = run(WEAK_GB_PER_WORKER * w, w)
            r["rep"] = rep
            report["weak"].append(r)
            print("weak   rep%d w=%3d wall=%7.1fs agg=%8.1f MB/s enc/core=%6.1f" %
                  (rep, w, r["wall_s"], r["agg_mbs"], r["enc_core_mbs"]), flush=True)
            flush()
    print("DONE")


if __name__ == "__main__":
    main()

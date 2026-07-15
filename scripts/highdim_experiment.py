"""High-dimensional scalability experiment for the v3 revision.

Sweeps synthetic dimensionality N in {64, 256, 1024, 2048, 4096} (grouped,
correlated channels; T=2000) and anchors with the widest real dataset on hand
(LD2011 electricity, 370 clients). For each N reports: encode/decode
throughput (fast + archival entropy settings), compressed ratio, metadata
fraction of the artifact, predictor-selection overhead share, predictor mix,
and the verified error bound.
"""

import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from tracq import lattice

OUT = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "highdim.json")
EPS = 1e-3


def gen_grouped(n_vars, n_time=2000, groups=16, seed=0):
    rng = np.random.default_rng(seed)
    factors = np.cumsum(rng.normal(0, 1, (groups, n_time)), axis=1)
    g = rng.integers(0, groups, n_vars)
    gain = rng.uniform(0.5, 2.0, n_vars)[:, None]
    base = rng.uniform(10, 1000, n_vars)[:, None]
    noise = rng.normal(0, 0.3, (n_vars, n_time))
    return base + gain * factors[g] + np.cumsum(noise, axis=1) * 0.1


def run_one(name, data):
    n, t = data.shape
    mb = data.nbytes / 1e6
    res = {"n_vars": n, "n_time": t, "mb": mb}

    t0 = time.perf_counter()
    blob_f, _, hdr = lattice.encode(data, eps=EPS, mode="abs", predictors="bank", zstd_level=1)
    res["enc_fast_mbs"] = mb / (time.perf_counter() - t0)

    t0 = time.perf_counter()
    blob_a, _, hdr_a = lattice.encode(data, eps=EPS, mode="abs", predictors="bank", zstd_level=19)
    res["enc_archival_mbs"] = mb / (time.perf_counter() - t0)

    t0 = time.perf_counter()
    recon, _ = lattice.decode(blob_a)
    res["dec_mbs"] = mb / (time.perf_counter() - t0)

    # predictor-selection + ordering overhead share (bank vs p1 encode time)
    t0 = time.perf_counter()
    lattice.encode(data, eps=EPS, mode="abs", predictors="p1", zstd_level=1)
    t_p1 = time.perf_counter() - t0
    t_bank = mb / res["enc_fast_mbs"]
    res["selection_share"] = max(0.0, (t_bank - t_p1) / t_bank)
    res["enc_p1_fast_mbs"] = mb / t_p1
    blob_p1, _, _ = lattice.encode(data, eps=EPS, mode="abs", predictors="p1", zstd_level=19)
    res["ratio_p1"] = len(blob_p1) / data.nbytes
    del blob_p1

    rng_v = data.max(axis=1) - data.min(axis=1)
    rng_v = np.where(rng_v <= 0, 1.0, rng_v)
    err = np.abs(recon - data)
    res["ratio"] = len(blob_a) / data.nbytes
    res["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    res["bound_ok"] = bool((err <= EPS * rng_v[:, None] * (1 + 1e-9)).all())
    hdr_bytes = len(json.dumps(hdr_a, separators=(",", ":")).encode())
    res["metadata_frac"] = hdr_bytes / len(blob_a)
    import collections
    mix = collections.Counter(hdr_a["pred"])
    res["pred_mix"] = {str(k): v for k, v in sorted(mix.items())}
    print(f"{name:14s} N={n:5d} encF {res['enc_fast_mbs']:6.1f} encA {res['enc_archival_mbs']:6.1f} "
          f"dec {res['dec_mbs']:6.1f} MB/s  ratio {res['ratio']:.4f}  meta {res['metadata_frac']*100:.2f}%  "
          f"sel {res['selection_share']*100:.0f}%  bound_ok {res['bound_ok']}  mix {res['pred_mix']}")
    return res


def main():
    out = {"synthetic": [], "eps": EPS}
    for n in (64, 256, 1024, 2048, 4096):
        data = gen_grouped(n)
        out["synthetic"].append(run_one(f"synth{n}", data))
        del data

    # real anchor: LD2011 electricity, 370 clients
    ld = (r"C:\Users\Abdulfatah\Desktop\Personal\Research\Multi Agent Improvement"
          r"\GTC-research\LD2011_2014.txt")
    try:
        import pandas as pd
        df = pd.read_csv(ld, sep=";", decimal=",", index_col=0, nrows=60000)
        data = df.values.T.astype(np.float64)[:, -10000:]
        data = np.where(np.isfinite(data), data, 0.0)
        out["ld2011"] = run_one("LD2011", data)
    except Exception as e:
        print("LD2011 skipped:", e)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()

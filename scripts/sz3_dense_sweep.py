#!/usr/bin/env python3
"""Dense SZ3 sweep (absolute-tolerance mode via the hdf5plugin SZ3 filter),
same protocol as the ZFP sweep in iso_rmse_experiment.py: canonical UCI
datasets plus full-length MetroPT-3, measured RMSE and encoded ratio per
tolerance. Runs on Linux (WSL); writes paper_results/lattice/sz3_dense.json.
"""

import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import h5py
import hdf5plugin

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "sz3_dense.json")

TOLS = np.logspace(-5, 3, 20)


def load_uci(fname):
    p = os.path.join(PROJECT_ROOT, "data", "processed", fname)
    d = pd.read_csv(p, header=None).values.T.astype(np.float64)
    return np.where(np.isfinite(d), d, 0.0)[:, :5000]


def load_metropt():
    p = os.path.join(PROJECT_ROOT, "data", "raw", "metropt3", "MetroPT3(AirCompressor).csv")
    df = pd.read_csv(p)
    num = df.select_dtypes(include=[np.number])
    num = num.drop(columns=[c for c in num.columns if "unnamed" in c.lower()], errors="ignore")
    d = num.values.T.astype(np.float64)
    return np.where(np.isfinite(d), d, 0.0)


def sz3_point(data, tol, tmpdir):
    path = os.path.join(tmpdir, "sz3.h5")
    t0 = time.perf_counter()
    with h5py.File(path, "w") as f:
        f.create_dataset("d", data=data, chunks=data.shape,
                         **hdf5plugin.SZ3(absolute=float(tol)))
    enc_s = time.perf_counter() - t0
    stored = os.path.getsize(path)
    with h5py.File(path, "r") as f:
        rec = f["d"][...]
    err = rec - data
    return {
        "tol": float(tol),
        "ratio": stored / data.nbytes,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_err": float(np.abs(err).max()),
        "encode_s": enc_s,
    }


def main():
    datasets = {
        "air_quality": load_uci("uci_air_quality.csv"),
        "appliances": load_uci("uci_appliances_energy.csv"),
        "metro_traffic": load_uci("uci_metro_traffic.csv"),
        "metropt3": load_metropt(),
    }
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in datasets.items():
            pts = []
            for tol in TOLS:
                try:
                    r = sz3_point(np.ascontiguousarray(data), tol, tmp)
                    pts.append(r)
                    print(f"{name:14s} tol {tol:9.2e}  ratio {r['ratio']:.4f}  "
                          f"rmse {r['rmse']:10.4g}  maxerr {r['max_err']:10.4g}")
                except Exception as e:
                    print(f"{name:14s} tol {tol:9.2e}  FAILED: {e}")
            out[name] = {"shape": list(data.shape), "sz3": pts}
            del data
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()

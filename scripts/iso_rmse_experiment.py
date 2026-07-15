"""Iso-RMSE comparison: at matched measured RMSE, compare encoded size of the
enhanced TRACQ variant against ZFP, as the RMSE target increases.

Both codecs are swept densely (TRACQ over its error tolerance, ZFP over its
absolute tolerance), the two rate-distortion curves are interpolated in
log-log space onto a common RMSE grid inside their overlap band, and the
size advantage ZFP_ratio / TRACQ_ratio is reported per dataset.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from tracq import lattice

import zfpy

OUT = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "iso_rmse.json")

TRACQ_EPS = np.logspace(-6, -0.5, 14)
ZFP_TOLS = np.logspace(-5, 3, 20)


def load_uci(fname):
    p = os.path.join(PROJECT_ROOT, "data", "processed", fname)
    d = pd.read_csv(p, header=None).values.T.astype(np.float64)
    return np.where(np.isfinite(d), d, 0.0)[:, :5000]


def load_metropt():
    p = os.path.join(PROJECT_ROOT, "data", "raw", "metropt3", "MetroPT3(AirCompressor).csv")
    df = pd.read_csv(p)
    num = df.select_dtypes(include=[np.number])
    num = num.drop(columns=[c for c in num.columns if "unnamed" in str(c).lower()],
                   errors="ignore")
    d = num.values.T.astype(np.float64)
    return np.where(np.isfinite(d), d, 0.0)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def sweep(data):
    orig = data.nbytes
    tr = []
    for eps in TRACQ_EPS:
        blob, _, _ = lattice.encode(data, eps=float(eps), mode="abs", predictors="bank")
        recon, _ = lattice.decode(blob)
        tr.append({"eps": float(eps), "ratio": len(blob) / orig, "rmse": rmse(data, recon)})
        del blob, recon
    zf = []
    arr = np.ascontiguousarray(data)
    for tol in ZFP_TOLS:
        try:
            zb = zfpy.compress_numpy(arr, tolerance=float(tol))
            rec = zfpy.decompress_numpy(zb)
            zf.append({"tol": float(tol), "ratio": len(zb) / orig, "rmse": rmse(data, rec)})
            del zb, rec
        except Exception:
            pass
    return tr, zf


def iso_compare(tr, zf, n_grid=12):
    t = sorted([(p["rmse"], p["ratio"]) for p in tr if p["rmse"] > 0])
    z = sorted([(p["rmse"], p["ratio"]) for p in zf if p["rmse"] > 0])
    lo = max(t[0][0], z[0][0])
    hi = min(t[-1][0], z[-1][0])
    if not (hi > lo):
        return []
    grid = np.logspace(np.log10(lo * 1.001), np.log10(hi * 0.999), n_grid)

    def interp(curve, x):
        xs = np.log10([c[0] for c in curve])
        ys = np.log10([c[1] for c in curve])
        return 10 ** np.interp(np.log10(x), xs, ys)

    rows = []
    for g in grid:
        rt = float(interp(t, g))
        rz = float(interp(z, g))
        rows.append({"rmse": float(g), "tracq_ratio": rt, "zfp_ratio": rz,
                     "advantage": rz / rt})
    return rows


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    datasets = {
        "air_quality": lambda: load_uci("uci_air_quality.csv"),
        "appliances": lambda: load_uci("uci_appliances_energy.csv"),
        "metro_traffic": lambda: load_uci("uci_metro_traffic.csv"),
        "metropt3": load_metropt,
    }
    if only:
        datasets = {only: datasets[only]}
    out = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            out = json.load(f)
    datasets = {k: v() for k, v in datasets.items()}
    for name, data in datasets.items():
        t0 = time.perf_counter()
        tr, zf = sweep(data)
        iso = iso_compare(tr, zf)
        out[name] = {"tracq": tr, "zfp": zf, "iso": iso,
                     "shape": list(data.shape)}
        adv = [r["advantage"] for r in iso]
        print(f"{name:14s} ({data.shape[0]}x{data.shape[1]})  "
              f"overlap {iso[0]['rmse']:.2g}..{iso[-1]['rmse']:.2g}  "
              f"advantage min {min(adv):.2f}x  median {np.median(adv):.2f}x  "
              f"max {max(adv):.2f}x   [{time.perf_counter()-t0:.0f}s]")
        del data

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()

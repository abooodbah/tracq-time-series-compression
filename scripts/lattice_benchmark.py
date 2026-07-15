"""Stage-3 bake-off: TRACQ-2 lattice candidates vs the paper's baselines.

Runs C1 (p1 predictor), C2 (predictor bank), C3 (bank + dead-zone) over an
eps sweep on the three UCI datasets, and prints them against the strongest
baseline numbers already measured by scripts/realworld_benchmark.py
(paper_results/realworld/*_results.json).
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
from tracq.metrics import rmse as calc_rmse, mae as calc_mae

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BASE_DIR = os.path.join(PROJECT_ROOT, "paper_results", "realworld")
OUT_DIR = os.path.join(PROJECT_ROOT, "paper_results", "lattice")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["uci_air_quality", "uci_appliances_energy", "uci_metro_traffic"]
EPS_SWEEP = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]

CANDIDATES = {
    "C1_p1": dict(predictors="p1", tau=0.0),
    "C2_bank": dict(predictors="bank", tau=0.0),
    "C3_deadzone": dict(predictors="bank", tau=1.0),
}


def smape(a, b):
    denom = (np.abs(a) + np.abs(b)) / 2.0
    mask = denom > 1e-12
    out = np.zeros_like(a)
    out[mask] = np.abs(a - b)[mask] / denom[mask]
    return float(np.mean(out))


def corr(a, b):
    af, bf = a.ravel(), b.ravel()
    if np.std(af) == 0 or np.std(bf) == 0:
        return 1.0
    return float(np.corrcoef(af, bf)[0, 1])


def main():
    all_results = {}
    for ds in DATASETS:
        path = os.path.join(DATA_DIR, ds + ".csv")
        data = pd.read_csv(path, header=None).values.T.astype(np.float64)
        data = np.where(np.isfinite(data), data, 0.0)
        orig_bytes = data.nbytes
        print(f"\n=== {ds}  ({data.shape[0]} vars x {data.shape[1]} steps, {orig_bytes} B) ===")
        results = {}

        for cname, copts in CANDIDATES.items():
            for mode in ("abs", "rel"):
                for eps in EPS_SWEEP:
                    key = f"{cname}_{mode}_eps{eps:g}"
                    t0 = time.perf_counter()
                    blob, grid, header = lattice.encode(data, eps=eps, mode=mode, **copts)
                    enc_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    recon, _ = lattice.decode(blob)
                    dec_s = time.perf_counter() - t0
                    results[key] = {
                        "candidate": cname,
                        "mode": mode,
                        "eps": eps,
                        "bytes": len(blob),
                        "ratio": len(blob) / orig_bytes,
                        "rmse": calc_rmse(data, recon),
                        "mae": calc_mae(data, recon),
                        "smape": smape(data, recon),
                        "corr": corr(data, recon),
                        "max_err": float(np.abs(data - recon).max()),
                        "encode_s": enc_s,
                        "decode_s": dec_s,
                        "escapes": int((grid == lattice.ESCAPE).sum()),
                    }

        all_results[ds] = results

        # ---- print table with baseline context ----
        base_path = os.path.join(BASE_DIR, ds + "_results.json")
        baselines = {}
        if os.path.exists(base_path):
            with open(base_path) as f:
                bj = json.load(f)
            for name in ["delta_zstd", "tracq_enh_8bit_anchors", "tracq_enh_16bit_anchors",
                         "paa", "gorilla_like", "zfp_tol_0.1", "zfp_tol_0.01",
                         "zfp_tol_0.001", "zfp_tol_0.0001"]:
                r = bj.get("results", bj).get(name)
                if r and "metrics" in r:
                    baselines[name] = {"ratio": r["ratio"], "rmse": r["metrics"]["rmse"],
                                       "smape": r["metrics"].get("smape"), "corr": r["metrics"]["corr"]}

        print(f"  {'method':38s} {'ratio':>8s} {'RMSE':>12s} {'SMAPE':>8s} {'corr':>7s} {'maxerr':>10s}")
        for name, r in baselines.items():
            print(f"  {name:38s} {r['ratio']:8.4f} {r['rmse']:12.4g} {(r['smape'] or 0):8.4f} {r['corr']:7.4f} {'':>10s}")
        print("  " + "-" * 90)
        for key, r in sorted(results.items(), key=lambda kv: kv[1]["ratio"]):
            print(f"  {key:38s} {r['ratio']:8.4f} {r['rmse']:12.4g} {r['smape']:8.4f} {r['corr']:7.4f} {r['max_err']:10.4g}")

    with open(os.path.join(OUT_DIR, "lattice_results.json"), "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'lattice_results.json')}")


if __name__ == "__main__":
    main()

"""Full-length MetroPT-3 benchmark for TRACQ-2 lattice candidates.

The paper's negative results here: unanchored TRACQ diverged to RMSE ~3.8e95;
anchored TRACQ (RMSE 1.71 @ 0.0297) lost to PAA-1024 (RMSE 1.55 @ 0.0007).
This script tests whether TRACQ-2 removes the divergence and becomes
Pareto-competitive, against PAA-1024/4096, Delta+Zstd and ZFP on the same data.
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

CSV = os.path.join(PROJECT_ROOT, "data", "raw", "metropt3", "MetroPT3(AirCompressor).csv")
OUT = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "metropt3_lattice.json")


def smape(a, b):
    denom = (np.abs(a) + np.abs(b)) / 2.0
    mask = denom > 1e-12
    out = np.zeros_like(a)
    out[mask] = np.abs(a - b)[mask] / denom[mask]
    return float(np.mean(out))


def metrics(name, data, recon, nbytes, orig_bytes, enc_s, dec_s, extra=None):
    r = {
        "bytes": int(nbytes),
        "ratio": nbytes / orig_bytes,
        "rmse": float(np.sqrt(np.mean((data - recon) ** 2))),
        "smape": smape(data, recon),
        "max_err": float(np.abs(data - recon).max()),
        "corr": float(np.corrcoef(data.ravel()[::37], recon.ravel()[::37])[0, 1]),
        "encode_s": enc_s,
        "decode_s": dec_s,
    }
    if extra:
        r.update(extra)
    print(f"  {name:34s} ratio {r['ratio']:.5f}  RMSE {r['rmse']:10.4g}  SMAPE {r['smape']:.4f} "
          f"maxerr {r['max_err']:10.4g}  enc {enc_s:6.1f}s dec {dec_s:5.1f}s")
    return r


def main():
    print("Loading MetroPT-3 ...")
    df = pd.read_csv(CSV)
    num = df.select_dtypes(include=[np.number])
    num = num.drop(columns=[c for c in num.columns if "unnamed" in c.lower()], errors="ignore")
    data = num.values.T.astype(np.float64)
    data = np.where(np.isfinite(data), data, 0.0)
    n_vars, n_time = data.shape
    orig_bytes = data.nbytes
    print(f"{n_vars} vars x {n_time} steps = {orig_bytes/1e6:.1f} MB raw")

    results = {}

    group = sys.argv[1] if len(sys.argv) > 1 else "all"

    # --- TRACQ-2 candidates ---
    configs = []
    if group in ("all", "lattice"):
        configs += [("C1_p1", dict(predictors="p1", tau=0.0), m, e)
                    for m in ("abs", "rel") for e in (1e-1, 1e-2, 1e-3, 1e-4)]
        configs += [("C2_bank", dict(predictors="bank", tau=0.0), m, e)
                    for m in ("abs", "rel") for e in (1e-2, 1e-4)]
    if group in ("all", "deadzone"):
        configs += [("C3_deadzone_tau1", dict(predictors="p1", tau=1.0), "abs", 1e-2)]

    if True:
        for cname, copts, mode, eps in configs:
                key = f"{cname}_{mode}_eps{eps:g}"
                t0 = time.perf_counter()
                blob, grid, header = lattice.encode(data, eps=eps, mode=mode, **copts)
                enc_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                recon, _ = lattice.decode(blob)
                dec_s = time.perf_counter() - t0
                results[key] = metrics(key, data, recon, len(blob), orig_bytes, enc_s, dec_s,
                                       {"escapes": int((grid == lattice.ESCAPE).sum())})
                del blob, grid, recon

    # --- PAA baselines (the paper's winner here) ---
    from tracq.baselines import paa_compress, paa_decompress
    for segs in (1024, 4096, 16384) if group in ("all", "baselines") else ():
        t0 = time.perf_counter()
        comp, meta = paa_compress(data, segs)
        enc_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        recon = paa_decompress(comp, meta)
        dec_s = time.perf_counter() - t0
        nbytes = comp.nbytes if hasattr(comp, "nbytes") else len(comp)
        results[f"paa_{segs}"] = metrics(f"paa_{segs}", data, recon, nbytes, orig_bytes, enc_s, dec_s)
        del comp, recon

    # --- Delta+Zstd lossless floor ---
    import zstandard as zstd
    if group not in ("all", "baselines"):
        _save(results)
        return
    t0 = time.perf_counter()
    delta = np.diff(data, axis=1)
    packed = np.concatenate([data[:, :1], delta], axis=1).astype(np.float64).tobytes()
    dz = zstd.ZstdCompressor(level=19).compress(packed)
    enc_s = time.perf_counter() - t0
    results["delta_zstd"] = metrics("delta_zstd_lossless", data, data, len(dz), orig_bytes, enc_s, 0.0)
    del packed, dz, delta

    # --- ZFP sweep ---
    try:
        import zfpy
        for tol in (1e-1, 1e-2, 1e-3):
            t0 = time.perf_counter()
            zb = zfpy.compress_numpy(data, tolerance=tol)
            enc_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            recon = zfpy.decompress_numpy(zb)
            dec_s = time.perf_counter() - t0
            results[f"zfp_tol_{tol:g}"] = metrics(f"zfp_tol_{tol:g}", data, recon, len(zb), orig_bytes, enc_s, dec_s)
            del zb, recon
    except Exception as e:
        print("zfp failed:", e)

    _save(results)


def _save(results):
    merged = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            merged = json.load(f)
    merged.update(results)
    with open(OUT, "w") as f:
        json.dump(merged, f, indent=1)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()

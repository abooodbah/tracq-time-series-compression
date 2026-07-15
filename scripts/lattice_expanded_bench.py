"""Expanded-suite benchmark: TRACQ-2 vs ZFP / PAA / Delta+Zstd on new domains.

Datasets (all real, publicly sourced):
  - ecg:       MIT-BIH excerpt via scipy.datasets (1 x 108,000 @ 360 Hz)
  - jena:      Jena climate 2009-2016 weather station (14 x ~420k @ 10 min)
  - household: UCI individual household electric power (7 x ~2.05M @ 1 min)
  - btc_ticks: Kraken BTC/USD trades, price+volume (2 x ~60k, irregular ticks)
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
from tracq.baselines import paa_compress, paa_decompress

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
OUT = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "expanded_results.json")


def load_ecg():
    return np.load(os.path.join(RAW, "ecg.npy"))[None, :]


def load_jena():
    df = pd.read_csv(os.path.join(RAW, "jena_climate_2009_2016.csv"))
    df = df.drop(columns=["Date Time"])
    arr = df.values.T.astype(np.float64)
    # standard Jena cleaning: -9999 wind sentinels are recording errors
    arr[arr <= -9998.0] = np.nan
    return arr


def load_household():
    df = pd.read_csv(
        os.path.join(RAW, "household_power_consumption.txt"),
        sep=";", na_values="?", low_memory=False,
    )
    df = df.drop(columns=["Date", "Time"])
    df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    return df.values.T.astype(np.float64)


def load_btc():
    return np.load(os.path.join(RAW, "btc_ticks.npy"))


LOADERS = {"ecg": load_ecg, "jena": load_jena, "household": load_household, "btc_ticks": load_btc}


def smape(a, b):
    denom = (np.abs(a) + np.abs(b)) / 2.0
    mask = denom > 1e-12
    out = np.zeros_like(a)
    out[mask] = np.abs(a - b)[mask] / denom[mask]
    return float(np.mean(out))


def report(name, data, recon, nbytes, orig_bytes, enc_s, dec_s):
    r = {
        "bytes": int(nbytes),
        "ratio": nbytes / orig_bytes,
        "rmse": float(np.sqrt(np.mean((data - recon) ** 2))),
        "smape": smape(data, recon),
        "max_err": float(np.abs(data - recon).max()),
        "encode_s": enc_s,
        "decode_s": dec_s,
    }
    print(f"  {name:30s} ratio {r['ratio']:8.5f}  RMSE {r['rmse']:11.5g}  SMAPE {r['smape']:7.4f}  "
          f"maxerr {r['max_err']:10.4g}")
    return r


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    all_results = {}
    for ds, loader in LOADERS.items():
        if only and ds != only:
            continue
        try:
            data = loader()
        except Exception as e:
            print(f"SKIP {ds}: {e}")
            continue
        data = np.where(np.isfinite(data), data, 0.0)
        orig_bytes = data.nbytes
        print(f"\n=== {ds}  ({data.shape[0]} x {data.shape[1]}, {orig_bytes/1e6:.1f} MB) ===")
        res = {}

        # TRACQ-2 (winner config C2 bank), abs + rel
        for mode in ("abs", "rel"):
            for eps in (1e-2, 1e-3, 1e-4):
                key = f"tracq2_{mode}_eps{eps:g}"
                t0 = time.perf_counter()
                blob, grid, header = lattice.encode(data, eps=eps, mode=mode, predictors="bank")
                enc_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                recon, _ = lattice.decode(blob)
                dec_s = time.perf_counter() - t0
                res[key] = report(key, data, recon, len(blob), orig_bytes, enc_s, dec_s)
                del blob, grid, recon

        # ZFP sweep
        try:
            import zfpy
            zdata = np.ascontiguousarray(data)
            for tol in (1e-1, 1e-2, 1e-3):
                t0 = time.perf_counter()
                zb = zfpy.compress_numpy(zdata, tolerance=tol)
                enc_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                recon = zfpy.decompress_numpy(zb)
                dec_s = time.perf_counter() - t0
                res[f"zfp_tol_{tol:g}"] = report(f"zfp_tol_{tol:g}", data, recon, len(zb), orig_bytes, enc_s, dec_s)
                del zb, recon
        except Exception as e:
            print("  zfp failed:", e)

        # PAA (segments ~ T/64 per paper convention, plus a fine setting)
        T = data.shape[1]
        for segs in (max(T // 78, 4), max(T // 16, 8)):
            t0 = time.perf_counter()
            comp, meta = paa_compress(data, segs)
            enc_s = time.perf_counter() - t0
            recon = paa_decompress(comp, meta)
            nbytes = comp.nbytes
            res[f"paa_{segs}"] = report(f"paa_{segs}", data, recon, nbytes, orig_bytes, enc_s, 0.0)
            del comp, recon

        # Delta+Zstd lossless floor
        import zstandard as zstd
        t0 = time.perf_counter()
        packed = np.concatenate([data[:, :1], np.diff(data, axis=1)], axis=1).tobytes()
        dz = zstd.ZstdCompressor(level=19).compress(packed)
        enc_s = time.perf_counter() - t0
        res["delta_zstd"] = report("delta_zstd_lossless", data, data, len(dz), orig_bytes, enc_s, 0.0)
        del packed, dz

        all_results[ds] = res

    merged = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            merged = json.load(f)
    merged.update(all_results)
    with open(OUT, "w") as f:
        json.dump(merged, f, indent=1)
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()

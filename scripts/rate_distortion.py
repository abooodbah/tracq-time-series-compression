import argparse
import time
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Make repo root importable when run as a script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tracq.core import TimeSeriesGrid
from tracq.codec import ImageCodec
from tracq.container import pack as pack_zst
from tracq.metrics import rmse, mae, mape, smape, corr


def run_tracq_png(arr, bits, clamp, epsilon, png_level):
    grid = TimeSeriesGrid(arr, clamp_pct=clamp, epsilon=epsilon)
    if bits == 8:
        q, meta = grid.quantize_8bit()
    else:
        q, meta = grid.quantize_16bit()
    meta = dict(meta)
    meta["bit_depth"] = bits
    import os
    import tempfile
    t0 = time.perf_counter()
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        ImageCodec.save_png(tmp_path, q, meta, compress_level=png_level)
        png_size = os.path.getsize(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    t2 = time.perf_counter()
    recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q, meta)
    return {
        "bytes": int(png_size),
        "time_s": t2 - t0,
        "metrics": metrics(arr, recon),
    }


def run_tracq_zst(arr, bits, clamp, epsilon, level):
    grid = TimeSeriesGrid(arr, clamp_pct=clamp, epsilon=epsilon)
    if bits == 8:
        q, meta = grid.quantize_8bit()
    else:
        q, meta = grid.quantize_16bit()
    meta = dict(meta)
    meta["bit_depth"] = bits
    t0 = time.perf_counter()
    blob = pack_zst(meta, q, compress_level=level)
    t1 = time.perf_counter()
    recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q, meta)
    return {
        "bytes": len(blob),
        "time_s": t1 - t0,
        "metrics": metrics(arr, recon),
    }


def metrics(orig, recon):
    return {
        "rmse": rmse(orig, recon),
        "mae": mae(orig, recon),
        "mape": mape(orig, recon),
        "smape": smape(orig, recon),
        "corr": corr(orig, recon),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bits", nargs="*", type=int, default=[8, 16])
    ap.add_argument("--clamps", nargs="*", type=float, default=[200.0, 500.0])
    ap.add_argument("--png-levels", nargs="*", type=int, default=[0, 3, 6])
    ap.add_argument("--zstd-levels", nargs="*", type=int, default=[1, 3, 6])
    args = ap.parse_args()

    df = pd.read_csv(args.input, header=None)
    arr = df.values.T.astype(float)

    results = []
    for bits in args.bits:
        for clamp in args.clamps:
            for lvl in args.png_levels:
                r = run_tracq_png(arr, bits, clamp, 1e-9, lvl)
                r.update({"kind": "png", "bits": bits, "clamp": clamp, "png_level": lvl})
                results.append(r)
            for zl in args.zstd_levels:
                r = run_tracq_zst(arr, bits, clamp, 1e-9, zl)
                r.update({"kind": "zst", "bits": bits, "clamp": clamp, "zstd_level": zl})
                results.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

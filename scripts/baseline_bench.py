import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Make repo root importable when run as a script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tracq.baselines import (
    paa_compress, paa_decompress,
    pla_compress, pla_decompress,
    sax_compress, sax_decompress,
    gorilla_like_compress, gorilla_like_decompress,
)
from tracq.metrics import rmse, mae, mape, smape, corr


def bench(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, t1 - t0


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
    ap.add_argument("--segments", type=int, default=64)
    ap.add_argument("--alphabet", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.input, header=None)
    arr = df.values.T.astype(float)
    orig_bytes = args.input.stat().st_size

    results = {}

    (paa_coeffs, paa_meta), t_enc = bench(paa_compress, arr, args.segments)
    recon, t_dec = bench(paa_decompress, paa_coeffs, paa_meta)
    results["paa"] = {
        "bytes": paa_coeffs.nbytes,
        "time_s": t_enc + t_dec,
        "ratio": paa_coeffs.nbytes / orig_bytes,
        "metrics": metrics(arr, recon),
    }

    (pla_coeffs, pla_meta), t_enc = bench(pla_compress, arr, args.segments)
    recon, t_dec = bench(pla_decompress, pla_coeffs, pla_meta)
    results["pla"] = {
        "bytes": pla_coeffs.nbytes,
        "time_s": t_enc + t_dec,
        "ratio": pla_coeffs.nbytes / orig_bytes,
        "metrics": metrics(arr, recon),
    }

    (symbols, sax_meta), t_enc = bench(sax_compress, arr, args.segments, args.alphabet)
    recon, t_dec = bench(sax_decompress, symbols, sax_meta)
    results["sax"] = {
        "bytes": symbols.nbytes,
        "time_s": t_enc + t_dec,
        "ratio": symbols.nbytes / orig_bytes,
        "metrics": metrics(arr, recon),
    }

    (g_bytes, g_meta), t_enc = bench(gorilla_like_compress, arr)
    recon, t_dec = bench(gorilla_like_decompress, g_bytes, g_meta)
    results["gorilla_like"] = {
        "bytes": len(g_bytes),
        "time_s": t_enc + t_dec,
        "ratio": len(g_bytes) / orig_bytes,
        "metrics": metrics(arr, recon),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _json_sanitize(x):
        if isinstance(x, dict):
            return {k: _json_sanitize(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_json_sanitize(v) for v in x]
        if isinstance(x, float):
            if not np.isfinite(x):
                return None
            return float(x)
        if isinstance(x, (np.floating,)):
            x = float(x)
            return x if np.isfinite(x) else None
        if isinstance(x, (np.integer,)):
            return int(x)
        return x

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            _json_sanitize({"input": str(args.input), "orig_bytes": orig_bytes, "runs": results}),
            f,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()

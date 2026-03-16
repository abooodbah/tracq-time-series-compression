"""Compare TRACQ (best sweep point) vs PAA on a single dataset.

Produces:
 - JSON full benchmark (like `full_benchmark`)
 - Per-variable RMSE CSV comparing TRACQ and PAA
 - PNGs showing original vs reconstructions for worst variables

Usage:
  conda run -n gtc-env python scripts/compare_tracq_paa.py \
      --input data/processed/uci_appliances_energy.csv \
      --outdir paper_results/rerun_appliances --target-rmse 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
from pathlib import Path as _P
# make repo importable when run as a script
_REPO_ROOT = _P(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tracq.tooling import full_benchmark, _tracq_quantize_array
from tracq.baselines import paa_compress, paa_decompress
from tracq.core import TimeSeriesGrid


def per_var_rmse(orig: np.ndarray, recon: np.ndarray) -> np.ndarray:
    # orig/recon are (n_vars, n_time)
    return np.sqrt(np.mean((orig - recon) ** 2, axis=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("paper_results/rerun"))
    ap.add_argument("--target-rmse", type=float, default=25.0)
    ap.add_argument("--segments", type=int, default=64)
    args = ap.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)

    stats = full_benchmark(
        str(args.input),
        segments=args.segments,
        tracq_sweep_mode=True,
        tracq_target_rmse=float(args.target_rmse),
    )

    # Save full benchmark JSON
    out_json = args.outdir / (args.input.stem + ".full_benchmark.json")
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Load original data again
    df = pd.read_csv(args.input, header=None)
    orig = df.values.T.astype(float)

    # Reconstruct TRACQ using chosen best params
    best = stats.get("tracq_best")
    if best is None:
        raise SystemExit("no tracq_best found in benchmark output")
    params = best.get("params") or {}
    bits = int(params.get("bits", 16))
    clamp = float(params.get("clamp", 500.0))

    q, meta = _tracq_quantize_array(orig, bits=bits, clamp=clamp)
    recon_tracq, _ = TimeSeriesGrid.reconstruct_from_quantized(q, meta)

    # Reconstruct PAA
    paa_coeffs, paa_meta = paa_compress(orig, int(args.segments))
    recon_paa = paa_decompress(paa_coeffs, paa_meta)

    # Per-variable RMSE
    rmse_tracq = per_var_rmse(orig, recon_tracq)
    rmse_paa = per_var_rmse(orig, recon_paa)

    cmp_df = pd.DataFrame({
        "var_idx": np.arange(orig.shape[0]),
        "rmse_tracq": rmse_tracq,
        "rmse_paa": rmse_paa,
        "diff": rmse_tracq - rmse_paa,
    })
    cmp_csv = args.outdir / (args.input.stem + ".pervar_rmse.csv")
    cmp_df.to_csv(cmp_csv, index=False)

    # Save a few diagnostic plots (optional; require matplotlib)
    try:
        import matplotlib.pyplot as plt

        worst = cmp_df.sort_values("diff", ascending=False).head(3)["var_idx"].tolist()
        for vi in worst:
            t = np.arange(orig.shape[1])
            plt.figure(figsize=(8, 3))
            plt.plot(t, orig[vi], label="orig", linewidth=1)
            plt.plot(t, recon_tracq[vi], label="tracq", linewidth=1)
            plt.plot(t, recon_paa[vi], label="paa", linewidth=1)
            plt.title(f"var {vi} (rmse_tracq={rmse_tracq[vi]:.2f}, rmse_paa={rmse_paa[vi]:.2f})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.outdir / (f"{args.input.stem}.var{vi}.compare.png"))
            plt.close()
    except Exception:
        # matplotlib might be missing; that's not fatal
        pass

    print(f"Wrote: {out_json}\nWrote: {cmp_csv}\n(plots saved to {args.outdir} if matplotlib is available)")


if __name__ == "__main__":
    raise SystemExit(main())

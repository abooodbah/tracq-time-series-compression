"""Regenerate the paper's synthetic-suite tables with the lattice codec as the
"enhanced variant". Produces paper_results/lattice/paper_tables.json with:

  T1: 8-bit RMSE by data type (base vs enhanced), 50 vars x 1000 steps
  T2: multi-scale handling (base / enhanced-abs / enhanced-rel), scales 1e-2..1e4
  T3: drift on a 10,000-step financial sequence (RMSE @1k/5k/end)
  T4: compression vs accuracy trade-off (ratio + RMSE at 8-bit)
  SMAPE: per-variable SMAPE on Appliances (for the SMAPE figure)
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tracq import lattice
from tracq.core import TimeSeriesGrid

spec = importlib.util.spec_from_file_location(
    "te", os.path.join(PROJECT_ROOT, "scripts", "test_enhancements.py"))
te = importlib.util.module_from_spec(spec)
spec.loader.exec_module(te)

OUT = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "paper_tables.json")


def gen_multiscale(n_vars=50, n_time=1000, seed=42):
    """Variables spanning six orders of magnitude (1e-2 .. 1e4), paper spec."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n_vars, n_time))
    scales = np.logspace(-2, 4, n_vars)
    for i, s in enumerate(scales):
        noise = rng.normal(0, 0.02 * s, n_time)
        diurnal = 0.2 * s * np.sin(2 * np.pi * np.arange(n_time) / 167 + rng.uniform(0, 6))
        data[i] = s + diurnal + np.cumsum(noise) * 0.05
    return data


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def base_8bit(data):
    g = TimeSeriesGrid(data)
    q, meta = g.quantize_8bit()
    recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q, meta)
    import zstandard as zstd

    blob = zstd.ZstdCompressor(level=19).compress(q.tobytes() + json.dumps(meta).encode())
    return recon, len(blob)


def enh(data, eps=1e-3, mode="abs"):
    blob, grid, header = lattice.encode(data, eps=eps, mode=mode, predictors="bank")
    recon, _ = lattice.decode(blob)
    return recon, len(blob)


def main():
    results = {}

    # ---- T1: RMSE by data type, 8-bit grids ----
    t1 = {}
    for dt in ["sensor", "financial", "iot", "electricity"]:
        data = te.generate_test_data(n_vars=50, n_time=1000, data_type=dt)
        rb, _ = base_8bit(data)
        re_, _ = enh(data, eps=1e-3, mode="abs")
        t1[dt] = {"base": rmse(data, rb), "enh": rmse(data, re_),
                  "improvement_pct": 100 * (1 - rmse(data, re_) / rmse(data, rb))}
    data = gen_multiscale()
    rb, _ = base_8bit(data)
    re_, _ = enh(data, eps=1e-3, mode="abs")
    t1["multi_scale"] = {"base": rmse(data, rb), "enh": rmse(data, re_),
                         "improvement_pct": 100 * (1 - rmse(data, re_) / rmse(data, rb))}
    results["T1"] = t1
    print("T1:", json.dumps(t1, indent=1))

    # ---- T2: multi-scale handling ----
    data = gen_multiscale()
    rb, _ = base_8bit(data)
    ra, _ = enh(data, eps=1e-3, mode="abs")
    rr, _ = enh(data, eps=1e-3, mode="rel")
    def relerr(recon):
        scale = np.abs(data).mean(axis=1, keepdims=True)
        return float(np.mean(np.abs(recon - data) / scale) * 100)
    results["T2"] = {
        "base":    {"rmse": rmse(data, rb), "max_err": float(np.abs(data-rb).max()),  "rel_err_pct": relerr(rb)},
        "enh_abs": {"rmse": rmse(data, ra), "max_err": float(np.abs(data-ra).max()),  "rel_err_pct": relerr(ra)},
        "enh_rel": {"rmse": rmse(data, rr), "max_err": float(np.abs(data-rr).max()),  "rel_err_pct": relerr(rr)},
    }
    print("T2:", json.dumps(results["T2"], indent=1))

    # ---- T3: drift on 10k-step financial ----
    data = te.generate_test_data(n_vars=50, n_time=10000, data_type="financial")
    rb, _ = base_8bit(data)
    re_, _ = enh(data, eps=1e-3, mode="abs")
    def upto(recon, t):
        return rmse(data[:, :t], recon[:, :t])
    results["T3"] = {
        "base": {"rmse_1k": upto(rb, 1000), "rmse_5k": upto(rb, 5000), "rmse_end": rmse(data, rb)},
        "enh":  {"rmse_1k": upto(re_, 1000), "rmse_5k": upto(re_, 5000), "rmse_end": rmse(data, re_)},
    }
    print("T3:", json.dumps(results["T3"], indent=1))

    # ---- T4: ratio vs accuracy trade-off (sensor workload, paper setup) ----
    data = te.generate_test_data(n_vars=50, n_time=1000, data_type="sensor")
    orig_bytes = data.nbytes
    rb, bb = base_8bit(data)
    t4 = {"base": {"ratio": bb / orig_bytes, "rmse": rmse(data, rb)}}
    for eps, tag in [(1e-2, "enh_eps1e-2"), (1e-3, "enh_eps1e-3"), (1e-4, "enh_eps1e-4")]:
        re_, be = enh(data, eps=eps, mode="abs")
        t4[tag] = {"ratio": be / orig_bytes, "rmse": rmse(data, re_)}
    results["T4"] = t4
    print("T4:", json.dumps(t4, indent=1))

    # ---- per-variable SMAPE on Appliances (figure data) ----
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "uci_appliances_energy.csv"), header=None)
    data = df.values.T.astype(np.float64)
    def pv_smape(recon):
        out = []
        for i in range(data.shape[0]):
            denom = (np.abs(data[i]) + np.abs(recon[i])) / 2
            mask = denom > 1e-12
            out.append(float(np.mean(np.abs(data[i] - recon[i])[mask] / denom[mask])))
        return out
    g16 = TimeSeriesGrid(data)
    q16, meta16 = g16.quantize_16bit()
    rb16, _ = TimeSeriesGrid.reconstruct_from_quantized(q16, meta16)
    smape_data = {
        "var_mean_abs": np.abs(data).mean(axis=1).tolist(),
        "base_16bit": pv_smape(rb16),
    }
    for eps, tag in [(1e-2, "enh_rel_eps1e-2"), (1e-3, "enh_rel_eps1e-3")]:
        re_, _ = enh(data, eps=eps, mode="rel")
        smape_data[tag] = pv_smape(re_)
    try:
        import zfpy
        rz = zfpy.decompress_numpy(zfpy.compress_numpy(np.ascontiguousarray(data), tolerance=1e-1))
        smape_data["zfp_tol_0.1"] = pv_smape(rz)
    except Exception:
        pass
    results["SMAPE_pervar"] = smape_data

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()

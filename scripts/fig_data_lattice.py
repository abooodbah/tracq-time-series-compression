"""Collect every remaining measurement needed to regenerate the paper figures
with the lattice codec as the enhanced variant. Saves JSON/NPZ under
paper_results/lattice/figdata/."""

import ctypes
import importlib.util
import json
import os
import sys
import time

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

OUTDIR = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "figdata")
os.makedirs(OUTDIR, exist_ok=True)


def peak_rss_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 1e6


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def base_8bit_recon(data):
    g = TimeSeriesGrid(data)
    q, meta = g.quantize_8bit()
    recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q, meta)
    return recon


def enh_recon(data, eps=1e-3, mode="abs", predictors="bank"):
    blob, _, _ = lattice.encode(data, eps=eps, mode=mode, predictors=predictors)
    recon, _ = lattice.decode(blob)
    return recon, len(blob)


def main():
    out = {}

    # ---- fig1: ablation grid ----
    ab = {}
    for dt in ["sensor", "financial", "iot", "electricity"]:
        data = te.generate_test_data(50, 1000, dt)
        row = {"base": rmse(data, base_8bit_recon(data))}
        for tag, kw in [("lattice_p1", dict(predictors="p1", mode="abs")),
                        ("lattice_bank", dict(predictors="bank", mode="abs")),
                        ("lattice_rel", dict(predictors="bank", mode="rel"))]:
            r, _ = enh_recon(data, eps=1e-3, **kw)
            row[tag] = rmse(data, r)
        ab[dt] = row
        print("fig1", dt, {k: round(v, 3) for k, v in row.items()})
    out["ablation"] = ab

    # ---- fig2: multi-scale per-variable relative error ----
    spec2 = importlib.util.spec_from_file_location(
        "ptab", os.path.join(PROJECT_ROOT, "scripts", "paper_tables_lattice.py"))
    ptab = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ptab)
    data = ptab.gen_multiscale()
    scales = np.logspace(-2, 4, 50)
    ms = {"scales": scales.tolist()}
    for tag, recon in [
        ("base", base_8bit_recon(data)),
        ("enh_abs", enh_recon(data, 1e-3, "abs")[0]),
        ("enh_rel", enh_recon(data, 1e-3, "rel")[0]),
    ]:
        per_var = np.abs(recon - data).mean(axis=1) / np.abs(data).mean(axis=1) * 100
        ms[tag] = per_var.tolist()
    out["multiscale"] = ms
    print("fig2 done")

    # ---- fig3: drift curves ----
    data = te.generate_test_data(50, 10000, "financial")
    rb = base_8bit_recon(data)
    re_, _ = enh_recon(data, 1e-3, "abs")
    checkpoints = list(range(100, 10001, 100))
    drift = {"t": checkpoints,
             "base": [rmse(data[:, :t], rb[:, :t]) for t in checkpoints],
             "enh": [rmse(data[:, :t], re_[:, :t]) for t in checkpoints]}
    rng_i = data.max(axis=1) - data.min(axis=1)
    drift["bound"] = float(np.sqrt(np.mean((1e-3 * rng_i) ** 2)))
    out["drift"] = drift
    print("fig3 done, enh end:", drift["enh"][-1], "bound:", drift["bound"])

    # ---- fig4/5: RD sweep on sensor + ZFP ----
    data = te.generate_test_data(50, 1000, "sensor")
    orig = data.nbytes
    rd = {"base": [], "enh_abs": [], "enh_rel": [], "zfp": []}
    import zstandard as zstd
    for bits in (8, 16):
        g = TimeSeriesGrid(data)
        q, meta = g.quantize_8bit() if bits == 8 else g.quantize_16bit()
        recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q, meta)
        blob = zstd.ZstdCompressor(level=19).compress(q.tobytes() + json.dumps(meta).encode())
        rd["base"].append({"bits": bits, "ratio": len(blob) / orig, "rmse": rmse(data, recon)})
    for mode in ("abs", "rel"):
        for eps in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
            r, nb = enh_recon(data, eps, mode)
            rd[f"enh_{mode}"].append({"eps": eps, "ratio": nb / orig, "rmse": rmse(data, r)})
    try:
        import zfpy
        for tol in (1e0, 1e-1, 1e-2, 1e-3, 1e-4):
            zb = zfpy.compress_numpy(np.ascontiguousarray(data), tolerance=tol)
            rz = zfpy.decompress_numpy(zb)
            rd["zfp"].append({"tol": tol, "ratio": len(zb) / orig, "rmse": rmse(data, rz)})
    except Exception as e:
        print("zfp skip:", e)
    out["rd_sensor"] = rd
    print("fig4/5 done")

    # ---- fig6: throughput vs data size ----
    th = {}
    for nv, nt in [(50, 500), (100, 1000), (200, 2000), (500, 5000)]:
        data = te.generate_test_data(nv, nt, "sensor")
        mb = data.nbytes / 1e6
        entry = {}
        g = TimeSeriesGrid(data)
        t0 = time.perf_counter(); q, meta = g.quantize_8bit(); enc = time.perf_counter() - t0
        t0 = time.perf_counter(); TimeSeriesGrid.reconstruct_from_quantized(q, meta); dec = time.perf_counter() - t0
        entry["base"] = {"enc": mb / enc, "dec": mb / dec}
        t0 = time.perf_counter(); blob, _, _ = lattice.encode(data, eps=1e-3, mode="abs", predictors="p1", zstd_level=1); enc = time.perf_counter() - t0
        t0 = time.perf_counter(); lattice.decode(blob); dec = time.perf_counter() - t0
        entry["enh"] = {"enc": mb / enc, "dec": mb / dec}
        try:
            import zfpy
            arr = np.ascontiguousarray(data)
            t0 = time.perf_counter(); zb = zfpy.compress_numpy(arr, tolerance=1e-3); enc = time.perf_counter() - t0
            t0 = time.perf_counter(); zfpy.decompress_numpy(zb); dec = time.perf_counter() - t0
            entry["zfp"] = {"enc": mb / enc, "dec": mb / dec}
        except Exception:
            pass
        th[f"{nv}x{nt}"] = entry
        print("fig6", f"{nv}x{nt}", {k: (round(v['enc']), round(v['dec'])) for k, v in entry.items()})
    out["throughput_sizes"] = th

    # ---- fig8: streaming scaling + MetroPT-3 windowed ----
    stream = {"rows": [], "enc_mbs": [], "peak_rss_mb": []}
    rng = np.random.default_rng(0)
    for total_rows in (10**6, 10**7, 10**8):
        window = 100_000
        n_windows = total_rows // window
        t_enc = 0.0
        nbytes_total = 0
        for w in range(n_windows):
            base = 100 + 10 * rng.standard_normal((16, 1))
            wdata = base + np.cumsum(rng.normal(0, 0.05, (16, window)), axis=1)
            t0 = time.perf_counter()
            blob, _, _ = lattice.encode(wdata, eps=1e-3, mode="abs", predictors="p1", zstd_level=1)
            t_enc += time.perf_counter() - t0
            nbytes_total += len(blob)
            del wdata, blob
        mb = total_rows * 16 * 8 / 1e6
        stream["rows"].append(total_rows)
        stream["enc_mbs"].append(mb / t_enc)
        stream["peak_rss_mb"].append(peak_rss_mb())
        print("fig8 stream", total_rows, round(mb / t_enc, 1), "MB/s rss", round(peak_rss_mb()))
    out["streaming"] = stream

    mp = {}
    csv = os.path.join(PROJECT_ROOT, "data", "raw", "metropt3", "MetroPT3(AirCompressor).csv")
    df = pd.read_csv(csv)
    num = df.select_dtypes(include=[np.number])
    mdata = num.values.T.astype(np.float64)
    mdata = np.where(np.isfinite(mdata), mdata, 0.0)
    mb = mdata.nbytes / 1e6
    window = 10_000
    for tag, kw in [("enh_fast", dict(zstd_level=1)), ("enh_archival", dict(zstd_level=19))]:
        t_enc = 0.0
        nb = 0
        for s in range(0, mdata.shape[1], window):
            wdata = mdata[:, s:s + window]
            t0 = time.perf_counter()
            blob, _, _ = lattice.encode(wdata, eps=1e-3, mode="abs", predictors="p1", **kw)
            t_enc += time.perf_counter() - t0
            nb += len(blob)
        mp[tag] = {"enc_mbs": mb / t_enc, "ratio": nb / mdata.nbytes}
        print("fig8 metropt", tag, mp[tag])
    import gzip as _gz
    t0 = time.perf_counter()
    gz = _gz.compress(mdata.tobytes(), compresslevel=6)
    mp["gzip"] = {"enc_mbs": mb / (time.perf_counter() - t0), "ratio": len(gz) / mdata.nbytes}
    del gz
    out["metropt_stream"] = mp

    # ---- fig13: visual inspection grids ----
    dfa = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "uci_air_quality.csv"), header=None)
    adata = dfa.values.T.astype(np.float64)[:, :1000]
    anom = adata.copy()
    rng13 = np.random.default_rng(7)
    v_spike, v_shift, v_osc = 2, 5, 9
    anom[v_spike, 400] += 8 * adata[v_spike].std()
    anom[v_shift, 600:] += 5 * adata[v_shift].std()
    tt = np.arange(300)
    anom[v_osc, 500:800] += 2 * adata[v_osc].std() * np.sin(2 * np.pi * tt / 12)
    _, gn, _ = lattice.encode(adata, eps=1e-2, mode="rel", predictors="p1")
    _, ga, _ = lattice.encode(anom, eps=1e-2, mode="rel", predictors="p1")
    np.savez(os.path.join(OUTDIR, "visual_demo.npz"),
             normal=adata, anomalous=anom, grid_normal=gn, grid_anom=ga,
             marks=np.array([v_spike, v_shift, v_osc]))
    print("fig13 grids saved", gn.shape)

    # ---- fig14: anomaly throughput (decode+detect vs direct) ----
    spec3 = importlib.util.spec_from_file_location(
        "anom", os.path.join(PROJECT_ROOT, "scripts", "anomaly_detection_experiment.py"))
    am = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(am)
    la = importlib.util.spec_from_file_location(
        "lam", os.path.join(PROJECT_ROOT, "scripts", "lattice_anomaly_experiment.py"))
    lam = importlib.util.module_from_spec(la)
    la.loader.exec_module(lam)

    wdata = am.load_appliances_data(am.DEFAULT_DATA_DIR)
    windows, labels, _ = am.create_labeled_dataset(wdata)
    blobs = [lattice.encode(w, eps=3e-2, mode="rel", predictors="p1", zstd_level=3)[0] for w in windows]
    from sklearn.ensemble import IsolationForest

    # decode+detect: blob -> float64 -> numerical features -> IF
    t0 = time.perf_counter()
    feats_num = []
    for b in blobs:
        dec, _ = lattice.decode(b)
        feats_num.append(am.extract_numerical_features(dec))
    feats_num = np.nan_to_num(np.array(feats_num), nan=0, posinf=1e6, neginf=-1e6)
    iso = IsolationForest(contamination=0.5, random_state=42, n_estimators=100)
    iso.fit(feats_num)
    iso.predict(feats_num)
    t_decode_path = time.perf_counter() - t0

    # direct: blob -> grid -> per-row grid features -> IF
    def per_row_features(grid):
        r = grid.astype(np.int64) - 128
        r[grid == lattice.ESCAPE] = 127
        a = np.abs(r).astype(np.float64)
        traj = np.cumsum(r, axis=1).astype(np.float64)
        scale = np.maximum(np.percentile(a, 90, axis=1), 1.0)
        tn = traj / scale[:, None]
        half = tn.shape[1] // 2
        return np.column_stack([
            a.mean(axis=1), a.max(axis=1), (a > 2).mean(axis=1), a.std(axis=1),
            tn.max(axis=1) - tn.min(axis=1),
            np.abs(tn[:, half:].mean(axis=1) - tn[:, :half].mean(axis=1)),
        ]).ravel()

    t0 = time.perf_counter()
    feats_g = []
    for b in blobs:
        _, grid, _, _ = lattice._unpack(b)
        feats_g.append(per_row_features(grid))
    feats_g = np.nan_to_num(np.array(feats_g), nan=0, posinf=1e6, neginf=-1e6)
    iso2 = IsolationForest(contamination=0.5, random_state=42, n_estimators=100)
    iso2.fit(feats_g)
    p = (iso2.predict(feats_g) == -1).astype(int)
    t_direct_path = time.perf_counter() - t0

    from sklearn.metrics import f1_score
    out["anomaly_throughput"] = {
        "decode_detect_wps": len(windows) / t_decode_path,
        "direct_wps": len(windows) / t_direct_path,
        "speedup": t_decode_path / t_direct_path,
        "direct_f1_check": float(f1_score(labels, p)),
    }
    print("fig14", out["anomaly_throughput"])

    with open(os.path.join(OUTDIR, "figdata.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("Saved:", os.path.join(OUTDIR, "figdata.json"))


if __name__ == "__main__":
    main()

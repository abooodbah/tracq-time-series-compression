#!/usr/bin/env python
"""Paper-ready experiments: RD curves, Pareto frontiers, ablations, failure-mode logging, figures, and tables.

Outputs:
  - paper_results/experiments/rd_curves/          RD sweep JSONs per dataset
  - paper_results/experiments/figures/            PNG figures (RD curves, Pareto, reconstructions)
  - paper_results/experiments/tables/             CSV/LaTeX tables
  - paper_results/experiments/failure_modes.json  Params that produced NaN/Inf/explosive RMSE

Usage:
  conda run -n gtc-env python scripts/paper_experiments.py --run-all
  conda run -n gtc-env python scripts/paper_experiments.py --rd-sweeps
  conda run -n gtc-env python scripts/paper_experiments.py --figures
  conda run -n gtc-env python scripts/paper_experiments.py --tables
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracq.tooling import full_benchmark, tracq_sweep, _tracq_quantize_array, _metrics
from tracq.baselines import paa_compress, paa_decompress
from tracq.core import TimeSeriesGrid

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS: Dict[str, Path] = {
    "electricity": REPO_ROOT / "uci_electricity_sample.cleaned.csv",
    "appliances_energy": REPO_ROOT / "data" / "processed" / "uci_appliances_energy.csv",
    "air_quality": REPO_ROOT / "data" / "processed" / "uci_air_quality.csv",
}

# Sweep grid for RD curves (comprehensive)
# Include sub-8-bit quantization to visualize bit-depth tradeoffs.
SWEEP_BITS = (2, 4, 8, 16)
SWEEP_CLAMPS = (50, 100, 200, 500, 1000)
SWEEP_PNG_LEVELS = (0, 6)
SWEEP_ZSTD_LEVELS = (1, 3, 6)

# Ablation-specific grids
ABLATION_BITS = (8, 16)
ABLATION_CLAMPS = (100, 200, 500)

# Thresholds for failure detection
RMSE_EXPLOSION_THRESHOLD = 1e6

# Conservative defaults for what we consider a "good" (non-degenerate) TRACQ point
# when summarizing into a single row/figure marker.
DEFAULT_MIN_CORR = 0.95
DEFAULT_MAX_REL_RMSE = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_csv_as_array(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    return df.values.T.astype(float)


def compute_relative_metrics(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    """Compute scale-robust metrics: relative RMSE (RMSE/std), normalized RMSE, sMAPE."""
    from tracq.metrics import rmse, mae, smape, corr

    abs_rmse = float(rmse(orig, recon))
    abs_mae = float(mae(orig, recon))
    abs_smape = float(smape(orig, recon))
    abs_corr = float(corr(orig, recon))

    # Relative RMSE: RMSE / std(orig)
    std_orig = float(np.std(orig))
    rel_rmse = abs_rmse / std_orig if std_orig > 1e-9 else np.nan

    # Normalized RMSE: RMSE / (max - min)
    range_orig = float(np.max(orig) - np.min(orig))
    nrmse = abs_rmse / range_orig if range_orig > 1e-9 else np.nan

    return {
        "rmse": abs_rmse,
        "mae": abs_mae,
        "smape": abs_smape,
        "corr": abs_corr,
        "rel_rmse": rel_rmse,
        "nrmse": nrmse,
    }


def per_var_metrics(orig: np.ndarray, recon: np.ndarray) -> pd.DataFrame:
    """Compute per-variable RMSE/MAE/rel_rmse."""
    n_vars = orig.shape[0]
    rows = []
    for i in range(n_vars):
        o = orig[i]
        r = recon[i]
        rmse_i = float(np.sqrt(np.mean((o - r) ** 2)))
        mae_i = float(np.mean(np.abs(o - r)))
        std_i = float(np.std(o))
        rel_rmse_i = rmse_i / std_i if std_i > 1e-9 else np.nan
        rows.append({"var_idx": i, "rmse": rmse_i, "mae": mae_i, "std": std_i, "rel_rmse": rel_rmse_i})
    return pd.DataFrame(rows)


def is_failure(metrics: Dict[str, Any]) -> bool:
    """Detect NaN/Inf or explosive RMSE."""
    rmse_val = metrics.get("rmse")
    if rmse_val is None:
        return True
    if not np.isfinite(rmse_val):
        return True
    if rmse_val > RMSE_EXPLOSION_THRESHOLD:
        return True
    return False


def pareto_frontier(runs: List[Dict[str, Any]], *, x_key: str, y_metric_key: str) -> List[Dict[str, Any]]:
    """Pareto frontier for minimizing x_key and metrics[y_metric_key]."""
    pts: List[Dict[str, Any]] = []
    for r in runs:
        m = r.get("metrics", {})
        if is_failure(m):
            continue
        x = r.get(x_key)
        y = m.get(y_metric_key)
        if x is None or y is None:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        pts.append(r)

    pts.sort(key=lambda rr: (rr.get(x_key, 1 << 60), rr.get("metrics", {}).get(y_metric_key, float("inf"))))

    frontier: List[Dict[str, Any]] = []
    best_y = float("inf")
    for r in pts:
        y = float(r["metrics"][y_metric_key])
        if y < best_y:
            frontier.append(r)
            best_y = y
    return frontier


def pick_best_tracq_for_reporting(
    res: Dict[str, Any],
    *,
    min_corr: float = DEFAULT_MIN_CORR,
    max_rel_rmse: float = DEFAULT_MAX_REL_RMSE,
) -> Optional[Dict[str, Any]]:
    """Pick a single TRACQ run for summary/plots without rewarding pathological points."""
    runs = [r for r in res.get("tracq_runs", []) if not is_failure(r.get("metrics", {}))]
    if not runs:
        return None

    feasible: List[Dict[str, Any]] = []
    for r in runs:
        m = r.get("metrics", {})
        corr_v = m.get("corr")
        rel_v = m.get("rel_rmse")
        if corr_v is None or rel_v is None:
            continue
        if not np.isfinite(corr_v) or not np.isfinite(rel_v):
            continue
        if float(corr_v) >= float(min_corr) and float(rel_v) <= float(max_rel_rmse):
            feasible.append(r)

    if feasible:
        return min(feasible, key=lambda rr: (rr.get("bytes", 1 << 60), rr.get("metrics", {}).get("rmse", float("inf"))))

    frontier = pareto_frontier(runs, x_key="bytes", y_metric_key="rmse")
    if not frontier:
        return None
    return max(frontier, key=lambda rr: (
        float(rr.get("metrics", {}).get("corr", -1.0)),
        -float(rr.get("metrics", {}).get("rel_rmse", float("inf"))),
        -float(rr.get("bytes", 0)),
    ))


# ---------------------------------------------------------------------------
# RD Sweep (comprehensive)
# ---------------------------------------------------------------------------
def run_rd_sweep(dataset_name: str, csv_path: Path, out_dir: Path) -> Dict[str, Any]:
    """Run comprehensive TRACQ sweep + baselines, compute absolute + relative metrics, log failures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    orig = load_csv_as_array(csv_path)
    orig_bytes = os.path.getsize(csv_path)
    n_vars, n_time = orig.shape

    results: Dict[str, Any] = {
        "dataset": dataset_name,
        "csv_path": str(csv_path),
        "orig_bytes": int(orig_bytes),
        "shape": {"n_vars": n_vars, "n_time": n_time},
        "tracq_runs": [],
        "baselines": {},
        "failures": [],
    }

    # --- TRACQ sweep ---
    from tracq.container import pack as pack_zst, unpack as unpack_zst
    from tracq.codec import ImageCodec
    import tempfile
    import time

    for bits in SWEEP_BITS:
        for clamp in SWEEP_CLAMPS:
            try:
                q, meta = _tracq_quantize_array(orig, bits=int(bits), clamp=float(clamp))
                recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q, meta)
                base_metrics = compute_relative_metrics(orig, recon)
            except Exception as e:
                results["failures"].append({"bits": bits, "clamp": clamp, "stage": "quantize", "error": str(e)})
                continue

            if is_failure(base_metrics):
                results["failures"].append({"bits": bits, "clamp": clamp, "metrics": base_metrics, "stage": "metrics"})

            # PNG variants
            for png_lvl in SWEEP_PNG_LEVELS:
                try:
                    fd, tmp_png = tempfile.mkstemp(suffix=".png")
                    os.close(fd)
                    t0 = time.perf_counter()
                    ImageCodec.save_png(tmp_png, q, meta, compress_level=int(png_lvl))
                    enc_s = time.perf_counter() - t0
                    size = os.path.getsize(tmp_png)
                    t0 = time.perf_counter()
                    q2, meta2 = ImageCodec.load_png(tmp_png)
                    recon2, _ = TimeSeriesGrid.reconstruct_from_quantized(q2, meta2)
                    dec_s = time.perf_counter() - t0
                    os.remove(tmp_png)
                    run_metrics = compute_relative_metrics(orig, recon2)
                    results["tracq_runs"].append({
                        "variant": "png",
                        "bits": bits,
                        "clamp": clamp,
                        "png_level": png_lvl,
                        "bytes": int(size),
                        "ratio": float(size / orig_bytes),
                        "encode_s": float(enc_s),
                        "decode_s": float(dec_s),
                        "metrics": run_metrics,
                    })
                except Exception as e:
                    results["failures"].append({"bits": bits, "clamp": clamp, "png_level": png_lvl, "stage": "png", "error": str(e)})

            # Zstd variants
            for zstd_lvl in SWEEP_ZSTD_LEVELS:
                try:
                    t0 = time.perf_counter()
                    blob = pack_zst(meta, q, compress_level=int(zstd_lvl))
                    enc_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    q3, meta3 = unpack_zst(blob)
                    recon3, _ = TimeSeriesGrid.reconstruct_from_quantized(q3, meta3)
                    dec_s = time.perf_counter() - t0
                    run_metrics = compute_relative_metrics(orig, recon3)
                    results["tracq_runs"].append({
                        "variant": "zstd",
                        "bits": bits,
                        "clamp": clamp,
                        "zstd_level": zstd_lvl,
                        "bytes": int(len(blob)),
                        "ratio": float(len(blob) / orig_bytes),
                        "encode_s": float(enc_s),
                        "decode_s": float(dec_s),
                        "metrics": run_metrics,
                    })
                except Exception as e:
                    results["failures"].append({"bits": bits, "clamp": clamp, "zstd_level": zstd_lvl, "stage": "zstd", "error": str(e)})

    # --- Baselines ---
    import gzip
    import io

    # gzip (lossless)
    with open(csv_path, "rb") as f:
        raw = f.read()
    t0 = time.perf_counter()
    gz = gzip.compress(raw)
    enc_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = gzip.decompress(gz)
    dec_s = time.perf_counter() - t0
    results["baselines"]["gzip"] = {
        "bytes": len(gz),
        "ratio": float(len(gz) / orig_bytes),
        "encode_s": float(enc_s),
        "decode_s": float(dec_s),
        "metrics": {"rmse": 0.0, "mae": 0.0, "smape": 0.0, "corr": 1.0, "rel_rmse": 0.0, "nrmse": 0.0},
    }

    # PAA
    for segs in (32, 64, 128):
        try:
            t0 = time.perf_counter()
            coeffs, paa_meta = paa_compress(orig, segs)
            enc_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            recon_paa = paa_decompress(coeffs, paa_meta)
            dec_s = time.perf_counter() - t0
            paa_metrics = compute_relative_metrics(orig, recon_paa)
            results["baselines"][f"paa_{segs}"] = {
                "bytes": int(coeffs.nbytes),
                "ratio": float(coeffs.nbytes / orig_bytes),
                "encode_s": float(enc_s),
                "decode_s": float(dec_s),
                "metrics": paa_metrics,
            }
        except Exception as e:
            results["failures"].append({"baseline": f"paa_{segs}", "error": str(e)})

    # Save JSON
    out_json = out_dir / f"{dataset_name}_rd_sweep.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------
def generate_rd_curve_figure(sweep_results: Dict[str, Dict], out_dir: Path):
    """Figure 1: RD curves (bytes vs RMSE) for multiple datasets."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure generation")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(sweep_results), figsize=(5 * len(sweep_results), 4), squeeze=False)

    for idx, (dname, res) in enumerate(sweep_results.items()):
        ax = axes[0, idx]

        # TRACQ PNG
        png_runs = [r for r in res["tracq_runs"] if r["variant"] == "png" and not is_failure(r["metrics"])]
        if png_runs:
            xs = [r["bytes"] / 1024 for r in png_runs]
            ys = [r["metrics"]["rmse"] for r in png_runs]
            ax.scatter(xs, ys, label="TRACQ-PNG", alpha=0.7, marker="o")

        # TRACQ Zstd
        zstd_runs = [r for r in res["tracq_runs"] if r["variant"] == "zstd" and not is_failure(r["metrics"])]
        if zstd_runs:
            xs = [r["bytes"] / 1024 for r in zstd_runs]
            ys = [r["metrics"]["rmse"] for r in zstd_runs]
            ax.scatter(xs, ys, label="TRACQ-Zstd", alpha=0.7, marker="s")

        # PAA baselines
        for bname, bdata in res["baselines"].items():
            if bname.startswith("paa"):
                ax.scatter([bdata["bytes"] / 1024], [bdata["metrics"]["rmse"]], label=bname.upper(), marker="^", s=100)

        # gzip (lossless, RMSE=0)
        gzip_data = res["baselines"].get("gzip")
        if gzip_data:
            ax.axvline(gzip_data["bytes"] / 1024, color="gray", linestyle="--", label="gzip (lossless)")

        ax.set_xlabel("Compressed Size (KB)")
        ax.set_ylabel("RMSE")
        ax.set_title(dname)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        chosen = pick_best_tracq_for_reporting(res)
        if chosen is not None:
            ax.scatter(
                [chosen["bytes"] / 1024],
                [chosen["metrics"]["rmse"]],
                label="TRACQ (chosen)",
                marker="*",
                s=180,
                c="black",
            )

    plt.tight_layout()
    fig.savefig(out_dir / "figure1_rd_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'figure1_rd_curves.png'}")


def generate_pareto_figure(sweep_results: Dict[str, Dict], out_dir: Path):
    """Figure 2: Useful Pareto frontiers (per-dataset).

    Produces:
      - figure2_pareto_bitdepth.png        bytes vs correlation (min bytes, max corr)
      - figure2d_pareto_ratio_vs_rmse.png  ratio vs RMSE (min ratio, min rmse)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    def _runs_by_bits(res: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
        by: Dict[int, List[Dict[str, Any]]] = {}
        for r in res.get("tracq_runs", []):
            if is_failure(r.get("metrics", {})):
                continue
            b = int(r.get("bits", 0))
            by.setdefault(b, []).append(r)
        return by

    def _frontier_minmin(points: List[Tuple[float, float, Dict[str, Any]]]) -> List[Tuple[float, float, Dict[str, Any]]]:
        points = [(x, y, r) for (x, y, r) in points if np.isfinite(x) and np.isfinite(y)]
        points.sort(key=lambda t: (t[0], t[1]))
        out: List[Tuple[float, float, Dict[str, Any]]] = []
        best_y = float("inf")
        for x, y, r in points:
            if y < best_y:
                out.append((x, y, r))
                best_y = y
        return out

    def _frontier_minmax(points: List[Tuple[float, float, Dict[str, Any]]]) -> List[Tuple[float, float, Dict[str, Any]]]:
        # minimize x, maximize y
        points = [(x, y, r) for (x, y, r) in points if np.isfinite(x) and np.isfinite(y)]
        points.sort(key=lambda t: (t[0], -t[1]))
        out: List[Tuple[float, float, Dict[str, Any]]] = []
        best_y = -float("inf")
        for x, y, r in points:
            if y > best_y:
                out.append((x, y, r))
                best_y = y
        return out

    # Build color map from observed bits
    all_bits: List[int] = []
    for _, res in sweep_results.items():
        for run in res.get("tracq_runs", []):
            all_bits.append(int(run.get("bits", 0)))
    colors = _bit_color_map(all_bits or [2, 4, 8, 16])
    markers = {"png": "o", "zstd": "s"}

    # --- A) Bytes vs Correlation (per-dataset, actual frontier per bit) ---
    fig, axes = plt.subplots(1, len(sweep_results), figsize=(5 * len(sweep_results), 4), squeeze=False)
    for idx, (dname, res) in enumerate(sweep_results.items()):
        ax = axes[0, idx]
        by_bits = _runs_by_bits(res)
        for b, runs in sorted(by_bits.items(), key=lambda kv: kv[0]):
            # faint scatter
            for r in runs:
                x = float(r["bytes"]) / 1024.0
                y = float(r.get("metrics", {}).get("corr", np.nan))
                ax.scatter(x, y, c=colors.get(str(b), "gray"), alpha=0.18, s=22, marker=markers.get(r.get("variant"), "o"))

            pts = []
            for r in runs:
                x = float(r["bytes"]) / 1024.0
                y = float(r.get("metrics", {}).get("corr", np.nan))
                pts.append((x, y, r))
            fr = _frontier_minmax(pts)
            if fr:
                xs = [t[0] for t in fr]
                ys = [t[1] for t in fr]
                ax.plot(xs, ys, color=colors.get(str(b), "gray"), linewidth=1.6, label=f"{b}-bit frontier")

        chosen = pick_best_tracq_for_reporting(res)
        if chosen is not None:
            ax.scatter([chosen["bytes"] / 1024], [chosen["metrics"]["corr"]], marker="*", s=180, c="black", label="chosen")

        ax.set_title(dname)
        ax.set_xlabel("Compressed Size (KB)")
        ax.set_ylabel("Correlation")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.02)
        ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_dir / "figure2_pareto_bitdepth.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'figure2_pareto_bitdepth.png'}")

    # --- B) Ratio vs RMSE (per-dataset, frontier per bit) ---
    fig, axes = plt.subplots(1, len(sweep_results), figsize=(5 * len(sweep_results), 4), squeeze=False)
    for idx, (dname, res) in enumerate(sweep_results.items()):
        ax = axes[0, idx]
        by_bits = _runs_by_bits(res)
        rmse_vals: List[float] = []
        for b, runs in sorted(by_bits.items(), key=lambda kv: kv[0]):
            for r in runs:
                x = float(r.get("ratio", np.nan))
                y = float(r.get("metrics", {}).get("rmse", np.nan))
                if np.isfinite(x) and np.isfinite(y):
                    rmse_vals.append(y)
                ax.scatter(x, y, c=colors.get(str(b), "gray"), alpha=0.18, s=22, marker=markers.get(r.get("variant"), "o"))

            pts = []
            for r in runs:
                x = float(r.get("ratio", np.nan))
                y = float(r.get("metrics", {}).get("rmse", np.nan))
                pts.append((x, y, r))
            fr = _frontier_minmin(pts)
            if fr:
                xs = [t[0] for t in fr]
                ys = [t[1] for t in fr]
                ax.plot(xs, ys, color=colors.get(str(b), "gray"), linewidth=1.6, label=f"{b}-bit frontier")

        chosen = pick_best_tracq_for_reporting(res)
        if chosen is not None:
            ax.scatter([chosen["ratio"]], [chosen["metrics"]["rmse"]], marker="*", s=180, c="black", label="chosen")

        ax.set_title(dname)
        ax.set_xlabel("Compression Ratio (compressed/original)")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

        # If RMSE range is huge, log-scale helps (avoids plots being dominated by outliers).
        if rmse_vals:
            rmin = float(np.nanmin(rmse_vals))
            rmax = float(np.nanmax(rmse_vals))
            if np.isfinite(rmin) and np.isfinite(rmax) and rmin > 0 and (rmax / max(rmin, 1e-12)) > 200:
                ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(out_dir / "figure2d_pareto_ratio_vs_rmse.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'figure2d_pareto_ratio_vs_rmse.png'}")


def _bit_color_map(bits_values: List[int]):
    """Deterministic colors for bit depths."""
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    uniq = sorted(set(int(b) for b in bits_values))
    return {str(b): palette[i % len(palette)] for i, b in enumerate(uniq)}


def summarize_runs_by_bits_clamp(res: Dict[str, Any], *, variant: str = "zstd") -> pd.DataFrame:
    """Collapse multiple codec levels into one best point per (bits, clamp).

    We pick the smallest-bytes run for each (bits, clamp) within the chosen variant.
    This makes parameter plots interpretable (bits/clamp choices, not zstd/png-level noise).
    """
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[int, float], Dict[str, Any]] = {}
    for r in res.get("tracq_runs", []):
        if r.get("variant") != variant:
            continue
        if is_failure(r.get("metrics", {})):
            continue
        key = (int(r.get("bits", 0)), float(r.get("clamp", 0.0)))
        cur = grouped.get(key)
        if cur is None or int(r.get("bytes", 1 << 60)) < int(cur.get("bytes", 1 << 60)):
            grouped[key] = r

    for (bits, clamp), r in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        m = r.get("metrics", {})
        rows.append(
            {
                "bits": bits,
                "clamp": clamp,
                "bytes": int(r.get("bytes", 0)),
                "ratio": float(r.get("ratio", np.nan)),
                "rmse": float(m.get("rmse", np.nan)),
                "rel_rmse": float(m.get("rel_rmse", np.nan)),
                "nrmse": float(m.get("nrmse", np.nan)),
                "corr": float(m.get("corr", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def generate_normalized_frontier_figures(sweep_results: Dict[str, Dict], out_dir: Path):
    """More meaningful parameter plots using normalized error.

    - figure2e_frontier_ratio_vs_relrmse.png: Pareto frontier (min ratio, min rel_RMSE) per bit-depth
      using one point per (bits, clamp) (best-by-bytes), zstd variant.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping normalized frontier figures")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    all_bits: List[int] = []
    for _, res in sweep_results.items():
        for run in res.get("tracq_runs", []):
            all_bits.append(int(run.get("bits", 0)))
    colors = _bit_color_map(all_bits or [2, 4, 8, 16])

    def _frontier_minmin_xy(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
        d = df[[x, y, "bits", "clamp"]].copy()
        d = d[np.isfinite(d[x]) & np.isfinite(d[y])]
        d = d.sort_values([x, y], ascending=[True, True])
        keep_idx = []
        best_y = float("inf")
        for idx, row in d.iterrows():
            if float(row[y]) < best_y:
                keep_idx.append(idx)
                best_y = float(row[y])
        return d.loc[keep_idx]

    fig, axes = plt.subplots(1, len(sweep_results), figsize=(5 * len(sweep_results), 4), squeeze=False)
    for idx, (dname, res) in enumerate(sweep_results.items()):
        ax = axes[0, idx]
        df = summarize_runs_by_bits_clamp(res, variant="zstd")
        if df.empty:
            ax.set_title(dname)
            ax.text(0.5, 0.5, "no valid zstd runs", ha="center", va="center")
            continue

        # Focus on normalized error (rel_rmse). It's comparable across datasets.
        for bits in sorted(df["bits"].unique().tolist()):
            dfi = df[df["bits"] == bits]
            # faint points (each clamp)
            ax.scatter(dfi["ratio"], dfi["rel_rmse"], c=colors.get(str(bits), "gray"), alpha=0.25, s=35)
            fr = _frontier_minmin_xy(dfi, "ratio", "rel_rmse")
            if not fr.empty:
                ax.plot(fr["ratio"].tolist(), fr["rel_rmse"].tolist(), color=colors.get(str(bits), "gray"), linewidth=1.8, label=f"{bits}-bit")

        chosen = pick_best_tracq_for_reporting(res)
        if chosen is not None and np.isfinite(chosen["ratio"]) and np.isfinite(chosen["metrics"].get("rel_rmse", np.nan)):
            ax.scatter([chosen["ratio"]], [chosen["metrics"]["rel_rmse"]], marker="*", s=180, c="black", label="chosen")

        ax.set_title(dname)
        ax.set_xlabel("Compression Ratio (compressed/original)")
        ax.set_ylabel("rel_RMSE (RMSE / std)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        # log-scale if extremely wide
        vals = df["rel_rmse"].replace([np.inf, -np.inf], np.nan).dropna().values
        if vals.size:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            if vmin > 0 and vmax / max(vmin, 1e-12) > 200:
                ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(out_dir / "figure2e_frontier_ratio_vs_relrmse.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'figure2e_frontier_ratio_vs_relrmse.png'}")


def generate_bits_clamp_heatmap(sweep_results: Dict[str, Dict], out_dir: Path):
    """Heatmap over (bits, clamp) using best-by-bytes zstd point.

    Produces:
      - figure2f_heatmap_relrmse_bits_clamp.png
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(sweep_results), figsize=(5 * len(sweep_results), 4), squeeze=False)

    for idx, (dname, res) in enumerate(sweep_results.items()):
        ax = axes[0, idx]
        df = summarize_runs_by_bits_clamp(res, variant="zstd")
        if df.empty:
            ax.set_title(dname)
            ax.text(0.5, 0.5, "no valid zstd runs", ha="center", va="center")
            continue

        bits_vals = sorted(df["bits"].unique().tolist())
        clamp_vals = sorted(df["clamp"].unique().tolist())
        mat = np.full((len(bits_vals), len(clamp_vals)), np.nan, dtype=float)
        for i, b in enumerate(bits_vals):
            for j, c in enumerate(clamp_vals):
                sub = df[(df["bits"] == b) & (df["clamp"] == c)]
                if not sub.empty:
                    mat[i, j] = float(sub.iloc[0]["rel_rmse"])

        im = ax.imshow(mat, aspect="auto", interpolation="nearest")
        ax.set_title(dname)
        ax.set_xlabel("Clamp (%)")
        ax.set_ylabel("Bit depth")
        ax.set_xticks(list(range(len(clamp_vals))))
        ax.set_xticklabels([str(int(c)) for c in clamp_vals], rotation=45, ha="right")
        ax.set_yticks(list(range(len(bits_vals))))
        ax.set_yticklabels([str(int(b)) for b in bits_vals])
        # annotate cells (small)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if mat[i, j] > np.nanmedian(mat) else "black")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="rel_RMSE")
    plt.tight_layout()
    fig.savefig(out_dir / "figure2f_heatmap_relrmse_bits_clamp.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'figure2f_heatmap_relrmse_bits_clamp.png'}")


def generate_parameter_choice_figures(sweep_results: Dict[str, Dict], out_dir: Path):
    """Extra plots to make parameter choices obvious.

    - bytes vs RMSE colored by bit depth
    - ratio vs RMSE colored by bit depth
    - clamp vs RMSE (best-by-bytes per clamp) lines per bit depth
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping parameter-choice figures")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect bit-depths present
    all_bits: List[int] = []
    for _, res in sweep_results.items():
        for run in res.get("tracq_runs", []):
            all_bits.append(int(run.get("bits", 0)))
    colors = _bit_color_map(all_bits or [2, 4, 8, 16])

    # 1) Bytes vs RMSE
    fig, axes = plt.subplots(1, len(sweep_results), figsize=(5 * len(sweep_results), 4), squeeze=False)
    for idx, (dname, res) in enumerate(sweep_results.items()):
        ax = axes[0, idx]
        for run in res.get("tracq_runs", []):
            if is_failure(run.get("metrics", {})):
                continue
            b = str(int(run.get("bits", 0)))
            ax.scatter(run["bytes"] / 1024, run["metrics"]["rmse"], c=colors.get(b, "gray"), alpha=0.45, s=28)
        ax.set_title(dname)
        ax.set_xlabel("Compressed Size (KB)")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
    # legend
    for b, c in colors.items():
        axes[0, 0].scatter([], [], c=c, label=f"{b}-bit", s=40)
    axes[0, 0].legend(fontsize=7, loc="best")
    plt.tight_layout()
    fig.savefig(out_dir / "figure2a_bytes_vs_rmse_bits.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'figure2a_bytes_vs_rmse_bits.png'}")

    # 2) Ratio vs RMSE
    fig, axes = plt.subplots(1, len(sweep_results), figsize=(5 * len(sweep_results), 4), squeeze=False)
    for idx, (dname, res) in enumerate(sweep_results.items()):
        ax = axes[0, idx]
        for run in res.get("tracq_runs", []):
            if is_failure(run.get("metrics", {})):
                continue
            b = str(int(run.get("bits", 0)))
            ax.scatter(run["ratio"], run["metrics"]["rmse"], c=colors.get(b, "gray"), alpha=0.45, s=28)
        ax.set_title(dname)
        ax.set_xlabel("Compression Ratio (compressed/original)")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
    for b, c in colors.items():
        axes[0, 0].scatter([], [], c=c, label=f"{b}-bit", s=40)
    axes[0, 0].legend(fontsize=7, loc="best")
    plt.tight_layout()
    fig.savefig(out_dir / "figure2b_ratio_vs_rmse_bits.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'figure2b_ratio_vs_rmse_bits.png'}")

    # 3) Clamp vs RMSE lines (using zstd runs; choose smallest-bytes for each (bits, clamp))
    fig, axes = plt.subplots(1, len(sweep_results), figsize=(5 * len(sweep_results), 4), squeeze=False)
    for idx, (dname, res) in enumerate(sweep_results.items()):
        ax = axes[0, idx]
        runs = [r for r in res.get("tracq_runs", []) if r.get("variant") == "zstd" and not is_failure(r.get("metrics", {}))]
        if not runs:
            ax.set_title(dname)
            ax.text(0.5, 0.5, "no valid runs", ha="center", va="center")
            continue

        # group by (bits, clamp)
        grouped: Dict[Tuple[int, float], Dict[str, Any]] = {}
        for r in runs:
            key = (int(r.get("bits", 0)), float(r.get("clamp", 0.0)))
            cur = grouped.get(key)
            if cur is None or int(r.get("bytes", 1 << 60)) < int(cur.get("bytes", 1 << 60)):
                grouped[key] = r

        bits_set = sorted(set(k[0] for k in grouped.keys()))
        for b in bits_set:
            xs = []
            ys = []
            for clamp in sorted(set(k[1] for k in grouped.keys() if k[0] == b)):
                r = grouped[(b, clamp)]
                xs.append(clamp)
                ys.append(float(r["metrics"]["rmse"]))
            ax.plot(xs, ys, label=f"{b}-bit", color=colors.get(str(b), "gray"), marker="o", linewidth=1)

        ax.set_title(dname)
        ax.set_xlabel("Clamp (%)")
        ax.set_ylabel("RMSE (best-by-bytes per clamp)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    fig.savefig(out_dir / "figure2c_clamp_tradeoff_rmse.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'figure2c_clamp_tradeoff_rmse.png'}")


def generate_reconstruction_figure(out_dir: Path, csv_path: Path, dataset_name: str, var_indices: List[int], bits: int = 8, clamp: float = 200.0):
    """Figure 3: Representative reconstructions for selected variables."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    orig = load_csv_as_array(csv_path)

    # TRACQ reconstruction
    q, meta = _tracq_quantize_array(orig, bits=int(bits), clamp=float(clamp))
    recon_tracq, _ = TimeSeriesGrid.reconstruct_from_quantized(q, meta)

    # PAA reconstruction
    coeffs, paa_meta = paa_compress(orig, 64)
    recon_paa = paa_decompress(coeffs, paa_meta)

    n_plots = len(var_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, vi in enumerate(var_indices):
        ax = axes[i]
        t = np.arange(orig.shape[1])
        ax.plot(t, orig[vi], label="Original", linewidth=1)
        ax.plot(t, recon_tracq[vi], label="TRACQ", linewidth=1, alpha=0.8)
        ax.plot(t, recon_paa[vi], label="PAA", linewidth=1, alpha=0.8)

        rmse_g = float(np.sqrt(np.mean((orig[vi] - recon_tracq[vi]) ** 2)))
        rmse_p = float(np.sqrt(np.mean((orig[vi] - recon_paa[vi]) ** 2)))
        ax.set_title(f"var {vi} (RMSE: TRACQ={rmse_g:.2f}, PAA={rmse_p:.2f})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / f"figure3_reconstructions_{dataset_name}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / f'figure3_reconstructions_{dataset_name}.png'}")


# ---------------------------------------------------------------------------
# Table Generation
# ---------------------------------------------------------------------------
def generate_summary_table(sweep_results: Dict[str, Dict], out_dir: Path):
    """Table 1: Per-dataset numeric summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for dname, res in sweep_results.items():
        orig_bytes = res["orig_bytes"]

        # Best TRACQ for reporting (quality-constrained; Pareto fallback)
        best_tracq = pick_best_tracq_for_reporting(res)

        gzip_data = res["baselines"].get("gzip", {})
        paa_64 = res["baselines"].get("paa_64", {})

        row = {
            "dataset": dname,
            "orig_bytes": orig_bytes,
            "tracq_bytes": best_tracq["bytes"] if best_tracq else None,
            "tracq_ratio": best_tracq["ratio"] if best_tracq else None,
            "tracq_rmse": best_tracq["metrics"]["rmse"] if best_tracq else None,
            "tracq_rel_rmse": best_tracq["metrics"]["rel_rmse"] if best_tracq else None,
            "tracq_corr": best_tracq["metrics"]["corr"] if best_tracq else None,
            "gzip_bytes": gzip_data.get("bytes"),
            "gzip_ratio": gzip_data.get("ratio"),
            "paa64_bytes": paa_64.get("bytes"),
            "paa64_rmse": paa_64.get("metrics", {}).get("rmse"),
            "paa64_rel_rmse": paa_64.get("metrics", {}).get("rel_rmse"),
            "paa64_corr": paa_64.get("metrics", {}).get("corr"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "table1_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # LaTeX version (optional, requires jinja2)
    try:
        latex_path = out_dir / "table1_summary.tex"
        latex_path.write_text(df.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
        print(f"Saved: {latex_path}")
    except ImportError:
        print("  [SKIP] LaTeX table (jinja2 not installed)")


def generate_ablation_table(sweep_results: Dict[str, Dict], out_dir: Path):
    """Table 2: Ablation results (8 vs 16 bits; clamp effects)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for dname, res in sweep_results.items():
        for bits in ABLATION_BITS:
            for clamp in ABLATION_CLAMPS:
                # Find best zstd run for this (bits, clamp) combo
                matching = [
                    r for r in res["tracq_runs"]
                    if r["variant"] == "zstd" and r["bits"] == bits and r["clamp"] == clamp and not is_failure(r["metrics"])
                ]
                if matching:
                    best = min(matching, key=lambda r: r["bytes"])
                    rows.append({
                        "dataset": dname,
                        "bits": bits,
                        "clamp": clamp,
                        "bytes": best["bytes"],
                        "ratio": best["ratio"],
                        "rmse": best["metrics"]["rmse"],
                        "rel_rmse": best["metrics"]["rel_rmse"],
                        "nrmse": best["metrics"]["nrmse"],
                        "corr": best["metrics"]["corr"],
                    })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "table2_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # LaTeX version (optional, requires jinja2)
    try:
        latex_path = out_dir / "table2_ablation.tex"
        latex_path.write_text(df.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
        print(f"Saved: {latex_path}")
    except ImportError:
        print("  [SKIP] LaTeX ablation table (jinja2 not installed)")


def save_failure_modes(sweep_results: Dict[str, Dict], out_dir: Path):
    """Save failure modes to JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = {}
    for dname, res in sweep_results.items():
        failures[dname] = res.get("failures", [])

    out_path = out_dir / "failure_modes.json"
    out_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Run paper experiments: RD sweeps, figures, tables")
    ap.add_argument("--run-all", action="store_true", help="Run all experiments")
    ap.add_argument("--rd-sweeps", action="store_true", help="Run RD sweeps for all datasets")
    ap.add_argument("--figures", action="store_true", help="Generate figures from existing sweep JSONs")
    ap.add_argument("--tables", action="store_true", help="Generate tables from existing sweep JSONs")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "paper_results" / "experiments")
    args = ap.parse_args(argv)

    rd_dir = args.outdir / "rd_curves"
    fig_dir = args.outdir / "figures"
    tbl_dir = args.outdir / "tables"

    sweep_results: Dict[str, Dict] = {}

    # Load existing sweep JSONs if available
    def load_existing():
        nonlocal sweep_results
        for dname in DATASETS:
            json_path = rd_dir / f"{dname}_rd_sweep.json"
            if json_path.exists():
                sweep_results[dname] = json.loads(json_path.read_text(encoding="utf-8"))

    if args.run_all or args.rd_sweeps:
        print("=== Running RD Sweeps ===")
        for dname, csv_path in DATASETS.items():
            if not csv_path.exists():
                print(f"  [SKIP] {dname}: {csv_path} not found")
                continue
            print(f"  [RUN] {dname} ...")
            res = run_rd_sweep(dname, csv_path, rd_dir)
            sweep_results[dname] = res
            print(f"    -> {len(res['tracq_runs'])} TRACQ runs, {len(res['failures'])} failures")

    if args.run_all or args.figures:
        print("=== Generating Figures ===")
        load_existing()
        if sweep_results:
            generate_rd_curve_figure(sweep_results, fig_dir)
            generate_pareto_figure(sweep_results, fig_dir)
            generate_parameter_choice_figures(sweep_results, fig_dir)
            generate_normalized_frontier_figures(sweep_results, fig_dir)
            generate_bits_clamp_heatmap(sweep_results, fig_dir)
            # Reconstruction figures for each dataset (use chosen TRACQ params)
            for dname, csv_path in DATASETS.items():
                if not csv_path.exists():
                    continue
                n_vars = load_csv_as_array(csv_path).shape[0]
                candidates = [0, n_vars // 2, max(0, n_vars - 1)]
                var_indices: List[int] = []
                for v in candidates:
                    v = int(max(0, min(n_vars - 1, v)))
                    if v not in var_indices:
                        var_indices.append(v)

                chosen = pick_best_tracq_for_reporting(sweep_results.get(dname, {}))
                bits = int(chosen.get("bits", 16)) if chosen else 16
                clamp = float(chosen.get("clamp", 200.0)) if chosen else 200.0
                generate_reconstruction_figure(fig_dir, csv_path, dname, var_indices, bits=bits, clamp=clamp)
        else:
            print("  No sweep results found. Run --rd-sweeps first.")

    if args.run_all or args.tables:
        print("=== Generating Tables ===")
        load_existing()
        if sweep_results:
            generate_summary_table(sweep_results, tbl_dir)
            generate_ablation_table(sweep_results, tbl_dir)
            save_failure_modes(sweep_results, args.outdir)
        else:
            print("  No sweep results found. Run --rd-sweeps first.")

    print("\n=== Done ===")
    print(f"Outputs in: {args.outdir}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Full-length MetroPT-3 rate-distortion benchmark.

This script benchmarks TRACQ/TRACQ and selected baselines on the official
MetroPT-3 dataset (1.5M+ rows, 15 numeric variables) and generates:
  - metropt3_rate_distortion.json
  - metropt3_rate_distortion.csv
  - figure_metropt3_rate_distortion.png / .pdf

The figure contains two views of the same operating points:
  1. Compression ratio vs RMSE
  2. Compression ratio vs SMAPE
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracq.bigdata import load_numeric_csv_matrix
from tracq.container import pack as pack_zst, unpack as unpack_zst
from tracq.core import TimeSeriesGrid
from tracq.core_enhanced import EnhancedTimeSeriesGrid
from tracq.metrics import corr as calc_corr, mae as calc_mae, rmse as calc_rmse, smape as calc_smape
from tracq.baselines import (
    HAS_HDF5PLUGIN,
    HAS_ZFP,
    delta_zstd_compress,
    delta_zstd_decompress,
    gorilla_like_compress,
    gorilla_like_decompress,
    paa_compress,
    paa_decompress,
    sax_compress,
    sax_decompress,
)

if HAS_ZFP:
    from tracq.baselines import zfp_compress, zfp_decompress
if HAS_HDF5PLUGIN:
    from tracq.baselines import sz3_compress, sz3_decompress


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paper_results" / "bigdata_rd"
DEFAULT_METROPT_CSV = PROJECT_ROOT / "data" / "raw" / "metropt3" / "MetroPT3(AirCompressor).csv"
FULL_WIDTH = 7.16  # inches (IEEE \textwidth)

plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

COLORS = {
    "gzip": "#7f8c8d",
    "delta_zstd": "#2c3e50",
    "tracq_base_16bit": "#c0392b",
    "tracq_enh_8bit": "#3498db",
    "tracq_enh_16bit": "#2980b9",
    "tracq_enh_8bit_anchors_100": "#6f42c1",
    "tracq_enh_8bit_anchors_10000": "#9b59b6",
    "tracq_enh_8bit_anchors_1000": "#8e44ad",
    "tracq_enh_16bit_anchors_1000": "#6c3483",
    "paa_256": "#e67e22",
    "paa_1024": "#f39c12",
    "sax_256": "#d35400",
    "gorilla_like": "#1abc9c",
    "zfp_tol_0.1": "#27ae60",
    "zfp_tol_0.01": "#239b56",
    "zfp_tol_0.001": "#1e8449",
    "sz3_abs_0.1": "#16a085",
    "sz3_abs_0.01": "#138d75",
    "sz3_abs_0.001": "#117864",
}

LABELS = {
    "gzip": "Gzip",
    "delta_zstd": "Delta+Zstd",
    "tracq_base_16bit": "Base TRACQ 16b",
    "tracq_enh_8bit": "Enh. TRACQ 8b",
    "tracq_enh_16bit": "Enh. TRACQ 16b",
    "tracq_enh_8bit_anchors_100": "Enh. TRACQ 8b+A100",
    "tracq_enh_8bit_anchors_10000": "Enh. TRACQ 8b+A10k",
    "tracq_enh_8bit_anchors_1000": "Enh. TRACQ 8b+A1k",
    "tracq_enh_16bit_anchors_1000": "Enh. TRACQ 16b+A1k",
    "paa_256": "PAA-256",
    "paa_1024": "PAA-1024",
    "sax_256": "SAX-256",
    "gorilla_like": "Gorilla",
    "zfp_tol_0.1": "ZFP (1e-1)",
    "zfp_tol_0.01": "ZFP (1e-2)",
    "zfp_tol_0.001": "ZFP (1e-3)",
    "sz3_abs_0.1": "SZ3 (1e-1)",
    "sz3_abs_0.01": "SZ3 (1e-2)",
    "sz3_abs_0.001": "SZ3 (1e-3)",
}


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return result, float(t1 - t0)


def compute_metrics(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(calc_rmse(orig, recon)),
        "mae": float(calc_mae(orig, recon)),
        "corr": float(calc_corr(orig, recon)),
        "smape": float(calc_smape(orig, recon)),
        "max_error": float(np.max(np.abs(orig - recon))),
    }


def sanitize_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: sanitize_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    return x


def benchmark_gzip(orig: np.ndarray) -> Dict[str, Any]:
    import gzip

    raw = orig.tobytes(order="C")
    blob, enc_s = timed(gzip.compress, raw)
    decoded, dec_s = timed(gzip.decompress, blob)
    recon = np.frombuffer(decoded, dtype=orig.dtype).reshape(orig.shape)
    return {
        "category": "Lossless",
        "bytes": int(len(blob)),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "total_s": enc_s + dec_s,
        "metrics": compute_metrics(orig, recon),
    }


def benchmark_delta_zstd(orig: np.ndarray) -> Dict[str, Any]:
    (blob, meta), enc_s = timed(delta_zstd_compress, orig, 3)
    recon, dec_s = timed(delta_zstd_decompress, blob, meta)
    return {
        "category": "Lossless",
        "bytes": int(len(blob)),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "total_s": enc_s + dec_s,
        "metrics": compute_metrics(orig, recon),
        "params": {"level": 3},
    }


def benchmark_tracq_base(orig: np.ndarray, *, bits: int, clamp_pct: float = 500.0, zstd_level: int = 3) -> Dict[str, Any]:
    grid = TimeSeriesGrid(orig, clamp_pct=clamp_pct)
    quantize = grid.quantize_8bit if int(bits) == 8 else grid.quantize_16bit
    (q, meta), quant_s = timed(quantize)
    blob, pack_s = timed(pack_zst, meta, q, compress_level=zstd_level)
    (q2, meta2), unpack_s = timed(unpack_zst, blob)
    (recon, _), recon_s = timed(TimeSeriesGrid.reconstruct_from_quantized, q2, meta2)
    return {
        "category": "TRACQ",
        "bytes": int(len(blob)),
        "encode_s": quant_s + pack_s,
        "decode_s": unpack_s + recon_s,
        "total_s": quant_s + pack_s + unpack_s + recon_s,
        "metrics": compute_metrics(orig, recon),
        "params": {"bits": int(bits), "clamp_pct": float(clamp_pct), "zstd_level": int(zstd_level)},
    }


def benchmark_tracq_enhanced(
    orig: np.ndarray,
    *,
    bits: int,
    clamp_pct: float = 500.0,
    zstd_level: int = 3,
    anchor_interval: int = 0,
    auto_offset: bool = True,
) -> Dict[str, Any]:
    grid = EnhancedTimeSeriesGrid(
        orig,
        clamp_pct=clamp_pct,
        adaptive_clamp=True,
        use_mu_law=True,
        anchor_interval=int(anchor_interval),
        auto_offset=bool(auto_offset),
    )
    quantize = grid.quantize_8bit if int(bits) == 8 else grid.quantize_16bit
    (q, meta), quant_s = timed(quantize)
    blob, pack_s = timed(pack_zst, meta, q, compress_level=zstd_level)
    (q2, meta2), unpack_s = timed(unpack_zst, blob)
    (recon, _), recon_s = timed(EnhancedTimeSeriesGrid.reconstruct_from_quantized, q2, meta2)
    return {
        "category": "TRACQ",
        "bytes": int(len(blob)),
        "encode_s": quant_s + pack_s,
        "decode_s": unpack_s + recon_s,
        "total_s": quant_s + pack_s + unpack_s + recon_s,
        "metrics": compute_metrics(orig, recon),
        "params": {
            "bits": int(bits),
            "clamp_pct": float(clamp_pct),
            "zstd_level": int(zstd_level),
            "anchor_interval": int(anchor_interval),
            "auto_offset": bool(auto_offset),
        },
    }


def benchmark_paa(orig: np.ndarray, *, segments: int) -> Dict[str, Any]:
    (coeffs, meta), enc_s = timed(paa_compress, orig, int(segments))
    recon, dec_s = timed(paa_decompress, coeffs, meta)
    return {
        "category": "Symbolic",
        "bytes": int(coeffs.nbytes),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "total_s": enc_s + dec_s,
        "metrics": compute_metrics(orig, recon),
        "params": {"segments": int(segments)},
    }


def benchmark_sax(orig: np.ndarray, *, segments: int, alphabet: int = 8) -> Dict[str, Any]:
    (symbols, meta), enc_s = timed(sax_compress, orig, int(segments), int(alphabet))
    recon, dec_s = timed(sax_decompress, symbols, meta)
    return {
        "category": "Symbolic",
        "bytes": int(symbols.nbytes),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "total_s": enc_s + dec_s,
        "metrics": compute_metrics(orig, recon),
        "params": {"segments": int(segments), "alphabet": int(alphabet)},
    }


def benchmark_gorilla_like(orig: np.ndarray) -> Dict[str, Any]:
    (blob, meta), enc_s = timed(gorilla_like_compress, orig)
    recon, dec_s = timed(gorilla_like_decompress, blob, meta)
    return {
        "category": "Delta",
        "bytes": int(len(blob)),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "total_s": enc_s + dec_s,
        "metrics": compute_metrics(orig, recon),
    }


def benchmark_zfp(orig: np.ndarray, *, tolerance: float) -> Dict[str, Any]:
    (blob, meta), enc_s = timed(zfp_compress, orig, mode="tolerance", tolerance=float(tolerance))
    recon, dec_s = timed(zfp_decompress, blob, meta)
    return {
        "category": "HPC",
        "bytes": int(len(blob)),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "total_s": enc_s + dec_s,
        "metrics": compute_metrics(orig, recon),
        "params": {"tolerance": float(tolerance)},
    }


def benchmark_sz3(orig: np.ndarray, *, abs_error: float) -> Dict[str, Any]:
    (blob, meta), enc_s = timed(sz3_compress, orig, mode="abs", abs_error=float(abs_error))
    recon, dec_s = timed(sz3_decompress, blob, meta)
    return {
        "category": "HPC",
        "bytes": int(len(blob)),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "total_s": enc_s + dec_s,
        "metrics": compute_metrics(orig, recon),
        "params": {"abs_error": float(abs_error)},
    }


def candidate_methods() -> List[Tuple[str, Any]]:
    methods: List[Tuple[str, Any]] = [
        ("gzip", benchmark_gzip),
        ("delta_zstd", benchmark_delta_zstd),
        ("tracq_base_16bit", lambda data: benchmark_tracq_base(data, bits=16)),
        ("tracq_enh_8bit", lambda data: benchmark_tracq_enhanced(data, bits=8, anchor_interval=0)),
        ("tracq_enh_16bit", lambda data: benchmark_tracq_enhanced(data, bits=16, anchor_interval=0)),
        ("tracq_enh_8bit_anchors_100", lambda data: benchmark_tracq_enhanced(data, bits=8, anchor_interval=100)),
        ("tracq_enh_8bit_anchors_10000", lambda data: benchmark_tracq_enhanced(data, bits=8, anchor_interval=10000)),
        ("tracq_enh_8bit_anchors_1000", lambda data: benchmark_tracq_enhanced(data, bits=8, anchor_interval=1000)),
        ("tracq_enh_16bit_anchors_1000", lambda data: benchmark_tracq_enhanced(data, bits=16, anchor_interval=1000)),
        ("paa_256", lambda data: benchmark_paa(data, segments=256)),
        ("paa_1024", lambda data: benchmark_paa(data, segments=1024)),
        ("sax_256", lambda data: benchmark_sax(data, segments=256, alphabet=8)),
        ("gorilla_like", benchmark_gorilla_like),
    ]
    if HAS_ZFP:
        methods.extend(
            [
                ("zfp_tol_0.1", lambda data: benchmark_zfp(data, tolerance=1e-1)),
                ("zfp_tol_0.01", lambda data: benchmark_zfp(data, tolerance=1e-2)),
                ("zfp_tol_0.001", lambda data: benchmark_zfp(data, tolerance=1e-3)),
            ]
        )
    if HAS_HDF5PLUGIN:
        methods.extend(
            [
                ("sz3_abs_0.1", lambda data: benchmark_sz3(data, abs_error=1e-1)),
                ("sz3_abs_0.01", lambda data: benchmark_sz3(data, abs_error=1e-2)),
                ("sz3_abs_0.001", lambda data: benchmark_sz3(data, abs_error=1e-3)),
            ]
        )
    return methods


def flatten_results(results: Dict[str, Dict[str, Any]], orig_bytes: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, res in results.items():
        metrics = res.get("metrics", {})
        rows.append(
            {
                "method": method,
                "label": LABELS.get(method, method),
                "category": res.get("category"),
                "bytes": res.get("bytes"),
                "ratio": float(res["bytes"] / orig_bytes) if res.get("bytes") is not None else None,
                "encode_s": res.get("encode_s"),
                "decode_s": res.get("decode_s"),
                "total_s": res.get("total_s"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "smape": metrics.get("smape"),
                "corr": metrics.get("corr"),
                "max_error": metrics.get("max_error"),
                "params": json.dumps(res.get("params", {}), separators=(",", ":")),
            }
        )
    return rows


def save_summary_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "label",
        "category",
        "bytes",
        "ratio",
        "encode_s",
        "decode_s",
        "total_s",
        "rmse",
        "mae",
        "smape",
        "corr",
        "max_error",
        "params",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_figure(path: Path, rows: List[Dict[str, Any]], dataset_info: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.25))

    for row in rows:
        ratio = row.get("ratio")
        rmse = row.get("rmse")
        smape = row.get("smape")
        method = row["method"]
        label = row["label"]
        color = COLORS.get(method, "#333333")

        if ratio is None:
            continue

        if rmse is not None and np.isfinite(rmse) and rmse > 0:
            ax1.scatter(ratio, rmse, s=48, color=color, alpha=0.9, label=label)
        if smape is not None and np.isfinite(smape):
            ax2.scatter(ratio, smape, s=48, color=color, alpha=0.9, label=label)

    ax1.set_xlabel("")
    ax1.set_ylabel("RMSE")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("MetroPT-3 Rate-Distortion (RMSE)", fontsize=10, fontweight="bold")

    ax2.set_xlabel("")
    ax2.set_ylabel("SMAPE")
    ax2.set_ylim(bottom=0.0)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("MetroPT-3 Rate-Distortion (SMAPE)", fontsize=10, fontweight="bold")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        f"Full-Length MetroPT-3: {dataset_info['n_rows']:,} Rows, {dataset_info['n_vars']} Variables",
        fontsize=11,
        fontweight="bold",
    )
    fig.supxlabel("Compression Ratio", y=0.13, fontsize=9)
    fig.tight_layout(rect=[0, 0.18, 1, 0.93])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run full-length rate-distortion evaluation on MetroPT-3.")
    ap.add_argument("--metropt-csv", type=Path, default=DEFAULT_METROPT_CSV)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke runs.")
    args = ap.parse_args(argv)

    if not args.metropt_csv.exists():
        raise SystemExit(f"MetroPT-3 CSV not found at {args.metropt_csv}")

    print(f"Loading MetroPT-3 from {args.metropt_csv} ...")
    data, dataset_info = load_numeric_csv_matrix(args.metropt_csv)
    if args.max_rows is not None and data.shape[1] > int(args.max_rows):
        data = np.ascontiguousarray(data[:, : int(args.max_rows)])
        dataset_info["n_rows"] = int(data.shape[1])
        dataset_info["input_bytes"] = int(data.nbytes)
    orig_bytes = int(data.nbytes)
    dataset_info["orig_bytes"] = orig_bytes
    print(f"Loaded shape={data.shape[0]} vars x {data.shape[1]} rows ({orig_bytes / (1024 * 1024):.1f} MiB float64)")

    results: Dict[str, Dict[str, Any]] = {}
    for method_name, fn in candidate_methods():
        print(f"Running {method_name} ...")
        gc.collect()
        try:
            res = fn(data)
            res["ratio"] = float(res["bytes"] / orig_bytes) if res.get("bytes") is not None else None
            results[method_name] = sanitize_jsonable(res)
            print(
                f"  ratio={results[method_name]['ratio']:.4f} "
                f"rmse={results[method_name]['metrics']['rmse']:.6g} "
                f"smape={results[method_name]['metrics']['smape']:.6g}"
            )
        except Exception as exc:
            results[method_name] = {"error": str(exc)}
            print(f"  failed: {exc}")

    rows = flatten_results({k: v for k, v in results.items() if "metrics" in v}, orig_bytes)

    payload = {
        "dataset": dataset_info,
        "results": results,
        "rd_points": rows,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "metropt3_rate_distortion.json").write_text(
        json.dumps(sanitize_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    save_summary_csv(args.outdir / "metropt3_rate_distortion.csv", rows)
    generate_figure(args.outdir / "figure_metropt3_rate_distortion.png", rows, dataset_info)

    print(f"Saved MetroPT-3 rate-distortion outputs to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

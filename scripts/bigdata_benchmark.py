#!/usr/bin/env python
"""
Big-data benchmark for reviewer-facing scale concerns.

Runs two complementary experiments:
1. MetroPT-3 streaming benchmark on fixed-size windows.
2. Synthetic scaling benchmark from 1M to 100M rows without materializing the
   full sequence in memory.

Outputs to paper_results/bigdata/ by default:
  - metropt3_streaming_results.json
  - synthetic_scaling_results.json
  - bigdata_summary.csv
  - figure_bigdata_scaling.png / .pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracq.bigdata import benchmark_window_stream, iter_numeric_csv_windows, run_synthetic_scaling


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paper_results" / "bigdata"
DEFAULT_METROPT_WINDOW_ROWS = 10_000
DEFAULT_SYNTHETIC_WINDOW_ROWS = 100_000
DEFAULT_SYNTHETIC_ROWS = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]
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


def save_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_summary_csv(path: Path, metropt: List[Dict[str, Any]], synthetic: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "section",
                "name",
                "method",
                "n_rows",
                "n_vars",
                "n_windows",
                "input_bytes",
                "compressed_bytes",
                "compression_ratio",
                "encode_mbps",
                "decode_mbps",
                "peak_rss_mb",
                "window_rows",
            ],
        )
        writer.writeheader()

        for item in metropt:
            writer.writerow(
                {
                    "section": "metropt3",
                    "name": "metropt3_streaming",
                    "method": item["method"],
                    "n_rows": item["n_rows"],
                    "n_vars": item["n_vars"],
                    "n_windows": item["n_windows"],
                    "input_bytes": item["input_bytes"],
                    "compressed_bytes": item["compressed_bytes"],
                    "compression_ratio": item["compression_ratio"],
                    "encode_mbps": item["encode_mbps"],
                    "decode_mbps": item["decode_mbps"],
                    "peak_rss_mb": item["peak_rss_mb"],
                    "window_rows": item["window_rows"],
                }
            )

        for item in synthetic:
            writer.writerow(
                {
                    "section": "synthetic_scaling",
                    "name": "synthetic_scaling",
                    "method": item["method"],
                    "n_rows": item["n_rows"],
                    "n_vars": item["n_vars"],
                    "n_windows": item["n_windows"],
                    "input_bytes": item["input_bytes"],
                    "compressed_bytes": item["compressed_bytes"],
                    "compression_ratio": item["compression_ratio"],
                    "encode_mbps": item["encode_mbps"],
                    "decode_mbps": item["decode_mbps"],
                    "peak_rss_mb": item["peak_rss_mb"],
                    "window_rows": item["window_rows"],
                }
            )


def generate_figure(path: Path, metropt: List[Dict[str, Any]], synthetic: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.2))

    synthetic = sorted(synthetic, key=lambda x: x["n_rows"])
    x = [r["n_rows"] for r in synthetic]
    encode = [r["encode_mbps"] for r in synthetic]
    peak = [r["peak_rss_mb"] for r in synthetic]

    ax1.plot(x, encode, marker="o", linewidth=2.0, color="#1f77b4", label="Encode throughput")
    ax1.set_xscale("log")
    ax1.set_xlabel("Total rows processed")
    ax1.set_ylabel("Encode throughput (MB/s)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Synthetic Streaming Scaling", fontsize=10, fontweight="bold")

    ax1b = ax1.twinx()
    ax1b.plot(x, peak, marker="s", linewidth=2.0, color="#d62728", label="Peak RSS")
    ax1b.set_ylabel("Peak RSS (MB)", color="#d62728")
    ax1b.tick_params(axis="y", labelcolor="#d62728")

    method_labels = {
        "tracq_base": "TRACQ Base",
        "tracq_enhanced": "TRACQ Enhanced",
        "gzip": "Gzip",
    }
    methods = [method_labels.get(r["method"], r["method"]) for r in metropt]
    m_x = range(len(methods))
    width = 0.36
    ratio_vals = [r["compression_ratio"] for r in metropt]
    metro_encode = [r["encode_mbps"] for r in metropt]

    ax2.bar([i - width / 2 for i in m_x], metro_encode, width=width, label="Encode MB/s", color="#2ca02c")
    ax2.set_ylabel("Encode throughput (MB/s)")
    ax2.set_xticks(list(m_x))
    ax2.set_xticklabels(methods, rotation=12)
    ax2.set_title("MetroPT-3 Streaming Benchmark", fontsize=10, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.bar([i + width / 2 for i in m_x], ratio_vals, width=width, label="Compression ratio", color="#9467bd")
    ax2b.set_ylabel("Compression ratio")

    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=7)

    fig.suptitle(
        "Big-Data Evidence: Real Streaming Data\nand Synthetic Scaling",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_metropt_streaming(
    metropt_csv: Path,
    *,
    window_rows: int,
    bits: int,
    clamp_pct: float,
    zstd_level: int,
) -> List[Dict[str, Any]]:
    methods = ["tracq_base", "tracq_enhanced", "gzip"]
    results: List[Dict[str, Any]] = []

    for method in methods:
        print(f"  MetroPT-3 method={method} window_rows={window_rows}")
        windows = iter_numeric_csv_windows(metropt_csv, window_rows=window_rows)
        res = benchmark_window_stream(
            windows,
            method=method,
            bits=bits,
            clamp_pct=clamp_pct,
            zstd_level=zstd_level,
            auto_offset=True,
        )
        res["window_rows"] = int(window_rows)
        res["source_csv"] = str(metropt_csv)
        results.append(res)

    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run MetroPT-3 and synthetic big-data benchmarks.")
    ap.add_argument("--metropt-csv", type=Path, default=PROJECT_ROOT / "data" / "raw" / "metropt3.csv")
    ap.add_argument(
        "--window-rows",
        type=int,
        default=None,
        help="Compatibility override that applies the same window size to both MetroPT-3 and synthetic runs.",
    )
    ap.add_argument("--metropt-window-rows", type=int, default=DEFAULT_METROPT_WINDOW_ROWS)
    ap.add_argument("--synthetic-window-rows", type=int, default=DEFAULT_SYNTHETIC_WINDOW_ROWS)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--clamp-pct", type=float, default=500.0)
    ap.add_argument("--zstd-level", type=int, default=3)
    ap.add_argument("--synthetic-vars", type=int, default=16)
    ap.add_argument(
        "--synthetic-rows",
        nargs="*",
        type=int,
        default=DEFAULT_SYNTHETIC_ROWS,
        help="Total rows for synthetic scaling, e.g. 1000000 5000000 10000000 50000000 100000000",
    )
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args(argv)

    metropt_window_rows = int(args.window_rows or args.metropt_window_rows)
    synthetic_window_rows = int(args.window_rows or args.synthetic_window_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)

    metropt_results: List[Dict[str, Any]] = []
    if args.metropt_csv.exists():
        print(f"Running MetroPT-3 streaming benchmark from {args.metropt_csv} ...")
        metropt_results = run_metropt_streaming(
            args.metropt_csv,
            window_rows=metropt_window_rows,
            bits=args.bits,
            clamp_pct=args.clamp_pct,
            zstd_level=args.zstd_level,
        )
        save_json(args.outdir / "metropt3_streaming_results.json", metropt_results)
    else:
        print(f"MetroPT-3 CSV not found at {args.metropt_csv}; synthetic scaling will still run.")

    print("Running synthetic scaling benchmark ...")
    for total_rows in args.synthetic_rows:
        print(f"  Synthetic total_rows={int(total_rows)} window_rows={synthetic_window_rows}")
    synthetic_results = run_synthetic_scaling(
        args.synthetic_rows,
        n_vars=args.synthetic_vars,
        window_rows=synthetic_window_rows,
        bits=args.bits,
        clamp_pct=args.clamp_pct,
        zstd_level=args.zstd_level,
        auto_offset=True,
        seed=0,
    )
    save_json(args.outdir / "synthetic_scaling_results.json", synthetic_results)

    save_summary_csv(args.outdir / "bigdata_summary.csv", metropt_results, synthetic_results)
    generate_figure(args.outdir / "figure_bigdata_scaling.png", metropt_results, synthetic_results)

    print(f"Saved big-data outputs to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scaled benchmark for FGCS paper - Full dataset evaluation with HPC compressors.

This script addresses reviewer concerns about:
1. Limited dataset scale (extends to full datasets)
2. Missing SZ3/ZFP comparisons (includes HPC compressors)
3. Parallel throughput evaluation (multi-core scaling analysis)
4. Resource usage monitoring (memory tracking)

Example usage:
  # Run on full UCI Electricity dataset with all compressors
  python scripts/scaled_benchmark.py --dataset electricity --full-length --include-hpc --parallel-scaling

  # Run comprehensive paper experiments
  python scripts/scaled_benchmark.py --all-datasets --full-length --include-hpc --parallel-scaling --outdir scaled_results
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import multiprocessing

import numpy as np

# Ensure repo root importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracq.tooling import (
    full_benchmark,
    benchmark_parallel_scaling,
    parallel_tracq_encode,
    measure_throughput,
    _metrics,
    _bench,
)
from tracq.core import TimeSeriesGrid


# ============================================================================
# Memory monitoring utilities
# ============================================================================

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback for systems without psutil
        return 0.0


def monitor_memory(func):
    """Decorator to monitor memory usage before/after function call."""
    def wrapper(*args, **kwargs):
        gc.collect()
        mem_before = get_memory_usage_mb()
        result = func(*args, **kwargs)
        gc.collect()
        mem_after = get_memory_usage_mb()
        return result, {"mem_before_mb": mem_before, "mem_after_mb": mem_after, "mem_delta_mb": mem_after - mem_before}
    return wrapper


# ============================================================================
# Dataset specifications (including full-length versions)
# ============================================================================

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    inner_path: Optional[str] = None
    sep: str = ","
    decimal: str = "."
    # For scaled experiments: max_rows=None means use full dataset
    default_max_rows: Optional[int] = None


DATASETS: Dict[str, DatasetSpec] = {
    "uci_air_quality": DatasetSpec(
        name="uci_air_quality",
        url="https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        inner_path="AirQualityUCI.csv",
        sep=";",
        decimal=",",
        default_max_rows=None,  # ~9358 rows
    ),
    "uci_appliances_energy": DatasetSpec(
        name="uci_appliances_energy",
        url="https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
        inner_path="energydata_complete.csv",
        sep=",",
        decimal=".",
        default_max_rows=None,  # ~19735 rows
    ),
    "uci_metro_traffic": DatasetSpec(
        name="uci_metro_traffic",
        url="https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip",
        inner_path="Metro_Interstate_Traffic_Volume.csv.gz",
        sep=",",
        decimal=".",
        default_max_rows=None,  # ~48204 rows
    ),
    "uci_electricity": DatasetSpec(
        name="uci_electricity",
        url="https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        inner_path="LD2011_2014.txt",
        sep=";",
        decimal=",",
        default_max_rows=None,  # 370 vars × 140256 time steps (FULL)
    ),
}


# ============================================================================
# Data preparation
# ============================================================================

def _download(url: str, out_path: Path) -> None:
    import urllib.request
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(f"  Downloading {url}...")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"    {total / (1024*1024):.1f} MB downloaded...", end="\r")
    print()
    tmp.replace(out_path)


def prepare_dataset(
    spec: DatasetSpec,
    raw_dir: Path,
    processed_dir: Path,
    max_rows: Optional[int],
    force: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Download + convert to headerless numeric CSV. Returns (csv_path, info_dict)."""
    import pandas as pd

    suffix = "" if max_rows is None else f"_{max_rows}"
    out_csv = processed_dir / f"{spec.name}{suffix}.csv"

    if out_csv.exists() and not force:
        # Load existing and return info
        df = pd.read_csv(out_csv, header=None)
        info = {
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "file_size_bytes": out_csv.stat().st_size,
        }
        return out_csv, info

    raw_zip = raw_dir / f"{spec.name}.zip"
    if not raw_zip.exists():
        _download(spec.url, raw_zip)

    extract_dir = raw_dir / spec.name
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {raw_zip}...")
    with zipfile.ZipFile(raw_zip, "r") as zf:
        zf.extractall(extract_dir)

    in_path = extract_dir / spec.inner_path if spec.inner_path else extract_dir
    if spec.inner_path and not in_path.exists():
        candidates = list(extract_dir.rglob("*.csv")) + list(extract_dir.rglob("*.csv.gz")) + list(extract_dir.rglob("*.txt"))
        candidates = [p for p in candidates if p.is_file()]
        if len(candidates) == 1:
            in_path = candidates[0]
        else:
            # Try to find the specified file with different extensions
            for cand in candidates:
                if spec.inner_path.split(".")[0] in cand.name:
                    in_path = cand
                    break

    print(f"  Reading {in_path}...")
    df = pd.read_csv(in_path, sep=spec.sep, decimal=spec.decimal, low_memory=False)

    # Drop non-numeric columns
    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] == 0:
        coerced = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        num = coerced.select_dtypes(include=["number"]).copy()

    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {spec.name}")

    # Fill missing values
    num = num.dropna(axis=1, how="all")
    num = num.ffill().bfill()

    if max_rows is not None:
        num = num.iloc[:int(max_rows)]

    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to {out_csv} ({num.shape[0]} rows × {num.shape[1]} cols)...")
    num.to_csv(out_csv, index=False, header=False)

    info = {
        "n_rows": num.shape[0],
        "n_cols": num.shape[1],
        "file_size_bytes": out_csv.stat().st_size,
    }
    return out_csv, info


# ============================================================================
# Scaled benchmark runner
# ============================================================================

def run_scaled_benchmark(
    csv_path: Path,
    *,
    include_hpc: bool = True,
    include_baselines: bool = True,
    include_throughput: bool = True,
    parallel_scaling: bool = False,
    max_workers_scaling: Tuple[int, ...] = (1, 2, 4, 8),
    tracq_bits: int = 8,
    tracq_clamp: float = 500.0,
    zfp_tolerances: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4),
    sz3_abs_errors: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4),
) -> Dict[str, Any]:
    """Run comprehensive benchmark with memory monitoring and parallel scaling."""
    import pandas as pd

    print(f"\n{'='*80}")
    print(f"Running scaled benchmark on: {csv_path}")
    print(f"{'='*80}")

    # Load data and track memory
    gc.collect()
    mem_start = get_memory_usage_mb()

    df = pd.read_csv(csv_path, header=None)
    orig = df.values.T.astype(np.float64)
    n_vars, n_time = orig.shape
    orig_bytes = csv_path.stat().st_size
    data_bytes = orig.nbytes

    mem_after_load = get_memory_usage_mb()

    print(f"  Shape: {n_vars} vars × {n_time} time steps")
    print(f"  CSV size: {orig_bytes / (1024*1024):.2f} MB")
    print(f"  Array size: {data_bytes / (1024*1024):.2f} MB")
    print(f"  Memory usage: {mem_after_load:.1f} MB (delta: {mem_after_load - mem_start:.1f} MB)")

    results: Dict[str, Any] = {
        "input": str(csv_path),
        "shape": {"n_vars": n_vars, "n_time": n_time},
        "orig_bytes": orig_bytes,
        "data_bytes": data_bytes,
        "memory": {
            "initial_mb": mem_start,
            "after_load_mb": mem_after_load,
        },
        "runs": {},
    }

    # Run full benchmark
    print("\n  Running compression benchmarks...")
    try:
        stats = full_benchmark(
            str(csv_path),
            tracq_bits=tracq_bits,
            tracq_clamp=tracq_clamp,
            include_baselines=include_baselines,
            include_hpc_compressors=include_hpc,
            zfp_tolerances=zfp_tolerances,
            sz3_abs_errors=sz3_abs_errors,
            include_throughput=include_throughput,
        )
        results["runs"] = stats.get("runs", {})
        if "hpc_compressors" in stats:
            results["hpc_compressors"] = stats["hpc_compressors"]
        if "throughput" in stats:
            results["throughput"] = stats["throughput"]
    except Exception as e:
        results["benchmark_error"] = str(e)
        traceback.print_exc()

    # Parallel scaling analysis
    if parallel_scaling:
        print("\n  Running parallel scaling analysis...")
        try:
            scaling_results = benchmark_parallel_scaling(
                orig,
                worker_counts=max_workers_scaling,
                bits=tracq_bits,
                clamp=tracq_clamp,
            )
            results["parallel_scaling"] = scaling_results
            print(f"    Workers: {[r['n_workers'] for r in scaling_results]}")
            print(f"    Speedups: {[f\"{r['speedup']:.2f}x\" for r in scaling_results]}")
        except Exception as e:
            results["parallel_scaling_error"] = str(e)
            traceback.print_exc()

    # Final memory snapshot
    gc.collect()
    results["memory"]["final_mb"] = get_memory_usage_mb()

    return results


def generate_rate_distortion_table(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate a rate-distortion comparison table from benchmark results."""
    rows = []
    orig_bytes = results.get("orig_bytes", 1)

    # Extract runs
    for method, r in results.get("runs", {}).items():
        if "metrics" not in r:
            continue
        rows.append({
            "method": method,
            "bytes": r.get("bytes"),
            "ratio": r.get("ratio", r.get("bytes", 0) / orig_bytes),
            "rmse": r["metrics"].get("rmse"),
            "mae": r["metrics"].get("mae"),
            "corr": r["metrics"].get("corr"),
            "encode_s": r.get("encode_s"),
            "decode_s": r.get("decode_s"),
            "category": "tracq" if "tracq" in method else ("hpc" if method in ["zfp_best", "sz3_best"] else "baseline"),
        })

    # Add HPC sweep details
    if "hpc_compressors" in results:
        for compressor in ["zfp", "sz3"]:
            if compressor in results["hpc_compressors"]:
                for r in results["hpc_compressors"][compressor]:
                    if "metrics" not in r:
                        continue
                    param_name = "tolerance" if compressor == "zfp" else "abs_error"
                    rows.append({
                        "method": f"{compressor}_{r.get(param_name, 'unk')}",
                        "bytes": r.get("bytes"),
                        "ratio": r.get("ratio"),
                        "rmse": r["metrics"].get("rmse"),
                        "mae": r["metrics"].get("mae"),
                        "corr": r["metrics"].get("corr"),
                        "encode_s": r.get("encode_s"),
                        "decode_s": r.get("decode_s"),
                        "max_error": r.get("max_error"),
                        "category": "hpc_sweep",
                        "param": r.get(param_name),
                    })

    return rows


# ============================================================================
# Main
# ============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Scaled benchmark for FGCS paper - addresses reviewer concerns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset, full length
  python scripts/scaled_benchmark.py --dataset uci_electricity --full-length

  # All datasets with HPC compressors and parallel scaling
  python scripts/scaled_benchmark.py --all-datasets --full-length --include-hpc --parallel-scaling

  # Custom max rows (for quick testing)
  python scripts/scaled_benchmark.py --dataset uci_electricity --max-rows 10000
        """,
    )

    # Dataset selection
    ap.add_argument("--dataset", type=str, choices=list(DATASETS.keys()), help="Single dataset to benchmark")
    ap.add_argument("--all-datasets", action="store_true", help="Run on all datasets")
    ap.add_argument("--datasets", nargs="*", type=str, default=[], help="List of datasets to benchmark")

    # Scale options
    ap.add_argument("--full-length", action="store_true", help="Use full dataset length (no truncation)")
    ap.add_argument("--max-rows", type=int, default=None, help="Maximum rows to use (overrides --full-length)")

    # Benchmark options
    ap.add_argument("--include-hpc", action="store_true", default=True, help="Include SZ3/ZFP (default: True)")
    ap.add_argument("--no-hpc", action="store_true", help="Skip HPC compressors")
    ap.add_argument("--no-baselines", action="store_true", help="Skip PAA/PLA/SAX baselines")
    ap.add_argument("--throughput", action="store_true", default=True, help="Measure throughput (default: True)")
    ap.add_argument("--parallel-scaling", action="store_true", help="Run parallel scaling analysis")
    ap.add_argument("--max-workers", nargs="*", type=int, default=[1, 2, 4, 8], help="Worker counts for scaling")

    # TRACQ options
    ap.add_argument("--tracq-bits", type=int, default=8, help="TRACQ bit depth")
    ap.add_argument("--tracq-clamp", type=float, default=500.0, help="TRACQ clamp percentage")

    # HPC options
    ap.add_argument("--zfp-tolerances", nargs="*", type=float, default=[1e-1, 1e-2, 1e-3, 1e-4])
    ap.add_argument("--sz3-abs-errors", nargs="*", type=float, default=[1e-1, 1e-2, 1e-3, 1e-4])

    # Output options
    ap.add_argument("--outdir", type=Path, default=Path("scaled_results"), help="Output directory")
    ap.add_argument("--prepare-data", action="store_true", help="Download/prepare datasets")
    ap.add_argument("--force-prepare", action="store_true", help="Force re-preparation of datasets")

    args = ap.parse_args(argv)

    # Determine datasets to run
    dataset_keys = []
    if args.all_datasets:
        dataset_keys = list(DATASETS.keys())
    elif args.dataset:
        dataset_keys = [args.dataset]
    elif args.datasets:
        dataset_keys = args.datasets
    else:
        ap.print_help()
        return 1

    # Determine max_rows
    max_rows = args.max_rows
    if max_rows is None and not args.full_length:
        max_rows = 5000  # Default truncation for backward compatibility

    raw_dir = REPO_ROOT / "data" / "raw"
    processed_dir = REPO_ROOT / "data" / "processed"
    args.outdir.mkdir(parents=True, exist_ok=True)

    include_hpc = args.include_hpc and not args.no_hpc

    all_results: Dict[str, Any] = {}
    all_rd_rows: List[Dict[str, Any]] = []

    print(f"\n{'#'*80}")
    print("# SCALED BENCHMARK FOR FGCS PAPER")
    print(f"# Datasets: {dataset_keys}")
    print(f"# Max rows: {'FULL' if max_rows is None else max_rows}")
    print(f"# Include HPC: {include_hpc}")
    print(f"# Parallel scaling: {args.parallel_scaling}")
    print(f"{'#'*80}")

    for dataset_key in dataset_keys:
        if dataset_key not in DATASETS:
            print(f"WARNING: Unknown dataset '{dataset_key}', skipping")
            continue

        spec = DATASETS[dataset_key]
        print(f"\n\n{'='*80}")
        print(f"DATASET: {dataset_key}")
        print(f"{'='*80}")

        # Prepare dataset
        try:
            csv_path, info = prepare_dataset(
                spec, raw_dir, processed_dir, max_rows, force=args.force_prepare
            )
            print(f"  Prepared: {csv_path}")
            print(f"  Info: {info}")
        except Exception as e:
            print(f"  ERROR preparing dataset: {e}")
            traceback.print_exc()
            continue

        # Run benchmark
        try:
            results = run_scaled_benchmark(
                csv_path,
                include_hpc=include_hpc,
                include_baselines=not args.no_baselines,
                include_throughput=args.throughput,
                parallel_scaling=args.parallel_scaling,
                max_workers_scaling=tuple(args.max_workers),
                tracq_bits=args.tracq_bits,
                tracq_clamp=args.tracq_clamp,
                zfp_tolerances=tuple(args.zfp_tolerances),
                sz3_abs_errors=tuple(args.sz3_abs_errors),
            )
            results["dataset_info"] = info
            all_results[dataset_key] = results

            # Generate rate-distortion table
            rd_rows = generate_rate_distortion_table(results)
            for row in rd_rows:
                row["dataset"] = dataset_key
            all_rd_rows.extend(rd_rows)

            # Save individual results
            out_json = args.outdir / f"{dataset_key}.json"
            out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            print(f"\n  Saved results to: {out_json}")

        except Exception as e:
            print(f"  ERROR running benchmark: {e}")
            traceback.print_exc()
            all_results[dataset_key] = {"error": str(e)}

    # Save aggregate results
    aggregate_json = args.outdir / "all_results.json"
    aggregate_json.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\n\nSaved aggregate results to: {aggregate_json}")

    # Save rate-distortion CSV
    if all_rd_rows:
        rd_csv = args.outdir / "rate_distortion.csv"
        fieldnames = ["dataset", "method", "category", "bytes", "ratio", "rmse", "mae", "corr", "encode_s", "decode_s", "max_error", "param"]
        with rd_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rd_rows)
        print(f"Saved rate-distortion table to: {rd_csv}")

    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for dataset_key, results in all_results.items():
        if "error" in results:
            print(f"  {dataset_key}: ERROR - {results['error']}")
        else:
            shape = results.get("shape", {})
            runs = results.get("runs", {})
            print(f"\n  {dataset_key} ({shape.get('n_vars', '?')} × {shape.get('n_time', '?')}):")

            # Find best methods
            for method in ["tracq_zst", "tracq_png", "zfp_best", "sz3_best", "gzip"]:
                if method in runs and "metrics" in runs[method]:
                    r = runs[method]
                    print(f"    {method:15s}: ratio={r['ratio']*100:6.2f}%  RMSE={r['metrics']['rmse']:10.4f}  Corr={r['metrics']['corr']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

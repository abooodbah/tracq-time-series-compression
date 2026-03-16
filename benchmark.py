from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tracq.tooling import full_benchmark


def main(argv=None):
    p = argparse.ArgumentParser(prog="tracq-benchmark", description="Consolidated benchmark for CSV compression (TRACQ + baselines + gzip/parquet + HPC compressors)")
    p.add_argument("input_csv", help="Input CSV file to benchmark")
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    p.add_argument("--out", type=Path, default=None, help="Write JSON results to a file (UTF-8)")
    p.add_argument("--segments", type=int, default=64, help="Segments for PAA/PLA/SAX")
    p.add_argument("--alphabet", type=int, default=8, help="Alphabet size for SAX")
    p.add_argument("--tracq-bits", type=int, default=8, choices=[8, 16], help="TRACQ quantization bit-depth")
    p.add_argument("--tracq-clamp", type=float, default=500.0, help="TRACQ clamp_pct")
    p.add_argument("--tracq-png-level", type=int, default=6, help="PNG compress_level (0-9)")
    p.add_argument("--tracq-zstd-level", type=int, default=3, help="Zstd compress_level")
    p.add_argument("--tracq-sweep", action="store_true", help="Sweep multiple TRACQ points and pick best")
    p.add_argument("--tracq-sweep-bits", nargs="*", type=int, default=[8, 16])
    p.add_argument("--tracq-sweep-clamps", nargs="*", type=float, default=[200.0, 500.0])
    p.add_argument("--tracq-sweep-png-levels", nargs="*", type=int, default=[0, 3, 6])
    p.add_argument("--tracq-sweep-zstd-levels", nargs="*", type=int, default=[1, 3, 6])
    p.add_argument("--tracq-target-rmse", type=float, default=None, help="Select smallest TRACQ bytes with RMSE <= target")
    p.add_argument("--tracq-target-ratio", type=float, default=None, help="Select lowest RMSE with (bytes/orig_bytes) <= target")
    p.add_argument("--no-baselines", action="store_true", help="Skip baseline methods")

    # HPC compressor options (SZ3/ZFP)
    p.add_argument("--include-hpc", action="store_true", default=True, help="Include SZ3/ZFP error-bounded compressors (default: True)")
    p.add_argument("--no-hpc", action="store_true", help="Skip HPC compressors (SZ3/ZFP)")
    p.add_argument("--zfp-tolerances", nargs="*", type=float, default=[1e-1, 1e-2, 1e-3, 1e-4], help="ZFP tolerance levels to sweep")
    p.add_argument("--sz3-abs-errors", nargs="*", type=float, default=[1e-1, 1e-2, 1e-3, 1e-4], help="SZ3 absolute error bounds to sweep")

    # Throughput options
    p.add_argument("--throughput", action="store_true", help="Measure encoding/decoding throughput (MB/s)")
    p.add_argument("--throughput-iterations", type=int, default=3, help="Iterations for throughput measurement")

    args = p.parse_args(argv)

    include_hpc = args.include_hpc and not args.no_hpc

    stats = full_benchmark(
        args.input_csv,
        segments=args.segments,
        alphabet=args.alphabet,
        tracq_bits=args.tracq_bits,
        tracq_clamp=args.tracq_clamp,
        tracq_png_level=args.tracq_png_level,
        tracq_zstd_level=args.tracq_zstd_level,
        tracq_sweep_mode=bool(args.tracq_sweep),
        tracq_sweep_bits=tuple(args.tracq_sweep_bits),
        tracq_sweep_clamps=tuple(args.tracq_sweep_clamps),
        tracq_sweep_png_levels=tuple(args.tracq_sweep_png_levels),
        tracq_sweep_zstd_levels=tuple(args.tracq_sweep_zstd_levels),
        tracq_target_rmse=args.tracq_target_rmse,
        tracq_target_ratio=args.tracq_target_ratio,
        include_baselines=not args.no_baselines,
        include_hpc_compressors=include_hpc,
        zfp_tolerances=tuple(args.zfp_tolerances),
        sz3_abs_errors=tuple(args.sz3_abs_errors),
        include_throughput=args.throughput,
        n_throughput_iterations=args.throughput_iterations,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        orig_bytes = stats["orig_bytes"]
        print(f"CSV bytes: {orig_bytes}")
        print(f"shape: vars={stats['shape']['n_vars']} time={stats['shape']['n_time']}")
        print(
            f"\n{'Method':<16} {'Bytes':>10} {'Ratio':>8} {'Enc(s)':>9} {'Dec(s)':>9} {'Tot(s)':>9} {'RMSE':>10} {'MAE':>10} {'sMAPE':>10} {'Corr':>8}"
        )
        print("=" * 100)

        order = [
            "tracq_best",
            "tracq_zst",
            "tracq_png",
            "zfp_best",
            "sz3_best",
            "gzip",
            "parquet",
            "paa",
            "pla",
            "sax",
            "gorilla_like",
        ]
        for key in order:
            if key not in stats["runs"]:
                continue
            r = stats["runs"][key]
            size = r.get("bytes")
            ratio = (float(size) / float(orig_bytes) * 100.0) if size is not None else float("nan")
            m = r.get("metrics") or {}
            def _fmt(x, nd=6):
                return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "N/A"

            print(
                f"{key:<16}"
                f" {int(size):>10,}"
                f" {ratio:>7.2f}%"
                f" {_fmt(r.get('encode_s'), 6):>9}"
                f" {_fmt(r.get('decode_s'), 6):>9}"
                f" {_fmt(r.get('total_s'), 6):>9}"
                f" {_fmt(m.get('rmse'), 4):>10}"
                f" {_fmt(m.get('mae'), 4):>10}"
                f" {_fmt(m.get('smape'), 6):>10}"
                f" {_fmt(m.get('corr'), 6):>8}"
            )

        # Print HPC compressor sweep details if available
        if "hpc_compressors" in stats:
            hpc = stats["hpc_compressors"]
            print("\n" + "=" * 100)
            print("HPC Compressor Details (Rate-Distortion Sweep)")
            print("=" * 100)

            avail = hpc.get("availability", {})
            print(f"Availability: ZFP={avail.get('zfp', False)}, SZ3={avail.get('sz3', False)}")

            if "zfp" in hpc:
                print("\nZFP (Error Tolerance Sweep):")
                print(f"  {'Tolerance':>12} {'Bytes':>12} {'Ratio':>8} {'RMSE':>12} {'MaxErr':>12} {'Corr':>8}")
                for r in hpc["zfp"]:
                    if "error" in r:
                        print(f"  {r['tolerance']:>12.1e} ERROR: {r['error']}")
                    else:
                        print(f"  {r['tolerance']:>12.1e} {r['bytes']:>12,} {r['ratio']*100:>7.2f}% {r['metrics']['rmse']:>12.4f} {r['max_error']:>12.4e} {r['metrics']['corr']:>8.4f}")

            if "sz3" in hpc:
                print("\nSZ3 (Absolute Error Sweep):")
                print(f"  {'AbsError':>12} {'Bytes':>12} {'Ratio':>8} {'RMSE':>12} {'MaxErr':>12} {'Corr':>8}")
                for r in hpc["sz3"]:
                    if "error" in r:
                        print(f"  {r['abs_error']:>12.1e} ERROR: {r['error']}")
                    else:
                        print(f"  {r['abs_error']:>12.1e} {r['bytes']:>12,} {r['ratio']*100:>7.2f}% {r['metrics']['rmse']:>12.4f} {r['max_error']:>12.4e} {r['metrics']['corr']:>8.4f}")

        # Print throughput metrics if available
        if "throughput" in stats:
            print("\n" + "=" * 100)
            print("Throughput Metrics (MB/s)")
            print("=" * 100)
            print(f"  {'Method':>12} {'Encode MB/s':>14} {'Decode MB/s':>14} {'Total MB/s':>14} {'DataSize MB':>12}")
            for method, tp in stats["throughput"].items():
                print(f"  {method:>12} {tp['encode_throughput_mbps']:>14.2f} {tp['decode_throughput_mbps']:>14.2f} {tp['total_throughput_mbps']:>14.2f} {tp['data_size_mb']:>12.2f}")


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import json

from .tooling import quick_benchmark_gzip_parquet


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tracq-benchmark", description="Quick gzip/parquet benchmark for a CSV")
    p.add_argument("input_csv", help="Input CSV file to benchmark")
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = p.parse_args(argv)

    stats = quick_benchmark_gzip_parquet(args.input_csv)
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"CSV bytes: {stats['csv_bytes']}")
        print(f"GZIP bytes: {stats['gzip_bytes']} (time {stats['gzip_time_s']:.6f}s)")
        print(f"Parquet bytes: {stats['parquet_bytes']} (time {stats['parquet_time_s']:.6f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

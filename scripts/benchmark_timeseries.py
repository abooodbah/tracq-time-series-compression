"""Benchmark helpers for time-series compression comparisons.

Usage (from PowerShell via conda run):
  conda run -n gtc-env python scripts/benchmark_timeseries.py --input path/to/clean.csv --outdir results

This script will:
 - read the input CSV with pandas
 - measure gzip compression time/size (writing raw CSV -> .gz)
 - measure parquet write time/size (pyarrow) using snappy and none
 - emit a JSON results file and print a summary

Requires: pandas, pyarrow
"""
import argparse
import json
import os
import time
from pathlib import Path

try:
    import pandas as pd
except Exception:
    raise SystemExit('pandas is required. Install into your environment (conda install -n gtc-env pandas -y)')

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

try:
    import zstandard as zstd  # type: ignore
    HAS_ZSTD = True
except Exception:
    HAS_ZSTD = False

try:
    import brotli  # type: ignore
    HAS_BROTLI = True
except Exception:
    HAS_BROTLI = False

import gzip
import sys

# tracq imports are deferred until we add repo root to sys.path inside run_all
HAS_TRACQ = False
TRACQ_IMPORT_ERROR = ""


def time_fn(fn, *args, repeats=1, **kwargs):
    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed / max(1, repeats)


def write_gzip(in_path: Path, out_path: Path):
    # stream-copy original CSV into gzip file
    with in_path.open('rb') as fin, gzip.open(out_path, 'wb') as fout:
        while True:
            chunk = fin.read(1 << 20)
            if not chunk:
                break
            fout.write(chunk)


def write_parquet(df: pd.DataFrame, out_path: Path, compression: str = None):
    # use pyarrow if available
    if not HAS_PYARROW:
        raise RuntimeError('pyarrow not available')
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(out_path), compression=compression)


def write_zstd(in_path: Path, out_path: Path, level: int = 3):
    if not HAS_ZSTD:
        raise RuntimeError('zstandard not available')
    cctx = zstd.ZstdCompressor(level=level)
    with in_path.open('rb') as fin, out_path.open('wb') as fout:
        cctx.copy_stream(fin, fout)


def write_brotli(in_path: Path, out_path: Path, quality: int = 5):
    if not HAS_BROTLI:
        raise RuntimeError('brotli not available')
    with in_path.open('rb') as fin:
        data = fin.read()
    compressed = brotli.compress(data, quality=quality)
    with out_path.open('wb') as fout:
        fout.write(compressed)


def write_tracq_inproc(df: pd.DataFrame, out_path: Path, bits: int = 8, clamp_pct: float = 500.0, epsilon: float = 1e-9, png_level: int = 6):
    if not HAS_TRACQ:
        raise RuntimeError(f'tracq not importable in-process: {TRACQ_IMPORT_ERROR}')
    arr = df.values.T.astype(float)
    grid = TimeSeriesGrid(arr, clamp_pct=clamp_pct, epsilon=epsilon)
    if bits == 8:
        q, meta = grid.quantize_8bit()
    else:
        q, meta = grid.quantize_16bit()
    meta = dict(meta)
    meta['bit_depth'] = bits
    ImageCodec.save_png(str(out_path), q, meta, compress_level=png_level)


def run_all(input_csv: Path, outdir: Path, tracq_bits: int = 8, tracq_png_level: int = 6, tracq_clamp: float = 500.0, tracq_epsilon: float = 1e-9):
    outdir.mkdir(parents=True, exist_ok=True)
    results = {
        'input': str(input_csv.resolve()),
        'original_bytes': input_csv.stat().st_size,
        'runs': {}
    }

    print(f'Reading CSV {input_csv} ...')
    df = pd.read_csv(input_csv, header=None)

    # Make tracq importable by adding repo root to sys.path
    global HAS_TRACQ, TRACQ_IMPORT_ERROR, TimeSeriesGrid, ImageCodec
    if not HAS_TRACQ:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        try:
            from tracq.core import TimeSeriesGrid  # type: ignore
            from tracq.codec import ImageCodec  # type: ignore
            HAS_TRACQ = True
            TRACQ_IMPORT_ERROR = ""
        except Exception as e:
            TRACQ_IMPORT_ERROR = str(e)
            HAS_TRACQ = False

    # gzip
    gz_out = outdir / (input_csv.name + '.gz')
    print('Benchmarking gzip...')
    t_gz = time_fn(write_gzip, input_csv, gz_out, repeats=1)
    size_gz = gz_out.stat().st_size
    results['runs']['gzip'] = {'time_s': t_gz, 'bytes': size_gz, 'ratio': size_gz / results['original_bytes']}

    # zstd (if available)
    if HAS_ZSTD:
        zstd_out = outdir / (input_csv.name + '.zst')
        print('Benchmarking zstd...')
        t_zstd = time_fn(write_zstd, input_csv, zstd_out, repeats=1)
        size_zstd = zstd_out.stat().st_size
        results['runs']['zstd'] = {'time_s': t_zstd, 'bytes': size_zstd, 'ratio': size_zstd / results['original_bytes']}
    else:
        print('zstandard not installed; skipping zstd benchmark')

    # brotli (if available)
    if HAS_BROTLI:
        br_out = outdir / (input_csv.name + '.br')
        print('Benchmarking brotli...')
        t_br = time_fn(write_brotli, input_csv, br_out, repeats=1)
        size_br = br_out.stat().st_size
        results['runs']['brotli'] = {'time_s': t_br, 'bytes': size_br, 'ratio': size_br / results['original_bytes']}
    else:
        print('brotli not installed; skipping brotli benchmark')

    # parquet with snappy and none (if pyarrow present)
    if HAS_PYARROW:
        for comp in ('snappy', None):
            suf = '.parquet' if comp is None else f'.parquet.{comp}'
            outp = outdir / (input_csv.name + suf)
            print(f'Benchmarking parquet (compression={comp})...')
            if comp is None:
                t_parquet = time_fn(write_parquet, df, outp, repeats=1)
            else:
                t_parquet = time_fn(write_parquet, df, outp, repeats=1, compression=comp)
            size_parquet = outp.stat().st_size
            results['runs'][f'parquet_{comp or "none"}'] = {'time_s': t_parquet, 'bytes': size_parquet, 'ratio': size_parquet / results['original_bytes']}
    else:
        print('pyarrow not installed; skipping parquet benchmarks')

    # in-process tracq (no extra interpreter spin-up)
    if HAS_TRACQ:
        tracq_out = outdir / (input_csv.name + '.tracq.inproc.png')
        print(f'Benchmarking tracq in-process (bits={tracq_bits}, png_level={tracq_png_level})...')
        t_tracq = time_fn(
            write_tracq_inproc,
            df,
            tracq_out,
            repeats=1,
            bits=tracq_bits,
            clamp_pct=tracq_clamp,
            epsilon=tracq_epsilon,
            png_level=tracq_png_level,
        )
        size_tracq = tracq_out.stat().st_size
        results['runs'][f'tracq_inproc_bits{tracq_bits}'] = {'time_s': t_tracq, 'bytes': size_tracq, 'ratio': size_tracq / results['original_bytes'], 'png_level': tracq_png_level}
    else:
        print(f'tracq package not importable; skipping in-process tracq benchmark (reason: {TRACQ_IMPORT_ERROR})')

    # Save results
    out_json = outdir / (input_csv.name + '.benchmark.json')
    with out_json.open('w', encoding='utf8') as f:
        json.dump(results, f, indent=2)
    print('\nBenchmark complete. Results:')
    print(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', type=Path, required=True)
    ap.add_argument('--outdir', '-o', type=Path, default=Path('bench_results'))
    ap.add_argument('--tracq-bits', type=int, default=8, choices=[8, 16], help='Bit depth for tracq in-process benchmark')
    ap.add_argument('--tracq-png-level', type=int, default=6, help='PNG compression level for tracq (0-9)')
    ap.add_argument('--tracq-clamp', type=float, default=500.0)
    ap.add_argument('--tracq-epsilon', type=float, default=1e-9)
    args = ap.parse_args()
    run_all(args.input, args.outdir, tracq_bits=args.tracq_bits, tracq_png_level=args.tracq_png_level, tracq_clamp=args.tracq_clamp, tracq_epsilon=args.tracq_epsilon)

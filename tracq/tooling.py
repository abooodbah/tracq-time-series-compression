from __future__ import annotations

import time
import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional, Tuple, Dict, Any, List, Callable

import numpy as np

from .core import TimeSeriesGrid, rmse
from .codec import ImageCodec


# ============================================================================
# Throughput measurement utilities
# ============================================================================

def measure_throughput(
    data: np.ndarray,
    compress_fn: Callable,
    decompress_fn: Callable,
    n_iterations: int = 3,
) -> Dict[str, float]:
    """
    Measure encoding/decoding throughput in MB/s.

    Returns dict with encode_throughput_mbps, decode_throughput_mbps, total_throughput_mbps.
    """
    data_size_mb = data.nbytes / (1024 * 1024)

    # Warm-up run
    compressed = compress_fn(data)
    _ = decompress_fn(compressed)

    # Encode timing
    encode_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        compressed = compress_fn(data)
        t1 = time.perf_counter()
        encode_times.append(t1 - t0)

    # Decode timing
    decode_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = decompress_fn(compressed)
        t1 = time.perf_counter()
        decode_times.append(t1 - t0)

    avg_encode_time = np.median(encode_times)
    avg_decode_time = np.median(decode_times)

    encode_throughput = data_size_mb / avg_encode_time if avg_encode_time > 0 else 0
    decode_throughput = data_size_mb / avg_decode_time if avg_decode_time > 0 else 0
    total_time = avg_encode_time + avg_decode_time
    total_throughput = data_size_mb / total_time if total_time > 0 else 0

    return {
        "encode_throughput_mbps": float(encode_throughput),
        "decode_throughput_mbps": float(decode_throughput),
        "total_throughput_mbps": float(total_throughput),
        "encode_time_s": float(avg_encode_time),
        "decode_time_s": float(avg_decode_time),
        "data_size_mb": float(data_size_mb),
    }


def _parallel_encode_chunk(args):
    """Worker function for parallel encoding."""
    chunk_data, bits, clamp, epsilon = args
    grid = TimeSeriesGrid(chunk_data, clamp_pct=clamp, epsilon=epsilon)
    if bits == 8:
        return grid.quantize_8bit()
    else:
        return grid.quantize_16bit()


def parallel_tracq_encode(
    data: np.ndarray,
    bits: int = 8,
    clamp: float = 500.0,
    epsilon: float = 1e-9,
    n_workers: Optional[int] = None,
    chunk_vars: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, float]]:
    """
    Parallel TRACQ encoding by splitting across variables.

    Args:
        data: 2D array (vars x time)
        bits: Quantization bit depth (8 or 16)
        clamp: Clamp percentage
        epsilon: Division epsilon
        n_workers: Number of parallel workers (default: CPU count)
        chunk_vars: Variables per chunk (0 = auto)

    Returns:
        (quantized_grid, metadata, timing_info)
    """
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), data.shape[0])

    if chunk_vars <= 0:
        chunk_vars = max(1, data.shape[0] // n_workers)

    n_vars, n_time = data.shape

    # Single-threaded path for small data or single worker
    if n_workers == 1 or n_vars <= chunk_vars:
        t0 = time.perf_counter()
        grid = TimeSeriesGrid(data, clamp_pct=clamp, epsilon=epsilon)
        if bits == 8:
            q, meta = grid.quantize_8bit()
        else:
            q, meta = grid.quantize_16bit()
        t1 = time.perf_counter()

        timing = {
            "n_workers": 1,
            "encode_time_s": t1 - t0,
            "parallel": False,
        }
        return q, meta, timing

    # Split data into chunks
    chunks = []
    for i in range(0, n_vars, chunk_vars):
        chunk = data[i:min(i + chunk_vars, n_vars), :]
        chunks.append((chunk, bits, clamp, epsilon))

    t0 = time.perf_counter()

    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_parallel_encode_chunk, chunks))

    # Merge results
    q_parts = [r[0] for r in results]
    meta_parts = [r[1] for r in results]

    # Concatenate quantized grids
    q = np.concatenate(q_parts, axis=0)

    # Merge metadata (combine baselines)
    all_baselines = []
    for m in meta_parts:
        all_baselines.extend(m.get("baseline", []))

    meta = meta_parts[0].copy()
    meta["n_vars"] = n_vars
    meta["baseline"] = all_baselines

    t1 = time.perf_counter()

    timing = {
        "n_workers": n_workers,
        "n_chunks": len(chunks),
        "encode_time_s": t1 - t0,
        "parallel": True,
    }

    return q, meta, timing


def benchmark_parallel_scaling(
    data: np.ndarray,
    worker_counts: Tuple[int, ...] = (1, 2, 4, 8),
    bits: int = 8,
    clamp: float = 500.0,
) -> List[Dict[str, Any]]:
    """
    Benchmark TRACQ encoding across different worker counts for scalability analysis.

    Returns list of timing results for each worker count.
    """
    results = []
    max_workers = multiprocessing.cpu_count()

    for n_workers in worker_counts:
        if n_workers > max_workers:
            continue

        # Run multiple iterations for stable timing
        times = []
        for _ in range(3):
            _, _, timing = parallel_tracq_encode(
                data, bits=bits, clamp=clamp, n_workers=n_workers
            )
            times.append(timing["encode_time_s"])

        avg_time = float(np.median(times))
        data_size_mb = data.nbytes / (1024 * 1024)
        throughput = data_size_mb / avg_time if avg_time > 0 else 0

        results.append({
            "n_workers": n_workers,
            "encode_time_s": avg_time,
            "throughput_mbps": throughput,
            "data_size_mb": data_size_mb,
            "speedup": results[0]["encode_time_s"] / avg_time if results else 1.0,
        })

    return results


def _bench(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def _metrics(orig, recon) -> Dict[str, float]:
    from .metrics import rmse as _rmse, mae as _mae, mape as _mape, smape as _smape, corr as _corr
    return {
        "rmse": float(_rmse(orig, recon)),
        "mae": float(_mae(orig, recon)),
        "mape": float(_mape(orig, recon)),
        "smape": float(_smape(orig, recon)),
        "corr": float(_corr(orig, recon)),
    }


def _sanitize_jsonable(x):
    """Make output strict-JSON-safe (no NaN/Infinity) and native Python types."""
    if isinstance(x, dict):
        return {k: _sanitize_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float):
        return float(x) if np.isfinite(x) else None
    return x


def _tracq_quantize_array(orig: np.ndarray, *, bits: int, clamp: float, epsilon: float = 1e-9):
    bits_i = int(bits)
    if bits_i <= 0:
        raise ValueError("bits must be positive")
    grid = TimeSeriesGrid(orig, clamp_pct=clamp, epsilon=epsilon)

    # Native TRACQ implementations
    if bits_i == 8:
        q, meta = grid.quantize_8bit()
        meta = dict(meta)
        meta["bit_depth"] = 8
        meta["levels"] = 256
        return q, meta
    if bits_i == 16:
        q, meta = grid.quantize_16bit()
        meta = dict(meta)
        meta["bit_depth"] = 16
        meta["levels"] = 65536
        return q, meta

    # Sub-8-bit quantization encoded into uint8 with fewer levels.
    # This lets us sweep bit_depth={2,4,6,...} while storing as bytes.
    if bits_i < 8:
        baseline, pct_grid = grid.compute_percent_grid()
        clamp_f = float(grid.clamp_pct)
        levels = 2 ** bits_i
        if pct_grid.size == 0:
            q = np.zeros((grid.n_vars, 0), dtype=np.uint8)
        else:
            pct_clipped = np.clip(pct_grid, -clamp_f, clamp_f)
            scaled = (pct_clipped + clamp_f) / (2.0 * clamp_f) * float(levels - 1)
            q = np.round(scaled).astype(np.uint8)
        meta = {
            "n_vars": int(grid.n_vars),
            "n_time": int(grid.n_time),
            "clamp_pct": float(grid.clamp_pct),
            "epsilon": float(grid.epsilon),
            "dtype": "uint8",
            "baseline": baseline.tolist(),
            "var_names": grid.var_names if grid.var_names is not None else None,
            "bit_depth": int(bits_i),
            "levels": int(levels),
        }
        return q, meta

    raise ValueError("bits must be 2..8 or 16")


def tracq_sweep(
    orig: np.ndarray,
    *,
    bits: Tuple[int, ...] = (8, 16),
    clamps: Tuple[float, ...] = (200.0, 500.0),
    png_levels: Tuple[int, ...] = (0, 3, 6),
    zstd_levels: Tuple[int, ...] = (1, 3, 6),
) -> Dict[str, Any]:
    """Evaluate multiple TRACQ configurations.

    Returns {"runs": [ ... ]} where each run has bytes/ratio/encode_s/decode_s/total_s/metrics/params.
    """
    import os
    import tempfile
    from .container import pack as pack_zst, unpack as unpack_zst

    if orig.ndim == 1:
        orig = orig[np.newaxis, :]
    if orig.ndim != 2:
        raise ValueError("orig must be 1D or 2D (vars x time)")

    out_runs = []

    for b in bits:
        for clamp in clamps:
            q, meta = _tracq_quantize_array(orig, bits=int(b), clamp=float(clamp))
            recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q, meta)
            base_metrics = _metrics(orig, recon)

            # PNG levels (measure encode + load + reconstruct)
            for lvl in png_levels:
                tmp_png = None
                try:
                    fd, tmp_png = tempfile.mkstemp(suffix=".png")
                    os.close(fd)
                    _, enc_s = _bench(ImageCodec.save_png, tmp_png, q, meta, compress_level=int(lvl))
                    size = os.path.getsize(tmp_png)
                    (q2, meta2), dec_load_s = _bench(ImageCodec.load_png, tmp_png)
                    (_, _), dec_recon_s = _bench(TimeSeriesGrid.reconstruct_from_quantized, q2, meta2)
                    out_runs.append(
                        {
                            "name": "tracq_png",
                            "bytes": int(size),
                            "encode_s": float(enc_s),
                            "decode_s": float(dec_load_s + dec_recon_s),
                            "total_s": float(enc_s + dec_load_s + dec_recon_s),
                            "metrics": base_metrics,
                            "params": {"bits": int(b), "clamp": float(clamp), "png_level": int(lvl)},
                        }
                    )
                finally:
                    if tmp_png and os.path.exists(tmp_png):
                        try:
                            os.remove(tmp_png)
                        except Exception:
                            pass

            # Zstd levels (measure pack + unpack + reconstruct)
            for zl in zstd_levels:
                blob, enc_s = _bench(pack_zst, meta, q, compress_level=int(zl))
                (q3, meta3), dec_unpack_s = _bench(unpack_zst, blob)
                (_, _), dec_recon_s = _bench(TimeSeriesGrid.reconstruct_from_quantized, q3, meta3)
                out_runs.append(
                    {
                        "name": "tracq_zst",
                        "bytes": int(len(blob)),
                        "encode_s": float(enc_s),
                        "decode_s": float(dec_unpack_s + dec_recon_s),
                        "total_s": float(enc_s + dec_unpack_s + dec_recon_s),
                        "metrics": base_metrics,
                        "params": {"bits": int(b), "clamp": float(clamp), "zstd_level": int(zl)},
                    }
                )

    return {"runs": _sanitize_jsonable(out_runs)}


def select_best_tracq(
    sweep: Dict[str, Any],
    *,
    orig_bytes: int,
    target_rmse: Optional[float] = None,
    target_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Pick the best TRACQ point from a sweep.

    - If target_rmse is set: choose the smallest bytes among runs with rmse <= target_rmse.
    - If target_ratio is set: choose the lowest rmse among runs with (bytes/orig_bytes) <= target_ratio.
    """
    runs = list(sweep.get("runs") or [])
    if not runs:
        raise ValueError("empty sweep")
    if (target_rmse is None) == (target_ratio is None):
        raise ValueError("set exactly one of target_rmse or target_ratio")

    def ratio_of(r):
        return float(r["bytes"]) / float(orig_bytes)

    if target_rmse is not None:
        feasible = [r for r in runs if r.get("metrics", {}).get("rmse") is not None and r["metrics"]["rmse"] <= float(target_rmse)]
        if not feasible:
            # If nothing meets target, fall back to minimum RMSE.
            best = min(runs, key=lambda r: (r.get("metrics", {}).get("rmse") is None, r.get("metrics", {}).get("rmse", float("inf")), r.get("bytes", 1 << 60)))
            best = dict(best)
            best["selection"] = {"mode": "target_rmse", "target": float(target_rmse), "met": False}
            best["ratio"] = ratio_of(best)
            return _sanitize_jsonable(best)
        best = min(feasible, key=lambda r: (r.get("bytes", 1 << 60), r.get("total_s", float("inf"))))
        best = dict(best)
        best["selection"] = {"mode": "target_rmse", "target": float(target_rmse), "met": True}
        best["ratio"] = ratio_of(best)
        return _sanitize_jsonable(best)

    # target_ratio
    feasible = [r for r in runs if ratio_of(r) <= float(target_ratio)]
    if not feasible:
        best = min(runs, key=lambda r: (r.get("bytes", 1 << 60), r.get("total_s", float("inf"))))
        best = dict(best)
        best["selection"] = {"mode": "target_ratio", "target": float(target_ratio), "met": False}
        best["ratio"] = ratio_of(best)
        return _sanitize_jsonable(best)
    best = min(feasible, key=lambda r: (r.get("metrics", {}).get("rmse", float("inf")), r.get("bytes", 1 << 60)))
    best = dict(best)
    best["selection"] = {"mode": "target_ratio", "target": float(target_ratio), "met": True}
    best["ratio"] = ratio_of(best)
    return _sanitize_jsonable(best)


def auto_tune_bits(
    data: np.ndarray,
    clamp_pct: float = 500.0,
    epsilon: float = 1e-9,
    max_rmse: Optional[float] = None,
    sample_max_cols: int = 512,
) -> Tuple[int, float]:
    """
    Auto-tune quantization bit-depth for a time-series array (vars x time).

    Binary-search across mid-bit values in [4..16] on a data sample and returns a
    chosen physical bit depth (8 or 16) and the evaluated RMSE on the sample.

    Returns (chosen_bit_depth, sample_rmse).
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    sample_cols = min(sample_max_cols, max(1, data.shape[1]))
    data_sample = data[:, :sample_cols]

    if max_rmse is None:
        mean_mag = float(np.mean(np.abs(data_sample))) if data_sample.size else 1.0
        target_rmse = 0.01 * (mean_mag if mean_mag > 0 else 1.0)
    else:
        target_rmse = float(max_rmse)

    low_bits = 4
    high_bits = 16
    best_bits_mid: Optional[int] = None
    best_rmse = float("inf")

    while low_bits <= high_bits:
        mid = (low_bits + high_bits) // 2
        # treat mid as number of bits; convert to levels
        levels = 2 ** mid
        if data_sample.shape[1] <= 1:
            test_rmse = 0.0
        else:
            sample_grid = TimeSeriesGrid(data_sample, clamp_pct=clamp_pct, epsilon=epsilon)
            baseline, sample_pct = sample_grid.compute_percent_grid()
            # quantize into integer levels mapped to either 8/16 space depending on levels
            if levels <= 256:
                # map into 8-bit space
                scaled = (np.clip(sample_pct, -clamp_pct, clamp_pct) + clamp_pct) / (2.0 * clamp_pct) * 255.0
                q_levels = np.round(scaled).astype(np.uint8)
                metadata = {
                    "n_vars": int(sample_grid.n_vars),
                    "n_time": int(sample_grid.n_time),
                    "clamp_pct": float(clamp_pct),
                    "epsilon": float(epsilon),
                    "dtype": "uint8",
                    "baseline": baseline.tolist(),
                }
                try:
                    recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q_levels, metadata)
                    test_rmse = rmse(data_sample, recon)
                except Exception:
                    test_rmse = float("inf")
            else:
                # map into 16-bit space
                scaled = (np.clip(sample_pct, -clamp_pct, clamp_pct) + clamp_pct) / (2.0 * clamp_pct) * 65535.0
                q_levels = np.round(scaled).astype(np.uint16)
                metadata = {
                    "n_vars": int(sample_grid.n_vars),
                    "n_time": int(sample_grid.n_time),
                    "clamp_pct": float(clamp_pct),
                    "epsilon": float(epsilon),
                    "dtype": "uint16",
                    "baseline": baseline.tolist(),
                }
                try:
                    recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q_levels, metadata)
                    test_rmse = rmse(data_sample, recon)
                except Exception:
                    test_rmse = float("inf")

        if np.isfinite(test_rmse) and test_rmse <= target_rmse:
            best_bits_mid = mid
            best_rmse = test_rmse
            high_bits = mid - 1
        else:
            low_bits = mid + 1

    if best_bits_mid is None:
        chosen_bits = 16
        final_rmse = float("inf")
    else:
        chosen_bits = 8 if best_bits_mid <= 8 else 16
        final_rmse = best_rmse

    return chosen_bits, float(final_rmse)


def rgb_image_from_first_three(q8: np.ndarray) -> np.ndarray:
    """
    Map first three rows of an 8-bit quantized grid into an HxW x 3 RGB uint8 image.
    q8 shape expected: (n_vars, n_time-1) or (n_vars, n_time)
    Returns numpy uint8 array shape (H, W, 3).
    """
    if q8 is None:
        raise ValueError("q8 must be provided")
    if q8.ndim != 2:
        raise ValueError("q8 must be 2D array (n_vars, n_time-1)")
    if q8.shape[0] < 3:
        raise ValueError("q8 must have at least 3 variables/rows to map to RGB")

    r = q8[0].astype(np.uint8)
    g = q8[1].astype(np.uint8)
    b = q8[2].astype(np.uint8)
    # stack into HxW x 3, where H = n_vars (we'll use rows==variables) and W = time-1
    # stacking along last axis requires matching shapes r,g,b -> (H, W)
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def safety_check(original: np.ndarray, reconstructed: np.ndarray, max_rmse: float) -> Dict[str, Any]:
    """
    Compute RMSE between original and reconstructed timeseries and return a dict
    with rmse and a boolean flag 'ok' indicating whether rmse <= max_rmse.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed must have the same shape for RMSE check")
    cur_rmse = rmse(original, reconstructed)
    ok = bool(cur_rmse <= float(max_rmse))
    return {"rmse": float(cur_rmse), "ok": ok, "max_rmse": float(max_rmse)}


def save_png_selfcontained(path: str, uint_grid: np.ndarray, metadata: Dict[str, Any], compress_level: int = 6) -> None:
    """Helper that uses ImageCodec.save_png to emit a self-contained PNG file (grid + metadata).

    Kept small to be used by tests and benchmarks.
    """
    ImageCodec.save_png(path, uint_grid, metadata, compress_level=compress_level)


def quick_benchmark_gzip_parquet(input_csv: str) -> Dict[str, Any]:
    """Backward-compatible helper.

    Kept for existing callers; prefer full_benchmark(...) for research-grade outputs.
    """
    out = full_benchmark(input_csv, include_baselines=False)
    # Maintain legacy keys expected by benchmark.py in earlier versions.
    stats = {
        "csv_bytes": out["orig_bytes"],
        "gzip_bytes": out["runs"]["gzip"]["bytes"],
        "gzip_time_s": out["runs"]["gzip"]["encode_s"],
        "parquet_bytes": out["runs"]["parquet"]["bytes"],
        "parquet_time_s": out["runs"]["parquet"]["encode_s"],
        "tracq_png_bytes": out["runs"]["tracq_png"]["bytes"],
        "tracq_png_time_s": out["runs"]["tracq_png"]["encode_s"],
        "tracq_png_error": out["runs"]["tracq_png"]["metrics"]["rmse"],
        "tracq_zst_bytes": out["runs"]["tracq_zst"]["bytes"],
        "tracq_zst_time_s": out["runs"]["tracq_zst"]["encode_s"],
        "tracq_zst_error": out["runs"]["tracq_zst"]["metrics"]["rmse"],
    }
    return stats


def full_benchmark(
    input_csv: str,
    *,
    segments: int = 64,
    alphabet: int = 8,
    tracq_bits: int = 8,
    tracq_clamp: float = 500.0,
    tracq_png_level: int = 6,
    tracq_zstd_level: int = 3,
    tracq_sweep_mode: bool = False,
    tracq_sweep_bits: Tuple[int, ...] = (8, 16),
    tracq_sweep_clamps: Tuple[float, ...] = (200.0, 500.0),
    tracq_sweep_png_levels: Tuple[int, ...] = (0, 3, 6),
    tracq_sweep_zstd_levels: Tuple[int, ...] = (1, 3, 6),
    tracq_target_rmse: Optional[float] = None,
    tracq_target_ratio: Optional[float] = None,
    include_baselines: bool = True,
    include_hpc_compressors: bool = True,
    zfp_tolerances: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4),
    sz3_abs_errors: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4),
    include_throughput: bool = False,
    n_throughput_iterations: int = 3,
) -> Dict[str, Any]:
    """Run a consolidated benchmark with consistent metrics.

    Returns strict-JSON-serializable dict:
      {
        input, orig_bytes, shape,
        runs: { method: {bytes, ratio, encode_s, decode_s, total_s, metrics} }
      }

    All lossy methods compute metrics against the same original array.
    Lossless methods (gzip/parquet) also compute metrics by round-tripping.
    """
    import gzip
    import io
    import os
    import tempfile
    import pandas as pd

    from .container import pack as pack_zst, unpack as unpack_zst

    orig_bytes = os.path.getsize(input_csv)
    df = pd.read_csv(input_csv, header=None)
    orig = df.values.T.astype(float)
    n_vars, n_time = orig.shape

    runs: Dict[str, Any] = {}

    # --- GZIP (lossless) ---
    with open(input_csv, "rb") as f:
        raw_csv = f.read()
    gz_bytes, enc_s = _bench(gzip.compress, raw_csv)
    decoded_raw, dec_s = _bench(gzip.decompress, gz_bytes)
    # Decode to array (count towards decode time for apples-to-apples metrics)
    df_gz, parse_s = _bench(pd.read_csv, io.BytesIO(decoded_raw), header=None)
    recon_gz = df_gz.values.T.astype(float)
    runs["gzip"] = {
        "bytes": int(len(gz_bytes)),
        "ratio": float(len(gz_bytes) / orig_bytes),
        "encode_s": float(enc_s),
        "decode_s": float(dec_s + parse_s),
        "total_s": float(enc_s + dec_s + parse_s),
        "metrics": _metrics(orig, recon_gz),
    }

    # --- Parquet (lossless) ---
    parquet_path = input_csv + ".tmp.parquet"
    try:
        _, p_enc_s = _bench(df.to_parquet, parquet_path, index=False)
        parquet_size = os.path.getsize(parquet_path)
        df_par, p_dec_s = _bench(pd.read_parquet, parquet_path)
        recon_par = df_par.values.T.astype(float)
        runs["parquet"] = {
            "bytes": int(parquet_size),
            "ratio": float(parquet_size / orig_bytes),
            "encode_s": float(p_enc_s),
            "decode_s": float(p_dec_s),
            "total_s": float(p_enc_s + p_dec_s),
            "metrics": _metrics(orig, recon_par),
        }
    finally:
        try:
            os.remove(parquet_path)
        except Exception:
            pass

    tracq_extra: Dict[str, Any] = {}
    if tracq_sweep_mode:
        sweep = tracq_sweep(
            orig,
            bits=tuple(int(x) for x in tracq_sweep_bits),
            clamps=tuple(float(x) for x in tracq_sweep_clamps),
            png_levels=tuple(int(x) for x in tracq_sweep_png_levels),
            zstd_levels=tuple(int(x) for x in tracq_sweep_zstd_levels),
        )
        best = select_best_tracq(
            sweep,
            orig_bytes=int(orig_bytes),
            target_rmse=tracq_target_rmse,
            target_ratio=tracq_target_ratio,
        )

        # Present the chosen best run as a method entry
        runs["tracq_best"] = {
            "bytes": int(best["bytes"]),
            "ratio": float(best["ratio"]),
            "encode_s": float(best["encode_s"]),
            "decode_s": float(best["decode_s"]),
            "total_s": float(best["total_s"]),
            "metrics": best.get("metrics"),
            "params": best.get("params"),
            "variant": best.get("name"),
            "selection": best.get("selection"),
        }
        tracq_extra["tracq_sweep"] = sweep
        tracq_extra["tracq_best"] = best
    else:
        # --- Fixed TRACQ PNG ---
        q, meta = _tracq_quantize_array(orig, bits=int(tracq_bits), clamp=float(tracq_clamp))
        tmp_png = None
        try:
            fd, tmp_png = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            _, enc_s = _bench(ImageCodec.save_png, tmp_png, q, meta, compress_level=int(tracq_png_level))
            png_size = os.path.getsize(tmp_png)
            (q2, meta2), dec_load_s = _bench(ImageCodec.load_png, tmp_png)
            (recon, _), dec_recon_s = _bench(TimeSeriesGrid.reconstruct_from_quantized, q2, meta2)
            runs["tracq_png"] = {
                "bytes": int(png_size),
                "ratio": float(png_size / orig_bytes),
                "encode_s": float(enc_s),
                "decode_s": float(dec_load_s + dec_recon_s),
                "total_s": float(enc_s + dec_load_s + dec_recon_s),
                "metrics": _metrics(orig, recon),
                "params": {"bits": int(tracq_bits), "clamp": float(tracq_clamp), "png_level": int(tracq_png_level)},
            }
        finally:
            if tmp_png and os.path.exists(tmp_png):
                try:
                    os.remove(tmp_png)
                except Exception:
                    pass

        # --- Fixed TRACQ Zstd ---
        q, meta = _tracq_quantize_array(orig, bits=int(tracq_bits), clamp=float(tracq_clamp))
        blob, enc_s = _bench(pack_zst, meta, q, compress_level=int(tracq_zstd_level))
        (q3, meta3), dec_unpack_s = _bench(unpack_zst, blob)
        (recon, _), dec_recon_s = _bench(TimeSeriesGrid.reconstruct_from_quantized, q3, meta3)
        runs["tracq_zst"] = {
            "bytes": int(len(blob)),
            "ratio": float(len(blob) / orig_bytes),
            "encode_s": float(enc_s),
            "decode_s": float(dec_unpack_s + dec_recon_s),
            "total_s": float(enc_s + dec_unpack_s + dec_recon_s),
            "metrics": _metrics(orig, recon),
            "params": {"bits": int(tracq_bits), "clamp": float(tracq_clamp), "zstd_level": int(tracq_zstd_level)},
        }

    # --- Baselines (lossy) ---
    if include_baselines:
        from .baselines import (
            paa_compress, paa_decompress,
            pla_compress, pla_decompress,
            sax_compress, sax_decompress,
            gorilla_like_compress, gorilla_like_decompress,
        )

        (paa_coeffs, paa_meta), enc_s = _bench(paa_compress, orig, int(segments))
        recon, dec_s = _bench(paa_decompress, paa_coeffs, paa_meta)
        runs["paa"] = {
            "bytes": int(paa_coeffs.nbytes),
            "ratio": float(paa_coeffs.nbytes / orig_bytes),
            "encode_s": float(enc_s),
            "decode_s": float(dec_s),
            "total_s": float(enc_s + dec_s),
            "metrics": _metrics(orig, recon),
            "params": {"segments": int(segments)},
        }

        (pla_coeffs, pla_meta), enc_s = _bench(pla_compress, orig, int(segments))
        recon, dec_s = _bench(pla_decompress, pla_coeffs, pla_meta)
        runs["pla"] = {
            "bytes": int(pla_coeffs.nbytes),
            "ratio": float(pla_coeffs.nbytes / orig_bytes),
            "encode_s": float(enc_s),
            "decode_s": float(dec_s),
            "total_s": float(enc_s + dec_s),
            "metrics": _metrics(orig, recon),
            "params": {"segments": int(segments)},
        }

        (symbols, sax_meta), enc_s = _bench(sax_compress, orig, int(segments), int(alphabet))
        recon, dec_s = _bench(sax_decompress, symbols, sax_meta)
        runs["sax"] = {
            "bytes": int(symbols.nbytes),
            "ratio": float(symbols.nbytes / orig_bytes),
            "encode_s": float(enc_s),
            "decode_s": float(dec_s),
            "total_s": float(enc_s + dec_s),
            "metrics": _metrics(orig, recon),
            "params": {"segments": int(segments), "alphabet": int(alphabet)},
        }

        (g_bytes, g_meta), enc_s = _bench(gorilla_like_compress, orig)
        recon, dec_s = _bench(gorilla_like_decompress, g_bytes, g_meta)
        runs["gorilla_like"] = {
            "bytes": int(len(g_bytes)),
            "ratio": float(len(g_bytes) / orig_bytes),
            "encode_s": float(enc_s),
            "decode_s": float(dec_s),
            "total_s": float(enc_s + dec_s),
            "metrics": _metrics(orig, recon),
            "params": {},
        }

        # --- Delta+Zstd (near-lossless) ---
        from .baselines import delta_zstd_compress, delta_zstd_decompress

        try:
            (dz_bytes, dz_meta), enc_s = _bench(delta_zstd_compress, orig, 3)
            recon, dec_s = _bench(delta_zstd_decompress, dz_bytes, dz_meta)
            runs["delta_zstd"] = {
                "bytes": int(len(dz_bytes)),
                "ratio": float(len(dz_bytes) / orig_bytes),
                "encode_s": float(enc_s),
                "decode_s": float(dec_s),
                "total_s": float(enc_s + dec_s),
                "metrics": _metrics(orig, recon),
                "params": {"level": 3},
            }
        except ImportError:
            pass  # zstandard not installed

    # --- HPC Error-Bounded Compressors (SZ3/ZFP) ---
    hpc_results: Dict[str, Any] = {}
    if include_hpc_compressors:
        from .baselines import (
            HAS_ZFP, HAS_HDF5PLUGIN,
            zfp_compress, zfp_decompress,
            sz3_compress, sz3_decompress,
            get_available_hpc_compressors,
        )

        hpc_results["availability"] = get_available_hpc_compressors()

        # ZFP benchmarks across tolerance levels
        if HAS_ZFP:
            zfp_runs = []
            for tol in zfp_tolerances:
                try:
                    (compressed, meta), enc_s = _bench(zfp_compress, orig, mode="tolerance", tolerance=tol)
                    decompressed, dec_s = _bench(zfp_decompress, compressed, meta)

                    zfp_runs.append({
                        "tolerance": float(tol),
                        "bytes": int(len(compressed)),
                        "ratio": float(len(compressed) / orig_bytes),
                        "encode_s": float(enc_s),
                        "decode_s": float(dec_s),
                        "total_s": float(enc_s + dec_s),
                        "metrics": _metrics(orig, decompressed),
                        "max_error": float(np.max(np.abs(orig - decompressed))),
                    })
                except Exception as e:
                    zfp_runs.append({
                        "tolerance": float(tol),
                        "error": str(e),
                    })

            hpc_results["zfp"] = zfp_runs

            # Select best ZFP config (lowest ratio meeting error threshold)
            valid_zfp = [r for r in zfp_runs if "metrics" in r]
            if valid_zfp:
                # Pick configuration with best balance (lowest ratio with reasonable RMSE)
                best_zfp = min(valid_zfp, key=lambda r: (r["ratio"], r["metrics"]["rmse"]))
                runs["zfp_best"] = {
                    "bytes": int(best_zfp["bytes"]),
                    "ratio": float(best_zfp["ratio"]),
                    "encode_s": float(best_zfp["encode_s"]),
                    "decode_s": float(best_zfp["decode_s"]),
                    "total_s": float(best_zfp["total_s"]),
                    "metrics": best_zfp["metrics"],
                    "params": {"tolerance": best_zfp["tolerance"]},
                    "max_error": float(best_zfp["max_error"]),
                }

        # SZ3 benchmarks across absolute error levels
        if HAS_HDF5PLUGIN:
            sz3_runs = []
            for err in sz3_abs_errors:
                try:
                    (compressed, meta), enc_s = _bench(sz3_compress, orig, mode="abs", abs_error=err)
                    decompressed, dec_s = _bench(sz3_decompress, compressed, meta)

                    sz3_runs.append({
                        "abs_error": float(err),
                        "bytes": int(len(compressed)),
                        "ratio": float(len(compressed) / orig_bytes),
                        "encode_s": float(enc_s),
                        "decode_s": float(dec_s),
                        "total_s": float(enc_s + dec_s),
                        "metrics": _metrics(orig, decompressed),
                        "max_error": float(np.max(np.abs(orig - decompressed))),
                    })
                except Exception as e:
                    sz3_runs.append({
                        "abs_error": float(err),
                        "error": str(e),
                    })

            hpc_results["sz3"] = sz3_runs

            # Select best SZ3 config
            valid_sz3 = [r for r in sz3_runs if "metrics" in r]
            if valid_sz3:
                best_sz3 = min(valid_sz3, key=lambda r: (r["ratio"], r["metrics"]["rmse"]))
                runs["sz3_best"] = {
                    "bytes": int(best_sz3["bytes"]),
                    "ratio": float(best_sz3["ratio"]),
                    "encode_s": float(best_sz3["encode_s"]),
                    "decode_s": float(best_sz3["decode_s"]),
                    "total_s": float(best_sz3["total_s"]),
                    "metrics": best_sz3["metrics"],
                    "params": {"abs_error": best_sz3["abs_error"]},
                    "max_error": float(best_sz3["max_error"]),
                }

    # --- Throughput Metrics ---
    throughput_results: Dict[str, Any] = {}
    if include_throughput:
        data_size_mb = orig.nbytes / (1024 * 1024)

        # TRACQ throughput
        def tracq_encode(d):
            grid = TimeSeriesGrid(d, clamp_pct=tracq_clamp)
            if tracq_bits == 8:
                return grid.quantize_8bit()
            return grid.quantize_16bit()

        def tracq_decode(qm):
            q, meta = qm
            return TimeSeriesGrid.reconstruct_from_quantized(q, meta)

        tracq_tp = measure_throughput(orig, tracq_encode, tracq_decode, n_throughput_iterations)
        throughput_results["tracq"] = tracq_tp

        # Gzip throughput
        import gzip as gzip_mod
        def gzip_encode(d):
            return gzip_mod.compress(d.tobytes())
        def gzip_decode(b):
            return np.frombuffer(gzip_mod.decompress(b), dtype=orig.dtype).reshape(orig.shape)

        gzip_tp = measure_throughput(orig, gzip_encode, gzip_decode, n_throughput_iterations)
        throughput_results["gzip"] = gzip_tp

        # ZFP throughput (if available)
        if include_hpc_compressors and HAS_ZFP:
            def zfp_enc(d):
                return zfp_compress(d, mode="tolerance", tolerance=1e-3)
            def zfp_dec(cm):
                return zfp_decompress(cm[0], cm[1])
            zfp_tp = measure_throughput(orig, zfp_enc, zfp_dec, n_throughput_iterations)
            throughput_results["zfp"] = zfp_tp

    out = {
        "input": str(input_csv),
        "orig_bytes": int(orig_bytes),
        "shape": {"n_vars": int(n_vars), "n_time": int(n_time)},
        "runs": runs,
    }
    out.update(tracq_extra)

    if hpc_results:
        out["hpc_compressors"] = hpc_results

    if throughput_results:
        out["throughput"] = throughput_results

    return _sanitize_jsonable(out)

from __future__ import annotations

import gc
import gzip
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .container import pack as pack_zst, unpack as unpack_zst
from .core import TimeSeriesGrid
from .core_enhanced import EnhancedTimeSeriesGrid


def get_memory_usage_mb() -> float:
    """Return current process RSS in MiB if psutil is available."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 * 1024))
    except Exception:
        return 0.0


def _select_numeric_columns(
    df: Any,
    *,
    drop_name_patterns: Sequence[str] = ("unnamed", "index", "timestamp", "date", "time", "label", "failure", "anomaly", "target"),
) -> Any:
    """Return a numeric-only dataframe, dropping common label/time columns when present."""
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}
    drop_cols = []
    for pattern in drop_name_patterns:
        for lower_name, original_name in lower_map.items():
            if pattern in lower_name:
                drop_cols.append(original_name)
    if drop_cols:
        df = df.drop(columns=list(dict.fromkeys(drop_cols)), errors="ignore")

    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] == 0:
        coerced = df.apply(lambda s: np.asarray(s).astype(str))
        coerced = coerced.apply(lambda s: np.asarray(s))
        import pandas as pd

        coerced_df = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in df.columns})
        num = coerced_df.select_dtypes(include=["number"]).copy()

    num = num.dropna(axis=1, how="all")
    if num.shape[1] == 0:
        raise ValueError("no numeric sensor columns found after preprocessing")

    num = num.ffill().bfill()
    return num


def iter_numeric_csv_windows(
    csv_path: str | Path,
    *,
    window_rows: int,
    drop_name_patterns: Sequence[str] = ("unnamed", "index", "timestamp", "date", "time", "label", "failure", "anomaly", "target"),
) -> Iterator[np.ndarray]:
    """
    Yield numeric windows from a CSV as arrays shaped (n_vars, n_time).

    The input CSV may include headers and mixed column types. Common timestamp
    or label columns are removed by name when present.
    """
    import pandas as pd

    if int(window_rows) <= 0:
        raise ValueError("window_rows must be positive")

    for chunk in pd.read_csv(csv_path, chunksize=int(window_rows), low_memory=False):
        num = _select_numeric_columns(chunk, drop_name_patterns=drop_name_patterns)
        arr = num.to_numpy(dtype=np.float64, copy=False)
        if arr.size == 0:
            continue
        yield np.ascontiguousarray(arr.T)


def load_numeric_csv_matrix(
    csv_path: str | Path,
    *,
    drop_name_patterns: Sequence[str] = ("unnamed", "index", "timestamp", "date", "time", "label", "failure", "anomaly", "target"),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a CSV into one numeric matrix shaped (n_vars, n_time).

    This is intended for full-dataset experiments where the entire sequence must
    be evaluated at once, such as rate-distortion analysis on MetroPT-3.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    num = _select_numeric_columns(df, drop_name_patterns=drop_name_patterns)
    del df

    arr = num.to_numpy(dtype=np.float64, copy=True)
    del num
    gc.collect()

    if arr.size == 0:
        raise ValueError("no numeric samples found in CSV")

    matrix = np.ascontiguousarray(arr.T)
    return matrix, {
        "source_csv": str(csv_path),
        "n_rows": int(arr.shape[0]),
        "n_vars": int(arr.shape[1]),
        "input_bytes": int(matrix.nbytes),
    }


def make_synthetic_sensor_window(
    start_index: int,
    rows: int,
    n_vars: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a deterministic multivariate sensor-like window shaped (n_vars, rows).
    """
    if rows <= 0 or n_vars <= 0:
        raise ValueError("rows and n_vars must be positive")

    t = np.arange(start_index, start_index + rows, dtype=np.float64)
    rng = np.random.default_rng(seed)

    amplitudes = rng.uniform(0.5, 8.0, size=(n_vars, 1))
    drifts = rng.uniform(-2e-4, 2e-4, size=(n_vars, 1))
    periods = rng.uniform(200.0, 5000.0, size=(n_vars, 1))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_vars, 1))
    baselines = rng.uniform(5.0, 30.0, size=(n_vars, 1))
    noise_scales = rng.uniform(0.01, 0.08, size=(n_vars, 1))

    tt = t[np.newaxis, :]
    seasonal = amplitudes * np.sin((2.0 * np.pi * tt / periods) + phases)
    harmonic = 0.25 * amplitudes * np.sin((4.0 * np.pi * tt / periods) + 0.5 * phases)
    drift = drifts * tt

    noise_rng = np.random.default_rng(seed + int(start_index))
    noise = noise_rng.normal(0.0, 1.0, size=(n_vars, rows)) * noise_scales

    data = baselines + seasonal + harmonic + drift + noise

    # Add sparse, smooth event bursts on a subset of channels.
    event_channels = np.arange(0, n_vars, max(1, n_vars // 4))
    for idx, ch in enumerate(event_channels):
        center = int((idx + 1) * rows / (len(event_channels) + 1))
        width = max(8, rows // 50)
        pulse = np.exp(-0.5 * ((np.arange(rows) - center) / width) ** 2)
        data[ch] += 0.6 * amplitudes[ch, 0] * pulse

    return np.ascontiguousarray(data.astype(np.float64))


def _encode_tracq_window(
    window: np.ndarray,
    *,
    enhanced: bool,
    bits: int = 8,
    clamp_pct: float = 500.0,
    zstd_level: int = 3,
    auto_offset: bool = True,
) -> Dict[str, float]:
    """
    Encode/decode one window and return byte and timing statistics.
    """
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")

    encoder_cls = EnhancedTimeSeriesGrid if enhanced else TimeSeriesGrid
    encoder_kwargs: Dict[str, Any] = {"clamp_pct": float(clamp_pct)}
    if enhanced:
        encoder_kwargs.update(
            {
                "adaptive_clamp": True,
                "use_mu_law": True,
                "auto_offset": bool(auto_offset),
            }
        )

    t0 = time.perf_counter()
    grid = encoder_cls(window, **encoder_kwargs)
    if bits == 8:
        q, meta = grid.quantize_8bit()
    else:
        q, meta = grid.quantize_16bit()
    blob = pack_zst(meta, q, compress_level=int(zstd_level))
    t1 = time.perf_counter()

    q2, meta2 = unpack_zst(blob)
    if enhanced:
        EnhancedTimeSeriesGrid.reconstruct_from_quantized(q2, meta2)
    else:
        TimeSeriesGrid.reconstruct_from_quantized(q2, meta2)
    t2 = time.perf_counter()

    return {
        "input_bytes": float(window.nbytes),
        "compressed_bytes": float(len(blob)),
        "encode_s": float(t1 - t0),
        "decode_s": float(t2 - t1),
    }


def _encode_gzip_window(window: np.ndarray) -> Dict[str, float]:
    raw = window.tobytes(order="C")
    t0 = time.perf_counter()
    blob = gzip.compress(raw)
    t1 = time.perf_counter()
    _ = np.frombuffer(gzip.decompress(blob), dtype=window.dtype).reshape(window.shape)
    t2 = time.perf_counter()
    return {
        "input_bytes": float(window.nbytes),
        "compressed_bytes": float(len(blob)),
        "encode_s": float(t1 - t0),
        "decode_s": float(t2 - t1),
    }


def benchmark_window_stream(
    windows: Iterable[np.ndarray],
    *,
    method: str,
    bits: int = 8,
    clamp_pct: float = 500.0,
    zstd_level: int = 3,
    auto_offset: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark a stream of windows with one method and aggregate bytes/timings.
    """
    if method not in {"tracq_base", "tracq_enhanced", "gzip"}:
        raise ValueError(f"unsupported method: {method}")

    total_input_bytes = 0.0
    total_output_bytes = 0.0
    total_encode_s = 0.0
    total_decode_s = 0.0
    peak_rss_mb = get_memory_usage_mb()
    n_windows = 0
    n_rows = 0
    n_vars: Optional[int] = None

    for window in windows:
        n_windows += 1
        n_rows += int(window.shape[1])
        n_vars = int(window.shape[0]) if n_vars is None else n_vars

        if method == "gzip":
            stats = _encode_gzip_window(window)
        else:
            stats = _encode_tracq_window(
                window,
                enhanced=(method == "tracq_enhanced"),
                bits=bits,
                clamp_pct=clamp_pct,
                zstd_level=zstd_level,
                auto_offset=auto_offset,
            )

        total_input_bytes += stats["input_bytes"]
        total_output_bytes += stats["compressed_bytes"]
        total_encode_s += stats["encode_s"]
        total_decode_s += stats["decode_s"]

        peak_rss_mb = max(peak_rss_mb, get_memory_usage_mb())
        del window
        gc.collect()

    if n_windows == 0:
        raise ValueError("no windows were provided")

    encode_mbps = (total_input_bytes / (1024 * 1024)) / total_encode_s if total_encode_s > 0 else 0.0
    decode_mbps = (total_input_bytes / (1024 * 1024)) / total_decode_s if total_decode_s > 0 else 0.0

    return {
        "method": method,
        "n_windows": int(n_windows),
        "n_rows": int(n_rows),
        "n_vars": int(n_vars or 0),
        "input_bytes": int(total_input_bytes),
        "compressed_bytes": int(total_output_bytes),
        "compression_ratio": float(total_output_bytes / total_input_bytes) if total_input_bytes > 0 else math.nan,
        "encode_s": float(total_encode_s),
        "decode_s": float(total_decode_s),
        "encode_mbps": float(encode_mbps),
        "decode_mbps": float(decode_mbps),
        "peak_rss_mb": float(peak_rss_mb),
        "bits": int(bits),
        "clamp_pct": float(clamp_pct),
        "zstd_level": int(zstd_level),
        "auto_offset": bool(auto_offset),
    }


def run_synthetic_scaling(
    total_rows_list: Sequence[int],
    *,
    n_vars: int = 16,
    window_rows: int = 10_000,
    bits: int = 8,
    clamp_pct: float = 500.0,
    zstd_level: int = 3,
    auto_offset: bool = True,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Run a streaming synthetic scaling benchmark across total sequence lengths.
    """
    results: List[Dict[str, Any]] = []

    for total_rows in total_rows_list:
        def _windows() -> Iterator[np.ndarray]:
            start = 0
            while start < int(total_rows):
                rows = min(int(window_rows), int(total_rows) - start)
                yield make_synthetic_sensor_window(start, rows, n_vars, seed=seed)
                start += rows

        res = benchmark_window_stream(
            _windows(),
            method="tracq_enhanced",
            bits=bits,
            clamp_pct=clamp_pct,
            zstd_level=zstd_level,
            auto_offset=auto_offset,
        )
        res["total_rows_target"] = int(total_rows)
        res["window_rows"] = int(window_rows)
        results.append(res)

    return results

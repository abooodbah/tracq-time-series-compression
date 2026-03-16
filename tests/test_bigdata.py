from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tracq.bigdata import (
    benchmark_window_stream,
    iter_numeric_csv_windows,
    load_numeric_csv_matrix,
    make_synthetic_sensor_window,
)


def test_iter_numeric_csv_windows_drops_time_and_label():
    csv_dir = Path("tests_artifacts")
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / "metropt_like.csv"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=6, freq="h").astype(str),
            "sensor_a": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "sensor_b": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "failure": [0, 0, 0, 1, 0, 0],
        }
    )
    try:
        df.to_csv(csv_path, index=False)

        windows = list(iter_numeric_csv_windows(csv_path, window_rows=4))
        assert len(windows) == 2
        assert windows[0].shape == (2, 4)
        assert windows[1].shape == (2, 2)
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_load_numeric_csv_matrix_drops_unnamed_and_transposes():
    csv_dir = Path("tests_artifacts")
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / "metropt_full_like.csv"
    df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="h").astype(str),
            "sensor_a": [1.0, 1.1, 1.2],
            "sensor_b": [2.0, 2.1, 2.2],
            "failure": [0, 0, 1],
        }
    )
    try:
        df.to_csv(csv_path, index=False)

        matrix, info = load_numeric_csv_matrix(csv_path)
        assert matrix.shape == (2, 3)
        assert info["n_rows"] == 3
        assert info["n_vars"] == 2
        assert np.allclose(matrix[0], [1.0, 1.1, 1.2])
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_make_synthetic_sensor_window_is_deterministic():
    a = make_synthetic_sensor_window(0, 128, 6, seed=7)
    b = make_synthetic_sensor_window(0, 128, 6, seed=7)
    assert a.shape == (6, 128)
    assert np.allclose(a, b)


def test_benchmark_window_stream_enhanced_small_window():
    window = make_synthetic_sensor_window(0, 256, 4, seed=0)
    result = benchmark_window_stream([window], method="tracq_enhanced", bits=8)

    assert result["n_windows"] == 1
    assert result["n_rows"] == 256
    assert result["n_vars"] == 4
    assert result["input_bytes"] > 0
    assert result["compressed_bytes"] > 0
    assert result["encode_s"] >= 0
    assert result["decode_s"] >= 0

"""
Test script to compare original vs enhanced TRACQ implementations.

Evaluates:
1. Reconstruction error (RMSE, MAE)
2. Compression ratio impact
3. Error drift over long sequences
4. Performance on different data characteristics
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import os
import shutil
from uuid import uuid4

from tracq.core import TimeSeriesGrid
from tracq.core_enhanced import EnhancedTimeSeriesGrid, compare_methods, rmse, mae
from tracq.codec import ImageCodec

TEST_RUNTIME_ROOT = REPO_ROOT / "test_runtime"
TEST_RUNTIME_ROOT.mkdir(exist_ok=True)


def generate_test_data(n_vars: int, n_time: int, data_type: str = "sensor") -> np.ndarray:
    """Generate test data with different characteristics."""
    np.random.seed(42)

    if data_type == "sensor":
        # Typical sensor data: smooth with occasional spikes
        data = np.zeros((n_vars, n_time))
        for i in range(n_vars):
            base = np.random.uniform(50, 200)
            noise = np.random.normal(0, base * 0.01, n_time)
            trend = np.linspace(0, np.random.uniform(-5, 5), n_time)
            spikes = np.random.choice([0, 1], n_time, p=[0.98, 0.02]) * np.random.normal(0, base * 0.1, n_time)
            data[i] = base + noise + trend + spikes

    elif data_type == "financial":
        # Financial returns: heavy-tailed, clustered volatility
        data = np.zeros((n_vars, n_time))
        for i in range(n_vars):
            base = 100
            returns = np.random.laplace(0, 0.01, n_time - 1)  # Heavy tails
            # Volatility clustering
            vol_regime = np.random.choice([0.5, 1.0, 2.0], n_time - 1, p=[0.7, 0.2, 0.1])
            returns *= vol_regime
            prices = base * np.cumprod(1 + returns)
            data[i] = np.concatenate([[base], prices])

    elif data_type == "iot":
        # IoT data: multiple scales, some flat periods, occasional dropouts
        data = np.zeros((n_vars, n_time))
        for i in range(n_vars):
            scale = 10 ** np.random.uniform(0, 3)  # Different scales
            base = np.random.uniform(0.1, 10) * scale
            noise = np.random.normal(0, base * 0.005, n_time)
            # Flat periods
            flat_mask = np.random.choice([True, False], n_time, p=[0.1, 0.9])
            noise[flat_mask] = 0
            data[i] = base + noise

    elif data_type == "electricity":
        # Electricity load: diurnal patterns, different scales across customers
        data = np.zeros((n_vars, n_time))
        for i in range(n_vars):
            base = np.random.uniform(100, 5000)
            diurnal = base * 0.3 * np.sin(2 * np.pi * np.arange(n_time) / 24)
            noise = np.random.normal(0, base * 0.05, n_time)
            data[i] = base + diurnal + noise

    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return data


def test_reconstruction_accuracy():
    """Test reconstruction accuracy across different configurations."""
    print("\n" + "=" * 80)
    print("TEST 1: Reconstruction Accuracy")
    print("=" * 80)

    for data_type in ["sensor", "financial", "iot", "electricity"]:
        print(f"\n--- Data type: {data_type} ---")
        data = generate_test_data(n_vars=50, n_time=1000, data_type=data_type)

        for bits in [8, 16]:
            print(f"\n  {bits}-bit quantization:")
            results = compare_methods(data, bits=bits, clamp_pct=500.0)

            for method, metrics in results.items():
                print(f"    {method:20s}: RMSE={metrics['rmse']:12.6f}  MAE={metrics['mae']:12.6f}")


def test_compression_ratio():
    """Test compression ratio with PNG encoding."""
    print("\n" + "=" * 80)
    print("TEST 2: Compression Ratio (with PNG encoding)")
    print("=" * 80)

    data = generate_test_data(n_vars=100, n_time=500, data_type="sensor")

    tmpdir = TEST_RUNTIME_ROOT / f"enhancements_{uuid4().hex[:8]}"
    tmpdir.mkdir(parents=True, exist_ok=False)
    try:
        # Original method
        orig_grid = TimeSeriesGrid(data, clamp_pct=500.0)
        q_orig, meta_orig = orig_grid.quantize_8bit()

        orig_png = os.path.join(tmpdir, "original.png")
        ImageCodec.save_png(orig_png, q_orig, meta_orig, compress_level=9)
        orig_size = os.path.getsize(orig_png)
        orig_recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q_orig, meta_orig)
        orig_rmse = rmse(data, orig_recon)

        # Enhanced with mu-law only (better compression compatibility)
        enh_grid1 = EnhancedTimeSeriesGrid(
            data, clamp_pct=500.0,
            adaptive_clamp=False, use_mu_law=True, reorder_variables=False
        )
        q_enh1, meta_enh1 = enh_grid1.quantize_8bit()

        enh1_png = os.path.join(tmpdir, "mulaw_only.png")
        ImageCodec.save_png(enh1_png, q_enh1, meta_enh1, compress_level=9)
        enh1_size = os.path.getsize(enh1_png)
        enh1_recon, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q_enh1, meta_enh1)
        enh1_rmse = rmse(data, enh1_recon)

        # Enhanced with adaptive clamping (best accuracy, larger files)
        enh_grid2 = EnhancedTimeSeriesGrid(
            data, clamp_pct=500.0,
            adaptive_clamp=True, use_mu_law=True, reorder_variables=False
        )
        q_enh2, meta_enh2 = enh_grid2.quantize_8bit()

        enh2_png = os.path.join(tmpdir, "adaptive.png")
        ImageCodec.save_png(enh2_png, q_enh2, meta_enh2, compress_level=9)
        enh2_size = os.path.getsize(enh2_png)
        enh2_recon, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q_enh2, meta_enh2)
        enh2_rmse = rmse(data, enh2_recon)

        csv_size = data.nbytes

        print(f"\n  {'Method':<25} {'PNG Size':>12} {'Ratio':>8} {'RMSE':>12}")
        print("  " + "-" * 60)
        print(f"  {'Raw data':<25} {csv_size:>12,} {'100.0%':>8}")
        print(f"  {'Original (uniform)':<25} {orig_size:>12,} {orig_size/csv_size*100:>7.1f}% {orig_rmse:>12.4f}")
        print(f"  {'Mu-law only':<25} {enh1_size:>12,} {enh1_size/csv_size*100:>7.1f}% {enh1_rmse:>12.4f}")
        print(f"  {'Adaptive + mu-law':<25} {enh2_size:>12,} {enh2_size/csv_size*100:>7.1f}% {enh2_rmse:>12.4f}")

        print(f"\n  Analysis:")
        print(f"    - Mu-law alone: {abs(orig_size - enh1_size)/orig_size*100:.1f}% size change, {(orig_rmse - enh1_rmse)/orig_rmse*100:.1f}% RMSE improvement")
        print(f"    - Adaptive:     {abs(orig_size - enh2_size)/orig_size*100:.1f}% size increase, {(orig_rmse - enh2_rmse)/orig_rmse*100:.1f}% RMSE improvement")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_error_drift():
    """Test error accumulation over long sequences."""
    print("\n" + "=" * 80)
    print("TEST 3: Error Drift Over Long Sequences")
    print("=" * 80)

    # Long sequence to observe drift
    n_time = 10000
    data = generate_test_data(n_vars=10, n_time=n_time, data_type="financial")

    # Test with and without anchors
    configs = [
        ("Original", TimeSeriesGrid, {"clamp_pct": 500.0}),
        ("Enhanced (no anchors)", EnhancedTimeSeriesGrid,
         {"clamp_pct": 500.0, "adaptive_clamp": True, "use_mu_law": True}),
        ("Enhanced (anchors@1000)", EnhancedTimeSeriesGrid,
         {"clamp_pct": 500.0, "adaptive_clamp": True, "use_mu_law": True, "anchor_interval": 1000}),
    ]

    print(f"\n  Sequence length: {n_time}")
    print(f"  {'Method':30s} {'RMSE@1000':>12} {'RMSE@5000':>12} {'RMSE@end':>12}")
    print("  " + "-" * 68)

    for name, GridClass, kwargs in configs:
        grid = GridClass(data, **kwargs)
        q, meta = grid.quantize_8bit()
        recon, _ = GridClass.reconstruct_from_quantized(q, meta)

        # Measure RMSE at different points
        rmse_1000 = rmse(data[:, :1000], recon[:, :1000])
        rmse_5000 = rmse(data[:, :5000], recon[:, :5000])
        rmse_end = rmse(data, recon)

        print(f"  {name:30s} {rmse_1000:12.4f} {rmse_5000:12.4f} {rmse_end:12.4f}")


def test_variable_scales():
    """Test handling of variables with different scales."""
    print("\n" + "=" * 80)
    print("TEST 4: Handling Different Variable Scales")
    print("=" * 80)

    # Create data with vastly different scales
    n_vars, n_time = 10, 500
    data = np.zeros((n_vars, n_time))
    scales = [1, 10, 100, 1000, 10000, 0.1, 0.01, 0.001, 500, 5000]

    for i, scale in enumerate(scales):
        base = scale
        data[i] = base + np.random.normal(0, base * 0.01, n_time)

    print(f"\n  Variable scales: {scales}")

    # Compare with global vs adaptive clamp
    configs = [
        ("Global clamp (original)", False, False),
        ("Adaptive clamp", True, False),
        ("Adaptive + mu-law", True, True),
    ]

    print(f"\n  {'Method':30s} {'RMSE':>12} {'Max Error':>12} {'Rel Error':>12}")
    print("  " + "-" * 68)

    for name, adaptive, mu_law in configs:
        grid = EnhancedTimeSeriesGrid(
            data, clamp_pct=500.0,
            adaptive_clamp=adaptive, use_mu_law=mu_law
        )
        q, meta = grid.quantize_8bit()
        recon, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, meta)

        err = rmse(data, recon)
        max_err = np.max(np.abs(data - recon))

        # Relative error (normalized by scale)
        rel_errors = np.abs(data - recon) / (np.abs(data) + 1e-10)
        rel_err = np.mean(rel_errors)

        print(f"  {name:30s} {err:12.6f} {max_err:12.6f} {rel_err:12.6f}")


def test_edge_cases():
    """Test edge cases: near-zero values, outliers, flat regions."""
    print("\n" + "=" * 80)
    print("TEST 5: Edge Cases")
    print("=" * 80)

    # Near-zero baseline
    print("\n  --- Near-zero baseline ---")
    data1 = np.random.normal(0.001, 0.0001, (5, 100))
    for name, grid_class, kwargs in [
        ("Original", TimeSeriesGrid, {}),
        ("Enhanced", EnhancedTimeSeriesGrid, {"adaptive_clamp": True, "use_mu_law": True}),
    ]:
        g = grid_class(data1, **kwargs)
        q, m = g.quantize_8bit()
        r, _ = grid_class.reconstruct_from_quantized(q, m)
        print(f"    {name:15s}: RMSE = {rmse(data1, r):.2e}")

    # Large outliers
    print("\n  --- Large outliers ---")
    data2 = np.random.normal(100, 1, (5, 100))
    data2[0, 50] = 10000  # Huge outlier
    for name, grid_class, kwargs in [
        ("Original", TimeSeriesGrid, {}),
        ("Enhanced", EnhancedTimeSeriesGrid, {"adaptive_clamp": True, "use_mu_law": True}),
    ]:
        g = grid_class(data2, **kwargs)
        q, m = g.quantize_8bit()
        r, _ = grid_class.reconstruct_from_quantized(q, m)
        print(f"    {name:15s}: RMSE = {rmse(data2, r):.2e}")

    # Flat regions (no change)
    print("\n  --- Flat regions ---")
    data3 = np.ones((5, 100)) * 100
    data3[:, 30:70] += np.random.normal(0, 0.01, (5, 40))  # Small variation in middle
    for name, grid_class, kwargs in [
        ("Original", TimeSeriesGrid, {}),
        ("Enhanced", EnhancedTimeSeriesGrid, {"adaptive_clamp": True, "use_mu_law": True}),
    ]:
        g = grid_class(data3, **kwargs)
        q, m = g.quantize_8bit()
        r, _ = grid_class.reconstruct_from_quantized(q, m)
        print(f"    {name:15s}: RMSE = {rmse(data3, r):.2e}")


def run_all_tests():
    """Run all tests."""
    test_reconstruction_accuracy()
    test_compression_ratio()
    test_error_drift()
    test_variable_scales()
    test_edge_cases()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()

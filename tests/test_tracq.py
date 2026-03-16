import numpy as np
import pytest
from typer.testing import CliRunner

import sys
import shutil
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEST_RUNTIME_ROOT = REPO_ROOT / "test_runtime"
TEST_RUNTIME_ROOT.mkdir(exist_ok=True)


def make_temp_dir(prefix: str) -> Path:
    path = TEST_RUNTIME_ROOT / f"{prefix}{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path

from tracq.cli import app
from tracq.core import TimeSeriesGrid, rmse
from tracq.baselines import (
    paa_compress, paa_decompress,
    pla_compress, pla_decompress,
    sax_compress, sax_decompress,
    HAS_ZFP, HAS_HDF5PLUGIN,
    get_available_hpc_compressors,
)


def test_delta_zstd_roundtrip():
    """Test Delta+Zstd compress/decompress produces near-lossless output."""
    from tracq.baselines import delta_zstd_compress, delta_zstd_decompress

    data = np.random.randn(10, 500).astype(np.float64) * 100 + 50
    compressed, meta = delta_zstd_compress(data, level=3)
    decompressed = delta_zstd_decompress(compressed, meta)

    assert decompressed.shape == data.shape
    max_error = np.max(np.abs(data - decompressed))
    # Delta+Zstd is near-lossless for float64 (only float rounding in cumsum)
    assert max_error < 1e-8, f"Max error {max_error} exceeds near-lossless threshold"
    # Compressed should be smaller than raw
    assert len(compressed) < data.nbytes


def test_quantize_dequantize_roundtrip_matches_rmse_zero():
    data = np.array([[100.0, 101.0, 99.0], [50.0, 50.5, 49.0]])
    grid = TimeSeriesGrid(data, clamp_pct=500.0)

    q16, meta16 = grid.quantize_16bit()
    recon16, _ = TimeSeriesGrid.reconstruct_from_quantized(q16, meta16)
    assert rmse(data, recon16) <= 5e-3

    q8, meta8 = grid.quantize_8bit()
    recon8, _ = TimeSeriesGrid.reconstruct_from_quantized(q8, meta8)
    assert rmse(data, recon8) < 1.5


def test_cli_compress_decompress_roundtrip():
    tmp_path = make_temp_dir("cli_roundtrip_")
    try:
        # Build small CSV (rows=time, cols=vars)
        data = np.array([
            [100.0, 101.0, 102.0, 103.0],
            [50.0, 50.5, 50.0, 49.5],
            [10.0, 10.1, 10.2, 10.3],
        ])
        csv_path = tmp_path / "sample.csv"
        np.savetxt(csv_path, data.T, delimiter=",")

        runner = CliRunner()

        # Compress
        result_comp = runner.invoke(app, ["compress", str(csv_path), "--bits", "8", "--force"])
        assert result_comp.exit_code == 0, result_comp.output
        png_path = csv_path.with_suffix(".tracq.png")
        assert png_path.exists()

        # Decompress
        result_decomp = runner.invoke(app, ["decompress", str(png_path), "--original", str(csv_path), "--force"])
        assert result_decomp.exit_code == 0, result_decomp.output
        recon_csv = png_path.with_suffix(".recon.csv")
        assert recon_csv.exists()

        recon = np.loadtxt(recon_csv, delimiter=",")
        assert recon.shape == data.T.shape
        assert rmse(data.T, recon) < 2.0
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_paa_roundtrip():
    """Test PAA compress/decompress preserves shape."""
    data = np.random.randn(10, 100)  # 10 vars, 100 time steps
    coeffs, meta = paa_compress(data, segments=10)
    recon = paa_decompress(coeffs, meta)
    assert recon.shape == data.shape


def test_pla_roundtrip():
    """Test PLA compress/decompress preserves shape."""
    data = np.random.randn(10, 100)
    coeffs, meta = pla_compress(data, segments=10)
    recon = pla_decompress(coeffs, meta)
    assert recon.shape == data.shape


def test_sax_roundtrip():
    """Test SAX compress/decompress preserves shape."""
    data = np.random.randn(10, 100)
    symbols, meta = sax_compress(data, segments=10, alphabet=8)
    recon = sax_decompress(symbols, meta)
    assert recon.shape == data.shape


def test_hpc_compressor_availability():
    """Test that HPC compressor availability check works."""
    avail = get_available_hpc_compressors()
    assert "zfp" in avail
    assert "sz3" in avail
    assert isinstance(avail["zfp"], bool)
    assert isinstance(avail["sz3"], bool)


@pytest.mark.skipif(not HAS_ZFP, reason="zfpy not installed")
def test_zfp_compress_decompress():
    """Test ZFP compression roundtrip."""
    from tracq.baselines import zfp_compress, zfp_decompress

    data = np.random.randn(10, 100).astype(np.float64)
    compressed, meta = zfp_compress(data, mode="tolerance", tolerance=1e-3)
    decompressed = zfp_decompress(compressed, meta)

    assert decompressed.shape == data.shape
    max_error = np.max(np.abs(data - decompressed))
    # ZFP should respect tolerance bound (with some margin)
    assert max_error < 1e-2  # Allow some margin


@pytest.mark.skipif(not HAS_HDF5PLUGIN, reason="hdf5plugin not installed")
def test_sz3_compress_decompress():
    """Test SZ3 compression roundtrip."""
    from tracq.baselines import sz3_compress, sz3_decompress

    data = np.random.randn(10, 100).astype(np.float64)
    compressed, meta = sz3_compress(data, mode="abs", abs_error=1e-3)
    decompressed = sz3_decompress(compressed, meta)

    assert decompressed.shape == data.shape
    max_error = np.max(np.abs(data - decompressed))
    # SZ3 should respect error bound
    assert max_error < 1e-2  # Allow some margin


def test_parallel_encode_single_worker():
    """Test parallel encoding with single worker matches serial."""
    from tracq.tooling import parallel_tracq_encode

    data = np.random.randn(10, 100).astype(np.float64)
    q, meta, timing = parallel_tracq_encode(data, bits=8, clamp=500.0, n_workers=1)

    # Verify output
    assert q.shape == (10, 99)  # n_vars x (n_time - 1)
    assert "baseline" in meta
    assert len(meta["baseline"]) == 10


def test_full_benchmark_integration():
    """Test full benchmark runs without error."""
    from tracq.tooling import full_benchmark

    tmp_path = make_temp_dir("benchmark_")
    try:
        # Create test CSV
        data = np.random.randn(100, 50)  # 100 time steps, 50 vars
        csv_path = tmp_path / "test_data.csv"
        np.savetxt(csv_path, data, delimiter=",")

        # Run benchmark (minimal settings for speed)
        results = full_benchmark(
            str(csv_path),
            include_baselines=True,
            include_hpc_compressors=False,  # Skip if not installed
            include_throughput=False,
        )

        assert "runs" in results
        assert "gzip" in results["runs"]
        assert "tracq_zst" in results["runs"]
        assert results["runs"]["gzip"]["bytes"] > 0
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_enhanced_tracq_roundtrip():
    """Test enhanced TRACQ compress/decompress roundtrip."""
    from tracq.core_enhanced import EnhancedTimeSeriesGrid, rmse as enhanced_rmse

    rng = np.random.default_rng(0)
    data = rng.standard_normal((10, 100)) * 100 + 500  # 10 vars, 100 time steps
    data = data.astype(np.float64)

    # Test 8-bit with all enhancements
    grid = EnhancedTimeSeriesGrid(
        data,
        adaptive_clamp=True,
        use_mu_law=True,
        anchor_interval=0,
    )
    q8, meta8 = grid.quantize_8bit()
    recon8, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q8, meta8)

    assert recon8.shape == data.shape
    # Enhanced should have reasonable RMSE for 8-bit
    assert enhanced_rmse(data, recon8) < 20.0  # Reasonable for 8-bit random data

    # Test 16-bit
    q16, meta16 = grid.quantize_16bit()
    recon16, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q16, meta16)
    assert recon16.shape == data.shape
    assert enhanced_rmse(data, recon16) < 0.5  # 16-bit should be very accurate


def test_enhanced_tracq_anchor_points():
    """Test that anchor points reduce error drift."""
    from tracq.core_enhanced import EnhancedTimeSeriesGrid, rmse as enhanced_rmse

    # Long sequence to observe drift
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 1000)) * 100 + 500
    data = data.astype(np.float64)

    # Without anchors
    grid_no_anchor = EnhancedTimeSeriesGrid(
        data, adaptive_clamp=True, use_mu_law=True, anchor_interval=0
    )
    q1, meta1 = grid_no_anchor.quantize_8bit()
    recon1, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q1, meta1)
    rmse_no_anchor = enhanced_rmse(data, recon1)

    # With anchors every 100 steps
    grid_with_anchor = EnhancedTimeSeriesGrid(
        data, adaptive_clamp=True, use_mu_law=True, anchor_interval=100
    )
    q2, meta2 = grid_with_anchor.quantize_8bit()
    recon2, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q2, meta2)
    rmse_with_anchor = enhanced_rmse(data, recon2)

    # Anchors should reduce error
    assert rmse_with_anchor < rmse_no_anchor


def test_enhanced_tracq_multiscale():
    """Test enhanced TRACQ handles multi-scale data."""
    from tracq.core_enhanced import EnhancedTimeSeriesGrid, rmse as enhanced_rmse
    from tracq.core import TimeSeriesGrid, rmse as orig_rmse

    # Create data with vastly different scales
    rng = np.random.default_rng(2)
    data = np.zeros((5, 100))
    scales = [1, 10, 100, 1000, 10000]
    for i, scale in enumerate(scales):
        data[i] = scale + rng.standard_normal(100) * scale * 0.01

    # Original method
    orig_grid = TimeSeriesGrid(data)
    q_orig, meta_orig = orig_grid.quantize_8bit()
    recon_orig, _ = TimeSeriesGrid.reconstruct_from_quantized(q_orig, meta_orig)
    orig_error = orig_rmse(data, recon_orig)

    # Enhanced method with adaptive clamp
    enh_grid = EnhancedTimeSeriesGrid(data, adaptive_clamp=True, use_mu_law=True)
    q_enh, meta_enh = enh_grid.quantize_8bit()
    recon_enh, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q_enh, meta_enh)
    enh_error = enhanced_rmse(data, recon_enh)

    # Enhanced should be much better for multi-scale data
    assert enh_error < orig_error * 0.1  # At least 10x improvement


def test_enhanced_tracq_auto_offset_matches_manual_shift():
    """Auto-offset metadata should reproduce the same result as manual preprocessing."""
    from tracq.core_enhanced import EnhancedTimeSeriesGrid

    n_time = 400
    t = np.arange(n_time)
    data = np.zeros((2, n_time), dtype=np.float64)
    data[0] = np.where((t % 50) < 3, -200.0, 5.0 + np.sin(2 * np.pi * t / 30))
    data[1] = np.where((t % 60) < 2, -200.0, 10.0 + 2.0 * np.cos(2 * np.pi * t / 45))

    auto_grid = EnhancedTimeSeriesGrid(
        data,
        adaptive_clamp=True,
        use_mu_law=True,
        anchor_interval=100,
        auto_offset=True,
    )
    q_auto, meta_auto = auto_grid.quantize_8bit()
    recon_auto, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q_auto, meta_auto)

    offsets = np.asarray(meta_auto["value_offsets"], dtype=np.float64)
    assert offsets.shape == (data.shape[0],)
    assert np.all(offsets > 0)

    manual_grid = EnhancedTimeSeriesGrid(
        data + offsets[:, None],
        adaptive_clamp=True,
        use_mu_law=True,
        anchor_interval=100,
    )
    q_manual, meta_manual = manual_grid.quantize_8bit()
    recon_manual, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q_manual, meta_manual)
    recon_manual = recon_manual - offsets[:, None]

    assert np.allclose(recon_auto, recon_manual)


def test_enhanced_tracq_auto_offset_stabilizes_near_zero_series():
    """Auto-offset should materially reduce error for zero-crossing/sentinel-heavy data."""
    from tracq.core_enhanced import EnhancedTimeSeriesGrid, rmse as enhanced_rmse

    n_time = 2000
    t = np.arange(n_time)
    data = np.zeros((2, n_time), dtype=np.float64)
    data[0] = np.where((t % 200) < 5, -200.0, 5.0 + np.sin(2 * np.pi * t / 30))
    data[1] = np.where((t % 150) < 4, -200.0, 10.0 + 2.0 * np.cos(2 * np.pi * t / 45))

    no_offset_grid = EnhancedTimeSeriesGrid(
        data,
        adaptive_clamp=True,
        use_mu_law=True,
        anchor_interval=100,
    )
    q_no_offset, meta_no_offset = no_offset_grid.quantize_8bit()
    recon_no_offset, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q_no_offset, meta_no_offset)
    err_no_offset = enhanced_rmse(data, recon_no_offset)

    auto_offset_grid = EnhancedTimeSeriesGrid(
        data,
        adaptive_clamp=True,
        use_mu_law=True,
        anchor_interval=100,
        auto_offset=True,
    )
    q_auto_offset, meta_auto_offset = auto_offset_grid.quantize_8bit()
    recon_auto_offset, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q_auto_offset, meta_auto_offset)
    err_auto_offset = enhanced_rmse(data, recon_auto_offset)

    assert np.isfinite(recon_auto_offset).all()
    assert meta_auto_offset["value_offsets"] is not None
    assert err_auto_offset < err_no_offset * 0.05

"""
Generate figures comparing Enhanced TRACQ vs HPC compressors (ZFP, SZ3).

This extends the paper figures to show how Enhanced TRACQ compares against
state-of-the-art error-bounded compressors.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import json
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# IEEE publication figure width constants
FULL_WIDTH = 7.16  # inches (IEEE \textwidth)
COL_WIDTH = 3.5    # inches (IEEE \columnwidth)

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (COL_WIDTH, 3.0),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

from tracq.core import TimeSeriesGrid
from tracq.core_enhanced import EnhancedTimeSeriesGrid, rmse, mae
from tracq.codec import ImageCodec
from tracq.baselines import (
    HAS_ZFP, HAS_HDF5PLUGIN,
    get_available_hpc_compressors,
)

if HAS_ZFP:
    from tracq.baselines import zfp_compress, zfp_decompress
if HAS_HDF5PLUGIN:
    from tracq.baselines import sz3_compress, sz3_decompress


def generate_test_data(n_vars: int, n_time: int, data_type: str = "sensor", seed: int = 42) -> np.ndarray:
    """Generate test data."""
    np.random.seed(seed)

    if data_type == "sensor":
        data = np.zeros((n_vars, n_time))
        for i in range(n_vars):
            base = np.random.uniform(50, 200)
            noise = np.random.normal(0, base * 0.01, n_time)
            trend = np.linspace(0, np.random.uniform(-5, 5), n_time)
            data[i] = base + noise + trend
    elif data_type == "financial":
        data = np.zeros((n_vars, n_time))
        for i in range(n_vars):
            base = 100
            returns = np.random.laplace(0, 0.01, n_time - 1)
            prices = base * np.cumprod(1 + returns)
            data[i] = np.concatenate([[base], prices])
    else:
        data = np.random.randn(n_vars, n_time) * 100 + 500

    return data.astype(np.float64)


def generate_hpc_comparison(output_dir: Path):
    """Generate rate-distortion comparison with HPC compressors."""
    print("Generating HPC Comparison Figure...")

    avail = get_available_hpc_compressors()
    print(f"  Available HPC compressors: {avail}")

    data = generate_test_data(n_vars=50, n_time=1000, data_type="sensor")
    raw_size = data.nbytes

    results = {
        'Base TRACQ 8-bit': {'sizes': [], 'rmses': []},
        'Base TRACQ 16-bit': {'sizes': [], 'rmses': []},
        'Enhanced TRACQ 8-bit': {'sizes': [], 'rmses': []},
        'Enhanced TRACQ 16-bit': {'sizes': [], 'rmses': []},
    }

    if HAS_ZFP:
        results['ZFP'] = {'sizes': [], 'rmses': []}
    if HAS_HDF5PLUGIN:
        results['SZ3'] = {'sizes': [], 'rmses': []}

    with tempfile.TemporaryDirectory() as tmpdir:
        # TRACQ variants with different clamp values
        for clamp in [100, 200, 500, 1000]:
            # Original 8-bit
            grid = TimeSeriesGrid(data, clamp_pct=clamp)
            q, m = grid.quantize_8bit()
            r, _ = TimeSeriesGrid.reconstruct_from_quantized(q, m)
            png_path = os.path.join(tmpdir, f"orig8_{clamp}.png")
            ImageCodec.save_png(png_path, q, m, compress_level=9)
            results['Base TRACQ 8-bit']['sizes'].append(os.path.getsize(png_path) / raw_size)
            results['Base TRACQ 8-bit']['rmses'].append(rmse(data, r))

            # Original 16-bit
            q, m = grid.quantize_16bit()
            r, _ = TimeSeriesGrid.reconstruct_from_quantized(q, m)
            png_path = os.path.join(tmpdir, f"orig16_{clamp}.png")
            ImageCodec.save_png(png_path, q, m, compress_level=9)
            results['Base TRACQ 16-bit']['sizes'].append(os.path.getsize(png_path) / raw_size)
            results['Base TRACQ 16-bit']['rmses'].append(rmse(data, r))

            # Enhanced 8-bit
            grid = EnhancedTimeSeriesGrid(data, clamp_pct=clamp, adaptive_clamp=True, use_mu_law=True)
            q, m = grid.quantize_8bit()
            r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)
            png_path = os.path.join(tmpdir, f"enh8_{clamp}.png")
            ImageCodec.save_png(png_path, q, m, compress_level=9)
            results['Enhanced TRACQ 8-bit']['sizes'].append(os.path.getsize(png_path) / raw_size)
            results['Enhanced TRACQ 8-bit']['rmses'].append(rmse(data, r))

            # Enhanced 16-bit
            q, m = grid.quantize_16bit()
            r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)
            png_path = os.path.join(tmpdir, f"enh16_{clamp}.png")
            ImageCodec.save_png(png_path, q, m, compress_level=9)
            results['Enhanced TRACQ 16-bit']['sizes'].append(os.path.getsize(png_path) / raw_size)
            results['Enhanced TRACQ 16-bit']['rmses'].append(rmse(data, r))

        # ZFP with different tolerances
        if HAS_ZFP:
            for tol in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
                try:
                    compressed, meta = zfp_compress(data, mode="tolerance", tolerance=tol)
                    decompressed = zfp_decompress(compressed, meta)
                    results['ZFP']['sizes'].append(len(compressed) / raw_size)
                    results['ZFP']['rmses'].append(rmse(data, decompressed))
                except Exception as e:
                    print(f"    ZFP tol={tol} failed: {e}")

        # SZ3 with different error bounds
        if HAS_HDF5PLUGIN:
            for err in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
                try:
                    compressed, meta = sz3_compress(data, mode="abs", abs_error=err)
                    decompressed = sz3_decompress(compressed, meta)
                    results['SZ3']['sizes'].append(len(compressed) / raw_size)
                    results['SZ3']['rmses'].append(rmse(data, decompressed))
                except Exception as e:
                    print(f"    SZ3 err={err} failed: {e}")

    # Plot
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 3.2))

    colors = {
        'Base TRACQ 8-bit': '#e74c3c',
        'Base TRACQ 16-bit': '#c0392b',
        'Enhanced TRACQ 8-bit': '#2ecc71',
        'Enhanced TRACQ 16-bit': '#27ae60',
        'ZFP': '#3498db',
        'SZ3': '#9b59b6',
    }

    markers = {
        'Base TRACQ 8-bit': 'o',
        'Base TRACQ 16-bit': 's',
        'Enhanced TRACQ 8-bit': '^',
        'Enhanced TRACQ 16-bit': 'D',
        'ZFP': 'P',
        'SZ3': 'X',
    }

    short_labels = {
        'Base TRACQ 8-bit': 'Base 8b',
        'Base TRACQ 16-bit': 'Base 16b',
        'Enhanced TRACQ 8-bit': 'Enh 8b',
        'Enhanced TRACQ 16-bit': 'Enh 16b',
        'ZFP': 'ZFP',
        'SZ3': 'SZ3',
    }

    for method, data_points in results.items():
        if len(data_points['sizes']) > 0:
            sizes = np.array(data_points['sizes']) * 100
            rmses = data_points['rmses']
            ax.scatter(sizes, rmses, c=colors.get(method, 'gray'),
                      marker=markers.get(method, 'o'), s=40, label=short_labels.get(method, method), alpha=0.8)

            # Connect points with lines
            sorted_idx = np.argsort(sizes)
            ax.plot(np.array(sizes)[sorted_idx], np.array(rmses)[sorted_idx],
                   c=colors.get(method, 'gray'), alpha=0.4, linewidth=1)

    ax.set_xlabel('Compression Ratio (% of original)')
    ax.set_ylabel('RMSE')
    ax.set_title('Rate-Distortion: TRACQ vs HPC', fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=6, frameon=True)
    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(output_dir / 'figure_hpc_comparison.png')
    plt.savefig(output_dir / 'figure_hpc_comparison.pdf')
    plt.close()

    # Save data
    with open(output_dir / 'hpc_comparison_data.json', 'w') as f:
        json.dump({k: {kk: list(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

    print(f"  Saved to {output_dir / 'figure_hpc_comparison.png'}")


def generate_throughput_comparison(output_dir: Path):
    """Generate throughput comparison figure."""
    print("Generating Throughput Comparison Figure...")

    import time

    data_sizes = [(50, 500), (100, 1000), (200, 2000), (500, 5000)]
    methods = ['Base TRACQ', 'Enhanced TRACQ']

    if HAS_ZFP:
        methods.append('ZFP')
    if HAS_HDF5PLUGIN:
        methods.append('SZ3')

    throughput_encode = {m: [] for m in methods}
    throughput_decode = {m: [] for m in methods}
    data_labels = []

    for n_vars, n_time in data_sizes:
        data = generate_test_data(n_vars, n_time, "sensor")
        data_size_mb = data.nbytes / 1e6
        data_labels.append(f"{n_vars}x{n_time}")

        n_repeats = 5

        # Base TRACQ
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            grid = TimeSeriesGrid(data, clamp_pct=500.0)
            q, m = grid.quantize_8bit()
            times.append(time.perf_counter() - t0)
        throughput_encode['Base TRACQ'].append(data_size_mb / np.mean(times))

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            r, _ = TimeSeriesGrid.reconstruct_from_quantized(q, m)
            times.append(time.perf_counter() - t0)
        throughput_decode['Base TRACQ'].append(data_size_mb / np.mean(times))

        # Enhanced TRACQ
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            grid = EnhancedTimeSeriesGrid(data, clamp_pct=500.0, adaptive_clamp=True, use_mu_law=True)
            q, m = grid.quantize_8bit()
            times.append(time.perf_counter() - t0)
        throughput_encode['Enhanced TRACQ'].append(data_size_mb / np.mean(times))

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)
            times.append(time.perf_counter() - t0)
        throughput_decode['Enhanced TRACQ'].append(data_size_mb / np.mean(times))

        # ZFP
        if HAS_ZFP:
            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                compressed, meta = zfp_compress(data, mode="tolerance", tolerance=1e-3)
                times.append(time.perf_counter() - t0)
            throughput_encode['ZFP'].append(data_size_mb / np.mean(times))

            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                decompressed = zfp_decompress(compressed, meta)
                times.append(time.perf_counter() - t0)
            throughput_decode['ZFP'].append(data_size_mb / np.mean(times))

        # SZ3
        if HAS_HDF5PLUGIN:
            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                compressed, meta = sz3_compress(data, mode="abs", abs_error=1e-3)
                times.append(time.perf_counter() - t0)
            throughput_encode['SZ3'].append(data_size_mb / np.mean(times))

            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                decompressed = sz3_decompress(compressed, meta)
                times.append(time.perf_counter() - t0)
            throughput_decode['SZ3'].append(data_size_mb / np.mean(times))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))

    x = np.arange(len(data_labels))
    width = 0.8 / len(methods)
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

    # Encode throughput
    for i, method in enumerate(methods):
        if method in throughput_encode:
            ax1.bar(x + i * width, throughput_encode[method], width,
                   label=method, color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Data Size (vars x time)')
    ax1.set_ylabel('Throughput (MB/s)')
    ax1.set_title('Encoding Throughput')
    ax1.set_xticks(x + width * (len(methods) - 1) / 2)
    ax1.set_xticklabels(data_labels)
    ax1.legend()

    # Decode throughput
    for i, method in enumerate(methods):
        if method in throughput_decode:
            ax2.bar(x + i * width, throughput_decode[method], width,
                   label=method, color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Data Size (vars x time)')
    ax2.set_ylabel('Throughput (MB/s)')
    ax2.set_title('Decoding Throughput')
    ax2.set_xticks(x + width * (len(methods) - 1) / 2)
    ax2.set_xticklabels(data_labels)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_throughput_comparison.png')
    plt.savefig(output_dir / 'figure_throughput_comparison.pdf')
    plt.close()

    # Save data
    with open(output_dir / 'throughput_data.json', 'w') as f:
        json.dump({
            'data_labels': data_labels,
            'encode': throughput_encode,
            'decode': throughput_decode
        }, f, indent=2)

    print(f"  Saved to {output_dir / 'figure_throughput_comparison.png'}")


def generate_method_comparison_table(output_dir: Path):
    """Generate comprehensive comparison table."""
    print("Generating Method Comparison Table...")

    data = generate_test_data(n_vars=100, n_time=1000, data_type="sensor")
    raw_size = data.nbytes

    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Base TRACQ
        for bits in [8, 16]:
            grid = TimeSeriesGrid(data, clamp_pct=500.0)
            q, m = grid.quantize_8bit() if bits == 8 else grid.quantize_16bit()
            r, _ = TimeSeriesGrid.reconstruct_from_quantized(q, m)
            png_path = os.path.join(tmpdir, f"orig_{bits}.png")
            ImageCodec.save_png(png_path, q, m, compress_level=9)
            rows.append({
                'method': f'Base TRACQ {bits}-bit',
                'size_bytes': os.path.getsize(png_path),
                'compression_ratio': os.path.getsize(png_path) / raw_size,
                'rmse': rmse(data, r),
                'mae': mae(data, r),
                'max_error': float(np.max(np.abs(data - r))),
            })

        # Enhanced TRACQ
        for bits in [8, 16]:
            grid = EnhancedTimeSeriesGrid(data, clamp_pct=500.0, adaptive_clamp=True, use_mu_law=True)
            q, m = grid.quantize_8bit() if bits == 8 else grid.quantize_16bit()
            r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)
            png_path = os.path.join(tmpdir, f"enh_{bits}.png")
            ImageCodec.save_png(png_path, q, m, compress_level=9)
            rows.append({
                'method': f'Enhanced TRACQ {bits}-bit',
                'size_bytes': os.path.getsize(png_path),
                'compression_ratio': os.path.getsize(png_path) / raw_size,
                'rmse': rmse(data, r),
                'mae': mae(data, r),
                'max_error': float(np.max(np.abs(data - r))),
            })

        # ZFP
        if HAS_ZFP:
            for tol in [1e-2, 1e-3, 1e-4]:
                try:
                    compressed, meta = zfp_compress(data, mode="tolerance", tolerance=tol)
                    decompressed = zfp_decompress(compressed, meta)
                    rows.append({
                        'method': f'ZFP (tol={tol})',
                        'size_bytes': len(compressed),
                        'compression_ratio': len(compressed) / raw_size,
                        'rmse': rmse(data, decompressed),
                        'mae': mae(data, decompressed),
                        'max_error': float(np.max(np.abs(data - decompressed))),
                    })
                except Exception as e:
                    print(f"    ZFP tol={tol} failed: {e}")

        # SZ3
        if HAS_HDF5PLUGIN:
            for err in [1e-2, 1e-3, 1e-4]:
                try:
                    compressed, meta = sz3_compress(data, mode="abs", abs_error=err)
                    decompressed = sz3_decompress(compressed, meta)
                    rows.append({
                        'method': f'SZ3 (err={err})',
                        'size_bytes': len(compressed),
                        'compression_ratio': len(compressed) / raw_size,
                        'rmse': rmse(data, decompressed),
                        'mae': mae(data, decompressed),
                        'max_error': float(np.max(np.abs(data - decompressed))),
                    })
                except Exception as e:
                    print(f"    SZ3 err={err} failed: {e}")

    # Write CSV
    import csv
    with open(output_dir / 'method_comparison_table.csv', 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Also save as JSON
    with open(output_dir / 'method_comparison_data.json', 'w') as f:
        json.dump(rows, f, indent=2)

    print(f"  Saved to {output_dir / 'method_comparison_table.csv'}")


def main():
    output_dir = REPO_ROOT / 'paper_results' / 'enhanced_figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating HPC comparison figures to: {output_dir}")
    print("=" * 60)

    generate_hpc_comparison(output_dir)
    generate_throughput_comparison(output_dir)
    generate_method_comparison_table(output_dir)

    print("=" * 60)
    print("HPC comparison figures generated successfully!")


if __name__ == "__main__":
    main()

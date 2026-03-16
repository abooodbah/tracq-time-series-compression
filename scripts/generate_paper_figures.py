"""
Generate paper figures comparing Original vs Enhanced TRACQ.

This script generates publication-quality figures demonstrating:
1. Reconstruction accuracy improvement across data types
2. Rate-distortion curves (compression vs accuracy tradeoff)
3. Error drift over long sequences
4. Multi-scale data handling
5. Ablation study of enhancement features
6. Compression ratio comparison
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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter

# Set publication-quality defaults
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


# =============================================================================
# Data Generation
# =============================================================================

def generate_test_data(n_vars: int, n_time: int, data_type: str = "sensor", seed: int = 42) -> np.ndarray:
    """Generate test data with different characteristics."""
    np.random.seed(seed)

    if data_type == "sensor":
        data = np.zeros((n_vars, n_time))
        for i in range(n_vars):
            base = np.random.uniform(50, 200)
            noise = np.random.normal(0, base * 0.01, n_time)
            trend = np.linspace(0, np.random.uniform(-5, 5), n_time)
            spikes = np.random.choice([0, 1], n_time, p=[0.98, 0.02]) * np.random.normal(0, base * 0.1, n_time)
            data[i] = base + noise + trend + spikes

    elif data_type == "financial":
        data = np.zeros((n_vars, n_time))
        for i in range(n_vars):
            base = 100
            returns = np.random.laplace(0, 0.01, n_time - 1)
            vol_regime = np.random.choice([0.5, 1.0, 2.0], n_time - 1, p=[0.7, 0.2, 0.1])
            returns *= vol_regime
            prices = base * np.cumprod(1 + returns)
            data[i] = np.concatenate([[base], prices])

    elif data_type == "iot":
        data = np.zeros((n_vars, n_time))
        for i in range(n_vars):
            scale = 10 ** np.random.uniform(0, 3)
            base = np.random.uniform(0.1, 10) * scale
            noise = np.random.normal(0, base * 0.005, n_time)
            flat_mask = np.random.choice([True, False], n_time, p=[0.1, 0.9])
            noise[flat_mask] = 0
            data[i] = base + noise

    elif data_type == "electricity":
        data = np.zeros((n_vars, n_time))
        for i in range(n_vars):
            base = np.random.uniform(100, 5000)
            diurnal = base * 0.3 * np.sin(2 * np.pi * np.arange(n_time) / 24)
            noise = np.random.normal(0, base * 0.05, n_time)
            data[i] = base + diurnal + noise

    elif data_type == "multiscale":
        data = np.zeros((n_vars, n_time))
        scales = np.logspace(-2, 4, n_vars)  # 0.01 to 10000
        for i, scale in enumerate(scales):
            data[i] = scale + np.random.normal(0, scale * 0.01, n_time)

    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return data


# =============================================================================
# Figure 1: Reconstruction Accuracy Comparison
# =============================================================================

def generate_accuracy_comparison(output_dir: Path):
    """Generate bar chart comparing RMSE across data types and methods."""
    print("Generating Figure 1: Accuracy Comparison...")

    data_types = ["sensor", "financial", "iot", "electricity"]
    methods = ["Original", "Adaptive Clamp", "Mu-Law", "Mu-Law + Adaptive"]

    results_8bit = {dt: [] for dt in data_types}
    results_16bit = {dt: [] for dt in data_types}

    for dt in data_types:
        data = generate_test_data(n_vars=50, n_time=1000, data_type=dt)

        for bits in [8, 16]:
            # Original
            orig = TimeSeriesGrid(data, clamp_pct=500.0)
            q, m = orig.quantize_8bit() if bits == 8 else orig.quantize_16bit()
            r, _ = TimeSeriesGrid.reconstruct_from_quantized(q, m)
            orig_rmse = rmse(data, r)

            # Adaptive clamp only
            enh1 = EnhancedTimeSeriesGrid(data, clamp_pct=500.0, adaptive_clamp=True, use_mu_law=False)
            q, m = enh1.quantize_8bit() if bits == 8 else enh1.quantize_16bit()
            r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)
            adapt_rmse = rmse(data, r)

            # Mu-law only
            enh2 = EnhancedTimeSeriesGrid(data, clamp_pct=500.0, adaptive_clamp=False, use_mu_law=True)
            q, m = enh2.quantize_8bit() if bits == 8 else enh2.quantize_16bit()
            r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)
            mulaw_rmse = rmse(data, r)

            # Both
            enh3 = EnhancedTimeSeriesGrid(data, clamp_pct=500.0, adaptive_clamp=True, use_mu_law=True)
            q, m = enh3.quantize_8bit() if bits == 8 else enh3.quantize_16bit()
            r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)
            both_rmse = rmse(data, r)

            if bits == 8:
                results_8bit[dt] = [orig_rmse, adapt_rmse, mulaw_rmse, both_rmse]
            else:
                results_16bit[dt] = [orig_rmse, adapt_rmse, mulaw_rmse, both_rmse]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))

    x = np.arange(len(data_types))
    width = 0.2
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    # 8-bit subplot
    for i, method in enumerate(methods):
        values = [results_8bit[dt][i] for dt in data_types]
        ax1.bar(x + i * width, values, width, label=method, color=colors[i], edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Data Type')
    ax1.set_ylabel('RMSE')
    ax1.set_title('8-bit Quantization')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([dt.capitalize() for dt in data_types])
    ax1.legend(loc='upper left')
    ax1.set_yscale('log')

    # 16-bit subplot
    for i, method in enumerate(methods):
        values = [results_16bit[dt][i] for dt in data_types]
        ax2.bar(x + i * width, values, width, label=method, color=colors[i], edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Data Type')
    ax2.set_ylabel('RMSE')
    ax2.set_title('16-bit Quantization')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([dt.capitalize() for dt in data_types])
    ax2.legend(loc='upper left')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_accuracy_comparison.png')
    plt.savefig(output_dir / 'figure1_accuracy_comparison.pdf')
    plt.close()

    # Save data as JSON
    with open(output_dir / 'figure1_data.json', 'w') as f:
        json.dump({'8bit': results_8bit, '16bit': results_16bit}, f, indent=2)

    print(f"  Saved to {output_dir / 'figure1_accuracy_comparison.png'}")


# =============================================================================
# Figure 2: Rate-Distortion Curves
# =============================================================================

def generate_rate_distortion_curves(output_dir: Path):
    """Generate rate-distortion curves comparing methods."""
    print("Generating Figure 2: Rate-Distortion Curves...")

    data = generate_test_data(n_vars=100, n_time=500, data_type="sensor")
    raw_size = data.nbytes

    results = {
        'original': {'sizes': [], 'rmses': [], 'configs': []},
        'mulaw_only': {'sizes': [], 'rmses': [], 'configs': []},
        'adaptive_only': {'sizes': [], 'rmses': [], 'configs': []},
        'enhanced': {'sizes': [], 'rmses': [], 'configs': []},
    }

    clamp_values = [100, 200, 500, 1000]

    with tempfile.TemporaryDirectory() as tmpdir:
        for bits in [8, 16]:
            for clamp in clamp_values:
                for zstd_level in [1, 6, 12]:
                    # Original
                    grid = TimeSeriesGrid(data, clamp_pct=clamp)
                    q, m = grid.quantize_8bit() if bits == 8 else grid.quantize_16bit()
                    r, _ = TimeSeriesGrid.reconstruct_from_quantized(q, m)

                    png_path = os.path.join(tmpdir, f"orig_{bits}_{clamp}_{zstd_level}.png")
                    ImageCodec.save_png(png_path, q, m, compress_level=9)
                    size = os.path.getsize(png_path)

                    results['original']['sizes'].append(size / raw_size)
                    results['original']['rmses'].append(rmse(data, r))
                    results['original']['configs'].append(f"bits={bits}, clamp={clamp}")

                    # Mu-law only
                    grid = EnhancedTimeSeriesGrid(data, clamp_pct=clamp, adaptive_clamp=False, use_mu_law=True)
                    q, m = grid.quantize_8bit() if bits == 8 else grid.quantize_16bit()
                    r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)

                    png_path = os.path.join(tmpdir, f"mulaw_{bits}_{clamp}_{zstd_level}.png")
                    ImageCodec.save_png(png_path, q, m, compress_level=9)
                    size = os.path.getsize(png_path)

                    results['mulaw_only']['sizes'].append(size / raw_size)
                    results['mulaw_only']['rmses'].append(rmse(data, r))
                    results['mulaw_only']['configs'].append(f"bits={bits}, clamp={clamp}")

                    # Adaptive only
                    grid = EnhancedTimeSeriesGrid(data, clamp_pct=clamp, adaptive_clamp=True, use_mu_law=False)
                    q, m = grid.quantize_8bit() if bits == 8 else grid.quantize_16bit()
                    r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)

                    png_path = os.path.join(tmpdir, f"adapt_{bits}_{clamp}_{zstd_level}.png")
                    ImageCodec.save_png(png_path, q, m, compress_level=9)
                    size = os.path.getsize(png_path)

                    results['adaptive_only']['sizes'].append(size / raw_size)
                    results['adaptive_only']['rmses'].append(rmse(data, r))
                    results['adaptive_only']['configs'].append(f"bits={bits}, clamp={clamp}")

                    # Enhanced (both)
                    grid = EnhancedTimeSeriesGrid(data, clamp_pct=clamp, adaptive_clamp=True, use_mu_law=True)
                    q, m = grid.quantize_8bit() if bits == 8 else grid.quantize_16bit()
                    r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)

                    png_path = os.path.join(tmpdir, f"enh_{bits}_{clamp}_{zstd_level}.png")
                    ImageCodec.save_png(png_path, q, m, compress_level=9)
                    size = os.path.getsize(png_path)

                    results['enhanced']['sizes'].append(size / raw_size)
                    results['enhanced']['rmses'].append(rmse(data, r))
                    results['enhanced']['configs'].append(f"bits={bits}, clamp={clamp}")

    # Plot
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 3.0))

    colors = {'original': '#e74c3c', 'mulaw_only': '#3498db', 'adaptive_only': '#2ecc71', 'enhanced': '#9b59b6'}
    labels = {'original': 'Base TRACQ', 'mulaw_only': 'Mu-Law Only', 'adaptive_only': 'Adaptive Clamp Only', 'enhanced': 'Enhanced (Both)'}
    markers = {'original': 'o', 'mulaw_only': 's', 'adaptive_only': '^', 'enhanced': 'D'}

    for method in results:
        sizes = np.array(results[method]['sizes']) * 100  # Convert to percentage
        rmses = results[method]['rmses']
        ax.scatter(sizes, rmses, c=colors[method], label=labels[method], marker=markers[method], s=60, alpha=0.7)

    ax.set_xlabel('Compression Ratio (% of original)')
    ax.set_ylabel('RMSE')
    ax.set_title('Rate-Distortion: Compression Ratio vs Reconstruction Error')
    ax.legend(loc='upper right')
    ax.set_yscale('log')

    # Add Pareto frontier for enhanced
    sizes = np.array(results['enhanced']['sizes']) * 100
    rmses = np.array(results['enhanced']['rmses'])

    # Sort by size and compute Pareto frontier
    sorted_idx = np.argsort(sizes)
    pareto_sizes = [sizes[sorted_idx[0]]]
    pareto_rmses = [rmses[sorted_idx[0]]]
    min_rmse = rmses[sorted_idx[0]]

    for i in sorted_idx[1:]:
        if rmses[i] < min_rmse:
            pareto_sizes.append(sizes[i])
            pareto_rmses.append(rmses[i])
            min_rmse = rmses[i]

    ax.plot(pareto_sizes, pareto_rmses, 'k--', alpha=0.5, linewidth=2, label='Pareto Frontier')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_rate_distortion.png')
    plt.savefig(output_dir / 'figure2_rate_distortion.pdf')
    plt.close()

    # Save data
    with open(output_dir / 'figure2_data.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir / 'figure2_rate_distortion.png'}")


# =============================================================================
# Figure 3: Error Drift Over Long Sequences
# =============================================================================

def generate_error_drift_figure(output_dir: Path):
    """Generate figure showing error accumulation over time."""
    print("Generating Figure 3: Error Drift Analysis...")

    n_time = 10000
    data = generate_test_data(n_vars=10, n_time=n_time, data_type="financial")

    configs = [
        ("Base TRACQ", TimeSeriesGrid, {"clamp_pct": 500.0}, None),
        ("Enhanced (no anchors)", EnhancedTimeSeriesGrid,
         {"clamp_pct": 500.0, "adaptive_clamp": True, "use_mu_law": True, "anchor_interval": 0}, None),
        ("Enhanced (anchors@500)", EnhancedTimeSeriesGrid,
         {"clamp_pct": 500.0, "adaptive_clamp": True, "use_mu_law": True, "anchor_interval": 500}, None),
        ("Enhanced (anchors@1000)", EnhancedTimeSeriesGrid,
         {"clamp_pct": 500.0, "adaptive_clamp": True, "use_mu_law": True, "anchor_interval": 1000}, None),
    ]

    # Compute RMSE at each time step
    time_points = np.arange(100, n_time + 1, 100)
    rmse_over_time = {}

    for name, GridClass, kwargs, _ in configs:
        grid = GridClass(data, **kwargs)
        q, meta = grid.quantize_8bit()
        recon, _ = GridClass.reconstruct_from_quantized(q, meta)

        rmses = []
        for t in time_points:
            rmses.append(rmse(data[:, :t], recon[:, :t]))
        rmse_over_time[name] = rmses

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    # Linear scale
    for i, (name, rmses) in enumerate(rmse_over_time.items()):
        ax1.plot(time_points, rmses, label=name, color=colors[i], linewidth=1.5)

    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Cumulative RMSE')
    ax1.set_title('Linear Scale')
    ax1.legend(loc='upper left', fontsize=7)

    # Log scale
    for i, (name, rmses) in enumerate(rmse_over_time.items()):
        ax2.plot(time_points, rmses, label=name, color=colors[i], linewidth=1.5)

    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Cumulative RMSE')
    ax2.set_title('Log Scale')
    ax2.set_yscale('log')
    ax2.legend(loc='upper left', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_error_drift.png')
    plt.savefig(output_dir / 'figure3_error_drift.pdf')
    plt.close()

    # Save data
    with open(output_dir / 'figure3_data.json', 'w') as f:
        json.dump({
            'time_points': time_points.tolist(),
            'rmse_over_time': {k: list(v) for k, v in rmse_over_time.items()}
        }, f, indent=2)

    print(f"  Saved to {output_dir / 'figure3_error_drift.png'}")


# =============================================================================
# Figure 4: Multi-Scale Data Handling
# =============================================================================

def generate_multiscale_figure(output_dir: Path):
    """Generate figure showing handling of multi-scale data."""
    print("Generating Figure 4: Multi-Scale Data Handling...")

    n_vars, n_time = 20, 500
    scales = np.logspace(-2, 4, n_vars)  # 0.01 to 10000

    data = np.zeros((n_vars, n_time))
    for i, scale in enumerate(scales):
        data[i] = scale + np.random.normal(0, scale * 0.01, n_time)

    methods = [
        ("Base TRACQ", TimeSeriesGrid, {"clamp_pct": 500.0}),
        ("Enhanced (Adaptive)", EnhancedTimeSeriesGrid, {"clamp_pct": 500.0, "adaptive_clamp": True, "use_mu_law": False}),
        ("Enhanced (Mu-Law)", EnhancedTimeSeriesGrid, {"clamp_pct": 500.0, "adaptive_clamp": False, "use_mu_law": True}),
        ("Enhanced (Both)", EnhancedTimeSeriesGrid, {"clamp_pct": 500.0, "adaptive_clamp": True, "use_mu_law": True}),
    ]

    # Compute per-variable relative RMSE
    per_var_rel_rmse = {}

    for name, GridClass, kwargs in methods:
        grid = GridClass(data, **kwargs)
        q, meta = grid.quantize_8bit()
        recon, _ = GridClass.reconstruct_from_quantized(q, meta)

        rel_rmse = []
        for i in range(n_vars):
            var_rmse = rmse(data[i:i+1], recon[i:i+1])
            rel_rmse.append(var_rmse / (np.mean(np.abs(data[i])) + 1e-10))
        per_var_rel_rmse[name] = rel_rmse

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    # Relative RMSE vs Scale
    for i, (name, rel_rmse) in enumerate(per_var_rel_rmse.items()):
        ax1.scatter(scales, rel_rmse, label=name, color=colors[i], s=20, alpha=0.7)
        # Connect with lines
        ax1.plot(scales, rel_rmse, color=colors[i], alpha=0.3, linewidth=1)

    ax1.set_xlabel('Variable Scale')
    ax1.set_ylabel('Relative RMSE')
    ax1.set_title('Relative Error vs Scale')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=6)

    # Box plot comparison
    box_data = [list(v) for v in per_var_rel_rmse.values()]
    box_labels = list(per_var_rel_rmse.keys())
    short_labels = ['Original', 'Adaptive', 'Mu-Law', 'Both']

    bp = ax2.boxplot(box_data, tick_labels=short_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Relative RMSE')
    ax2.set_title('Distribution of Relative RMSE')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_multiscale.png')
    plt.savefig(output_dir / 'figure4_multiscale.pdf')
    plt.close()

    # Save data
    with open(output_dir / 'figure4_data.json', 'w') as f:
        json.dump({
            'scales': scales.tolist(),
            'per_var_rel_rmse': {k: list(v) for k, v in per_var_rel_rmse.items()}
        }, f, indent=2)

    print(f"  Saved to {output_dir / 'figure4_multiscale.png'}")


# =============================================================================
# Figure 5: Ablation Study
# =============================================================================

def generate_ablation_study(output_dir: Path):
    """Generate ablation study figure."""
    print("Generating Figure 5: Ablation Study...")

    data_types = ["sensor", "financial", "iot", "electricity"]

    configs = [
        ("Baseline", {"adaptive_clamp": False, "use_mu_law": False}),
        ("+ Adaptive Clamp", {"adaptive_clamp": True, "use_mu_law": False}),
        ("+ Mu-Law", {"adaptive_clamp": False, "use_mu_law": True}),
        ("+ Both", {"adaptive_clamp": True, "use_mu_law": True}),
    ]

    results = {dt: [] for dt in data_types}

    for dt in data_types:
        data = generate_test_data(n_vars=50, n_time=1000, data_type=dt)

        for name, kwargs in configs:
            grid = EnhancedTimeSeriesGrid(data, clamp_pct=500.0, **kwargs)
            q, meta = grid.quantize_8bit()
            recon, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, meta)
            results[dt].append(rmse(data, recon))

    # Plot as heatmap
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.8))

    data_matrix = np.array([results[dt] for dt in data_types])

    im = ax.imshow(data_matrix, cmap='RdYlGn_r', aspect='auto')

    short_config_labels = ['Baseline', '+Adapt.', '+Mu-Law', '+Both']
    ax.set_xticks(np.arange(len(configs)))
    ax.set_yticks(np.arange(len(data_types)))
    ax.set_xticklabels(short_config_labels, fontsize=8)
    ax.set_yticklabels([dt.capitalize() for dt in data_types], fontsize=8)

    # Add text annotations
    for i in range(len(data_types)):
        for j in range(len(configs)):
            value = data_matrix[i, j]
            text_color = 'white' if value > np.median(data_matrix) else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center', color=text_color, fontsize=9)

    ax.set_title('Ablation: RMSE by Config. and Data Type', fontsize=9)
    plt.colorbar(im, ax=ax, label='RMSE', shrink=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_ablation_heatmap.png')
    plt.savefig(output_dir / 'figure5_ablation_heatmap.pdf')
    plt.close()

    # Also create bar chart
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 3.0))

    x = np.arange(len(data_types))
    width = 0.2
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for i, (name, _) in enumerate(configs):
        values = [results[dt][i] for dt in data_types]
        ax.bar(x + i * width, values, width, label=name, color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Data Type')
    ax.set_ylabel('RMSE')
    ax.set_title('Ablation Study: Impact of Each Enhancement')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([dt.capitalize() for dt in data_types])
    ax.legend(loc='upper left')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_ablation_bars.png')
    plt.savefig(output_dir / 'figure5_ablation_bars.pdf')
    plt.close()

    # Save data
    with open(output_dir / 'figure5_data.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir / 'figure5_ablation_heatmap.png'}")


# =============================================================================
# Figure 6: Compression vs Accuracy Tradeoff
# =============================================================================

def generate_compression_accuracy_tradeoff(output_dir: Path):
    """Generate figure showing compression ratio vs accuracy tradeoff."""
    print("Generating Figure 6: Compression vs Accuracy Tradeoff...")

    data = generate_test_data(n_vars=100, n_time=500, data_type="sensor")
    raw_size = data.nbytes

    methods = [
        ("Original", TimeSeriesGrid, {}),
        ("Mu-Law Only", EnhancedTimeSeriesGrid, {"adaptive_clamp": False, "use_mu_law": True}),
        ("Adaptive Only", EnhancedTimeSeriesGrid, {"adaptive_clamp": True, "use_mu_law": False}),
        ("Enhanced (Both)", EnhancedTimeSeriesGrid, {"adaptive_clamp": True, "use_mu_law": True}),
    ]

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, GridClass, kwargs in methods:
            grid = GridClass(data, clamp_pct=500.0, **kwargs)

            for bits in [8, 16]:
                q, meta = grid.quantize_8bit() if bits == 8 else grid.quantize_16bit()
                recon, _ = GridClass.reconstruct_from_quantized(q, meta)

                png_path = os.path.join(tmpdir, f"{name}_{bits}.png")
                ImageCodec.save_png(png_path, q, meta, compress_level=9)
                size = os.path.getsize(png_path)

                results.append({
                    'method': name,
                    'bits': bits,
                    'size': size,
                    'ratio': size / raw_size,
                    'rmse': rmse(data, recon),
                    'mae': mae(data, recon),
                })

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))

    colors = {'Original': '#e74c3c', 'Mu-Law Only': '#3498db', 'Adaptive Only': '#2ecc71', 'Enhanced (Both)': '#9b59b6'}
    markers = {8: 'o', 16: 's'}

    # Scatter plot
    for r in results:
        ax1.scatter(r['ratio'] * 100, r['rmse'],
                   c=colors[r['method']], marker=markers[r['bits']],
                   s=60, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Create legend
    method_handles = [mpatches.Patch(color=colors[m], label=m) for m in colors]
    bit_handles = [plt.Line2D([0], [0], marker=markers[b], color='gray', label=f'{b}-bit',
                              markersize=6, linestyle='None') for b in [8, 16]]

    ax1.legend(handles=method_handles + bit_handles, loc='upper right', ncol=2, fontsize=7)
    ax1.set_xlabel('Compression Ratio (% of original)')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Ratio vs Error')
    ax1.set_yscale('log')

    # Bar chart comparison
    methods_8bit = [r for r in results if r['bits'] == 8]
    x = np.arange(len(methods_8bit))
    width = 0.35

    ax2_twin = ax2.twinx()

    bars1 = ax2.bar(x - width/2, [r['ratio'] * 100 for r in methods_8bit], width,
                   label='Compression Ratio', color='#3498db', alpha=0.7)
    bars2 = ax2_twin.bar(x + width/2, [r['rmse'] for r in methods_8bit], width,
                        label='RMSE', color='#e74c3c', alpha=0.7)

    ax2.set_xlabel('Method')
    ax2.set_ylabel('Ratio (%)', color='#3498db')
    ax2_twin.set_ylabel('RMSE', color='#e74c3c')
    ax2.set_xticks(x)
    short_method_labels = ['Original', 'Mu-Law\nOnly', 'Adaptive\nOnly', 'Enhanced\n(Both)']
    ax2.set_xticklabels(short_method_labels, fontsize=7)
    ax2.set_title('8-bit: Compression vs Accuracy')

    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure6_compression_accuracy.png')
    plt.savefig(output_dir / 'figure6_compression_accuracy.pdf')
    plt.close()

    # Save data
    with open(output_dir / 'figure6_data.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir / 'figure6_compression_accuracy.png'}")


# =============================================================================
# Figure 7: Reconstruction Example
# =============================================================================

def generate_reconstruction_example(output_dir: Path):
    """Generate figure showing actual reconstruction vs original."""
    print("Generating Figure 7: Reconstruction Example...")

    # Generate data with clear patterns
    n_time = 500
    t = np.arange(n_time)

    # Create synthetic data with known patterns
    data = np.zeros((4, n_time))
    data[0] = 100 + 20 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 2, n_time)  # Sinusoidal
    data[1] = 50 + 0.1 * t + np.random.normal(0, 3, n_time)  # Linear trend
    data[2] = 200 + np.random.normal(0, 5, n_time)  # Noisy constant
    data[3] = 1000 * (1.001 ** t) + np.random.normal(0, 10, n_time)  # Exponential growth

    methods = [
        ("Base TRACQ", TimeSeriesGrid, {}),
        ("Enhanced TRACQ", EnhancedTimeSeriesGrid, {"adaptive_clamp": True, "use_mu_law": True}),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(FULL_WIDTH, 9.5), gridspec_kw={'hspace': 0.55, 'wspace': 0.35})
    pattern_names = ['Sinusoidal', 'Linear Trend', 'Noisy Constant', 'Exponential']

    colors = {'Base TRACQ': '#e74c3c', 'Enhanced TRACQ': '#2ecc71'}

    for i, pattern_name in enumerate(pattern_names):
        original = data[i]

        for j, (method_name, GridClass, kwargs) in enumerate(methods):
            ax = axes[i, j]

            grid = GridClass(data, clamp_pct=500.0, **kwargs)
            q, meta = grid.quantize_8bit()
            recon, _ = GridClass.reconstruct_from_quantized(q, meta)

            reconstructed = recon[i]
            error = rmse(original.reshape(1, -1), reconstructed.reshape(1, -1))

            ax.plot(t, original, 'b-', label='Original', alpha=0.7, linewidth=1)
            ax.plot(t, reconstructed, color=colors[method_name], linestyle='--',
                   label=f'Recon (RMSE={error:.2f})', alpha=0.8, linewidth=1)

            ax.set_title(f'{pattern_name} — {method_name}', fontsize=9)
            ax.legend(loc='upper right', fontsize=7)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure7_reconstruction_example.png')
    plt.savefig(output_dir / 'figure7_reconstruction_example.pdf')
    plt.close()

    print(f"  Saved to {output_dir / 'figure7_reconstruction_example.png'}")


# =============================================================================
# Figure 8: Improvement Summary
# =============================================================================

def generate_improvement_summary(output_dir: Path):
    """Generate summary figure showing improvement percentages."""
    print("Generating Figure 8: Improvement Summary...")

    data_types = ["sensor", "financial", "iot", "electricity"]

    improvements = {'8-bit': {}, '16-bit': {}}

    for dt in data_types:
        data = generate_test_data(n_vars=50, n_time=1000, data_type=dt)

        for bits, key in [(8, '8-bit'), (16, '16-bit')]:
            # Original
            orig = TimeSeriesGrid(data, clamp_pct=500.0)
            q, m = orig.quantize_8bit() if bits == 8 else orig.quantize_16bit()
            r, _ = TimeSeriesGrid.reconstruct_from_quantized(q, m)
            orig_rmse = rmse(data, r)

            # Enhanced
            enh = EnhancedTimeSeriesGrid(data, clamp_pct=500.0, adaptive_clamp=True, use_mu_law=True)
            q, m = enh.quantize_8bit() if bits == 8 else enh.quantize_16bit()
            r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)
            enh_rmse = rmse(data, r)

            improvement = (orig_rmse - enh_rmse) / orig_rmse * 100
            improvements[key][dt] = improvement

    # Plot
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.5))

    x = np.arange(len(data_types))
    width = 0.35

    bars1 = ax.bar(x - width/2, [improvements['8-bit'][dt] for dt in data_types],
                  width, label='8-bit', color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [improvements['16-bit'][dt] for dt in data_types],
                  width, label='16-bit', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Data Type')
    ax.set_ylabel('RMSE Improvement (%)')
    ax.set_title('Enhanced TRACQ: RMSE Improvement Over Original')
    ax.set_xticks(x)
    ax.set_xticklabels([dt.capitalize() for dt in data_types])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 108)

    # Add value labels on bars — offset alternating to avoid overlap
    for bars, y_offset in [(bars1, (0, 5)), (bars2, (0, -12))]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=y_offset,
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure8_improvement_summary.png')
    plt.savefig(output_dir / 'figure8_improvement_summary.pdf')
    plt.close()

    # Save data
    with open(output_dir / 'figure8_data.json', 'w') as f:
        json.dump(improvements, f, indent=2)

    print(f"  Saved to {output_dir / 'figure8_improvement_summary.png'}")


# =============================================================================
# Summary Table
# =============================================================================

def generate_summary_table(output_dir: Path):
    """Generate summary table as CSV."""
    print("Generating Summary Table...")

    data_types = ["sensor", "financial", "iot", "electricity", "multiscale"]

    rows = []

    for dt in data_types:
        data = generate_test_data(n_vars=50, n_time=1000, data_type=dt)

        for bits in [8, 16]:
            # Original
            orig = TimeSeriesGrid(data, clamp_pct=500.0)
            q, m = orig.quantize_8bit() if bits == 8 else orig.quantize_16bit()
            r, _ = TimeSeriesGrid.reconstruct_from_quantized(q, m)
            orig_rmse = rmse(data, r)
            orig_mae = mae(data, r)

            # Enhanced
            enh = EnhancedTimeSeriesGrid(data, clamp_pct=500.0, adaptive_clamp=True, use_mu_law=True)
            q, m = enh.quantize_8bit() if bits == 8 else enh.quantize_16bit()
            r, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q, m)
            enh_rmse = rmse(data, r)
            enh_mae = mae(data, r)

            improvement = (orig_rmse - enh_rmse) / orig_rmse * 100

            rows.append({
                'data_type': dt,
                'bits': bits,
                'original_rmse': orig_rmse,
                'original_mae': orig_mae,
                'enhanced_rmse': enh_rmse,
                'enhanced_mae': enh_mae,
                'rmse_improvement_pct': improvement,
            })

    # Write CSV
    import csv
    with open(output_dir / 'summary_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved to {output_dir / 'summary_table.csv'}")


# =============================================================================
# Main
# =============================================================================

def main():
    output_dir = REPO_ROOT / 'paper_results' / 'enhanced_figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures to: {output_dir}")
    print("=" * 60)

    # Generate all figures
    generate_accuracy_comparison(output_dir)
    generate_rate_distortion_curves(output_dir)
    generate_error_drift_figure(output_dir)
    generate_multiscale_figure(output_dir)
    generate_ablation_study(output_dir)
    generate_compression_accuracy_tradeoff(output_dir)
    generate_reconstruction_example(output_dir)
    generate_improvement_summary(output_dir)
    generate_summary_table(output_dir)

    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

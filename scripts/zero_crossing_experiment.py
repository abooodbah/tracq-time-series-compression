#!/usr/bin/env python
"""
Zero-Crossing Analysis for TRACQ Paper Revision (Reviewer 1).

Generates test signals that cross zero and compares behavior of:
  - TimeSeriesGrid (core.py): uses prev + eps in denominator
  - EnhancedTimeSeriesGrid (core_enhanced.py): uses max(|prev|, eps) in denominator

Test signals:
  1. Pure sine wave centered at 0: sin(2*pi*t/100)
  2. AC voltage simulation: 220*sin(2*pi*t/50)
  3. Noisy zero-centered random walk
  4. Linear trend crossing from positive to negative

Outputs to paper_results/zero_crossing/:
  - figure_zero_crossing.png / .pdf
  - zero_crossing_results.json
"""

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tracq.core import TimeSeriesGrid
from tracq.core_enhanced import EnhancedTimeSeriesGrid
from tracq.metrics import rmse as calc_rmse

DEFAULT_OUTPUT_DIR = Path(PROJECT_ROOT) / "paper_results" / "zero_crossing"


def generate_signals(n_time=1000):
    """Generate 4 zero-crossing test signals."""
    t = np.arange(n_time, dtype=np.float64)

    signals = {}

    # 1. Pure sine wave centered at 0
    signals["sine_wave"] = {
        "data": np.sin(2 * np.pi * t / 100).reshape(1, -1),
        "label": "Sine Wave (amplitude=1)",
    }

    # 2. AC voltage simulation
    signals["ac_voltage"] = {
        "data": (220.0 * np.sin(2 * np.pi * t / 50)).reshape(1, -1),
        "label": "AC Voltage (220V, 50Hz equiv.)",
    }

    # 3. Noisy zero-centered random walk
    np.random.seed(42)
    rw = np.cumsum(np.random.randn(n_time) * 0.5)
    rw -= rw.mean()  # center at zero
    signals["random_walk"] = {
        "data": rw.reshape(1, -1),
        "label": "Zero-Centered Random Walk",
    }

    # 4. Linear trend crossing zero
    linear = np.linspace(50, -50, n_time)
    signals["linear_cross"] = {
        "data": linear.reshape(1, -1),
        "label": "Linear Trend (+50 to -50)",
    }

    return signals


def run_experiment(signals):
    """Run both original and enhanced TRACQ on each signal."""
    results = {}

    for sig_name, sig_info in signals.items():
        data = sig_info["data"]
        label = sig_info["label"]
        print(f"  Processing: {label}")

        sig_results = {"label": label, "n_time": data.shape[1]}

        # --- Original TRACQ ---
        try:
            grid_orig = TimeSeriesGrid(data, clamp_pct=500.0)
            q8, meta8 = grid_orig.quantize_8bit()
            recon8, _ = TimeSeriesGrid.reconstruct_from_quantized(q8, meta8)

            orig_rmse = float(calc_rmse(data, recon8))
            orig_max_err = float(np.max(np.abs(data - recon8)))
            orig_errors = np.abs(data - recon8).ravel().tolist()

            sig_results["original_8bit"] = {
                "rmse": orig_rmse,
                "max_error": orig_max_err,
                "per_step_errors": orig_errors,
            }
        except Exception as e:
            sig_results["original_8bit"] = {"error": str(e)}

        # --- Enhanced TRACQ ---
        try:
            grid_enh = EnhancedTimeSeriesGrid(
                data, adaptive_clamp=True, use_mu_law=True, anchor_interval=0
            )
            eq8, emeta8 = grid_enh.quantize_8bit()
            erecon8, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(eq8, emeta8)

            enh_rmse = float(calc_rmse(data, erecon8))
            enh_max_err = float(np.max(np.abs(data - erecon8)))
            enh_errors = np.abs(data - erecon8).ravel().tolist()

            sig_results["enhanced_8bit"] = {
                "rmse": enh_rmse,
                "max_error": enh_max_err,
                "per_step_errors": enh_errors,
            }
        except Exception as e:
            sig_results["enhanced_8bit"] = {"error": str(e)}

        # --- Original 16-bit ---
        try:
            q16, meta16 = grid_orig.quantize_16bit()
            recon16, _ = TimeSeriesGrid.reconstruct_from_quantized(q16, meta16)
            sig_results["original_16bit"] = {
                "rmse": float(calc_rmse(data, recon16)),
                "max_error": float(np.max(np.abs(data - recon16))),
            }
        except Exception as e:
            sig_results["original_16bit"] = {"error": str(e)}

        # --- Enhanced 16-bit ---
        try:
            eq16, emeta16 = grid_enh.quantize_16bit()
            erecon16, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(eq16, emeta16)
            sig_results["enhanced_16bit"] = {
                "rmse": float(calc_rmse(data, erecon16)),
                "max_error": float(np.max(np.abs(data - erecon16))),
            }
        except Exception as e:
            sig_results["enhanced_16bit"] = {"error": str(e)}

        results[sig_name] = sig_results

    return results


def generate_figure(signals, results, output_dir):
    """Generate the zero-crossing figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # IEEE publication-quality defaults (full-width figure)
    FULL_WIDTH = 7.16  # inches (IEEE \textwidth)
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })

    fig, axes = plt.subplots(4, 3, figsize=(FULL_WIDTH, 7.0))
    fig.suptitle("Zero-Crossing Signal Behavior: Original vs Enhanced TRACQ", fontsize=11, fontweight="bold")

    sig_names = list(signals.keys())

    for row, sig_name in enumerate(sig_names):
        data = signals[sig_name]["data"].ravel()
        label = signals[sig_name]["label"]
        res = results[sig_name]
        n = len(data)
        t = np.arange(n)

        # Column 1: Original signal + reconstructions
        ax1 = axes[row, 0]
        ax1.plot(t, data, "b-", linewidth=0.8, alpha=0.7, label="Original")

        if "per_step_errors" in res.get("original_8bit", {}):
            orig_recon = data.copy()
            # We can approximate reconstruction from errors
            orig_err = np.array(res["original_8bit"]["per_step_errors"])
            ax1.set_title(f"{label}\nOriginal RMSE={res['original_8bit']['rmse']:.4f}")
        else:
            ax1.set_title(f"{label}\nOriginal: ERROR")

        if "per_step_errors" in res.get("enhanced_8bit", {}):
            enh_err = np.array(res["enhanced_8bit"]["per_step_errors"])
            ax1.set_title(f"{label}")

        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax1.set_ylabel("Value")
        if row == 0:
            ax1.set_title(f"Signal: {label}", fontsize=10)
        else:
            ax1.set_title(f"{label}", fontsize=10)

        # Column 2: Per-step absolute error comparison
        ax2 = axes[row, 1]
        if "per_step_errors" in res.get("original_8bit", {}):
            orig_errs = np.array(res["original_8bit"]["per_step_errors"])
            ax2.semilogy(t, np.maximum(orig_errs, 1e-15), "r-", linewidth=0.5, alpha=0.7, label=f"Original (RMSE={res['original_8bit']['rmse']:.4f})")
        if "per_step_errors" in res.get("enhanced_8bit", {}):
            enh_errs = np.array(res["enhanced_8bit"]["per_step_errors"])
            ax2.semilogy(t, np.maximum(enh_errs, 1e-15), "b-", linewidth=0.5, alpha=0.7, label=f"Enhanced (RMSE={res['enhanced_8bit']['rmse']:.4f})")
        ax2.set_ylabel("|Error|")
        ax2.legend(fontsize=7, loc="upper right")
        if row == 0:
            ax2.set_title("Per-Step Absolute Error (8-bit)", fontsize=10)

        # Column 3: Summary bars
        ax3 = axes[row, 2]
        methods = []
        rmses = []
        colors = []
        for method_key, color, method_label in [
            ("original_8bit", "#e74c3c", "Orig 8b"),
            ("enhanced_8bit", "#3498db", "Enh 8b"),
            ("original_16bit", "#e67e22", "Orig 16b"),
            ("enhanced_16bit", "#2ecc71", "Enh 16b"),
        ]:
            if method_key in res and "rmse" in res[method_key]:
                methods.append(method_label)
                rmses.append(res[method_key]["rmse"])
                colors.append(color)

        if methods:
            bars = ax3.bar(methods, rmses, color=colors, alpha=0.8)
            for bar, val in zip(bars, rmses):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax3.set_ylabel("RMSE")
        if row == 0:
            ax3.set_title("RMSE Comparison", fontsize=10)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time Step")

    plt.tight_layout()

    # Save
    png_path = os.path.join(output_dir, "figure_zero_crossing.png")
    pdf_path = os.path.join(output_dir, "figure_zero_crossing.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the zero-crossing TRACQ experiment")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args(argv)

    output_dir = str(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Generating zero-crossing test signals...")
    signals = generate_signals(n_time=1000)

    print("Running experiments...")
    results = run_experiment(signals)

    # Save results JSON (without per-step errors to keep file small)
    results_summary = {}
    for sig_name, sig_res in results.items():
        results_summary[sig_name] = {
            "label": sig_res["label"],
            "n_time": sig_res["n_time"],
        }
        for method in ["original_8bit", "enhanced_8bit", "original_16bit", "enhanced_16bit"]:
            if method in sig_res:
                entry = {k: v for k, v in sig_res[method].items() if k != "per_step_errors"}
                results_summary[sig_name][method] = entry

    json_path = os.path.join(output_dir, "zero_crossing_results.json")
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    print("Generating figure...")
    generate_figure(signals, results, output_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print("Zero-Crossing Experiment Summary")
    print("=" * 80)
    print(f"{'Signal':<25} {'Orig 8b RMSE':>14} {'Enh 8b RMSE':>14} {'Orig 16b RMSE':>14} {'Enh 16b RMSE':>14}")
    print("-" * 80)
    for sig_name, sig_res in results.items():
        vals = [sig_res["label"][:24]]
        for method in ["original_8bit", "enhanced_8bit", "original_16bit", "enhanced_16bit"]:
            if method in sig_res and "rmse" in sig_res[method]:
                vals.append(f"{sig_res[method]['rmse']:.6f}")
            else:
                vals.append("ERROR")
        print(f"{vals[0]:<25} {vals[1]:>14} {vals[2]:>14} {vals[3]:>14} {vals[4]:>14}")

    print("\nZero-crossing experiment complete!")


if __name__ == "__main__":
    main()

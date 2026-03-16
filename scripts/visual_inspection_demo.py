#!/usr/bin/env python
"""
Visual Inspection Demonstration for TRACQ Paper Revision (R1 + R2).

Demonstrates that TRACQ compressed artifacts (images) can be visually
inspected to detect anomalies without numerical decompression.

Steps:
  1. Load a real UCI sensor signal (appliances energy)
  2. Create a copy with an injected anomaly (spike + level shift)
  3. Encode both with TRACQ 8-bit
  4. Generate heatmap images using tracq.viewer.heatmap_from_grid()
  5. Create publication figure: side-by-side heatmaps + time-domain plots

Outputs to paper_results/visual_demo/:
  - figure_visual_inspection.png / .pdf
  - normal_signal.tracq.png (TRACQ heatmap artifact)
  - anomalous_signal.tracq.png (TRACQ heatmap artifact)
"""

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tracq.core_enhanced import EnhancedTimeSeriesGrid
from tracq.viewer import heatmap_from_grid

DEFAULT_OUTPUT_DIR = Path(PROJECT_ROOT) / "paper_results" / "visual_demo"
DEFAULT_DATA_DIR = Path(PROJECT_ROOT) / "data" / "processed"


def load_signal(data_dir):
    """Load a real UCI sensor signal for demo."""
    import pandas as pd

    # Try appliances energy first, fall back to others
    for fname in ["uci_appliances_energy.csv", "uci_air_quality.csv", "uci_metro_traffic.csv"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, header=None)
            data = df.values.T.astype(np.float64)
            # Handle NaN/Inf
            data = np.where(np.isfinite(data), data, 0.0)
            # Take first 8 variables and 500 time steps for clear visualization
            n_vars = min(8, data.shape[0])
            n_time = min(500, data.shape[1])
            data = data[:n_vars, :n_time]
            print(f"  Loaded {fname}: using {n_vars} vars x {n_time} steps")
            return data, fname

    # Fallback: synthetic
    print("  No UCI data found, generating synthetic signal")
    np.random.seed(42)
    n_vars, n_time = 8, 500
    t = np.arange(n_time, dtype=np.float64)
    data = np.zeros((n_vars, n_time))
    for i in range(n_vars):
        base = 100 + i * 50
        data[i] = base + np.sin(2 * np.pi * t / (50 + i * 10)) * (5 + i * 2) + np.random.randn(n_time) * 2
    return data, "synthetic"


def inject_anomaly(data):
    """Inject a clear anomaly into a copy of the data."""
    anomalous = data.copy()
    n_vars, n_time = data.shape

    # Anomaly 1: Spike on variable 0 at t=200
    spike_pos = min(200, n_time - 1)
    spike_magnitude = np.std(data[0]) * 10
    anomalous[0, spike_pos:spike_pos+3] += spike_magnitude

    # Anomaly 2: Level shift on variable 2 starting at t=300
    shift_pos = min(300, n_time - 1)
    shift_magnitude = np.std(data[min(2, n_vars-1)]) * 5
    var_idx = min(2, n_vars - 1)
    anomalous[var_idx, shift_pos:] += shift_magnitude

    # Anomaly 3: High-frequency oscillation on variable 4 between t=350-400
    osc_var = min(4, n_vars - 1)
    osc_start = min(350, n_time - 10)
    osc_end = min(400, n_time)
    osc_mag = np.std(data[osc_var]) * 3
    t_osc = np.arange(osc_end - osc_start)
    anomalous[osc_var, osc_start:osc_end] += osc_mag * np.sin(2 * np.pi * t_osc / 3)

    anomaly_info = {
        "spike": {"variable": 0, "position": spike_pos, "magnitude": float(spike_magnitude)},
        "level_shift": {"variable": int(var_idx), "position": shift_pos, "magnitude": float(shift_magnitude)},
        "oscillation": {"variable": int(osc_var), "start": osc_start, "end": osc_end, "magnitude": float(osc_mag)},
    }

    return anomalous, anomaly_info


def encode_and_visualize(data, label):
    """Encode data with TRACQ and generate heatmap."""
    grid = EnhancedTimeSeriesGrid(data, adaptive_clamp=True, use_mu_law=True)
    q8, meta8 = grid.quantize_8bit()

    # Generate heatmap from quantized grid
    heatmap_img = heatmap_from_grid(q8.astype(float))

    return q8, meta8, heatmap_img


def generate_figure(
    normal_data, anomalous_data, anomaly_info,
    normal_q8, anomalous_q8,
    normal_heatmap, anomalous_heatmap,
    output_dir,
):
    """Generate the publication figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

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

    fig = plt.figure(figsize=(FULL_WIDTH, 9.5))

    # Layout: 3 rows x 2 cols
    # Row 1: Time-domain signals (normal vs anomalous)
    # Row 2: TRACQ heatmaps (normal vs anomalous)
    # Row 3: Difference heatmap + annotations

    gs = fig.add_gridspec(3, 2, hspace=0.65, wspace=0.35)

    # --- Row 1, Col 1: Normal time-domain ---
    ax1 = fig.add_subplot(gs[0, 0])
    n_vars = normal_data.shape[0]
    for i in range(min(n_vars, 6)):
        offset = i * 1.0
        normalized = (normal_data[i] - normal_data[i].mean()) / (normal_data[i].std() + 1e-9)
        ax1.plot(normalized + offset, linewidth=0.6, alpha=0.8, label=f"Var {i}")
    ax1.set_title("(a) Normal Signal (Time Domain)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Normalized Value (offset)")
    ax1.legend(fontsize=7, ncol=3, loc="upper right")

    # --- Row 1, Col 2: Anomalous time-domain ---
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(min(n_vars, 6)):
        offset = i * 1.0
        normalized = (anomalous_data[i] - normal_data[i].mean()) / (normal_data[i].std() + 1e-9)
        ax2.plot(normalized + offset, linewidth=0.6, alpha=0.8, label=f"Var {i}")

    # Annotate anomalies
    spike = anomaly_info["spike"]
    ax2.axvline(x=spike["position"], color="red", linestyle="--", alpha=0.5, linewidth=1.5)
    ax2.text(spike["position"] + 5, ax2.get_ylim()[1] * 0.9, "Spike", color="red", fontsize=8)

    shift = anomaly_info["level_shift"]
    ax2.axvline(x=shift["position"], color="orange", linestyle="--", alpha=0.5, linewidth=1.5)
    ax2.text(shift["position"] + 5, ax2.get_ylim()[1] * 0.7, "Level Shift", color="orange", fontsize=8)

    osc = anomaly_info["oscillation"]
    ax2.axvspan(osc["start"], osc["end"], alpha=0.15, color="purple")
    ax2.text((osc["start"] + osc["end"]) / 2, ax2.get_ylim()[1] * 0.5, "Oscillation",
             color="purple", fontsize=8, ha="center")

    ax2.set_title("(b) Anomalous Signal (Time Domain)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Normalized Value (offset)")

    # --- Row 2, Col 1: Normal TRACQ heatmap ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(normal_q8.astype(float), aspect="auto", cmap="viridis", interpolation="nearest")
    ax3.set_title("(c) Normal — TRACQ 8-bit", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Variable")

    # --- Row 2, Col 2: Anomalous TRACQ heatmap ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(anomalous_q8.astype(float), aspect="auto", cmap="viridis", interpolation="nearest")
    ax4.set_title("(d) Anomalous — TRACQ 8-bit", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Variable")

    # Annotate anomaly locations on heatmap
    ax4.axvline(x=spike["position"], color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax4.axvline(x=shift["position"], color="orange", linestyle="--", alpha=0.7, linewidth=1.5)
    ax4.axvspan(osc["start"], osc["end"], alpha=0.2, color="purple")

    # --- Row 3: Difference heatmap ---
    ax5 = fig.add_subplot(gs[2, 0])
    diff = np.abs(anomalous_q8.astype(float) - normal_q8.astype(float))
    im = ax5.imshow(diff, aspect="auto", cmap="hot", interpolation="nearest")
    ax5.set_title("(e) Absolute Difference", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Variable")
    # Horizontal colorbar below the heatmap to avoid stealing column width
    cbar = plt.colorbar(im, ax=ax5, orientation="horizontal", shrink=0.85,
                        pad=0.25, label="Quant. Level Diff.")
    cbar.ax.tick_params(labelsize=7)

    # --- Row 3, Col 2: Key findings text ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    findings_text = (
        "Key Observations for Visual Inspection:\n\n"
        "1. SPIKE (red): Manifests as a bright/dark\n"
        "   spot at the affected variable and time step.\n"
        "   Easily detected as a color discontinuity.\n\n"
        "2. LEVEL SHIFT (orange): Appears as an abrupt\n"
        "   color change that persists beyond the\n"
        "   shift point.\n\n"
        "3. OSCILLATION (purple): Creates a visible\n"
        "   texture pattern (alternating colors)\n"
        "   within the affected region.\n\n"
        "4. The difference heatmap (e) directly\n"
        "   highlights anomalous regions, enabling\n"
        "   automated screening without full\n"
        "   numerical reconstruction."
    )
    ax6.text(0.5, 0.95, findings_text, transform=ax6.transAxes,
             fontsize=8, verticalalignment="top", horizontalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax6.set_title("(f) Visual Inspection Guide", fontsize=10, fontweight="bold")

    # Save
    png_path = os.path.join(output_dir, "figure_visual_inspection.png")
    pdf_path = os.path.join(output_dir, "figure_visual_inspection.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the TRACQ visual inspection demo")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args(argv)

    output_dir = str(args.output_dir)
    data_dir = str(args.data_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading real sensor signal...")
    normal_data, source_file = load_signal(data_dir)

    print("Injecting anomalies...")
    anomalous_data, anomaly_info = inject_anomaly(normal_data)

    print("Encoding with TRACQ...")
    normal_q8, normal_meta, normal_heatmap = encode_and_visualize(normal_data, "normal")
    anomalous_q8, anomalous_meta, anomalous_heatmap = encode_and_visualize(anomalous_data, "anomalous")

    # Save TRACQ heatmap artifacts
    normal_heatmap.save(os.path.join(output_dir, "normal_signal.tracq.png"))
    anomalous_heatmap.save(os.path.join(output_dir, "anomalous_signal.tracq.png"))
    print(f"  Saved TRACQ artifacts to {output_dir}")

    print("Generating publication figure...")
    generate_figure(
        normal_data, anomalous_data, anomaly_info,
        normal_q8, anomalous_q8,
        normal_heatmap, anomalous_heatmap,
        output_dir,
    )

    # Save metadata
    meta_path = os.path.join(output_dir, "visual_demo_info.json")
    with open(meta_path, "w") as f:
        json.dump({
            "source_file": source_file,
            "n_vars": int(normal_data.shape[0]),
            "n_time": int(normal_data.shape[1]),
            "anomaly_info": anomaly_info,
        }, f, indent=2, default=str)
    print(f"  Saved: {meta_path}")

    print("\nVisual inspection demo complete!")


if __name__ == "__main__":
    main()

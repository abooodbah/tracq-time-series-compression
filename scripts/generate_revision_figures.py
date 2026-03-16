#!/usr/bin/env python
"""
Generate All Revision Figures for TRACQ Paper.

Reads results from Phase 2-4 experiments and generates:
  - figure_r1_realworld_rd.png        Rate-distortion curves on UCI datasets
  - figure_r2_realworld_comparison.png Grouped bar chart: RMSE by method by dataset
  - figure_r3_zero_crossing.png        Zero-crossing signal reconstruction + error
  - figure_r4_visual_inspection.png    Normal vs anomalous TRACQ heatmaps
  - figure_r5_realworld_throughput.png  Throughput bar chart

All figures: 300 DPI, matplotlib, publication quality.
"""

import json
import os
import sys
import shutil

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "paper_results")
REALWORLD_DIR = os.path.join(RESULTS_DIR, "realworld")
ZERO_CROSS_DIR = os.path.join(RESULTS_DIR, "zero_crossing")
VISUAL_DIR = os.path.join(RESULTS_DIR, "visual_demo")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "paper_submission")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# IEEE publication-quality defaults (full-width figures)
FULL_WIDTH = 7.16  # inches (IEEE \textwidth)
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# ============================================================================
# Color scheme
# ============================================================================
COLORS = {
    "gzip": "#7f8c8d",
    "parquet": "#95a5a6",
    "delta_zstd": "#2c3e50",
    "tracq_orig_8bit": "#e74c3c",
    "tracq_orig_16bit": "#c0392b",
    "tracq_enh_8bit": "#3498db",
    "tracq_enh_16bit": "#2980b9",
    "tracq_enh_8bit_anchors": "#9b59b6",
    "tracq_enh_16bit_anchors": "#8e44ad",
    "paa": "#e67e22",
    "pla": "#f39c12",
    "sax": "#d35400",
    "gorilla_like": "#1abc9c",
    "zfp": "#27ae60",
    "sz3": "#16a085",
    "sz3_abs_0.1": "#16a085",
    "sz3_abs_0.01": "#1abc9c",
    "sz3_abs_0.001": "#138d75",
    "sz3_abs_0.0001": "#0e6655",
}

METHOD_LABELS = {
    "gzip": "Gzip",
    "parquet": "Parquet",
    "delta_zstd": "Delta+Zstd",
    "tracq_orig_8bit": "TRACQ 8b",
    "tracq_orig_16bit": "TRACQ 16b",
    "tracq_enh_8bit": "Enh. TRACQ 8b",
    "tracq_enh_16bit": "Enh. TRACQ 16b",
    "tracq_enh_8bit_anchors": "Enh. TRACQ 8b+A",
    "tracq_enh_16bit_anchors": "Enh. TRACQ 16b+A",
    "paa": "PAA-64",
    "pla": "PLA-64",
    "sax": "SAX-64",
    "gorilla_like": "Gorilla",
    "sz3_abs_0.1": "SZ3 (0.1)",
    "sz3_abs_0.01": "SZ3 (0.01)",
    "sz3_abs_0.001": "SZ3 (0.001)",
    "sz3_abs_0.0001": "SZ3 (0.0001)",
}

DATASET_LABELS = {
    "uci_air_quality": "Air Quality",
    "uci_appliances_energy": "Appliances Energy",
    "uci_metro_traffic": "Metro Traffic",
}


def load_json(path):
    """Load JSON file, return None if not found."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Figure R1: Rate-Distortion Curves
# ============================================================================
def generate_figure_r1():
    """Rate-distortion curves on 3 UCI datasets, all methods."""
    print("Generating figure_r1_realworld_rd...")
    rd_data = load_json(os.path.join(REALWORLD_DIR, "rd_data.json"))
    if rd_data is None:
        print("  Skipping (no data)")
        return

    n_datasets = len(rd_data)
    fig, axes = plt.subplots(1, n_datasets, figsize=(FULL_WIDTH, 2.8), squeeze=False)

    # Track which labels have been added to avoid duplicate legend entries
    legend_labels_added = set()

    for col, (dataset_name, points) in enumerate(rd_data.items()):
        ax = axes[0, col]
        dataset_label = DATASET_LABELS.get(dataset_name, dataset_name)

        for pt in points:
            method = pt["method"]
            ratio = pt["ratio"]
            rmse_val = pt["rmse"]

            # Skip overflow / inf points — they distort the log-scale axis
            if rmse_val is None or not np.isfinite(rmse_val) or rmse_val > 1e6:
                continue

            color = COLORS.get(method, "#333333")
            label = METHOD_LABELS.get(method, method)

            # Consolidate ZFP/SZ3 sweep into single legend entries
            if method.startswith("zfp_tol_"):
                color = COLORS.get("zfp", "#27ae60")
                label = "ZFP"
            elif method.startswith("sz3_abs_"):
                color = COLORS.get("sz3", "#16a085")
                label = "SZ3"

            # Only add label for first occurrence (for legend), suppress rest
            show_label = label if label not in legend_labels_added else None
            if show_label:
                legend_labels_added.add(label)

            ax.scatter(ratio, max(rmse_val, 1e-6), c=color, s=30, alpha=0.8,
                      zorder=5, label=show_label)

        ax.set_xlabel("Compression Ratio")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{dataset_label}", fontsize=10, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Add a single shared legend from the first axes
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6,
              bbox_to_anchor=(0.5, -0.20), frameon=True)
    fig.suptitle("Rate-Distortion: Real-World UCI Datasets", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    path = os.path.join(FIGURE_DIR, "figure_r1_realworld_rd.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Figure R2: Grouped Bar Chart
# ============================================================================
def generate_figure_r2():
    """Grouped bar chart: RMSE by method by dataset."""
    print("Generating figure_r2_realworld_comparison...")

    # Load all per-dataset results
    all_results = {}
    for dataset_name in DATASET_LABELS:
        data = load_json(os.path.join(REALWORLD_DIR, f"{dataset_name}_results.json"))
        if data is not None:
            all_results[dataset_name] = data["results"]

    # Merge SZ3 results if available
    sz3_data = load_json(os.path.join(REALWORLD_DIR, "sz3_results.json"))
    if sz3_data:
        for dataset_name, ds_data in sz3_data.items():
            if dataset_name not in all_results:
                all_results[dataset_name] = {}
            all_results[dataset_name].update(ds_data.get("results", {}))

    if not all_results:
        print("  Skipping (no data)")
        return

    # Methods to show in bar chart (core comparison — skip orig_8bit which overflows)
    show_methods = [
        "delta_zstd",
        "tracq_orig_16bit", "tracq_enh_8bit", "tracq_enh_16bit_anchors",
        "paa", "gorilla_like", "zfp_tol_0.1", "sz3_abs_0.1",
    ]

    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    n_methods = len(show_methods)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))
    x = np.arange(n_datasets)
    bar_width = 0.8 / n_methods

    # Cap RMSE at 1e5 for plotting; mark overflow with hatching
    RMSE_CAP = 1e5

    for i, method in enumerate(show_methods):
        rmses = []
        for ds in datasets:
            res = all_results[ds].get(method, {})
            if "metrics" in res:
                val = res["metrics"]["rmse"]
                if val is None or not np.isfinite(val) or val > RMSE_CAP:
                    rmses.append(RMSE_CAP)
                else:
                    rmses.append(max(val, 1e-6))  # avoid log(0)
            else:
                rmses.append(0)

        color = COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, rmses, bar_width * 0.9, color=color, alpha=0.85, label=label)

    ax.set_xlabel("Dataset", fontsize=9)
    ax.set_ylabel("RMSE", fontsize=9)
    ax.set_title("Real-World Dataset Comparison: RMSE by Method", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=8)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, "figure_r2_realworld_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Figure R3: Zero-Crossing
# ============================================================================
def generate_figure_r3():
    """Zero-crossing signal reconstruction + error."""
    print("Generating figure_r3_zero_crossing...")

    # Copy existing figure if it was generated
    src = os.path.join(ZERO_CROSS_DIR, "figure_zero_crossing.png")
    if os.path.exists(src):
        dst = os.path.join(FIGURE_DIR, "figure_r3_zero_crossing.png")
        shutil.copy2(src, dst)
        src_pdf = src.replace(".png", ".pdf")
        if os.path.exists(src_pdf):
            shutil.copy2(src_pdf, dst.replace(".png", ".pdf"))
        print(f"  Copied: {dst}")
    else:
        # Generate from results JSON
        results = load_json(os.path.join(ZERO_CROSS_DIR, "zero_crossing_results.json"))
        if results is None:
            print("  Skipping (no data)")
            return

        fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.5))
        sig_names = list(results.keys())

        for idx, sig_name in enumerate(sig_names[:4]):
            ax = axes[idx // 2, idx % 2]
            sig = results[sig_name]
            label = sig.get("label", sig_name)

            methods = ["original_8bit", "enhanced_8bit", "original_16bit", "enhanced_16bit"]
            method_labels = ["Orig 8b", "Enh 8b", "Orig 16b", "Enh 16b"]
            colors = ["#e74c3c", "#3498db", "#e67e22", "#2ecc71"]

            rmses = []
            for m in methods:
                if m in sig and "rmse" in sig[m]:
                    rmses.append(sig[m]["rmse"])
                else:
                    rmses.append(0)

            bars = ax.bar(method_labels, rmses, color=colors, alpha=0.85)
            for bar, val in zip(bars, rmses):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.4f}", ha="center", va="bottom", fontsize=8)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_ylabel("RMSE")

        fig.suptitle("Zero-Crossing Signal Behavior: RMSE Comparison", fontsize=11, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(FIGURE_DIR, "figure_r3_zero_crossing.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ============================================================================
# Figure R4: Visual Inspection
# ============================================================================
def generate_figure_r4():
    """Normal vs anomalous TRACQ heatmaps."""
    print("Generating figure_r4_visual_inspection...")

    src = os.path.join(VISUAL_DIR, "figure_visual_inspection.png")
    if os.path.exists(src):
        dst = os.path.join(FIGURE_DIR, "figure_r4_visual_inspection.png")
        shutil.copy2(src, dst)
        src_pdf = src.replace(".png", ".pdf")
        if os.path.exists(src_pdf):
            shutil.copy2(src_pdf, dst.replace(".png", ".pdf"))
        print(f"  Copied: {dst}")
    else:
        print("  Skipping (no visual demo data). Run visual_inspection_demo.py first.")


# ============================================================================
# Figure R5: Throughput
# ============================================================================
def generate_figure_r5():
    """Throughput bar chart with hardware specs."""
    print("Generating figure_r5_realworld_throughput...")

    # Collect encode/decode times from all datasets
    all_times = {}
    for dataset_name in DATASET_LABELS:
        data = load_json(os.path.join(REALWORLD_DIR, f"{dataset_name}_results.json"))
        if data is None:
            continue

        info = data["info"]
        data_size_mb = info["n_vars"] * info["n_time"] * 8 / (1024 * 1024)  # float64

        for method, res in data["results"].items():
            if "error" in res:
                continue
            if method not in all_times:
                all_times[method] = {"encode_tp": [], "decode_tp": []}

            enc_s = res.get("encode_s", 0)
            dec_s = res.get("decode_s", 0)
            if enc_s > 0:
                all_times[method]["encode_tp"].append(data_size_mb / enc_s)
            if dec_s > 0:
                all_times[method]["decode_tp"].append(data_size_mb / dec_s)

    if not all_times:
        print("  Skipping (no data)")
        return

    # Filter to key methods
    show_methods = [
        "gzip", "delta_zstd",
        "tracq_orig_16bit", "tracq_enh_8bit", "tracq_enh_16bit_anchors",
        "paa", "gorilla_like",
    ]
    show_methods = [m for m in show_methods if m in all_times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))

    # Encode throughput
    labels = [METHOD_LABELS.get(m, m) for m in show_methods]
    enc_means = [np.mean(all_times[m]["encode_tp"]) if all_times[m]["encode_tp"] else 0 for m in show_methods]
    colors = [COLORS.get(m, "#333") for m in show_methods]

    bars1 = ax1.barh(labels, enc_means, color=colors, alpha=0.85)
    ax1.set_xlabel("Encoding Throughput (MB/s)")
    ax1.set_title("Encoding Throughput", fontsize=10, fontweight="bold")
    for bar, val in zip(bars1, enc_means):
        if val > 0:
            ax1.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=8)
    ax1.grid(True, axis="x", alpha=0.3)

    # Decode throughput
    dec_means = [np.mean(all_times[m]["decode_tp"]) if all_times[m]["decode_tp"] else 0 for m in show_methods]
    bars2 = ax2.barh(labels, dec_means, color=colors, alpha=0.85)
    ax2.set_xlabel("Decoding Throughput (MB/s)")
    ax2.set_title("Decoding Throughput", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, dec_means):
        if val > 0:
            ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=8)
    ax2.grid(True, axis="x", alpha=0.3)

    import platform
    hw_note = (f"Platform: Python {platform.python_version()} + NumPy | "
               f"OS: {platform.system()} {platform.machine()}")
    fig.suptitle("Real-World Benchmark: Compression Throughput\n" + hw_note,
                 fontsize=10, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, "figure_r5_realworld_throughput.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("Generating Revision Figures")
    print("=" * 60)

    generate_figure_r1()
    generate_figure_r2()
    generate_figure_r3()
    generate_figure_r4()
    generate_figure_r5()

    print("\nAll revision figures generated!")
    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()

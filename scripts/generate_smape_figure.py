#!/usr/bin/env python
"""
Generate SMAPE analysis figure for TRACQ paper.

Creates figure_r6_smape_analysis.png showing:
  - Panel A: Per-variable SMAPE on Appliances Energy (28 vars), sorted by
    variable magnitude, comparing TRACQ Enh 16b+A vs PAA vs ZFP
  - Panel B: SMAPE vs Compression Ratio trade-off across all methods/datasets

This figure demonstrates the mu-law companding advantage: TRACQ achieves
more uniform relative fidelity across variables of different magnitudes.
"""

import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "paper_results", "realworld")
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


def load_json(path):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def generate_smape_figure():
    """Per-variable SMAPE analysis on Appliances Energy + SMAPE trade-off."""
    print("Generating figure_r6_smape_analysis...")

    # Load Appliances Energy results (28 vars, most multi-scale)
    app_data = load_json(os.path.join(RESULTS_DIR, "uci_appliances_energy_results.json"))
    if app_data is None:
        print("  Skipping (no data)")
        return

    results = app_data["results"]
    n_vars = app_data["info"]["n_vars"]

    # Load the actual dataset to get variable magnitudes
    import pandas as pd
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "uci_appliances_energy.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, header=None, nrows=5000)
        raw_data = df.values.T.astype(np.float64)
        var_magnitudes = np.mean(np.abs(raw_data), axis=1)
    else:
        var_magnitudes = np.arange(n_vars, dtype=float) + 1

    # Methods to compare in per-variable SMAPE
    methods_pervar = [
        ("tracq_orig_16bit", "TRACQ Orig 16b", "#c0392b", "-"),
        ("tracq_enh_8bit", "TRACQ Enh 8b (mu-law)", "#3498db", "-"),
        ("tracq_enh_16bit_anchors", "TRACQ Enh 16b+A", "#8e44ad", "-"),
        ("paa", "PAA-64", "#e67e22", "--"),
        ("zfp_tol_0.1", "ZFP (tol=0.1)", "#27ae60", "-."),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))

    # ==========================================================================
    # Panel A: Per-variable SMAPE sorted by variable magnitude
    # ==========================================================================
    sort_idx = np.argsort(var_magnitudes)
    sorted_magnitudes = var_magnitudes[sort_idx]

    for method_key, label, color, ls in methods_pervar:
        if method_key not in results or "metrics" not in results[method_key]:
            continue
        per_var = results[method_key]["metrics"].get("per_var_smape", [])
        if len(per_var) != n_vars:
            continue
        per_var = np.array(per_var)
        sorted_smape = per_var[sort_idx]
        # Clip for log scale display
        sorted_smape = np.maximum(sorted_smape, 1e-8)
        ax1.plot(range(n_vars), sorted_smape, label=label, color=color,
                 linestyle=ls, linewidth=1.5, alpha=0.85, marker=".", markersize=4)

    ax1.set_yscale("log")
    ax1.set_xlabel("Variable (sorted by mean magnitude →)", fontsize=9)
    ax1.set_ylabel("Per-Variable SMAPE", fontsize=9)
    ax1.set_title(
        "(a) Per-Variable SMAPE\nAppliances Energy (28 vars)",
        fontsize=10,
        fontweight="bold",
        pad=12,
    )
    ax1.legend(fontsize=6, loc="upper left", framealpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, n_vars - 0.5)

    # Add secondary x-axis showing magnitude range
    ax1_twin = ax1.twiny()
    ax1_twin.set_xlim(ax1.get_xlim())
    tick_positions = [0, n_vars // 4, n_vars // 2, 3 * n_vars // 4, n_vars - 1]
    ax1_twin.set_xticks(tick_positions)
    ax1_twin.set_xticklabels([f"{sorted_magnitudes[i]:.0f}" for i in tick_positions], fontsize=7)
    ax1_twin.set_xlabel("Mean |value|", fontsize=8, labelpad=6)

    # ==========================================================================
    # Panel B: SMAPE vs Compression Ratio across all datasets
    # ==========================================================================
    # Load all datasets
    datasets = {
        "Air Quality": "uci_air_quality_results.json",
        "Appliances": "uci_appliances_energy_results.json",
        "Metro Traffic": "uci_metro_traffic_results.json",
    }

    markers = {"Air Quality": "o", "Appliances": "s", "Metro Traffic": "^"}
    method_colors = {
        "tracq_orig_16bit": "#c0392b",
        "tracq_enh_8bit": "#3498db",
        "tracq_enh_8bit_anchors": "#9b59b6",
        "tracq_enh_16bit_anchors": "#8e44ad",
        "paa": "#e67e22",
        "gorilla_like": "#1abc9c",
        "delta_zstd": "#2c3e50",
        "zfp_tol_0.1": "#27ae60",
    }
    method_labels = {
        "tracq_orig_16bit": "TRACQ 16b",
        "tracq_enh_8bit": "Enh. 8b",
        "tracq_enh_8bit_anchors": "Enh. 8b+A",
        "tracq_enh_16bit_anchors": "Enh. 16b+A",
        "paa": "PAA-64",
        "gorilla_like": "Gorilla",
        "delta_zstd": "Delta+Zstd",
        "zfp_tol_0.1": "ZFP (0.1)",
    }

    plotted_labels = set()
    method_handles = []
    method_handle_labels = []
    for ds_name, ds_file in datasets.items():
        ds_data = load_json(os.path.join(RESULTS_DIR, ds_file))
        if ds_data is None:
            continue
        ds_results = ds_data["results"]
        marker = markers[ds_name]

        for method_key in method_colors:
            if method_key not in ds_results:
                continue
            res = ds_results[method_key]
            if "error" in res or "metrics" not in res:
                continue
            smape_val = res["metrics"].get("smape", None)
            if smape_val is None or not np.isfinite(smape_val):
                continue
            ratio = res["ratio"]
            color = method_colors[method_key]
            label = method_labels[method_key]

            # Only label once per method
            show_label = label if label not in plotted_labels else None
            if show_label:
                plotted_labels.add(label)

            scatter = ax2.scatter(
                ratio,
                max(smape_val, 1e-8),
                c=color,
                marker=marker,
                s=50,
                alpha=0.8,
                zorder=5,
                label=show_label,
                edgecolors="white",
                linewidths=0.3,
            )
            if show_label:
                method_handles.append(scatter)
                method_handle_labels.append(show_label)

    # Add dataset marker legend
    dataset_handles = []
    dataset_labels = []
    for ds_name, marker in markers.items():
        h = ax2.scatter([], [], c="gray", marker=marker, s=40)
        dataset_handles.append(h)
        dataset_labels.append(ds_name)

    ax2.set_xlabel("Compression Ratio", fontsize=9)
    ax2.set_ylabel("SMAPE", fontsize=9)
    ax2.set_title(
        "(b) SMAPE vs Compression Ratio\nAcross All Datasets",
        fontsize=10,
        fontweight="bold",
        pad=12,
    )
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    dataset_legend = ax2.legend(
        dataset_handles,
        dataset_labels,
        fontsize=6,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.95),
        borderaxespad=0.0,
        framealpha=0.75,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    ax2.add_artist(dataset_legend)
    ax2.text(1.18, 1.01, "Dataset", transform=ax2.transAxes,
             ha="center", va="bottom", fontsize=7)
    ax2.legend(
        method_handles,
        method_handle_labels,
        fontsize=6,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.68),
        borderaxespad=0.0,
        framealpha=0.9,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    ax2.text(1.18, 0.74, "Method", transform=ax2.transAxes,
             ha="center", va="bottom", fontsize=7)

    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, "figure_r6_smape_analysis.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    generate_smape_figure()

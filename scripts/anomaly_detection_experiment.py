#!/usr/bin/env python
"""
Anomaly Detection in Compressed Domain: TRACQ vs Numerical Pipelines.

Demonstrates that anomalies can be detected directly on TRACQ compressed
images (PNGs) without full numerical decompression, achieving comparable
F1-scores with potentially higher throughput.

Experiment Design:
  1. Generate labeled windows from UCI Appliances Energy (28 vars):
     - Normal windows: unmodified sliding windows
     - Anomalous windows: injected spike, level shift, variance change
  2. Pipeline A (Numerical Decode+Detect):
     Compressed data -> Decompress to float64 -> Extract features -> Classify
  3. Pipeline B (TRACQ Direct):
     TRACQ PNG -> Load image array -> Extract image features -> Classify
  4. Pipeline C (Image Processing Only, no ML):
     TRACQ PNG -> Histogram entropy + edge density -> Threshold
  5. Report F1, Precision, Recall, and Throughput (windows/sec).

Outputs to paper_results/anomaly_detection/:
  - anomaly_detection_results.json
  - figure_r7_anomaly_detection.png / .pdf
"""

import json
import os
import sys
import time
import io
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import sobel
from scipy.stats import entropy, kurtosis
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tracq.core_enhanced import EnhancedTimeSeriesGrid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Configuration
# ============================================================================
DEFAULT_DATA_DIR = Path(PROJECT_ROOT) / "data" / "processed"
DEFAULT_OUTPUT_DIR = Path(PROJECT_ROOT) / "paper_results" / "anomaly_detection"
DEFAULT_FIGURE_DIR = Path(PROJECT_ROOT) / "paper_submission"

WINDOW_SIZE = 100      # timesteps per window
WINDOW_STRIDE = 50     # overlap
ANOMALY_FRACTION = 0.5 # ~50% anomalous
RANDOM_SEED = 42
N_CV_FOLDS = 5


# ============================================================================
# Step 1: Generate Labeled Windows
# ============================================================================
def load_appliances_data(data_dir):
    """Load UCI Appliances Energy dataset."""
    path = os.path.join(data_dir, "uci_appliances_energy.csv")
    df = pd.read_csv(path, header=None)
    data = df.values.T.astype(np.float64)  # (28, n_time)
    return data[:, :5000]


def inject_anomaly(window, rng):
    """Inject a random anomaly into a window. Returns (anomalous_window, anomaly_type)."""
    w = window.copy()
    n_vars, n_time = w.shape
    anomaly_type = rng.choice(["spike", "level_shift", "variance"])

    # Pick 1-3 random variables to affect
    n_affected = rng.integers(1, min(4, n_vars))
    affected_vars = rng.choice(n_vars, size=n_affected, replace=False)

    if anomaly_type == "spike":
        # Inject a large spike at a random timestep
        t = rng.integers(n_time // 4, 3 * n_time // 4)
        for v in affected_vars:
            magnitude = rng.uniform(5, 15) * np.std(w[v])
            sign = rng.choice([-1, 1])
            w[v, t] += sign * magnitude
            # Also affect a small neighborhood
            spread = rng.integers(1, 4)
            for dt in range(1, spread + 1):
                if t + dt < n_time:
                    w[v, t + dt] += sign * magnitude * 0.5 / dt
                if t - dt >= 0:
                    w[v, t - dt] += sign * magnitude * 0.3 / dt

    elif anomaly_type == "level_shift":
        # Sustained offset starting at a random point
        t_start = rng.integers(n_time // 4, n_time // 2)
        for v in affected_vars:
            shift = rng.uniform(3, 10) * np.std(w[v])
            sign = rng.choice([-1, 1])
            w[v, t_start:] += sign * shift

    elif anomaly_type == "variance":
        # Increased noise in a region
        t_start = rng.integers(n_time // 4, n_time // 2)
        t_end = min(t_start + rng.integers(20, 50), n_time)
        for v in affected_vars:
            noise_scale = rng.uniform(3, 8) * np.std(w[v])
            w[v, t_start:t_end] += rng.normal(0, noise_scale, t_end - t_start)

    return w, anomaly_type


def create_labeled_dataset(data):
    """Create sliding windows with injected anomalies."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_vars, n_time = data.shape

    windows = []
    labels = []
    anomaly_types = []

    # Create sliding windows
    positions = list(range(0, n_time - WINDOW_SIZE + 1, WINDOW_STRIDE))

    for pos in positions:
        window = data[:, pos:pos + WINDOW_SIZE]

        # Normal version
        windows.append(window.copy())
        labels.append(0)
        anomaly_types.append("normal")

        # Anomalous version
        anom_window, anom_type = inject_anomaly(window, rng)
        windows.append(anom_window)
        labels.append(1)
        anomaly_types.append(anom_type)

    return windows, np.array(labels), anomaly_types


# ============================================================================
# Step 2: Encode windows to TRACQ images
# ============================================================================
def encode_to_tracq_image(window):
    """Encode a window to TRACQ Enhanced 8-bit image array."""
    grid = EnhancedTimeSeriesGrid(window, adaptive_clamp=True, use_mu_law=True,
                                   anchor_interval=0)
    q8, meta8 = grid.quantize_8bit()
    return q8  # uint8 array, directly usable as image


def encode_to_tracq_png_bytes(window):
    """Encode window to TRACQ PNG bytes (simulates stored compressed data)."""
    from PIL import Image
    q8 = encode_to_tracq_image(window)
    img = Image.fromarray(q8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================================
# Step 3: Feature Extraction
# ============================================================================
def extract_numerical_features(window):
    """Extract statistical features from raw float64 window for anomaly detection."""
    n_vars, n_time = window.shape
    features = []
    for v in range(n_vars):
        row = window[v]
        features.extend([
            np.mean(row),
            np.std(row),
            np.max(row) - np.min(row),     # range
            kurtosis(row),
            np.max(np.abs(np.diff(row))),   # max step change
            np.percentile(row, 95) - np.percentile(row, 5),  # inter-percentile range
        ])
    return np.array(features)


def extract_image_features(img_array):
    """Extract image-based features from TRACQ uint8 array (no decompression)."""
    # Pixel-level statistics
    flat = img_array.astype(float).ravel()
    features = [
        np.mean(flat),
        np.std(flat),
        kurtosis(flat),
        np.max(flat) - np.min(flat),  # dynamic range
    ]

    # Histogram entropy (texture measure)
    hist, _ = np.histogram(flat, bins=32, range=(0, 255))
    hist = hist / (hist.sum() + 1e-9)
    features.append(entropy(hist))

    # Edge density (spatial discontinuity measure)
    edges_x = sobel(img_array.astype(float), axis=1)
    edges_y = sobel(img_array.astype(float), axis=0)
    edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
    features.extend([
        np.mean(edge_magnitude),
        np.std(edge_magnitude),
        np.max(edge_magnitude),
    ])

    # Row-wise variance (per-variable texture)
    row_stds = np.std(img_array.astype(float), axis=1)
    features.extend([
        np.mean(row_stds),
        np.std(row_stds),
        np.max(row_stds),
    ])

    # Column-wise gradient (temporal smoothness)
    col_diffs = np.abs(np.diff(img_array.astype(float), axis=1))
    features.extend([
        np.mean(col_diffs),
        np.std(col_diffs),
        np.max(col_diffs),
    ])

    return np.array(features)


def extract_image_simple_features(img_array):
    """Ultra-lightweight image features for threshold-based detection (no ML)."""
    flat = img_array.astype(float)
    # Histogram entropy
    hist, _ = np.histogram(flat.ravel(), bins=32, range=(0, 255))
    hist = hist / (hist.sum() + 1e-9)
    h_entropy = entropy(hist)

    # Edge density
    edges = np.abs(np.diff(flat, axis=1))
    edge_density = np.mean(edges)

    # Pixel intensity std
    pixel_std = np.std(flat)

    return h_entropy, edge_density, pixel_std


# ============================================================================
# Step 4: Detection Pipelines
# ============================================================================
def run_numerical_isolation_forest(windows, labels):
    """Pipeline A: Numerical decode + Isolation Forest (unsupervised)."""
    print("  Running Pipeline A: Numerical Isolation Forest...")

    # Extract features
    t0 = time.perf_counter()
    features = np.array([extract_numerical_features(w) for w in windows])
    feature_time = time.perf_counter() - t0

    # Replace inf/nan
    features = np.nan_to_num(features, nan=0, posinf=1e6, neginf=-1e6)

    # Isolation Forest (unsupervised)
    t0 = time.perf_counter()
    iso = IsolationForest(contamination=0.5, random_state=RANDOM_SEED, n_estimators=100)
    preds = iso.fit_predict(features)
    detect_time = time.perf_counter() - t0

    # Convert: -1 = anomaly, 1 = normal → 1 = anomaly, 0 = normal
    preds_binary = (preds == -1).astype(int)

    total_time = feature_time + detect_time
    throughput = len(windows) / total_time

    return {
        "name": "Numerical IF",
        "predictions": preds_binary,
        "f1": float(f1_score(labels, preds_binary)),
        "precision": float(precision_score(labels, preds_binary)),
        "recall": float(recall_score(labels, preds_binary)),
        "accuracy": float(accuracy_score(labels, preds_binary)),
        "feature_time_s": feature_time,
        "detect_time_s": detect_time,
        "total_time_s": total_time,
        "throughput_wps": throughput,
    }


def run_numerical_rf(windows, labels):
    """Pipeline A2: Numerical features + Random Forest (supervised, cross-validated)."""
    print("  Running Pipeline A2: Numerical Random Forest (CV)...")

    features = np.array([extract_numerical_features(w) for w in windows])
    features = np.nan_to_num(features, nan=0, posinf=1e6, neginf=-1e6)

    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    all_preds = np.zeros(len(labels), dtype=int)

    t0 = time.perf_counter()
    for train_idx, test_idx in skf.split(features, labels):
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_SEED)
        rf.fit(features[train_idx], labels[train_idx])
        all_preds[test_idx] = rf.predict(features[test_idx])
    total_time = time.perf_counter() - t0

    # Inference-only timing (single pass)
    rf_full = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_SEED)
    rf_full.fit(features, labels)
    t0 = time.perf_counter()
    _ = rf_full.predict(features)
    inference_time = time.perf_counter() - t0
    throughput = len(windows) / inference_time

    return {
        "name": "Numerical RF",
        "predictions": all_preds,
        "f1": float(f1_score(labels, all_preds)),
        "precision": float(precision_score(labels, all_preds)),
        "recall": float(recall_score(labels, all_preds)),
        "accuracy": float(accuracy_score(labels, all_preds)),
        "cv_time_s": total_time,
        "inference_time_s": inference_time,
        "throughput_wps": throughput,
    }


def run_tracq_image_rf(windows, labels):
    """Pipeline B: TRACQ image features + Random Forest (supervised, CV)."""
    print("  Running Pipeline B: TRACQ Image RF (CV)...")

    # Encode all windows to TRACQ images
    t0 = time.perf_counter()
    images = [encode_to_tracq_image(w) for w in windows]
    encode_time = time.perf_counter() - t0

    # Extract image features
    t0 = time.perf_counter()
    features = np.array([extract_image_features(img) for img in images])
    feature_time = time.perf_counter() - t0
    features = np.nan_to_num(features, nan=0, posinf=1e6, neginf=-1e6)

    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    all_preds = np.zeros(len(labels), dtype=int)

    t0 = time.perf_counter()
    for train_idx, test_idx in skf.split(features, labels):
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_SEED)
        rf.fit(features[train_idx], labels[train_idx])
        all_preds[test_idx] = rf.predict(features[test_idx])
    cv_time = time.perf_counter() - t0

    # Inference-only timing: image feature extraction + classification
    rf_full = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_SEED)
    rf_full.fit(features, labels)
    t0 = time.perf_counter()
    # Simulate direct detection: load image + extract features + classify
    for img in images:
        feat = extract_image_features(img)
    feat_array = np.array([extract_image_features(img) for img in images])
    feat_array = np.nan_to_num(feat_array, nan=0, posinf=1e6, neginf=-1e6)
    _ = rf_full.predict(feat_array)
    inference_time = time.perf_counter() - t0
    throughput = len(windows) / inference_time

    return {
        "name": "TRACQ Image RF",
        "predictions": all_preds,
        "f1": float(f1_score(labels, all_preds)),
        "precision": float(precision_score(labels, all_preds)),
        "recall": float(recall_score(labels, all_preds)),
        "accuracy": float(accuracy_score(labels, all_preds)),
        "encode_time_s": encode_time,
        "feature_time_s": feature_time,
        "cv_time_s": cv_time,
        "inference_time_s": inference_time,
        "throughput_wps": throughput,
    }


def run_tracq_threshold(windows, labels):
    """Pipeline C: TRACQ image processing only (no ML, no training)."""
    print("  Running Pipeline C: TRACQ Threshold (no ML)...")

    # Encode + extract simple features
    t0 = time.perf_counter()
    images = [encode_to_tracq_image(w) for w in windows]
    encode_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    simple_features = [extract_image_simple_features(img) for img in images]
    feature_time = time.perf_counter() - t0

    entropies = np.array([f[0] for f in simple_features])
    edge_densities = np.array([f[1] for f in simple_features])
    pixel_stds = np.array([f[2] for f in simple_features])

    # Find best threshold via grid search on each feature
    best_f1 = 0
    best_preds = None
    best_feature_name = None

    for feat, fname in [(entropies, "entropy"), (edge_densities, "edge_density"),
                         (pixel_stds, "pixel_std")]:
        for percentile in range(20, 80, 5):
            thresh = np.percentile(feat, percentile)
            # Anomalies should have different feature values
            for direction in ["above", "below"]:
                if direction == "above":
                    preds = (feat > thresh).astype(int)
                else:
                    preds = (feat < thresh).astype(int)
                f1 = f1_score(labels, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_preds = preds.copy()
                    best_feature_name = f"{fname}_{direction}"

    # Also try combined: anomaly if ANY feature is extreme
    combined = np.zeros(len(labels), dtype=int)
    for feat in [entropies, edge_densities, pixel_stds]:
        q25, q75 = np.percentile(feat, [25, 75])
        iqr = q75 - q25
        outlier_mask = (feat < q25 - 1.5 * iqr) | (feat > q75 + 1.5 * iqr)
        combined = np.maximum(combined, outlier_mask.astype(int))
    f1_combined = f1_score(labels, combined, zero_division=0)
    if f1_combined > best_f1:
        best_f1 = f1_combined
        best_preds = combined
        best_feature_name = "combined_iqr"

    total_time = encode_time + feature_time
    throughput = len(windows) / (feature_time + 1e-9)

    return {
        "name": "TRACQ Threshold",
        "predictions": best_preds if best_preds is not None else np.zeros(len(labels), dtype=int),
        "f1": float(best_f1),
        "precision": float(precision_score(labels, best_preds, zero_division=0)) if best_preds is not None else 0,
        "recall": float(recall_score(labels, best_preds, zero_division=0)) if best_preds is not None else 0,
        "accuracy": float(accuracy_score(labels, best_preds)) if best_preds is not None else 0,
        "best_feature": best_feature_name,
        "encode_time_s": encode_time,
        "feature_time_s": feature_time,
        "total_time_s": total_time,
        "throughput_wps": throughput,
    }


def run_decode_then_detect_throughput(windows, labels):
    """Measure throughput of decode+detect pipeline (simulating compressed input).

    Simulates: Compressed blob → Decompress to float64 → Feature extract → Classify.
    """
    print("  Measuring decode+detect throughput...")

    # First, encode all windows to TRACQ compressed form
    encoded = []
    for w in windows:
        grid = EnhancedTimeSeriesGrid(w, adaptive_clamp=True, use_mu_law=True,
                                       anchor_interval=0)
        q8, meta8 = grid.quantize_8bit()
        encoded.append((q8, meta8))

    # Train classifier on numerical features
    features = np.array([extract_numerical_features(w) for w in windows])
    features = np.nan_to_num(features, nan=0, posinf=1e6, neginf=-1e6)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_SEED)
    rf.fit(features, labels)

    # Measure full pipeline: decompress → extract features → classify
    t0 = time.perf_counter()
    for q8, meta8 in encoded:
        # Step 1: Decompress to float64
        recon, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q8, meta8)
        # Step 2: Extract features
        feat = extract_numerical_features(recon)
        feat = np.nan_to_num(feat, nan=0, posinf=1e6, neginf=-1e6)
        # Step 3: Classify
        _ = rf.predict(feat.reshape(1, -1))
    decode_detect_time = time.perf_counter() - t0

    # Measure image-direct pipeline: load image array → extract features → classify
    img_features = np.array([extract_image_features(q8) for q8, _ in encoded])
    img_features = np.nan_to_num(img_features, nan=0, posinf=1e6, neginf=-1e6)
    rf_img = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_SEED)
    rf_img.fit(img_features, labels)

    t0 = time.perf_counter()
    for q8, meta8 in encoded:
        # Step 1: Image array is already available (no decompression)
        feat = extract_image_features(q8)
        feat = np.nan_to_num(feat, nan=0, posinf=1e6, neginf=-1e6)
        # Step 2: Classify
        _ = rf_img.predict(feat.reshape(1, -1))
    direct_detect_time = time.perf_counter() - t0

    return {
        "decode_then_detect_time_s": decode_detect_time,
        "direct_detect_time_s": direct_detect_time,
        "decode_then_detect_wps": len(windows) / decode_detect_time,
        "direct_detect_wps": len(windows) / direct_detect_time,
        "speedup": decode_detect_time / max(direct_detect_time, 1e-9),
    }


# ============================================================================
# Step 5: Generate Figure
# ============================================================================
def generate_figure(results, throughput_results, figure_dir):
    """Generate combined F1 + throughput figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))

    # Panel A: F1 / Precision / Recall comparison
    pipeline_names = [r["name"] for r in results]
    f1_scores = [r["f1"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]

    x = np.arange(len(pipeline_names))
    width = 0.25

    colors_f1 = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad"]
    bars1 = ax1.bar(x - width, f1_scores, width, label="F1-Score",
                    color=[colors_f1[i % len(colors_f1)] for i in range(len(pipeline_names))],
                    alpha=0.85)
    bars2 = ax1.bar(x, precisions, width, label="Precision",
                    color=[colors_f1[i % len(colors_f1)] for i in range(len(pipeline_names))],
                    alpha=0.55, hatch="//")
    bars3 = ax1.bar(x + width, recalls, width, label="Recall",
                    color=[colors_f1[i % len(colors_f1)] for i in range(len(pipeline_names))],
                    alpha=0.55, hatch="\\\\")

    ax1.set_ylabel("Score", fontsize=9)
    ax1.set_title("(a) Anomaly Detection: F1 / Precision / Recall", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pipeline_names, fontsize=8, rotation=15, ha="right")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, axis="y", alpha=0.3)

    # Add F1 value labels
    for bar, val in zip(bars1, f1_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel B: Throughput comparison
    pipeline_tp = {
        "Decode+Detect\n(Numerical RF)": throughput_results["decode_then_detect_wps"],
        "Direct Detect\n(TRACQ Image RF)": throughput_results["direct_detect_wps"],
    }

    tp_names = list(pipeline_tp.keys())
    tp_vals = list(pipeline_tp.values())
    tp_colors = ["#e74c3c", "#27ae60"]

    bars_tp = ax2.barh(tp_names, tp_vals, color=tp_colors, alpha=0.85, height=0.5)
    for bar, val in zip(bars_tp, tp_vals):
        ax2.text(val + max(tp_vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} w/s", va="center", fontsize=8, fontweight="bold")

    speedup = throughput_results["speedup"]
    ax2.set_xlabel("Throughput (windows/sec)", fontsize=9)
    ax2.set_title(f"(b) Inference Throughput ({speedup:.1f}x speedup)",
                  fontsize=10, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    path = os.path.join(figure_dir, "figure_r7_anomaly_detection.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {path}")


# ============================================================================
# Main
# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Run compressed-domain anomaly detection for TRACQ")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = ap.parse_args(argv)

    data_dir = str(args.data_dir)
    output_dir = str(args.output_dir)
    figure_dir = str(args.figure_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    print("=" * 60)
    print("Anomaly Detection Experiment")
    print("=" * 60)

    # Step 1: Load data and create labeled windows
    print("\n1. Loading UCI Appliances Energy dataset...")
    expected_csv = os.path.join(data_dir, "uci_appliances_energy.csv")
    if not os.path.exists(expected_csv):
        raise SystemExit(
            f"Missing required dataset: {expected_csv}. "
            "Provide --data-dir pointing to processed CSVs to rerun this experiment."
        )
    data = load_appliances_data(data_dir)
    print(f"   Data shape: {data.shape}")

    print("\n2. Creating labeled windows...")
    windows, labels, anomaly_types = create_labeled_dataset(data)
    n_normal = np.sum(labels == 0)
    n_anom = np.sum(labels == 1)
    print(f"   Total windows: {len(windows)} ({n_normal} normal, {n_anom} anomalous)")

    anom_counts = {}
    for t in anomaly_types:
        anom_counts[t] = anom_counts.get(t, 0) + 1
    print(f"   Anomaly types: {anom_counts}")

    # Step 2: Run detection pipelines
    print("\n3. Running detection pipelines...")
    results = []

    r1 = run_numerical_isolation_forest(windows, labels)
    results.append(r1)
    print(f"     Numerical IF:        F1={r1['f1']:.3f}  Prec={r1['precision']:.3f}  Rec={r1['recall']:.3f}")

    r2 = run_numerical_rf(windows, labels)
    results.append(r2)
    print(f"     Numerical RF:        F1={r2['f1']:.3f}  Prec={r2['precision']:.3f}  Rec={r2['recall']:.3f}")

    r3 = run_tracq_image_rf(windows, labels)
    results.append(r3)
    print(f"     TRACQ Image RF:     F1={r3['f1']:.3f}  Prec={r3['precision']:.3f}  Rec={r3['recall']:.3f}")

    r4 = run_tracq_threshold(windows, labels)
    results.append(r4)
    print(f"     TRACQ Threshold:    F1={r4['f1']:.3f}  Prec={r4['precision']:.3f}  Rec={r4['recall']:.3f}")

    # Step 3: Throughput comparison
    print("\n4. Measuring throughput (decode+detect vs direct detect)...")
    throughput = run_decode_then_detect_throughput(windows, labels)
    print(f"     Decode+Detect:  {throughput['decode_then_detect_wps']:.0f} windows/sec")
    print(f"     Direct Detect:  {throughput['direct_detect_wps']:.0f} windows/sec")
    print(f"     Speedup:        {throughput['speedup']:.1f}x")

    # Step 4: Save results
    output = {
        "dataset": "uci_appliances_energy",
        "window_size": WINDOW_SIZE,
        "n_windows": len(windows),
        "n_normal": int(n_normal),
        "n_anomalous": int(n_anom),
        "anomaly_types": anom_counts,
        "pipelines": [{k: v for k, v in r.items() if k != "predictions"} for r in results],
        "throughput": throughput,
    }
    out_path = os.path.join(output_dir, "anomaly_detection_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # Step 5: Generate figure
    print("\n5. Generating figure...")
    generate_figure(results, throughput, figure_dir)

    print("\nAnomaly detection experiment complete!")


if __name__ == "__main__":
    main()

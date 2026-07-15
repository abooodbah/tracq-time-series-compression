"""Compressed-domain anomaly detection: TRACQ-2 lattice grids vs old TRACQ grids.

Reuses the paper experiment's exact window generation (same seed -> identical
198 windows), feature extractors, and pipelines, swapping only the compressed
image. Adds a TRACQ-2-native feature set that exploits the residual grid's
structure (mid-gray = no change; anomalies = bright/dark pixels + escapes).
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tracq import lattice

# import the original experiment module (has __main__ guard)
spec = importlib.util.spec_from_file_location(
    "anom", os.path.join(PROJECT_ROOT, "scripts", "anomaly_detection_experiment.py"))
anom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(anom)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

OUT = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "anomaly_results.json")


def encode_tracq2_image(window, eps=1e-2, mode="rel"):
    _, grid, _ = lattice.encode(window, eps=eps, mode=mode, predictors="p1", zstd_level=3)
    return grid


def tracq2_native_features(grid):
    """Features native to the residual grid: activity, extremes, escapes."""
    g = grid.astype(np.float64)
    dev = np.abs(g - 128.0)
    n_escape = float((grid == lattice.ESCAPE).sum())
    active = dev > 2
    row_max = dev.max(axis=1)
    col_activity = active.mean(axis=0)
    feats = [
        dev.mean(), dev.std(), dev.max(),
        float(active.mean()),                     # fraction of moving pixels
        n_escape,
        row_max.mean(), row_max.max(),
        float(col_activity.max()),                # burstiest timestep
        float((dev > 20).sum()),                  # strong-move count
        float(np.percentile(dev, 99)),
    ]
    return np.array(feats)


def tracq2_trajectory_features(grid):
    """Zero-copy shape features: cumsum of residuals recovers the
    transform-domain trajectory (integer adds on the compressed grid, no
    float decode). Level shifts -> sustained offset; spikes -> jump+return;
    variance bursts -> local roughness."""
    r = grid.astype(np.int64) - 128
    r[grid == lattice.ESCAPE] = 127  # escapes = violent moves
    traj = np.cumsum(r, axis=1)      # per-row transform trajectory
    t = traj.astype(np.float64)
    n = t.shape[1]
    half = n // 2
    # normalize per row by robust scale of its steps
    step_scale = np.maximum(np.percentile(np.abs(r), 90, axis=1), 1.0)
    tn = t / step_scale[:, None]
    drift = np.abs(tn[:, -1])                       # end-vs-start offset
    span = tn.max(axis=1) - tn.min(axis=1)          # trajectory range
    mean_shift = np.abs(tn[:, half:].mean(axis=1) - tn[:, :half].mean(axis=1))
    rough = np.abs(np.diff(np.sign(np.diff(tn, axis=1) + 1e-9), axis=1)).mean(axis=1)
    feats = [
        drift.max(), drift.mean(),
        span.max(), span.mean(),
        mean_shift.max(), mean_shift.mean(),
        rough.max(),
        float(np.abs(r).max()),
        float((np.abs(r) >= 100).sum()),
    ]
    return np.array(feats)


def rf_cv(features, labels, name):
    features = np.nan_to_num(features, nan=0, posinf=1e6, neginf=-1e6)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(labels), dtype=int)
    for tr, te in skf.split(features, labels):
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(features[tr], labels[tr])
        preds[te] = rf.predict(features[te])
    return {
        "name": name,
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
    }


def threshold_search(feature_arrays, labels, name):
    """Same protocol as the paper's Pipeline C grid search."""
    best = {"name": name, "f1": 0.0, "precision": 0.0, "recall": 0.0, "feature": None}
    for fname, feat in feature_arrays.items():
        for pct in range(20, 80, 5):
            th = np.percentile(feat, pct)
            for direction in ("above", "below"):
                preds = (feat > th).astype(int) if direction == "above" else (feat < th).astype(int)
                f1 = f1_score(labels, preds, zero_division=0)
                if f1 > best["f1"]:
                    best.update({
                        "f1": float(f1),
                        "precision": float(precision_score(labels, preds, zero_division=0)),
                        "recall": float(recall_score(labels, preds, zero_division=0)),
                        "feature": f"{fname}_{direction}",
                    })
    return best


def main():
    data = anom.load_appliances_data(anom.DEFAULT_DATA_DIR)
    windows, labels, types = anom.create_labeled_dataset(data)
    print(f"{len(windows)} windows ({int(labels.sum())} anomalous): "
          f"{dict((t, types.count(t)) for t in set(types) if t != 'normal')}")

    results = {}

    # Reference: paper's numerical IF pipeline (identical windows)
    results["numerical_if"] = {
        k: v for k, v in anom.run_numerical_isolation_forest(windows, labels).items()
        if k in ("name", "f1", "precision", "recall")
    }

    # Old TRACQ grids: paper pipelines B and C
    results["old_image_rf"] = {
        k: v for k, v in anom.run_tracq_image_rf(windows, labels).items()
        if k in ("name", "f1", "precision", "recall")
    }
    results["old_threshold"] = {
        k: v for k, v in anom.run_tracq_threshold(windows, labels).items()
        if k in ("name", "f1", "precision", "recall", "best_feature")
    }

    # TRACQ-2 grids
    t0 = time.perf_counter()
    grids = [encode_tracq2_image(w) for w in windows]
    enc_s = time.perf_counter() - t0
    print(f"TRACQ-2 encoded {len(grids)} windows in {enc_s:.2f}s "
          f"({len(grids)/enc_s:.0f} windows/s)")

    # B': same paper image features on TRACQ-2 grids
    feats_paper = np.array([anom.extract_image_features(g) for g in grids])
    results["t2_image_rf_paper_feats"] = rf_cv(feats_paper, labels, "TRACQ-2 RF (paper feats)")

    # B'': native residual features
    t0 = time.perf_counter()
    feats_native = np.array([tracq2_native_features(g) for g in grids])
    feat_s = time.perf_counter() - t0
    results["t2_image_rf_native"] = rf_cv(feats_native, labels, "TRACQ-2 RF (native feats)")
    results["t2_image_rf_native"]["throughput_wps"] = len(grids) / (feat_s + 1e-9)

    # C': threshold search on native single features (no ML)
    names = ["dev_mean", "dev_std", "dev_max", "active_frac", "escapes",
             "rowmax_mean", "rowmax_max", "col_burst", "strong_moves", "p99"]
    farrs = {n: feats_native[:, i] for i, n in enumerate(names)}
    results["t2_threshold"] = threshold_search(farrs, labels, "TRACQ-2 Threshold (no ML)")

    # combined native+paper features
    results["t2_image_rf_all"] = rf_cv(
        np.hstack([feats_paper, feats_native]), labels, "TRACQ-2 RF (all feats)")

    # trajectory features (zero-copy cumsum shape recovery)
    t0 = time.perf_counter()
    feats_traj = np.array([tracq2_trajectory_features(g) for g in grids])
    traj_s = time.perf_counter() - t0
    tnames = ["drift_max", "drift_mean", "span_max", "span_mean",
              "mshift_max", "mshift_mean", "rough_max", "rmax", "big_moves"]
    results["t2_traj_rf"] = rf_cv(feats_traj, labels, "TRACQ-2 RF (trajectory)")
    results["t2_traj_rf"]["throughput_wps"] = len(grids) / (traj_s + 1e-9)
    results["t2_traj_threshold"] = threshold_search(
        {n: feats_traj[:, i] for i, n in enumerate(tnames)}, labels,
        "TRACQ-2 Traj Threshold (no ML)")

    # per-row (per-variable) features: same granularity as the numerical
    # pipeline's 6-per-variable stats, but computed zero-copy on the grid
    def per_row_features(grid):
        r = grid.astype(np.int64) - 128
        r[grid == lattice.ESCAPE] = 127
        a = np.abs(r).astype(np.float64)
        traj = np.cumsum(r, axis=1).astype(np.float64)
        scale = np.maximum(np.percentile(a, 90, axis=1), 1.0)
        tn = traj / scale[:, None]
        half = tn.shape[1] // 2
        f = np.column_stack([
            a.mean(axis=1),                                   # activity level
            a.max(axis=1),                                    # biggest move
            (a > 2).mean(axis=1),                             # moving fraction
            a.std(axis=1),                                    # burstiness
            tn.max(axis=1) - tn.min(axis=1),                  # trajectory span
            np.abs(tn[:, half:].mean(axis=1) - tn[:, :half].mean(axis=1)),  # level shift
        ])
        return f.ravel()

    t0 = time.perf_counter()
    feats_pr = np.array([per_row_features(g) for g in grids])
    pr_s = time.perf_counter() - t0
    results["t2_perrow_rf"] = rf_cv(feats_pr, labels, "TRACQ-2 RF (per-row feats)")
    results["t2_perrow_rf"]["throughput_wps"] = len(grids) / (pr_s + 1e-9)
    from sklearn.ensemble import IsolationForest as _IF
    fc = np.nan_to_num(feats_pr, nan=0, posinf=1e6, neginf=-1e6)
    iso_pr = _IF(contamination=0.5, random_state=42, n_estimators=100)
    p = (iso_pr.fit_predict(fc) == -1).astype(int)
    results["t2_perrow_if"] = {
        "name": "TRACQ-2 IF per-row (unsup)",
        "f1": float(f1_score(labels, p)),
        "precision": float(precision_score(labels, p)),
        "recall": float(recall_score(labels, p)),
    }

    # eps sweep for the per-row IF detector (archived-eps sensitivity)
    from sklearn.ensemble import IsolationForest as _IF2
    for det_eps in (3e-2, 1e-3):
        g2 = [encode_tracq2_image(w, eps=det_eps) for w in windows]
        fpr = np.nan_to_num(np.array([per_row_features(g) for g in g2]),
                            nan=0, posinf=1e6, neginf=-1e6)
        iso2 = _IF2(contamination=0.5, random_state=42, n_estimators=100)
        p2 = (iso2.fit_predict(fpr) == -1).astype(int)
        results[f"t2_perrow_if_eps{det_eps:g}"] = {
            "name": f"TRACQ-2 IF per-row eps={det_eps:g}",
            "f1": float(f1_score(labels, p2)),
            "precision": float(precision_score(labels, p2)),
            "recall": float(recall_score(labels, p2)),
        }

    # ensemble: per-row IF anomaly score + rowmax threshold (both unsupervised)
    scores = -iso_pr.score_samples(fc)  # higher = more anomalous
    rowmax_mean = feats_native[:, 5]
    def rank(x):
        r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r / len(x)
    combo = rank(scores) + rank(rowmax_mean)
    pens = (combo > np.median(combo)).astype(int)
    results["t2_ensemble"] = {
        "name": "TRACQ-2 IF+rowmax ensemble (unsup)",
        "f1": float(f1_score(labels, pens)),
        "precision": float(precision_score(labels, pens)),
        "recall": float(recall_score(labels, pens)),
    }

    # combined trajectory + native, RF and unsupervised IF
    both = np.hstack([feats_native, feats_traj])
    results["t2_full_rf"] = rf_cv(both, labels, "TRACQ-2 RF (native+traj)")
    from sklearn.ensemble import IsolationForest
    bothc = np.nan_to_num(both, nan=0, posinf=1e6, neginf=-1e6)
    iso = IsolationForest(contamination=0.5, random_state=42, n_estimators=100)
    ipred = (iso.fit_predict(bothc) == -1).astype(int)
    results["t2_if"] = {
        "name": "TRACQ-2 IsolationForest (unsup)",
        "f1": float(f1_score(labels, ipred)),
        "precision": float(precision_score(labels, ipred)),
        "recall": float(recall_score(labels, ipred)),
    }

    print()
    for k, r in results.items():
        extra = f"  [{r.get('feature') or r.get('best_feature') or ''}]"
        print(f"  {r['name']:34s} F1 {r['f1']:.3f}  P {r['precision']:.3f}  R {r['recall']:.3f}{extra}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()

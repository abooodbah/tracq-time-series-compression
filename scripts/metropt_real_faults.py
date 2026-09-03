# -*- coding: utf-8 -*-
"""Compressed-domain anomaly detection on MetroPT-3's REAL documented failures.

Labels come from the four air-leak failure reports in the dataset's own
documentation (Data Description_Metro.pdf), not from injected anomalies.
Both pipelines start from the same stored compressed artifact:

  A (decode-then-detect): blob -> lattice.decode -> float64 -> stat features -> IF
  B (compressed-domain):  grid -> integer grid features -> IF   (no decode)

Feature definitions and codec settings are copied from the paper's existing
scripts so the numbers are comparable to Table V.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

REPO = r"C:\Users\Abdulfatah\personal\research\tracq\tracq-time-series-compression"
sys.path.insert(0, REPO)
from tracq import lattice  # noqa: E402

CSV = os.path.join(REPO, "data", "raw", "metropt3", "MetroPT3(AirCompressor).csv")
WINDOW = 100          # matches the paper's WINDOW_SIZE
SEED = 42
EPS, MODE = 1e-2, "rel"   # matches encode_tracq2_image()

# air-leak failure reports from the dataset documentation
FAILURES = [("2020-04-18 00:00", "2020-04-18 23:59"),
            ("2020-05-29 23:30", "2020-05-30 06:00"),
            ("2020-06-05 10:00", "2020-06-07 14:30"),
            ("2020-07-15 14:30", "2020-07-15 19:00")]


def extract_numerical_features(window):
    """Paper's Pipeline-A features (anomaly_detection_experiment.py:193)."""
    feats = []
    for v in range(window.shape[0]):
        row = window[v]
        feats.extend([np.mean(row), np.std(row), np.max(row) - np.min(row),
                      kurtosis(row), np.max(np.abs(np.diff(row))),
                      np.percentile(row, 95) - np.percentile(row, 5)])
    return np.array(feats)


def tracq2_native_features(grid):
    """Paper's grid-native features (lattice_anomaly_experiment.py:40)."""
    g = grid.astype(np.float64)
    dev = np.abs(g - 128.0)
    active = dev > 2
    row_max = dev.max(axis=1)
    return np.array([dev.mean(), dev.std(), dev.max(), float(active.mean()),
                     float((grid == lattice.ESCAPE).sum()),
                     row_max.mean(), row_max.max(),
                     float(active.mean(axis=0).max()),
                     float((dev > 20).sum()), float(np.percentile(dev, 99))])


def tracq2_trajectory_features(grid):
    """Paper's trajectory features (lattice_anomaly_experiment.py:60)."""
    r = grid.astype(np.int64) - 128
    r[grid == lattice.ESCAPE] = 127
    t = np.cumsum(r, axis=1).astype(np.float64)
    half = t.shape[1] // 2
    scale = np.maximum(np.percentile(np.abs(r), 90, axis=1), 1.0)
    tn = t / scale[:, None]
    drift = np.abs(tn[:, -1])
    span = tn.max(axis=1) - tn.min(axis=1)
    shift = np.abs(tn[:, half:].mean(axis=1) - tn[:, :half].mean(axis=1))
    rough = np.abs(np.diff(np.sign(np.diff(tn, axis=1) + 1e-9), axis=1)).mean(axis=1)
    return np.array([drift.max(), drift.mean(), span.max(), span.mean(),
                     shift.max(), shift.mean(), rough.max(), rough.mean()])


def score(name, labels, preds, raw_scores, secs, n):
    return {"pipeline": name,
            "f1": f1_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "roc_auc": roc_auc_score(labels, raw_scores),
            "pr_auc": average_precision_score(labels, raw_scores),
            "win_per_s": n / secs}


def main():
    print("loading MetroPT-3 ...")
    df = pd.read_csv(CSV)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    ts = pd.to_datetime(df.pop("timestamp"))
    data = df.to_numpy(dtype=np.float64).T          # (n_vars, n_time)
    data = np.where(np.isfinite(data), data, 0.0)
    print(f"  {data.shape[0]} sensors x {data.shape[1]:,} steps")

    is_fault = np.zeros(len(ts), dtype=bool)
    for a, b in FAILURES:
        is_fault |= ((ts >= pd.Timestamp(a)) & (ts <= pd.Timestamp(b))).to_numpy()
    print(f"  {is_fault.sum():,} rows inside documented failures "
          f"({100 * is_fault.mean():.2f}%)")

    n_win = data.shape[1] // WINDOW
    windows = [data[:, i * WINDOW:(i + 1) * WINDOW] for i in range(n_win)]
    labels = np.array([int(is_fault[i * WINDOW:(i + 1) * WINDOW].any())
                       for i in range(n_win)])
    rate = labels.mean()
    print(f"  {n_win:,} non-overlapping windows of {WINDOW} steps, "
          f"{labels.sum()} positive ({100 * rate:.2f}%)\n")

    print("encoding every window once (storage step, not timed against either pipeline)")
    t0 = time.perf_counter()
    stored = []
    for w in windows:
        blob, grid, _ = lattice.encode(w.copy(), eps=EPS, mode=MODE,
                                       predictors="p1", zstd_level=3)
        stored.append((blob, grid))
    print(f"  encoded {n_win:,} windows in {time.perf_counter() - t0:.1f} s\n")

    # ---- A: decode-then-detect -------------------------------------------
    t0 = time.perf_counter()
    fa = []
    for blob, _ in stored:
        x, _ = lattice.decode(blob)
        fa.append(extract_numerical_features(x))
    fa = np.nan_to_num(np.array(fa), nan=0, posinf=1e6, neginf=-1e6)
    iso = IsolationForest(contamination=rate, random_state=SEED, n_estimators=100)
    pa = (iso.fit_predict(fa) == -1).astype(int)
    sa = -iso.score_samples(fa)
    ta = time.perf_counter() - t0

    # ---- B: compressed-domain --------------------------------------------
    t0 = time.perf_counter()
    fb = []
    for _, grid in stored:
        fb.append(np.concatenate([tracq2_native_features(grid),
                                  tracq2_trajectory_features(grid)]))
    fb = np.nan_to_num(np.array(fb), nan=0, posinf=1e6, neginf=-1e6)
    iso2 = IsolationForest(contamination=rate, random_state=SEED, n_estimators=100)
    pb = (iso2.fit_predict(fb) == -1).astype(int)
    sb = -iso2.score_samples(fb)
    tb = time.perf_counter() - t0

    rows = [score("Numerical IF (decode-then-detect)", labels, pa, sa, ta, n_win),
            score("TRACQ Direct IF (compressed domain)", labels, pb, sb, tb, n_win)]

    print(f"{'Pipeline':<38}{'F1':>7}{'Prec.':>8}{'Rec.':>7}"
          f"{'ROC':>7}{'PR':>7}{'win/s':>10}")
    for r in rows:
        print(f"{r['pipeline']:<38}{r['f1']:>7.3f}{r['precision']:>8.3f}"
              f"{r['recall']:>7.3f}{r['roc_auc']:>7.3f}{r['pr_auc']:>7.3f}"
              f"{r['win_per_s']:>10,.0f}")
    print(f"\nthroughput ratio: {rows[1]['win_per_s'] / rows[0]['win_per_s']:.1f}x")
    print(f"random-baseline PR-AUC (base rate): {rate:.4f}")


if __name__ == "__main__":
    main()

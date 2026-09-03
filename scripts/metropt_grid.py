# -*- coding: utf-8 -*-
"""Pre-specified sweep for MetroPT-3 real-fault detection.

Factors: window length, encoder tolerance, compressed feature set (with and
without the stored baseline indices m0), and detector (Isolation Forest vs
Local Outlier Factor). Every cell is reported; nothing is selected post hoc.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = r"C:\Users\Abdulfatah\personal\research\tracq\tracq-time-series-compression"
sys.path.insert(0, REPO)
from tracq import lattice  # noqa: E402

CSV = os.path.join(REPO, "data", "raw", "metropt3", "MetroPT3(AirCompressor).csv")
SEED = 42
FAILURES = [("2020-04-18 00:00", "2020-04-18 23:59"),
            ("2020-05-29 23:30", "2020-05-30 06:00"),
            ("2020-06-05 10:00", "2020-06-07 14:30"),
            ("2020-07-15 14:30", "2020-07-15 19:00")]


def numerical_features(w):
    f = []
    for v in range(w.shape[0]):
        r = w[v]
        f.extend([np.mean(r), np.std(r), np.max(r) - np.min(r), kurtosis(r),
                  np.max(np.abs(np.diff(r))),
                  np.percentile(r, 95) - np.percentile(r, 5)])
    return np.array(f)


def native_features(grid):
    g = grid.astype(np.float64)
    dev = np.abs(g - 128.0)
    active = dev > 2
    row_max = dev.max(axis=1)
    return np.array([dev.mean(), dev.std(), dev.max(), float(active.mean()),
                     float((grid == lattice.ESCAPE).sum()),
                     row_max.mean(), row_max.max(),
                     float(active.mean(axis=0).max()),
                     float((dev > 20).sum()), float(np.percentile(dev, 99))])


def trajectory_features(grid):
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


def detect(name, feats, labels):
    """Return (roc, pr, throughput-relevant fit seconds) for one detector."""
    X = np.nan_to_num(feats, nan=0, posinf=1e6, neginf=-1e6)
    rate = labels.mean()
    t0 = time.perf_counter()
    if name == "IF":
        m = IsolationForest(contamination=rate, random_state=SEED, n_estimators=100)
        m.fit(X)
        s = -m.score_samples(X)
    else:
        m = LocalOutlierFactor(n_neighbors=20, contamination=rate)
        m.fit_predict(X)
        s = -m.negative_outlier_factor_
    secs = time.perf_counter() - t0
    return roc_auc_score(labels, s), average_precision_score(labels, s), secs


def main():
    df = pd.read_csv(CSV)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    ts = pd.to_datetime(df.pop("timestamp"))
    data = df.to_numpy(dtype=np.float64).T
    data = np.where(np.isfinite(data), data, 0.0)
    fault = np.zeros(len(ts), dtype=bool)
    for a, b in FAILURES:
        fault |= ((ts >= pd.Timestamp(a)) & (ts <= pd.Timestamp(b))).to_numpy()

    print(f"{'W':>5} {'eps':>7} {'features':<26} {'det':>4} {'ROC':>7} {'PR':>7} {'lift':>6}")
    print("-" * 68)
    for W in (100, 500, 2000):
        n = data.shape[1] // W
        wins = [data[:, i * W:(i + 1) * W] for i in range(n)]
        labels = np.array([int(fault[i * W:(i + 1) * W].any()) for i in range(n)])
        rate = labels.mean()

        blobs, packs = [], {}
        for eps in (1e-2, 1e-3):
            grids, m0s, bl = [], [], []
            for w in wins:
                blob, grid, hdr = lattice.encode(w.copy(), eps=eps, mode="rel",
                                                 predictors="p1", zstd_level=3)
                grids.append(grid)
                m0s.append(np.asarray(hdr["m0"], dtype=np.float64))
                bl.append(blob)
            packs[eps] = (grids, np.array(m0s))
            if eps == 1e-2:
                blobs = bl

        num = np.array([numerical_features(lattice.decode(b)[0]) for b in blobs])

        sets = {"numerical (decode)": num}
        for eps in (1e-2, 1e-3):
            grids, m0s = packs[eps]
            base = np.array([np.concatenate([native_features(g), trajectory_features(g)])
                             for g in grids])
            sets[f"grid @{eps:g}"] = base
            sets[f"grid+m0 @{eps:g}"] = np.hstack([base, m0s])

        for fname, F in sets.items():
            for det in ("IF", "LOF"):
                roc, pr, _ = detect(det, F, labels)
                eps_lbl = fname.split("@")[-1] if "@" in fname else "-"
                print(f"{W:>5} {eps_lbl:>7} {fname.split(' @')[0]:<26} {det:>4} "
                      f"{roc:>7.3f} {pr:>7.3f} {pr / rate:>6.1f}x")
        print(f"      (n={n:,} windows, {labels.sum()} positive, base rate {rate:.4f})")
        print("-" * 68)


if __name__ == "__main__":
    main()

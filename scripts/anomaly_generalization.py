"""Zero-copy analytics generalization for the v3 revision: the same
compressed-grid detection protocol on all three UCI datasets and across
model families (Isolation Forest, Local Outlier Factor, One-Class SVM,
plus the no-ML threshold), against decoded-numerical counterparts."""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from tracq import lattice

spec = importlib.util.spec_from_file_location(
    "anom", os.path.join(PROJECT_ROOT, "scripts", "anomaly_detection_experiment.py"))
am = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am)

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

OUT = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "anomaly_generalization.json")

DATASETS = {
    "appliances": "uci_appliances_energy.csv",
    "air_quality": "uci_air_quality.csv",
    "metro": "uci_metro_traffic.csv",
}


def load(fname):
    path = os.path.join(PROJECT_ROOT, "data", "processed", fname)
    data = pd.read_csv(path, header=None).values.T.astype(np.float64)
    return np.where(np.isfinite(data), data, 0.0)[:, :5000]


def per_row_features(grid):
    r = grid.astype(np.int64) - 128
    r[grid == lattice.ESCAPE] = 127
    a = np.abs(r).astype(np.float64)
    traj = np.cumsum(r, axis=1).astype(np.float64)
    scale = np.maximum(np.percentile(a, 90, axis=1), 1.0)
    tn = traj / scale[:, None]
    half = tn.shape[1] // 2
    return np.column_stack([
        a.mean(axis=1), a.max(axis=1), (a > 2).mean(axis=1), a.std(axis=1),
        tn.max(axis=1) - tn.min(axis=1),
        np.abs(tn[:, half:].mean(axis=1) - tn[:, :half].mean(axis=1)),
    ]).ravel()


def unsup_f1(model_name, feats, labels):
    X = StandardScaler().fit_transform(np.nan_to_num(feats, nan=0, posinf=1e6, neginf=-1e6))
    if model_name == "if":
        m = IsolationForest(contamination=0.5, random_state=42, n_estimators=100)
        pred = (m.fit_predict(X) == -1).astype(int)
    elif model_name == "lof":
        m = LocalOutlierFactor(n_neighbors=20, contamination=0.5)
        pred = (m.fit_predict(X) == -1).astype(int)
    elif model_name == "ocsvm":
        m = OneClassSVM(nu=0.5, kernel="rbf", gamma="scale")
        pred = (m.fit_predict(X) == -1).astype(int)
    return float(f1_score(labels, pred))


def main():
    results = {}
    for ds, fname in DATASETS.items():
        data = load(fname)
        windows, labels, _ = am.create_labeled_dataset(data)
        grids = [lattice.encode(w, eps=3e-2, mode="rel", predictors="p1", zstd_level=3)[1]
                 for w in windows]
        feats_num = np.array([am.extract_numerical_features(w) for w in windows])
        feats_grid = np.array([per_row_features(g) for g in grids])
        row = {"n_windows": len(windows)}
        for model in ("if", "lof", "ocsvm"):
            row[f"num_{model}"] = unsup_f1(model, feats_num, labels)
            row[f"grid_{model}"] = unsup_f1(model, feats_grid, labels)
        results[ds] = row
        print(ds, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()

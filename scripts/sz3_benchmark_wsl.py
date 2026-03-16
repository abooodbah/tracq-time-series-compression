#!/usr/bin/env python3
"""
SZ3 Benchmark via hdf5plugin — designed to run in WSL where hdf5plugin is available.
Reads the 3 UCI datasets and runs SZ3 with multiple error tolerances.
Outputs results to paper_results/realworld/sz3_results.json
"""

import json
import os
import sys
import time

import numpy as np
import h5py
import hdf5plugin


def load_dataset(csv_path, max_time=5000):
    """Load a UCI CSV dataset, return (data, info)."""
    import csv
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            vals = []
            for v in row:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(np.nan)
            rows.append(vals)

    data = np.array(rows, dtype=np.float64).T  # (n_vars, n_time)
    if data.shape[1] > max_time:
        data = data[:, :max_time]

    # Drop rows that are all-NaN
    valid = ~np.all(np.isnan(data), axis=1)
    data = data[valid]

    # Fill remaining NaNs with forward-fill then zero
    for i in range(data.shape[0]):
        row = data[i]
        nans = np.isnan(row)
        if nans.any():
            # forward fill
            idx = np.where(~nans)[0]
            if len(idx) > 0:
                for j in range(len(row)):
                    if nans[j]:
                        prev = idx[idx < j]
                        if len(prev) > 0:
                            row[j] = row[prev[-1]]
                        else:
                            row[j] = 0.0
            else:
                row[:] = 0.0
            data[i] = row

    name = os.path.splitext(os.path.basename(csv_path))[0]
    zero_offset_vars = int(np.sum(data[:, 0] == 0.0))

    info = {
        "name": name,
        "n_vars": int(data.shape[0]),
        "n_time": int(data.shape[1]),
        "file": os.path.basename(csv_path),
        "zero_offset_vars": zero_offset_vars,
    }
    return data, info


def sz3_compress(data, tolerance):
    """Compress with SZ3 via hdf5plugin. Returns (compressed_bytes, encode_time)."""
    tmp_path = "/tmp/sz3_bench.h5"
    t0 = time.perf_counter()
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("data", data=data, **hdf5plugin.SZ3(absolute=tolerance))
    encode_s = time.perf_counter() - t0

    compressed_bytes = os.path.getsize(tmp_path)
    return tmp_path, compressed_bytes, encode_s


def sz3_decompress(tmp_path):
    """Decompress SZ3 from HDF5. Returns (reconstructed, decode_time)."""
    t0 = time.perf_counter()
    with h5py.File(tmp_path, "r") as f:
        recon = f["data"][:]
    decode_s = time.perf_counter() - t0
    return recon, decode_s


def compute_smape(a, b, eps=1e-9):
    """Symmetric Mean Absolute Percentage Error."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum((np.abs(a) + np.abs(b)) / 2.0, eps)
    return float(np.mean(np.abs(a - b) / denom))


def compute_metrics(original, reconstructed):
    """Compute RMSE, MAE, SMAPE, correlation, max_error."""
    diff = original - reconstructed
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    max_error = float(np.max(np.abs(diff)))
    smape = compute_smape(original, reconstructed)

    # Per-variable SMAPE
    per_var_smape = []
    for i in range(original.shape[0]):
        per_var_smape.append(compute_smape(original[i], reconstructed[i]))

    # Correlation (mean across variables)
    corrs = []
    for i in range(original.shape[0]):
        o = original[i]
        r = reconstructed[i]
        if np.std(o) > 0 and np.std(r) > 0:
            corrs.append(float(np.corrcoef(o, r)[0, 1]))
        else:
            corrs.append(1.0 if np.allclose(o, r) else 0.0)
    corr = float(np.mean(corrs))

    return {"rmse": rmse, "mae": mae, "smape": smape, "corr": corr,
            "max_error": max_error, "per_var_smape": per_var_smape}


def main():
    # Find project root (this script is in scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "processed")
    results_dir = os.path.join(project_root, "paper_results", "realworld")
    os.makedirs(results_dir, exist_ok=True)

    datasets = {
        "uci_air_quality": "uci_air_quality.csv",
        "uci_appliances_energy": "uci_appliances_energy.csv",
        "uci_metro_traffic": "uci_metro_traffic.csv",
    }

    tolerances = [0.1, 0.01, 0.001, 0.0001]

    all_results = {}

    for ds_name, ds_file in datasets.items():
        csv_path = os.path.join(data_dir, ds_file)
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping {ds_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        data, info = load_dataset(csv_path)
        print(f"  Shape: {data.shape} ({info['n_vars']} vars x {info['n_time']} time steps)")
        orig_bytes = data.nbytes
        print(f"  Original size: {orig_bytes:,} bytes ({orig_bytes/1024:.1f} KB)")

        ds_results = {}

        for tol in tolerances:
            method_name = f"sz3_abs_{tol}"
            print(f"\n  SZ3 absolute tolerance = {tol}")

            try:
                tmp_path, comp_bytes, encode_s = sz3_compress(data, tol)
                recon, decode_s = sz3_decompress(tmp_path)
                metrics = compute_metrics(data, recon)
                ratio = comp_bytes / orig_bytes

                # Separate per_var_smape (list) from scalar metrics for rounding
                per_var = metrics.pop("per_var_smape", [])
                rounded_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        rounded_metrics[k] = round(v, 10) if abs(v) < 1e10 else v
                    else:
                        rounded_metrics[k] = v
                rounded_metrics["per_var_smape"] = [round(x, 10) for x in per_var]

                ds_results[method_name] = {
                    "category": "HPC",
                    "bytes": comp_bytes,
                    "ratio": round(ratio, 6),
                    "encode_s": round(encode_s, 6),
                    "decode_s": round(decode_s, 6),
                    "metrics": rounded_metrics,
                    "params": {"absolute_tolerance": tol},
                }

                print(f"    Ratio: {ratio:.4f} | RMSE: {metrics['rmse']:.6f} | "
                      f"SMAPE: {metrics['smape']:.6f} | Corr: {metrics['corr']:.6f}")
                print(f"    Encode: {encode_s:.4f}s | Decode: {decode_s:.4f}s")

                # Clean up
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            except Exception as e:
                print(f"    ERROR: {e}")
                ds_results[method_name] = {"category": "HPC", "error": str(e)}

        all_results[ds_name] = {"info": info, "results": ds_results}

    # Save results
    out_path = os.path.join(results_dir, "sz3_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {out_path}")

    # Also print summary for easy copy to paper
    print("\n" + "=" * 80)
    print("SUMMARY FOR PAPER TABLE")
    print("=" * 80)
    print(f"{'Dataset':<25} {'Method':<15} {'Ratio':>8} {'RMSE':>12} {'SMAPE':>10} {'Corr':>8}")
    print("-" * 80)
    for ds_name, ds_data in all_results.items():
        for method, res in ds_data["results"].items():
            if "error" in res:
                print(f"{ds_name:<25} {method:<15} {'ERROR':>8}")
            else:
                m = res["metrics"]
                print(f"{ds_name:<25} {method:<15} {res['ratio']:>8.4f} "
                      f"{m['rmse']:>12.6f} {m['smape']:>10.6f} {m['corr']:>8.6f}")


if __name__ == "__main__":
    main()

"""Winner analysis: Pareto fronts of TRACQ-2 candidates vs baselines on UCI data."""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAT = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "lattice_results.json")
BASE = os.path.join(PROJECT_ROOT, "paper_results", "realworld")

BASELINE_KEYS = ["gzip", "delta_zstd", "tracq_enh_8bit_anchors", "tracq_enh_16bit_anchors",
                 "paa", "sax", "gorilla_like", "zfp_tol_0.1", "zfp_tol_0.01",
                 "zfp_tol_0.001", "zfp_tol_0.0001"]


def load_points(ds):
    pts = []
    with open(LAT) as f:
        lat = json.load(f)[ds]
    for k, r in lat.items():
        pts.append((k, r["ratio"], r["rmse"], r["smape"], r["candidate"]))
    with open(os.path.join(BASE, ds + "_results.json")) as f:
        bj = json.load(f)["results"]
    for k in BASELINE_KEYS:
        r = bj.get(k)
        if r and "metrics" in r:
            pts.append((k, r["ratio"], r["metrics"]["rmse"], r["metrics"].get("smape", 0), "baseline"))
    return pts


def pareto(pts, err_idx):
    """Points not dominated in (ratio, error) space."""
    front = []
    for p in pts:
        dominated = any(
            (o[1] <= p[1] and o[err_idx] <= p[err_idx]) and (o[1] < p[1] or o[err_idx] < p[err_idx])
            for o in pts
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p[1])


def main():
    wins = {}
    for ds in ["uci_air_quality", "uci_appliances_energy", "uci_metro_traffic"]:
        pts = [p for p in load_points(ds) if p[2] > 0 or "zstd" in p[0] or "gzip" in p[0]]
        print(f"\n=== {ds}: RMSE-ratio Pareto front ===")
        for name, ratio, rmse, smape_v, cand in pareto(pts, 2):
            marker = " <-- TRACQ-2" if cand != "baseline" else ""
            print(f"  {name:40s} ratio {ratio:8.4f}  RMSE {rmse:12.5g}{marker}")
            wins.setdefault(ds, {"front": 0, "tracq2": 0})
            wins[ds]["front"] += 1
            wins[ds]["tracq2"] += int(cand != "baseline")
        print(f"--- {ds}: SMAPE-ratio Pareto front ---")
        for name, ratio, rmse, smape_v, cand in pareto([p for p in pts if p[3] is not None], 3):
            marker = " <-- TRACQ-2" if cand != "baseline" else ""
            print(f"  {name:40s} ratio {ratio:8.4f}  SMAPE {smape_v:8.5f}{marker}")

    print("\nSummary: share of RMSE-ratio Pareto front held by TRACQ-2:")
    for ds, w in wins.items():
        print(f"  {ds}: {w['tracq2']}/{w['front']}")


if __name__ == "__main__":
    main()

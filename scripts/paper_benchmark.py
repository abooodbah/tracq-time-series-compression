"""Paper-style benchmark runner.

- Downloads a small set of public time-series datasets (optional)
- Converts them to the repo's numeric CSV format (headerless)
- Runs a consolidated benchmark with TRACQ sweep+auto-selection plus baselines
- Writes per-dataset JSON and an aggregate CSV for paper tables/plots

Example:
  conda run -n gtc-env python scripts/paper_benchmark.py --prepare-data --run --target-rmse 25 --outdir paper_results

Notes:
- We intentionally do NOT vendor datasets into the repo.
- Network access may be required for --prepare-data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional


# Ensure repo root importable when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracq.tooling import full_benchmark


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    # If provided, the extracted file to read (relative inside the zip)
    inner_path: Optional[str] = None
    # Pandas read_csv kwargs
    sep: str = ","
    decimal: str = "."


DATASETS: Dict[str, DatasetSpec] = {
    # UCI Air Quality (multivariate time series)
    "uci_air_quality": DatasetSpec(
        name="uci_air_quality",
        url="https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        inner_path="AirQualityUCI.csv",
        sep=";",
        decimal=",",
    ),
    # UCI Appliances energy prediction (multivariate time series)
    "uci_appliances_energy": DatasetSpec(
        name="uci_appliances_energy",
        url="https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
        inner_path="energydata_complete.csv",
        sep=",",
        decimal=".",
    ),
    # UCI Metro Interstate Traffic Volume (multivariate time series)
    "uci_metro_traffic": DatasetSpec(
        name="uci_metro_traffic",
        url="https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip",
        # The UCI zip contains a single gzipped CSV.
        inner_path="Metro_Interstate_Traffic_Volume.csv.gz",
        sep=",",
        decimal=".",
    ),
}


def _download(url: str, out_path: Path) -> None:
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(out_path)


def prepare_dataset(spec: DatasetSpec, raw_dir: Path, processed_dir: Path, max_rows: Optional[int]) -> Path:
    """Download + convert to headerless numeric CSV. Returns processed CSV path."""
    raw_zip = raw_dir / f"{spec.name}.zip"
    if not raw_zip.exists():
        _download(spec.url, raw_zip)

    extract_dir = raw_dir / spec.name
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip, "r") as zf:
        zf.extractall(extract_dir)

    in_path = extract_dir / spec.inner_path if spec.inner_path else extract_dir
    if spec.inner_path and not in_path.exists():
        # Be a bit forgiving: some UCI zips contain a .csv.gz or have slightly different paths.
        alt_gz = None
        if spec.inner_path.lower().endswith(".csv"):
            alt_gz = extract_dir / (spec.inner_path + ".gz")

        candidates: List[Path] = []
        candidates.extend(extract_dir.rglob("*.csv"))
        candidates.extend(extract_dir.rglob("*.csv.gz"))
        candidates = [p for p in candidates if p.is_file()]

        if alt_gz is not None and alt_gz.exists():
            in_path = alt_gz
        elif len(candidates) == 1:
            in_path = candidates[0]
        else:
            sample = "\n".join(str(p.relative_to(extract_dir)) for p in candidates[:20])
            raise FileNotFoundError(
                f"Expected {spec.inner_path} inside {raw_zip} but it was not found. "
                f"Found {len(candidates)} CSV-like files under {extract_dir}:\n{sample}"
            )

    try:
        import pandas as pd
    except Exception as e:
        raise SystemExit("pandas is required for dataset preparation") from e

    df = pd.read_csv(in_path, sep=spec.sep, decimal=spec.decimal, low_memory=False)

    # Drop non-numeric columns (timestamps, categorical, etc.)
    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] == 0:
        # Try coercion as a fallback
        coerced = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        num = coerced.select_dtypes(include=["number"]).copy()

    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {spec.name}")

    # Fill missing values and drop fully-missing columns
    num = num.dropna(axis=1, how="all")
    num = num.ffill().bfill()

    if max_rows is not None:
        num = num.iloc[: int(max_rows)]

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_csv = processed_dir / f"{spec.name}.csv"
    num.to_csv(out_csv, index=False, header=False)
    return out_csv


def flatten_results_to_rows(dataset_name: str, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    orig_bytes = int(stats["orig_bytes"])
    for method, r in (stats.get("runs") or {}).items():
        m = r.get("metrics") or {}
        row = {
            "dataset": dataset_name,
            "method": method,
            "bytes": r.get("bytes"),
            "ratio": r.get("ratio"),
            "encode_s": r.get("encode_s"),
            "decode_s": r.get("decode_s"),
            "total_s": r.get("total_s"),
            "rmse": m.get("rmse"),
            "mae": m.get("mae"),
            "mape": m.get("mape"),
            "smape": m.get("smape"),
            "corr": m.get("corr"),
            "orig_bytes": orig_bytes,
        }
        if method == "tracq_best":
            row["tracq_variant"] = r.get("variant")
            row["tracq_params"] = json.dumps(r.get("params"), separators=(",", ":")) if r.get("params") else ""
            row["tracq_selection"] = json.dumps(r.get("selection"), separators=(",", ":")) if r.get("selection") else ""
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run paper-style benchmarks across multiple datasets")
    ap.add_argument("--outdir", type=Path, default=Path("paper_results"))
    ap.add_argument("--datasets", nargs="*", default=["uci_air_quality", "uci_appliances_energy", "uci_metro_traffic"], help="Dataset keys or 'all'")
    ap.add_argument("--prepare-data", action="store_true", help="Download/prepare datasets into data/processed")
    ap.add_argument("--run", action="store_true", help="Run benchmarks")
    ap.add_argument("--no-prepare-missing", action="store_true", help="Do not auto-download/prepare missing processed CSVs during --run")
    ap.add_argument("--max-rows", type=int, default=5000, help="Max rows per dataset during preparation (keeps runs manageable)")

    # TRACQ sweep/selection
    ap.add_argument("--target-rmse", type=float, default=25.0, help="Pick smallest bytes with RMSE <= target")
    ap.add_argument("--target-ratio", type=float, default=None, help="Pick lowest RMSE with ratio <= target (set to use instead of target-rmse)")
    ap.add_argument("--bits", nargs="*", type=int, default=[8, 16])
    ap.add_argument("--clamps", nargs="*", type=float, default=[200.0, 500.0])
    ap.add_argument("--png-levels", nargs="*", type=int, default=[0, 3, 6])
    ap.add_argument("--zstd-levels", nargs="*", type=int, default=[1, 3, 6])

    # Baselines
    ap.add_argument("--segments", type=int, default=64)
    ap.add_argument("--alphabet", type=int, default=8)
    ap.add_argument("--no-baselines", action="store_true")

    args = ap.parse_args(argv)

    raw_dir = REPO_ROOT / "data" / "raw"
    processed_dir = REPO_ROOT / "data" / "processed"

    dataset_keys = args.datasets
    if len(dataset_keys) == 1 and dataset_keys[0].lower() == "all":
        dataset_keys = list(DATASETS.keys())

    prepared: Dict[str, Path] = {}
    if args.prepare_data:
        for k in dataset_keys:
            if k not in DATASETS:
                raise SystemExit(f"Unknown dataset key: {k}")
            out_csv = prepare_dataset(DATASETS[k], raw_dir, processed_dir, args.max_rows)
            prepared[k] = out_csv

    if not args.run:
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    all_stats: Dict[str, Any] = {}

    for k in dataset_keys:
        if k in prepared:
            csv_path = prepared[k]
        else:
            csv_path = processed_dir / f"{k}.csv"
            if not csv_path.exists():
                if (not args.no_prepare_missing) and (k in DATASETS):
                    csv_path = prepare_dataset(DATASETS[k], raw_dir, processed_dir, args.max_rows)
                else:
                    raise SystemExit(
                        f"Missing processed CSV for {k}: {csv_path}. Run with --prepare-data, or provide your own CSV in data/processed/{k}.csv"
                    )

        stats = full_benchmark(
            str(csv_path),
            segments=args.segments,
            alphabet=args.alphabet,
            tracq_sweep_mode=True,
            tracq_sweep_bits=tuple(args.bits),
            tracq_sweep_clamps=tuple(args.clamps),
            tracq_sweep_png_levels=tuple(args.png_levels),
            tracq_sweep_zstd_levels=tuple(args.zstd_levels),
            tracq_target_rmse=None if args.target_ratio is not None else float(args.target_rmse),
            tracq_target_ratio=float(args.target_ratio) if args.target_ratio is not None else None,
            include_baselines=not args.no_baselines,
        )

        out_json = args.outdir / f"{k}.json"
        out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        all_stats[k] = stats
        all_rows.extend(flatten_results_to_rows(k, stats))

    # Write aggregate JSON
    (args.outdir / "all_results.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")

    # Write aggregate CSV
    out_csv = args.outdir / "summary.csv"
    if all_rows:
        # Some fields (tracq_*) only exist for tracq_best rows; write a stable superset.
        preferred = [
            "dataset",
            "method",
            "bytes",
            "ratio",
            "encode_s",
            "decode_s",
            "total_s",
            "rmse",
            "mae",
            "mape",
            "smape",
            "corr",
            "orig_bytes",
            "tracq_variant",
            "tracq_params",
            "tracq_selection",
        ]
        extra = sorted({k for r in all_rows for k in r.keys()} - set(preferred))
        fieldnames = preferred + extra
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in all_rows:
                w.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

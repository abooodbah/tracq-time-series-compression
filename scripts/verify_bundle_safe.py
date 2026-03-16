#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_command(label: str, command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": "\n".join(completed.stdout.strip().splitlines()[-20:]),
        "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-20:]),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def expect_close(name: str, actual: float, expected: float, tolerance: float) -> dict[str, Any]:
    diff = abs(actual - expected)
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "tolerance": tolerance,
        "difference": diff,
        "passed": diff <= tolerance,
    }


def lookup_row(rows: list[dict[str, str]], **filters: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    raise KeyError(f"Missing row for filters={filters}")


def compare_committed_results() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    method_rows = read_csv_rows(REPO_ROOT / "paper_results" / "enhanced_figures" / "method_comparison_table.csv")
    base_8b = lookup_row(method_rows, method="Base TRACQ 8-bit")
    enh_8b = lookup_row(method_rows, method="Enhanced TRACQ 8-bit")

    checks.append(expect_close("tradeoff.base_ratio_pct", 100.0 * float(base_8b["compression_ratio"]), 3.1, 0.2))
    checks.append(expect_close("tradeoff.base_rmse", float(base_8b["rmse"]), 34.2, 0.2))
    checks.append(expect_close("tradeoff.enh_ratio_pct", 100.0 * float(enh_8b["compression_ratio"]), 12.2, 0.2))
    checks.append(expect_close("tradeoff.enh_rmse", float(enh_8b["rmse"]), 0.52, 0.05))

    realworld_rows = read_csv_rows(REPO_ROOT / "paper_results" / "realworld" / "summary.csv")
    aq_anchor = lookup_row(realworld_rows, dataset="uci_air_quality", method="tracq_enh_8bit_anchors")
    app_enh = lookup_row(realworld_rows, dataset="uci_appliances_energy", method="tracq_enh_8bit")
    app_anchor = lookup_row(realworld_rows, dataset="uci_appliances_energy", method="tracq_enh_8bit_anchors")
    metro_anchor = lookup_row(realworld_rows, dataset="uci_metro_traffic", method="tracq_enh_8bit_anchors")

    checks.append(expect_close("realworld.air_quality_anchor_rmse", float(aq_anchor["rmse"]), 108.3, 0.5))
    checks.append(expect_close("realworld.air_quality_anchor_ratio", float(aq_anchor["ratio"]), 0.114, 0.002))
    checks.append(expect_close("realworld.appliances_enh_rmse", float(app_enh["rmse"]), 22.0, 0.2))
    checks.append(expect_close("realworld.appliances_enh_ratio", float(app_enh["ratio"]), 0.077, 0.002))
    checks.append(expect_close("realworld.appliances_anchor_rmse", float(app_anchor["rmse"]), 7.35, 0.1))
    checks.append(expect_close("realworld.appliances_anchor_ratio", float(app_anchor["ratio"]), 0.082, 0.002))
    checks.append(expect_close("realworld.metro_anchor_rmse", float(metro_anchor["rmse"]), 180.3, 0.5))
    checks.append(expect_close("realworld.metro_anchor_ratio", float(metro_anchor["ratio"]), 0.057, 0.002))

    metro_rows = read_csv_rows(REPO_ROOT / "paper_results" / "bigdata_rd" / "metropt3_rate_distortion.csv")
    metro_tracq = lookup_row(metro_rows, method="tracq_enh_8bit_anchors_100")
    metro_paa = lookup_row(metro_rows, method="paa_1024")

    checks.append(expect_close("metropt3.anchor100_rmse", float(metro_tracq["rmse"]), 1.71, 0.05))
    checks.append(expect_close("metropt3.anchor100_ratio", float(metro_tracq["ratio"]), 0.0297, 0.001))
    checks.append(expect_close("metropt3.anchor100_smape", float(metro_tracq["smape"]), 0.379, 0.01))
    checks.append(expect_close("metropt3.anchor100_corr", float(metro_tracq["corr"]), 0.994, 0.002))
    checks.append(expect_close("metropt3.paa1024_rmse", float(metro_paa["rmse"]), 1.55, 0.05))
    checks.append(expect_close("metropt3.paa1024_ratio", float(metro_paa["ratio"]), 0.0007, 0.0001))

    missing = [
        {
            "name": "anomaly_table",
            "passed": None,
            "reason": (
                "The upload bundle does not include data/processed or the anomaly JSON results, "
                "so the F1/throughput claims cannot be recomputed from this bundle alone."
            ),
        }
    ]

    return {
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
        "not_recomputed": missing,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Safely verify the GitHub upload bundle without touching paper outputs.")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "verification_runs" / "latest")
    args = ap.parse_args(argv)

    outdir = args.outdir.resolve()
    generated_dir = outdir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    runs: list[dict[str, Any]] = []

    runs.append(run_command("pytest", [python, "-m", "pytest"], REPO_ROOT))
    runs.append(run_command("tracq_help", [python, "-m", "tracq", "--help"], REPO_ROOT))
    runs.append(run_command("benchmark_help", [python, "benchmark.py", "--help"], REPO_ROOT))
    runs.append(
        run_command(
            "zero_crossing",
            [python, "scripts/zero_crossing_experiment.py", "--output-dir", str(generated_dir / "zero_crossing")],
            REPO_ROOT,
        )
    )
    runs.append(
        run_command(
            "visual_demo",
            [python, "scripts/visual_inspection_demo.py", "--output-dir", str(generated_dir / "visual_demo")],
            REPO_ROOT,
        )
    )

    data_dir = REPO_ROOT / "data" / "processed"
    if (data_dir / "uci_appliances_energy.csv").exists():
        runs.append(
            run_command(
                "anomaly_detection",
                [
                    python,
                    "scripts/anomaly_detection_experiment.py",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(generated_dir / "anomaly_detection"),
                    "--figure-dir",
                    str(generated_dir / "figures"),
                ],
                REPO_ROOT,
            )
        )
    else:
        runs.append(
            {
                "label": "anomaly_detection",
                "command": [],
                "returncode": None,
                "stdout_tail": "",
                "stderr_tail": "",
                "skipped": True,
                "reason": "data/processed/uci_appliances_energy.csv is not present in the upload bundle",
            }
        )

    comparisons = compare_committed_results()
    report = {
        "repo_root": str(REPO_ROOT),
        "outdir": str(outdir),
        "generated_dir": str(generated_dir),
        "runs": runs,
        "comparisons": comparisons,
        "all_commands_passed": all(run.get("returncode") in (0, None) for run in runs),
    }

    (outdir / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_lines = [
        "# Safe Verification Report",
        "",
        f"Bundle root: `{REPO_ROOT}`",
        f"Generated outputs: `{generated_dir}`",
        "",
        "## Command Runs",
    ]
    for run in runs:
        if run.get("skipped"):
            summary_lines.append(f"- `{run['label']}`: skipped ({run['reason']})")
        else:
            status = "passed" if run["returncode"] == 0 else f"failed ({run['returncode']})"
            summary_lines.append(f"- `{run['label']}`: {status}")
    summary_lines.append("")
    summary_lines.append("## Paper Claim Checks")
    for check in comparisons["checks"]:
        status = "passed" if check["passed"] else "failed"
        summary_lines.append(
            f"- `{check['name']}`: {status} "
            f"(actual={check['actual']}, expected={check['expected']}, tol={check['tolerance']})"
        )
    for item in comparisons["not_recomputed"]:
        summary_lines.append(f"- `{item['name']}`: not recomputed ({item['reason']})")

    (outdir / "verification_report.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote verification report to {outdir}")
    return 0 if report["all_commands_passed"] and comparisons["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

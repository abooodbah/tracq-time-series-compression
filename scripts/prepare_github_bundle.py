#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = PROJECT_ROOT / "github_upload_bundle"

ROOT_CODE_FILES = [
    "README.md",
    "REPRODUCE_RESULTS.md",
    "requirements.txt",
    "benchmark.py",
    "viewer.py",
    "prepare_artifacts.py",
    "tracq_algorithm_diagram.py",
]

ROOT_DIAGRAM_FILES = [
    "tracq_algorithm_diagram.png",
]

RESULT_ROOTS = [
    PROJECT_ROOT / "paper_results",
    PROJECT_ROOT / "bench_results",
]

ROOT_RESULT_FILES = [
    "uci_electricity_results.csv",
]

CODE_DIRS = [
    PROJECT_ROOT / "tracq",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "tests",
]

MANUSCRIPT_FILES = [
    PROJECT_ROOT / "paper_submission" / "main.tex",
    PROJECT_ROOT / "paper_submission" / "references.bib",
]

EXCLUDED_SUBSTRINGS = (
    "backup before name change",
    "full length paper",
)

EXCLUDED_NAME_PREFIXES = (
    "tmp",
)

INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def copy_file(src: Path, bundle_root: Path) -> None:
    rel = src.relative_to(PROJECT_ROOT)
    dst = bundle_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src_dir: Path, bundle_root: Path) -> None:
    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue
        if any(part in {"__pycache__", ".pytest_cache", ".venv", ".pytest_tmp", ".pytest-tmp"} for part in path.parts):
            continue
        copy_file(path, bundle_root)


def parse_manuscript_figures(tex_path: Path) -> list[Path]:
    text = tex_path.read_text(encoding="utf-8")
    figures: list[Path] = []
    for raw_ref in INCLUDEGRAPHICS_RE.findall(text):
        ref = raw_ref.strip()
        if not ref:
            continue
        ref_path = tex_path.parent / ref
        if ref_path.suffix:
            candidates = [ref_path]
        else:
            candidates = [ref_path.with_suffix(ext) for ext in (".png", ".pdf", ".jpg", ".jpeg")]

        for candidate in candidates:
            if candidate.exists():
                figures.append(candidate)
                break
    return sorted(dict.fromkeys(figures))


def iter_result_csvs() -> Iterable[Path]:
    seen: set[Path] = set()

    for root in RESULT_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            rel_parts = [part.lower() for part in path.relative_to(root).parts]
            joined_rel = "/".join(rel_parts)
            if any(fragment in joined_rel for fragment in EXCLUDED_SUBSTRINGS):
                continue
            if path.name.lower().startswith(EXCLUDED_NAME_PREFIXES):
                continue
            if path not in seen:
                seen.add(path)
                yield path

    for name in ROOT_RESULT_FILES:
        path = PROJECT_ROOT / name
        if path.exists() and path not in seen:
            seen.add(path)
            yield path


def write_bundle_readme(bundle_root: Path, copied_code_dirs: list[str], copied_figures: int, copied_csvs: int) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    readme = bundle_root / "README_UPLOAD.md"
    readme.write_text(
        "\n".join(
            [
                "# GitHub Upload Bundle",
                "",
                f"Generated from the working repo on {timestamp}.",
                "",
                "Contents:",
                "- Source code under `tracq/`, `scripts/`, and `tests/`.",
                "- Manuscript sources in `paper_submission/`.",
                "- Final manuscript figures referenced by `paper_submission/main.tex`.",
                "- Result CSV files copied from `paper_results/`, `bench_results/`, and selected root-level outputs.",
                "",
                "Included code directories:",
                *[f"- `{name}/`" for name in copied_code_dirs],
                "",
                f"Figure files copied: {copied_figures}",
                f"Result CSV files copied: {copied_csvs}",
                "",
                "Excluded on purpose:",
                "- Raw datasets and processed data snapshots.",
                "- Virtual environments, caches, temporary TeX files, and backup manuscript folders.",
                "- Large binary archives and duplicate legacy release trees.",
                "",
                "If you need to regenerate this bundle, run:",
                "`python scripts/prepare_github_bundle.py`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_gitignore(bundle_root: Path) -> None:
    (bundle_root / ".gitignore").write_text(
        "\n".join(
            [
                "__pycache__/",
                ".pytest_cache/",
                ".venv/",
                "*.pyc",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare a clean GitHub upload bundle with code, figures, and result CSVs.")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--force", action="store_true", help="Replace the output directory if it already exists.")
    args = ap.parse_args(argv)

    outdir = args.outdir.resolve()
    if outdir.exists():
        if not args.force:
            raise SystemExit(f"{outdir} already exists. Use --force to rebuild it.")
        shutil.rmtree(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    copied_code_dirs: list[str] = []
    for src_dir in CODE_DIRS:
        if src_dir.exists():
            copy_tree(src_dir, outdir)
            copied_code_dirs.append(src_dir.name)

    for name in ROOT_CODE_FILES:
        path = PROJECT_ROOT / name
        if path.exists():
            copy_file(path, outdir)

    for name in ROOT_DIAGRAM_FILES:
        path = PROJECT_ROOT / name
        if path.exists():
            copy_file(path, outdir)

    for path in MANUSCRIPT_FILES:
        if path.exists():
            copy_file(path, outdir)

    figure_paths = parse_manuscript_figures(PROJECT_ROOT / "paper_submission" / "main.tex")
    for path in figure_paths:
        copy_file(path, outdir)

    result_csvs = list(iter_result_csvs())
    for path in result_csvs:
        copy_file(path, outdir)

    write_bundle_readme(outdir, copied_code_dirs, len(figure_paths) + sum((PROJECT_ROOT / name).exists() for name in ROOT_DIAGRAM_FILES), len(result_csvs))
    write_gitignore(outdir)

    print(f"Created GitHub bundle at {outdir}")
    print(f"Copied code directories: {', '.join(copied_code_dirs)}")
    print(f"Copied figure files: {len(figure_paths) + sum((PROJECT_ROOT / name).exists() for name in ROOT_DIAGRAM_FILES)}")
    print(f"Copied result CSV files: {len(result_csvs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

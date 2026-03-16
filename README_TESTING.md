# Testing

This public repository contains the TRACQ code and tests. Manuscript sources and precomputed paper artifacts are intentionally not included in the public repo during review.

## 1) Create an environment

Use Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run unit tests

```bash
python -m pytest tests/test_tracq.py
```

To run the full test suite:

```bash
python -m pytest
```

## 3) Basic CLI sanity check

```bash
python -m tracq compress path/to/data.csv --bits 8 --force
python -m tracq decompress path/to/data.tracq.png --force
```

## 4) Recreate benchmark-style results

For the main consolidated benchmark on a prepared CSV:

```bash
python benchmark.py path/to/data.csv --json
```

## Notes

- Compression ratio and reconstruction error metrics should match closely across machines. `encode_s`, `decode_s`, throughput, and wall-clock runtime may differ across machines.
- If you want to compare throughput numbers directly, record the host machine details with the run: CPU model, logical core count, RAM, OS, Python version, and optional libraries such as `zfpy`, `hdf5plugin`, and `pyarrow`.
- The upload bundle does not include `data/processed/` or raw datasets. Data-dependent scripts such as anomaly detection and full dataset-preparation flows require those inputs to be provided separately.
- Reference machine for manuscript throughput numbers: 13th Gen Intel(R) Core(TM) i7-13620H @ 2.40 GHz, 16.0 GB RAM, 64-bit Windows on an x64-based processor.
- Some reproduction scripts may require optional third-party compressors or external dataset downloads.

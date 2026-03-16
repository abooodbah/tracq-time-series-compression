# TRACQ

TRACQ compresses multivariate time-series into compact PNGs by storing per-step percentage changes. It supports 8-bit and 16-bit quantization, embeds reconstruction metadata in the PNG, and can reconstruct the original series with small error. A CLI is provided for compression/decompression/visualization plus a small CSV vs gzip/parquet benchmark.

## Install

Use any Python 3.9+ environment.

```bash
pip install -r requirements.txt
```

## CLI usage

All commands use the Typer CLI; run via `python -m tracq ...`.

- Compress CSV → TRACQ PNG:
  ```bash
  python -m tracq compress data.csv --bits 8 --clamp 500 --force
  ```
  Options: `--auto-tune` (search bits), `--max-rmse` target when auto-tuning, `--rgb` experimental 3-channel packing.

- Decompress PNG → CSV:
  ```bash
  python -m tracq decompress data.tracq.png --original data.csv
  ```
  Options: `--assume-baseline-zero` or `--manual-baseline "1.0,1.0"` if metadata was stripped; `--force` to overwrite.

- Visualize percent-change heatmap:
  ```bash
  python -m tracq view data.tracq.png --colormap viridis --force
  ```

- Quick size benchmark (CSV vs gzip/parquet):
  ```bash
  python -m tracq.benchmark data.csv --json
  ```

## Python API (lightweight)

```python
import numpy as np
from tracq import TimeSeriesGrid

data = np.random.rand(3, 100) * 100
q8, meta = TimeSeriesGrid(data).quantize_8bit()
recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q8, meta)
```

## Tests

```bash
python -m pytest
```

## Notes

- CSV reader prefers pandas for headers; without pandas it assumes rows=time, columns=variables.
- Parquet writing in the benchmark requires `pyarrow` (included in requirements).
- Matplotlib is optional for nicer heatmaps; fallback colors are used if unavailable.
- Compression ratio and reconstruction error metrics should reproduce closely across machines; throughput and wall-clock timing are hardware-dependent.
- Manuscript sources and precomputed paper artifacts are intentionally not included in the public repository during review.

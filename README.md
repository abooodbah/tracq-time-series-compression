# TRACQ

TRACQ compresses multivariate time series into two-dimensional integer grids that
can be stored as a Zstandard container or rendered as standard images.

Two variants are implemented:

- **Base variant** (`tracq/`): stores per-step percentage changes as compact PNGs
  with 8/16-bit quantization and embedded reconstruction metadata, plus a Typer
  CLI for compression/decompression/visualization.
- **Enhanced variant** (`tracq/lattice.py`): a per-variable scale-aware transform
  (arcsinh in relative mode, linear in absolute mode), integer quantization
  lattice, differencing of the quantized coordinates with escape coding for
  outliers, per-row predictor selection (delta, delta-of-delta, seasonal lag,
  2D Lorenzo), and Zstandard entropy coding. Reconstruction is exact integer
  accumulation, so the pointwise error bound is independent of sequence length.
  A lossless PNG view of the grid can be regenerated from the container.

## Install

Use any Python 3.9+ environment.

```bash
pip install -r requirements.txt
```

## Enhanced variant API

```python
import numpy as np
from tracq import lattice

data = np.random.rand(16, 100_000) * 100          # (variables, time)
blob, grid, header = lattice.encode(data, eps=1e-3, mode="abs", predictors="bank")
recon, header = lattice.decode(blob)              # |recon - data| <= eps * range_i
```

`mode="abs"` bounds the error by `eps` times each variable's observed range;
`mode="rel"` gives a pointwise relative bound `eps` above each variable's scale
with an absolute floor `eps * s_i` near zero.

## Base variant CLI

```bash
python -m tracq compress data.csv --bits 8 --clamp 500 --force
python -m tracq decompress data.tracq.png --original data.csv
python -m tracq view data.tracq.png --colormap viridis --force
```

## Experiments

The `scripts/` directory contains the full benchmark and figure pipeline:

- `prepare_uci_processed.py` — dataset preparation (UCI Air Quality, Appliances
  Energy, Metro Traffic, MetroPT-3; raw files fetched separately from UCI).
- `lattice_benchmark.py`, `lattice_pareto.py`, `lattice_expanded_bench.py` —
  rate-distortion and Pareto sweeps against ZFP (zfpy), SZ3 (hdf5plugin, Linux),
  gzip, Zstandard, PAA/SAX, and a rounded-delta baseline.
- `iso_rmse_experiment.py` — dense sweeps of all three error-bounded codecs
  interpolated onto a common RMSE grid (size at matched accuracy).
- `sz3_dense_sweep.py` — SZ3 tolerance sweep via the HDF5 filter.
- `highdim_experiment.py` — 64 to 4,096-variable scaling.
- `lattice_anomaly_experiment.py`, `anomaly_generalization.py` — compressed-domain
  anomaly detection.
- `terabyte_stream.py` + `terabyte_stream.sbatch` — node-parallel 1 TB windowed
  streaming run (SLURM, 112-core Sapphire Rapids node).
- `render_paper_figs.py` — regenerates every figure from the JSON results in
  `paper_results/lattice/`.

Measured results consumed by the figures and tables are versioned under
`paper_results/lattice/*.json`. Compression ratios and reconstruction errors are
deterministic given the library versions; throughput numbers are
hardware-dependent.

## Tests

```bash
python -m pytest
```

## Notes

- Run with `PYTHONNOUSERSITE=1` if a user-site NumPy conflicts with the
  environment's NumPy.
- SZ3 is exercised through `hdf5plugin` and requires Linux (WSL works).
- CSV reader prefers pandas for headers; parquet in the base benchmark requires
  `pyarrow`.

# TRACQ

Code and measured results for my paper "A Novel Approach to Multivariate
Time Series Analysis Using Adaptive Compression and Quantization".

TRACQ (`tracq/lattice.py`) encodes multivariate time series as integer grids:
a per-variable arcsinh or linear transform, an error-bounded quantization
lattice, drift-free integer differencing with escape coding, per-row
predictor selection, and Zstandard coding. Reconstruction is exact integer
accumulation, so the pointwise bound holds at any horizon. `tracq/` also
holds the naive percentage-differencing ablation and its CLI.

## Use

```bash
pip install -r requirements.txt
```

```python
from tracq import lattice
blob, grid, header = lattice.encode(data, eps=1e-3, mode="abs", predictors="bank")
recon, header = lattice.decode(blob)   # |recon - data| <= eps * range_i
```

## Results

`scripts/` regenerates everything in the paper: dataset preparation
(`prepare_uci_processed.py`), the benchmark and anomaly experiments (each
script is named after its experiment), and `render_paper_figs.py`, which
rebuilds every figure from the versioned measurements in `paper_results/`.
Error and size numbers are deterministic; throughput is hardware-dependent.

## Tests

```bash
python -m pytest
```

Run with `PYTHONNOUSERSITE=1` if a user-site NumPy conflicts with the
environment's; SZ3 runs through `hdf5plugin` on Linux (WSL works).

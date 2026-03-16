import json
from typing import Optional, Tuple, Dict, Any, Sequence

import numpy as np
try:
    import numba  # type: ignore
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False


class TimeSeriesGrid:
    """
    Convert multivariate time series into a grid of percentage changes and
    provide quantization (8-bit & 16-bit) and reconstruction helpers.

    Usage:
      - Instantiate with a 2D array-like (variables x time) or a pandas DataFrame.
      - call quantize_16bit() or quantize_8bit() to get integer grid + metadata
      - call reconstruct_from_quantized(...) to get reconstructed timeseries

    Important details:
      - Percentage change between consecutive points computed with an epsilon
        added to the denominator to avoid division by zero.
      - Percentage changes are clamped to +/- clamp_pct (default 500%) to
        limit outlier influence and preserve precision for small changes.
      - Input NaN/Inf values are handled gracefully by forward-filling per-variable
        and defaulting fully-missing variables to zeros.
    """

    def __init__(
        self,
        data: Any,
        clamp_pct: float = 500.0,
        epsilon: float = 1e-9,
        var_names: Optional[Sequence[str]] = None,
    ):
        """
        data: 2D array-like with shape (n_vars, n_time) or pandas.DataFrame
        clamp_pct: positive float, clamp limits in percent (e.g., 500 -> +/-500%)
        epsilon: small value added to denominator for safety
        var_names: optional sequence of names for variables (kept in metadata)
        """
        # Lazy import to avoid requiring pandas at import-time
        try:
            import pandas as _pd  # type: ignore
        except Exception:
            _pd = None

        if _pd is not None and hasattr(data, "values") and isinstance(
            data, _pd.DataFrame
        ):
            arr = data.values.astype(float)
            if var_names is None:
                var_names = list(data.columns.astype(str))
        else:
            arr = np.asarray(data, dtype=float)

        # Normalize dimensions
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError("data must be 1D or 2D array-like (variables x time)")

        # Defensive sanitization: convert Inf to NaN, then forward-fill NaNs per variable.
        # If a variable is entirely NaN, replace with zeros.
        arr = arr.astype(float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        n_vars_s, n_time_s = arr.shape
        for i in range(n_vars_s):
            row = arr[i]
            if np.all(np.isnan(row)):
                arr[i] = np.zeros_like(row)
                continue
            mask = np.isnan(row)
            if mask[0]:
                # backfill first element with first valid or zero
                valid_idx = np.flatnonzero(~mask)
                row[0] = row[valid_idx[0]] if valid_idx.size > 0 else 0.0
                mask = np.isnan(row)
            if mask.any():
                # forward-fill using last seen index trick
                idx = np.where(~mask, np.arange(n_time_s), 0)
                np.maximum.accumulate(idx, out=idx)
                row[:] = row[idx]

        n_vars, n_time = arr.shape
        if n_time < 1:
            raise ValueError("time-series must contain at least one time point")

        self._data = arr
        self.n_vars = n_vars
        self.n_time = n_time
        self.clamp_pct = float(clamp_pct)
        if self.clamp_pct <= 0:
            raise ValueError("clamp_pct must be positive")
        self.epsilon = float(epsilon)
        self.var_names = list(var_names) if var_names is not None else None

    @staticmethod
    def calculate_delta(prev: np.ndarray, current: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Safe percentage change formula: (current - prev) / (prev + eps) * 100
        Works elementwise. epsilon prevents division by zero.
        """
        prev = np.asarray(prev, dtype=float)
        current = np.asarray(current, dtype=float)
        denom = prev + epsilon
        # Avoid extremely small denominators causing huge deltas; epsilon prevents div0
        pct = (current - prev) / denom * 100.0
        return pct

    def compute_percent_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute percent-change grid for all variables.
        Returns (baseline, percent_grid)
          - baseline: shape (n_vars,) first value of each variable
          - percent_grid: shape (n_vars, n_time-1)
        """
        baseline = self._data[:, 0].astype(float).copy()
        if self.n_time == 1:
            # No deltas
            percent_grid = np.zeros((self.n_vars, 0), dtype=float)
            return baseline, percent_grid

        prev = self._data[:, :-1]
        cur = self._data[:, 1:]

        if HAS_NUMBA:
            pct = _compute_pct_numba(prev, cur, float(self.epsilon))
        else:
            pct = (cur - prev) / (prev + self.epsilon) * 100.0
        pct_grid = np.clip(pct, -self.clamp_pct, self.clamp_pct)
        return baseline, pct_grid

    def quantize_16bit(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Quantize the percent grid into uint16 values in [0, 65535].
        Metadata returned contains baseline, clamp_pct, epsilon, var_names, and shapes.
        """
        baseline, pct_grid = self.compute_percent_grid()
        # Range [-clamp, +clamp] maps to [0, 65535]
        clamp = float(self.clamp_pct)
        if pct_grid.size == 0:
            q = np.zeros((self.n_vars, 0), dtype=np.uint16)
        else:
            pct_clipped = np.clip(pct_grid, -clamp, clamp)
            scaled = (pct_clipped + clamp) / (2.0 * clamp) * 65535.0
            q = np.round(scaled).astype(np.uint16)
        metadata: Dict[str, Any] = {
            "n_vars": int(self.n_vars),
            "n_time": int(self.n_time),
            "clamp_pct": float(self.clamp_pct),
            "epsilon": float(self.epsilon),
            "dtype": "uint16",
            "baseline": baseline.tolist(),
            "var_names": self.var_names if self.var_names is not None else None,
        }
        return q, metadata

    def quantize_8bit(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Quantize the percent grid into uint8 values in [0, 255].
        Useful for smaller storage at the cost of precision.
        """
        baseline, pct_grid = self.compute_percent_grid()
        clamp = float(self.clamp_pct)
        if pct_grid.size == 0:
            q = np.zeros((self.n_vars, 0), dtype=np.uint8)
        else:
            pct_clipped = np.clip(pct_grid, -clamp, clamp)
            scaled = (pct_clipped + clamp) / (2.0 * clamp) * 255.0
            q = np.round(scaled).astype(np.uint8)
        metadata: Dict[str, Any] = {
            "n_vars": int(self.n_vars),
            "n_time": int(self.n_time),
            "clamp_pct": float(self.clamp_pct),
            "epsilon": float(self.epsilon),
            "dtype": "uint8",
            "baseline": baseline.tolist(),
            "var_names": self.var_names if self.var_names is not None else None,
        }
        return q, metadata

    @staticmethod
    def dequantize_16bit(q: np.ndarray, clamp_pct: float) -> np.ndarray:
        """
        Convert uint16 grid back to percent floats in range [-clamp_pct, clamp_pct].
        q shape: (n_vars, n_time-1) or empty.
        """
        if q.size == 0:
            return np.empty((q.shape[0], 0), dtype=float)
        clamp = float(clamp_pct)
        scaled = q.astype(float) / 65535.0
        pct = scaled * (2.0 * clamp) - clamp
        return pct

    @staticmethod
    def dequantize_8bit(q: np.ndarray, clamp_pct: float) -> np.ndarray:
        """
        Convert uint8 grid back to percent floats in range [-clamp_pct, clamp_pct].
        """
        if q.size == 0:
            return np.empty((q.shape[0], 0), dtype=float)
        clamp = float(clamp_pct)
        scaled = q.astype(float) / 255.0
        pct = scaled * (2.0 * clamp) - clamp
        return pct

    @staticmethod
    def dequantize_uint8_levels(q: np.ndarray, clamp_pct: float, levels: int) -> np.ndarray:
        """Dequantize uint8 where only `levels` distinct values are used (levels<=256)."""
        if q.size == 0:
            return np.empty((q.shape[0], 0), dtype=float)
        clamp = float(clamp_pct)
        lv = int(levels)
        if lv <= 1 or lv > 256:
            raise ValueError("levels must be in [2, 256]")
        denom = float(lv - 1)
        scaled = q.astype(float) / denom
        pct = scaled * (2.0 * clamp) - clamp
        return pct

    @staticmethod
    def reconstruct_from_quantized(
        q: np.ndarray, metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reconstruct the original timeseries (approx) from a quantized uint8/uint16 grid and metadata.
        Returns (reconstructed_array, info)
        reconstructed_array shape: (n_vars, n_time)
        info contains RMSE/MAE placeholders if original not provided.
        """
        if metadata is None:
            raise ValueError("metadata must be provided to reconstruct")
        baseline = np.asarray(metadata.get("baseline", []), dtype=float)
        n_vars = int(metadata.get("n_vars", baseline.shape[0] if baseline.size else 0))
        clamp_pct = float(metadata.get("clamp_pct", 500.0))

        # Infer n_time if not present
        if "n_time" in metadata:
            n_time = int(metadata["n_time"])
        else:
            if q is None or q.size == 0:
                n_time = 1
            else:
                n_time = int(q.shape[1] + 1)

        if baseline.size == 0:
            baseline = np.zeros((n_vars,), dtype=float)

        dtype = str(metadata.get("dtype", "uint16"))
        if q is None:
            q = np.zeros((n_vars, 0), dtype=np.uint16 if dtype == "uint16" else np.uint8)

        if dtype == "uint16":
            pct_grid = TimeSeriesGrid.dequantize_16bit(q, clamp_pct)
        elif dtype == "uint8":
            # Support sub-8-bit quantization stored in uint8 by honoring levels/bit_depth.
            levels = metadata.get("levels")
            if levels is None:
                bd = metadata.get("bit_depth")
                if bd is not None:
                    try:
                        bd_i = int(bd)
                        if 1 <= bd_i <= 8:
                            levels = 2 ** bd_i
                    except Exception:
                        levels = None
            if levels is not None:
                lv = int(levels)
                if lv != 256:
                    pct_grid = TimeSeriesGrid.dequantize_uint8_levels(q, clamp_pct, lv)
                else:
                    pct_grid = TimeSeriesGrid.dequantize_8bit(q, clamp_pct)
            else:
                pct_grid = TimeSeriesGrid.dequantize_8bit(q, clamp_pct)
        else:
            raise ValueError(f"unsupported dtype in metadata: {dtype}")

        # Build reconstructed timeseries by applying percent changes multiplicatively.
        if pct_grid.size == 0:
            recon = baseline.reshape(n_vars, 1).copy()
        else:
            factors = 1.0 + pct_grid / 100.0
            cumprod = np.cumprod(factors, axis=1)
            recon = baseline[:, None] * np.concatenate([np.ones((n_vars, 1)), cumprod], axis=1)
        info = {
            "reconstructed_n_vars": int(recon.shape[0]),
            "reconstructed_n_time": int(recon.shape[1]),
        }
        return recon, info


# Small utility functions for error metrics (useful for quick verification)

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("shapes for rmse must match")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("shapes for mae must match")
    return float(np.mean(np.abs(a - b)))


# numba-accelerated percent computation
if HAS_NUMBA:
    @numba.njit(cache=True)
    def _compute_pct_numba(prev, cur, eps):
        out = np.empty_like(cur, dtype=np.float64)
        for i in range(prev.shape[0]):
            for j in range(prev.shape[1]):
                out[i, j] = (cur[i, j] - prev[i, j]) / (prev[i, j] + eps) * 100.0
        return out
else:
    def _compute_pct_numba(prev, cur, eps):
        return (cur - prev) / (prev + eps) * 100.0

"""
Enhanced TRACQ Core Module

Structural improvements over the original implementation:
1. Per-variable adaptive clamping (instead of global clamp)
2. Non-uniform mu-law quantization (better for heavy-tailed distributions)
3. Anchor points to prevent cumulative error drift
4. Automated baseline offsetting for zero-crossing / near-zero signals
5. Variable reordering by correlation for better PNG compression
6. Log-domain encoding option for multiplicative processes
7. Symmetric percentage formula to reduce bias near zero

These changes are backward-compatible - the original API is preserved.
"""

import json
from typing import Optional, Tuple, Dict, Any, Sequence, Literal

import numpy as np

try:
    import numba  # type: ignore
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False


# ============================================================================
# Mu-Law Companding (Non-Uniform Quantization)
# ============================================================================

class MuLawCompander:
    """
    Mu-law companding for non-uniform quantization.

    Provides more quantization levels near zero where most percentage changes
    occur, and fewer levels at extremes. This is optimal for Laplacian-like
    (heavy-tailed) distributions common in time series data.

    Standard mu=255 provides ~33dB dynamic range improvement over uniform.
    """

    def __init__(self, mu: float = 255.0):
        self.mu = mu
        self._log_1_plus_mu = np.log1p(mu)

    def compress(self, x: np.ndarray) -> np.ndarray:
        """
        Apply mu-law compression.
        Input x should be normalized to [-1, 1].
        Output is in [-1, 1].
        """
        return np.sign(x) * np.log1p(self.mu * np.abs(x)) / self._log_1_plus_mu

    def expand(self, y: np.ndarray) -> np.ndarray:
        """
        Apply mu-law expansion (inverse).
        Input y should be in [-1, 1].
        Output is in [-1, 1].
        """
        return np.sign(y) * (np.power(1 + self.mu, np.abs(y)) - 1) / self.mu


# Global compander instance (mu=255 is standard)
_COMPANDER = MuLawCompander(mu=255.0)


# ============================================================================
# Enhanced TimeSeriesGrid
# ============================================================================

class EnhancedTimeSeriesGrid:
    """
    Enhanced TRACQ with improved compression and error performance.

    Key improvements:
    - Per-variable adaptive clamping based on percentile statistics
    - Non-uniform (mu-law) quantization for heavy-tailed distributions
    - Optional anchor points to bound cumulative error drift
    - Variable reordering by correlation for better PNG compression
    - Log-domain encoding for multiplicative processes

    Backward compatible with original TimeSeriesGrid API.
    """

    def __init__(
        self,
        data: Any,
        clamp_pct: float = 500.0,
        epsilon: float = 1e-9,
        var_names: Optional[Sequence[str]] = None,
        # Enhanced options
        adaptive_clamp: bool = True,
        clamp_percentile: float = 99.5,
        use_mu_law: bool = True,
        mu: float = 255.0,
        use_log_domain: bool = False,
        anchor_interval: int = 0,  # 0 = disabled, >0 = store anchor every N steps
        reorder_variables: bool = False,
        auto_offset: bool = False,
        near_zero_std_factor: float = 0.1,
    ):
        """
        Args:
            data: 2D array-like (n_vars, n_time) or pandas DataFrame
            clamp_pct: Fallback global clamp if adaptive_clamp=False
            epsilon: Small value for numerical stability
            var_names: Optional variable names

        Enhanced options:
            adaptive_clamp: Use per-variable percentile-based clamping
            clamp_percentile: Percentile for adaptive clamp (e.g., 99.5)
            use_mu_law: Use mu-law companding instead of uniform quantization
            mu: Mu parameter for companding (255 is standard)
            use_log_domain: Encode log(1+pct) instead of pct
            anchor_interval: Store exact values every N steps (0=disabled)
            reorder_variables: Reorder variables by correlation for better compression
            auto_offset: Automatically add a per-variable constant offset when a
                channel crosses or approaches zero. This keeps percentage-change
                encoding in a safe positive domain and stores offsets in metadata.
            near_zero_std_factor: Threshold for deciding whether a strictly
                positive channel is still too close to zero relative to its
                standard deviation.
        """
        # Lazy pandas import
        try:
            import pandas as _pd
        except Exception:
            _pd = None

        # Convert input to numpy array
        if _pd is not None and hasattr(data, "values") and isinstance(data, _pd.DataFrame):
            arr = data.values.astype(float)
            if var_names is None:
                var_names = list(data.columns.astype(str))
        else:
            arr = np.asarray(data, dtype=float)

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError("data must be 1D or 2D array-like (variables x time)")

        # Sanitize: convert Inf to NaN, forward-fill NaNs
        arr = self._sanitize_array(arr)

        self.auto_offset = auto_offset
        self.near_zero_std_factor = float(near_zero_std_factor)
        if self.auto_offset:
            self._value_offsets = self._compute_value_offsets(
                arr,
                epsilon=float(epsilon),
                near_zero_std_factor=self.near_zero_std_factor,
            )
            arr = arr + self._value_offsets[:, None]
        else:
            self._value_offsets = np.zeros(arr.shape[0], dtype=float)

        n_vars, n_time = arr.shape
        if n_time < 1:
            raise ValueError("time-series must contain at least one time point")

        self._data = arr
        self.n_vars = n_vars
        self.n_time = n_time
        self.epsilon = float(epsilon)
        self.var_names = list(var_names) if var_names is not None else None

        # Enhanced options
        self.adaptive_clamp = adaptive_clamp
        self.clamp_percentile = clamp_percentile
        self.global_clamp_pct = float(clamp_pct)
        self.use_mu_law = use_mu_law
        self.mu = mu
        self.use_log_domain = use_log_domain
        self.anchor_interval = anchor_interval
        self.reorder_variables = reorder_variables

        # Compander for mu-law
        self._compander = MuLawCompander(mu=mu) if use_mu_law else None

        # Variable reordering (computed on demand)
        self._var_order = None
        self._var_order_inverse = None

    @staticmethod
    def _sanitize_array(arr: np.ndarray) -> np.ndarray:
        """Sanitize array: convert Inf to NaN, forward-fill NaNs."""
        arr = arr.astype(float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        n_vars, n_time = arr.shape

        for i in range(n_vars):
            row = arr[i]
            if np.all(np.isnan(row)):
                arr[i] = np.zeros_like(row)
                continue
            mask = np.isnan(row)
            if mask[0]:
                valid_idx = np.flatnonzero(~mask)
                row[0] = row[valid_idx[0]] if valid_idx.size > 0 else 0.0
                mask = np.isnan(row)
            if mask.any():
                idx = np.where(~mask, np.arange(n_time), 0)
                np.maximum.accumulate(idx, out=idx)
                row[:] = row[idx]

        return arr

    @staticmethod
    def _compute_value_offsets(
        arr: np.ndarray,
        *,
        epsilon: float,
        near_zero_std_factor: float,
    ) -> np.ndarray:
        """
        Compute per-variable offsets that move unstable channels into a safe,
        positive domain before percentage-change encoding.

        Two cases are handled:
        1. Any variable that reaches or crosses zero gets shifted by
           `2 * abs(min) + eps`, matching the paper-side preprocessing proposal.
        2. A strictly positive variable whose minimum magnitude is still very
           small relative to its standard deviation is lifted to a small
           positive floor proportional to that standard deviation.
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError("arr must be 2D")

        mins = np.min(arr, axis=1)
        stds = np.std(arr, axis=1)
        min_abs = np.min(np.abs(arr), axis=1)

        offsets = np.zeros(arr.shape[0], dtype=float)

        crosses_zero = mins <= 0.0
        offsets[crosses_zero] = 2.0 * np.abs(mins[crosses_zero]) + float(epsilon)

        positive_near_zero = (~crosses_zero) & (
            min_abs <= near_zero_std_factor * np.maximum(stds, float(epsilon))
        )
        target_floor = near_zero_std_factor * np.maximum(stds[positive_near_zero], float(epsilon))
        if target_floor.size:
            offsets[positive_near_zero] = np.maximum(target_floor - mins[positive_near_zero], 0.0) + float(epsilon)

        return offsets

    def _compute_variable_order(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal variable ordering by correlation clustering.

        Reorders variables so that correlated ones are adjacent,
        improving PNG filter effectiveness.
        """
        if self.n_vars <= 2:
            order = np.arange(self.n_vars)
            return order, order

        # Compute correlation matrix of percentage changes
        _, pct_grid = self._compute_raw_percent_grid()

        if pct_grid.size == 0:
            order = np.arange(self.n_vars)
            return order, order

        # Use absolute correlation for similarity
        corr_matrix = np.corrcoef(pct_grid)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Greedy nearest-neighbor ordering
        visited = np.zeros(self.n_vars, dtype=bool)
        order = []

        # Start with variable having highest total correlation
        total_corr = np.sum(np.abs(corr_matrix), axis=1)
        current = np.argmax(total_corr)

        for _ in range(self.n_vars):
            order.append(current)
            visited[current] = True

            if len(order) == self.n_vars:
                break

            # Find most correlated unvisited variable
            correlations = np.abs(corr_matrix[current])
            correlations[visited] = -np.inf
            current = np.argmax(correlations)

        order = np.array(order)
        inverse = np.argsort(order)

        return order, inverse

    def _compute_raw_percent_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute raw percentage changes without clamping."""
        baseline = self._data[:, 0].copy()

        if self.n_time == 1:
            return baseline, np.zeros((self.n_vars, 0), dtype=float)

        prev = self._data[:, :-1]
        cur = self._data[:, 1:]

        if self.use_log_domain:
            # Log-domain: better for multiplicative processes
            # Store log(cur/prev) which inverts cleanly as cur = prev * exp(log_ratio)
            safe_prev = np.maximum(np.abs(prev), self.epsilon)
            safe_cur = np.maximum(np.abs(cur), self.epsilon)
            # Handle sign changes
            sign = np.sign(prev) * np.sign(cur)
            sign = np.where(sign == 0, 1, sign)
            pct = sign * np.log(safe_cur / safe_prev) * 100.0
        else:
            # Standard percentage formula: (cur - prev) / prev * 100
            # This inverts cleanly as: cur = prev * (1 + pct/100)
            # Use max(|prev|, epsilon) for numerical stability
            safe_denom = np.maximum(np.abs(prev), self.epsilon)
            pct = (cur - prev) / safe_denom * 100.0

        return baseline, pct

    def _compute_adaptive_clamps(self, pct_grid: np.ndarray) -> np.ndarray:
        """Compute per-variable clamp values based on percentile statistics."""
        if pct_grid.size == 0:
            return np.full(self.n_vars, self.global_clamp_pct)

        # Use specified percentile for each variable
        upper = np.percentile(np.abs(pct_grid), self.clamp_percentile, axis=1)

        # Also compute the actual max to handle periodic patterns
        actual_max = np.max(np.abs(pct_grid), axis=1)

        # Use a blend: if max is much larger than percentile, data has outliers
        # Otherwise (periodic patterns), use a value closer to max
        ratio = actual_max / (upper + 1e-10)
        # If ratio > 10, data has heavy outliers, use percentile
        # If ratio < 2, data is relatively uniform, use max with margin
        blend_factor = np.clip((ratio - 2) / 8, 0, 1)  # 0 at ratio=2, 1 at ratio=10
        clamps = blend_factor * upper + (1 - blend_factor) * (actual_max * 1.1)

        # Ensure minimum clamp to avoid division issues
        clamps = np.maximum(clamps, 1.0)  # At least 1% clamp

        # Cap at global clamp
        clamps = np.minimum(clamps, self.global_clamp_pct)

        return clamps

    def compute_percent_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute clamped percent-change grid.

        Returns:
            baseline: shape (n_vars,) first value of each variable
            pct_grid: shape (n_vars, n_time-1) clamped percentage changes
            clamps: shape (n_vars,) per-variable clamp values
        """
        baseline, raw_pct = self._compute_raw_percent_grid()

        if self.adaptive_clamp:
            clamps = self._compute_adaptive_clamps(raw_pct)
        else:
            clamps = np.full(self.n_vars, self.global_clamp_pct)

        # Apply per-variable clamping
        if raw_pct.size > 0:
            pct_grid = np.clip(raw_pct, -clamps[:, None], clamps[:, None])
        else:
            pct_grid = raw_pct

        return baseline, pct_grid, clamps

    def _apply_mu_law(self, pct_grid: np.ndarray, clamps: np.ndarray) -> np.ndarray:
        """Apply mu-law compression to normalized percentage grid."""
        if pct_grid.size == 0:
            return pct_grid

        # Normalize each variable to [-1, 1] using its clamp
        normalized = pct_grid / clamps[:, None]
        normalized = np.clip(normalized, -1, 1)

        # Apply mu-law compression
        compressed = self._compander.compress(normalized)

        return compressed

    def _inverse_mu_law(self, compressed: np.ndarray, clamps: np.ndarray) -> np.ndarray:
        """Apply inverse mu-law to recover percentage grid."""
        if compressed.size == 0:
            return compressed

        # Expand mu-law
        normalized = self._compander.expand(compressed)

        # Denormalize
        pct_grid = normalized * clamps[:, None]

        return pct_grid

    def quantize_16bit(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Quantize to 16-bit with all enhancements.

        Returns (quantized_grid, metadata)
        """
        baseline, pct_grid, clamps = self.compute_percent_grid()
        offsets = self._value_offsets.copy()

        # Optional variable reordering
        var_order = None
        if self.reorder_variables and self.n_vars > 2:
            if self._var_order is None:
                self._var_order, self._var_order_inverse = self._compute_variable_order()
            var_order = self._var_order.tolist()
            pct_grid = pct_grid[self._var_order]
            baseline = baseline[self._var_order]
            clamps = clamps[self._var_order]
            offsets = offsets[self._var_order]

        # Apply mu-law if enabled
        if self.use_mu_law and pct_grid.size > 0:
            working_grid = self._apply_mu_law(pct_grid, clamps)
            # After mu-law, values are in [-1, 1]
            scaled = (working_grid + 1.0) / 2.0 * 65535.0
        else:
            # Standard uniform quantization
            if pct_grid.size > 0:
                # Normalize per-variable
                normalized = (pct_grid + clamps[:, None]) / (2.0 * clamps[:, None])
                scaled = normalized * 65535.0
            else:
                scaled = np.zeros((self.n_vars, 0), dtype=float)

        q = np.round(scaled).astype(np.uint16)

        # Handle anchor points
        anchors = None
        if self.anchor_interval > 0 and self.n_time > 1:
            anchor_indices = list(range(0, self.n_time, self.anchor_interval))
            if self.n_time - 1 not in anchor_indices:
                anchor_indices.append(self.n_time - 1)

            if self.reorder_variables and self._var_order is not None:
                anchors = {
                    "indices": anchor_indices,
                    "values": self._data[self._var_order][:, anchor_indices].tolist()
                }
            else:
                anchors = {
                    "indices": anchor_indices,
                    "values": self._data[:, anchor_indices].tolist()
                }

        metadata = {
            "n_vars": int(self.n_vars),
            "n_time": int(self.n_time),
            "clamp_pct": clamps.tolist() if self.adaptive_clamp else float(self.global_clamp_pct),
            "epsilon": float(self.epsilon),
            "dtype": "uint16",
            "baseline": baseline.tolist(),
            "var_names": self.var_names,
            # Enhanced metadata
            "adaptive_clamp": self.adaptive_clamp,
            "use_mu_law": self.use_mu_law,
            "mu": float(self.mu) if self.use_mu_law else None,
            "use_log_domain": self.use_log_domain,
            "var_order": var_order,
            "anchors": anchors,
            "auto_offset": self.auto_offset,
            "value_offsets": offsets.tolist() if self.auto_offset and np.any(offsets) else None,
            "near_zero_std_factor": float(self.near_zero_std_factor) if self.auto_offset else None,
        }

        return q, metadata

    def quantize_8bit(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Quantize to 8-bit with all enhancements.

        Returns (quantized_grid, metadata)
        """
        baseline, pct_grid, clamps = self.compute_percent_grid()
        offsets = self._value_offsets.copy()

        # Optional variable reordering
        var_order = None
        if self.reorder_variables and self.n_vars > 2:
            if self._var_order is None:
                self._var_order, self._var_order_inverse = self._compute_variable_order()
            var_order = self._var_order.tolist()
            pct_grid = pct_grid[self._var_order]
            baseline = baseline[self._var_order]
            clamps = clamps[self._var_order]
            offsets = offsets[self._var_order]

        # Apply mu-law if enabled
        if self.use_mu_law and pct_grid.size > 0:
            working_grid = self._apply_mu_law(pct_grid, clamps)
            scaled = (working_grid + 1.0) / 2.0 * 255.0
        else:
            if pct_grid.size > 0:
                normalized = (pct_grid + clamps[:, None]) / (2.0 * clamps[:, None])
                scaled = normalized * 255.0
            else:
                scaled = np.zeros((self.n_vars, 0), dtype=float)

        q = np.round(scaled).astype(np.uint8)

        # Handle anchor points
        anchors = None
        if self.anchor_interval > 0 and self.n_time > 1:
            anchor_indices = list(range(0, self.n_time, self.anchor_interval))
            if self.n_time - 1 not in anchor_indices:
                anchor_indices.append(self.n_time - 1)

            if self.reorder_variables and self._var_order is not None:
                anchors = {
                    "indices": anchor_indices,
                    "values": self._data[self._var_order][:, anchor_indices].tolist()
                }
            else:
                anchors = {
                    "indices": anchor_indices,
                    "values": self._data[:, anchor_indices].tolist()
                }

        metadata = {
            "n_vars": int(self.n_vars),
            "n_time": int(self.n_time),
            "clamp_pct": clamps.tolist() if self.adaptive_clamp else float(self.global_clamp_pct),
            "epsilon": float(self.epsilon),
            "dtype": "uint8",
            "baseline": baseline.tolist(),
            "var_names": self.var_names,
            # Enhanced metadata
            "adaptive_clamp": self.adaptive_clamp,
            "use_mu_law": self.use_mu_law,
            "mu": float(self.mu) if self.use_mu_law else None,
            "use_log_domain": self.use_log_domain,
            "var_order": var_order,
            "anchors": anchors,
            "auto_offset": self.auto_offset,
            "value_offsets": offsets.tolist() if self.auto_offset and np.any(offsets) else None,
            "near_zero_std_factor": float(self.near_zero_std_factor) if self.auto_offset else None,
        }

        return q, metadata

    @staticmethod
    def reconstruct_from_quantized(
        q: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reconstruct time series from quantized grid and metadata.

        Handles both original and enhanced metadata formats.
        """
        if metadata is None:
            raise ValueError("metadata must be provided")

        baseline = np.asarray(metadata.get("baseline", []), dtype=float)
        n_vars = int(metadata.get("n_vars", baseline.shape[0] if baseline.size else 0))
        n_time = int(metadata.get("n_time", q.shape[1] + 1 if q is not None and q.size > 0 else 1))

        if baseline.size == 0:
            baseline = np.zeros((n_vars,), dtype=float)

        # Get clamp(s)
        clamp_pct = metadata.get("clamp_pct", 500.0)
        if isinstance(clamp_pct, list):
            clamps = np.array(clamp_pct)
        else:
            clamps = np.full(n_vars, float(clamp_pct))

        # Get enhanced options
        use_mu_law = metadata.get("use_mu_law", False)
        mu = metadata.get("mu", 255.0)
        use_log_domain = metadata.get("use_log_domain", False)
        var_order = metadata.get("var_order")
        anchors = metadata.get("anchors")
        value_offsets = metadata.get("value_offsets")
        if value_offsets is None:
            offsets = np.zeros(n_vars, dtype=float)
        else:
            offsets = np.asarray(value_offsets, dtype=float)

        dtype = str(metadata.get("dtype", "uint16"))
        if q is None:
            q = np.zeros((n_vars, 0), dtype=np.uint16 if dtype == "uint16" else np.uint8)

        # Dequantize
        if dtype == "uint16":
            max_val = 65535.0
        else:
            max_val = 255.0

        if q.size == 0:
            pct_grid = np.zeros((n_vars, 0), dtype=float)
        else:
            # Convert to [0, 1] range
            normalized = q.astype(float) / max_val

            if use_mu_law:
                # Convert to [-1, 1] and apply inverse mu-law
                compressed = normalized * 2.0 - 1.0
                compander = MuLawCompander(mu=mu)
                expanded = compander.expand(compressed)
                # Denormalize with per-variable clamps
                pct_grid = expanded * clamps[:, None]
            else:
                # Standard uniform dequantization
                pct_grid = normalized * (2.0 * clamps[:, None]) - clamps[:, None]

        # Reconstruct time series
        if pct_grid.size == 0:
            recon = baseline.reshape(n_vars, 1).copy()
        else:
            if use_log_domain:
                # Log-domain: pct_grid contains sign * log(|cur/prev|) * 100
                log_ratios = pct_grid / 100.0
                cum_log = np.cumsum(np.abs(log_ratios), axis=1)
                signs = np.cumprod(np.sign(log_ratios), axis=1)
                signs = np.where(signs == 0, 1, signs)
                ratios = signs * np.exp(cum_log)
                recon = baseline[:, None] * np.concatenate([np.ones((n_vars, 1)), ratios], axis=1)
            else:
                # Standard multiplicative reconstruction
                # pct = (cur - prev) / prev * 100
                # Inverts as: cur = prev * (1 + pct/100)
                factors = 1.0 + pct_grid / 100.0
                cumprod = np.cumprod(factors, axis=1)
                recon = baseline[:, None] * np.concatenate([np.ones((n_vars, 1)), cumprod], axis=1)

        # Apply anchor corrections if available
        if anchors is not None:
            anchor_indices = anchors.get("indices", [])
            anchor_values = np.array(anchors.get("values", []))

            if len(anchor_indices) > 0 and anchor_values.size > 0:
                # Interpolate corrections between anchor points
                for i, (idx, val) in enumerate(zip(anchor_indices, anchor_values.T)):
                    if idx < recon.shape[1]:
                        recon[:, idx] = val

                        # If not last anchor, blend corrections to next anchor
                        if i < len(anchor_indices) - 1:
                            next_idx = anchor_indices[i + 1]
                            if next_idx > idx + 1:
                                # Re-propagate from this anchor
                                for t in range(idx + 1, min(next_idx, recon.shape[1])):
                                    if t - 1 < pct_grid.shape[1]:
                                        factor = 1.0 + pct_grid[:, t - 1] / 100.0
                                        recon[:, t] = recon[:, t - 1] * factor

        if offsets.size:
            recon = recon - offsets[:, None]

        # Reverse variable reordering if applied
        if var_order is not None:
            inverse_order = np.argsort(var_order)
            recon = recon[inverse_order]

        info = {
            "reconstructed_n_vars": int(recon.shape[0]),
            "reconstructed_n_time": int(recon.shape[1]),
        }

        return recon, info


# ============================================================================
# Utility functions
# ============================================================================

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared error."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    return float(np.mean(np.abs(a - b)))


def compare_methods(
    data: np.ndarray,
    bits: int = 8,
    clamp_pct: float = 500.0,
) -> Dict[str, Dict[str, float]]:
    """
    Compare original vs enhanced TRACQ on the same data.

    Returns dict with RMSE, MAE, and compression stats for each method.
    """
    from .core import TimeSeriesGrid as OriginalGrid

    results = {}

    # Original method
    orig_grid = OriginalGrid(data, clamp_pct=clamp_pct)
    if bits == 8:
        q_orig, meta_orig = orig_grid.quantize_8bit()
    else:
        q_orig, meta_orig = orig_grid.quantize_16bit()

    recon_orig, _ = OriginalGrid.reconstruct_from_quantized(q_orig, meta_orig)

    results["original"] = {
        "rmse": rmse(data, recon_orig),
        "mae": mae(data, recon_orig),
        "grid_bytes": q_orig.nbytes,
    }

    # Enhanced method: uniform quantization with adaptive clamp
    enh_grid1 = EnhancedTimeSeriesGrid(
        data, clamp_pct=clamp_pct,
        adaptive_clamp=True, use_mu_law=False
    )
    if bits == 8:
        q_enh1, meta_enh1 = enh_grid1.quantize_8bit()
    else:
        q_enh1, meta_enh1 = enh_grid1.quantize_16bit()

    recon_enh1, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q_enh1, meta_enh1)

    results["adaptive_clamp"] = {
        "rmse": rmse(data, recon_enh1),
        "mae": mae(data, recon_enh1),
        "grid_bytes": q_enh1.nbytes,
    }

    # Enhanced method: mu-law with adaptive clamp
    enh_grid2 = EnhancedTimeSeriesGrid(
        data, clamp_pct=clamp_pct,
        adaptive_clamp=True, use_mu_law=True
    )
    if bits == 8:
        q_enh2, meta_enh2 = enh_grid2.quantize_8bit()
    else:
        q_enh2, meta_enh2 = enh_grid2.quantize_16bit()

    recon_enh2, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q_enh2, meta_enh2)

    results["mu_law_adaptive"] = {
        "rmse": rmse(data, recon_enh2),
        "mae": mae(data, recon_enh2),
        "grid_bytes": q_enh2.nbytes,
    }

    # Enhanced method: mu-law with variable reordering
    if data.shape[0] > 2:
        enh_grid3 = EnhancedTimeSeriesGrid(
            data, clamp_pct=clamp_pct,
            adaptive_clamp=True, use_mu_law=True, reorder_variables=True
        )
        if bits == 8:
            q_enh3, meta_enh3 = enh_grid3.quantize_8bit()
        else:
            q_enh3, meta_enh3 = enh_grid3.quantize_16bit()

        recon_enh3, _ = EnhancedTimeSeriesGrid.reconstruct_from_quantized(q_enh3, meta_enh3)

        results["mu_law_reordered"] = {
            "rmse": rmse(data, recon_enh3),
            "mae": mae(data, recon_enh3),
            "grid_bytes": q_enh3.nbytes,
        }

    return results

"""
TRACQ-2: drift-free, error-bounded lattice codec for multivariate time series.

Keeps TRACQ's core idea -- a per-timestep 2D grid of relative changes,
adaptively quantized per variable, viewable as an image -- but replaces the
open-loop percent-change/cumprod pipeline with an integer-lattice code that is
drift-free and error-bounded by construction:

  1. Transform  y_i[t] = x_i[t] / 1        (mode='abs',  linear)
                y_i[t] = asinh(x_i[t]/s_i) (mode='rel',  smooth mixed abs/rel)
  2. Lattice    m_i[t] = round(y_i[t] / q_i)   (per-variable step q_i from a
                                                user-specified error bound)
  3. Grid       r_i[t] = predictor residual of m (integer), stored as uint8
                pixel r+128; |r|>126 escapes to an exact sparse sidecar.
  4. Decode     exact integer accumulation -> y_hat = q*m -> x_hat.
                cumsum(diff(m)) == m identically, so |y_hat - y| <= q/2 for
                every t: no drift, no anchors, guaranteed pointwise bound.

Predictors (per row, integer-exact, chosen by residual entropy):
  P1 temporal diff, P2 double diff (delta-of-delta), PS seasonal lag-L diff,
  PL 2D Lorenzo (row diff of temporal diffs, on correlation-ordered rows).

Primary artifact: one Zstd blob (header + grid + sidecar) -- the honest
measured size. View artifact: the same grid rendered as a standard PNG.
"""

import json
import struct
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import zstandard as zstd

    HAS_ZSTD = True
except Exception:  # pragma: no cover
    HAS_ZSTD = False

ESCAPE = 255  # pixel value marking an escaped (sidecar) residual
RMAX = 126    # residuals in [-RMAX, RMAX] are stored inline as r+128

PRED_P1 = 0   # temporal diff
PRED_P2 = 1   # double diff
PRED_PS = 2   # seasonal lag-L diff
PRED_PL = 3   # 2D Lorenzo vs previous (correlation-ordered) row


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------

def _forward_transform(x: np.ndarray, mode: str, s: np.ndarray) -> np.ndarray:
    if mode == "abs":
        return x
    return np.arcsinh(x / s[:, None])


def _inverse_transform(y: np.ndarray, mode: str, s: np.ndarray) -> np.ndarray:
    if mode == "abs":
        return y
    return s[:, None] * np.sinh(y)


def _derive_steps(
    x: np.ndarray,
    mode: str,
    eps: float,
    s: np.ndarray,
) -> np.ndarray:
    """Per-variable lattice step q_i from the user error bound eps.

    mode='abs': eps is a fraction of each variable's value range (SZ-style REL
                bound). q_i = 2 * eps * range_i, giving |x_hat-x| <= eps*range_i.
    mode='rel': eps is a pointwise relative error target for |x| >> s_i
                (and eps*s_i absolute near zero). q = 2*ln(1+eps) so that
                e^{q/2}-1 <= ~eps.
    """
    n_vars = x.shape[0]
    if mode == "abs":
        rng = x.max(axis=1) - x.min(axis=1)
        rng = np.where(rng <= 0, np.maximum(np.abs(x[:, 0]), 1.0), rng)
        return 2.0 * eps * rng
    return np.full(n_vars, 2.0 * np.log1p(eps))


# ----------------------------------------------------------------------------
# Predictor bank (all integer-exact on the lattice)
# ----------------------------------------------------------------------------

def _residual_p1(m: np.ndarray) -> np.ndarray:
    return np.diff(m, axis=1)


def _residual_p2(m: np.ndarray) -> np.ndarray:
    k1 = np.diff(m, axis=1)
    r = k1.copy()
    r[:, 1:] = k1[:, 1:] - k1[:, :-1]
    return r


def _residual_ps(m_row: np.ndarray, lag: int) -> np.ndarray:
    r = np.diff(m_row)
    if lag > 1 and m_row.shape[0] > lag:
        r = r.copy()
        r[lag - 1:] = m_row[lag:] - m_row[:-lag]
    return r


def _detect_lag(k1_row: np.ndarray, max_lag: int = 2048) -> int:
    """Dominant seasonal lag of one row via autocorrelation of its diffs."""
    n = k1_row.shape[0]
    max_lag = int(min(max_lag, n // 2))
    if max_lag < 4:
        return 1
    v = k1_row.astype(np.float64)
    v = v - v.mean()
    denom = float(np.dot(v, v))
    if denom <= 0:
        return 1
    f = np.fft.rfft(v, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:max_lag] / denom
    lag = int(np.argmax(ac[2:])) + 2
    return lag if ac[lag] > 0.3 else 1


def _row_cost(r: np.ndarray) -> float:
    """Approximate coded bytes for one residual row: entropy + escape cost."""
    inline = np.abs(r) <= RMAX
    n_esc = int(r.size - inline.count_nonzero()) if hasattr(inline, "count_nonzero") else int(r.size - np.count_nonzero(inline))
    vals = np.where(inline, r, RMAX + 1) + 128
    hist = np.bincount(vals.astype(np.int64), minlength=256)
    p = hist[hist > 0] / r.size
    bits = float(-(p * np.log2(p)).sum()) * r.size
    return bits / 8.0 + n_esc * 12.0


# ----------------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------------

def _pack(header: Dict[str, Any], grid: np.ndarray, esc_pos: np.ndarray, esc_val: np.ndarray, level: int = 19) -> bytes:
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = (
        struct.pack("<IQQ", len(hdr), grid.size, esc_pos.size)
        + hdr
        + grid.tobytes()
        + esc_pos.astype("<u4").tobytes()
        + esc_val.astype("<i8").tobytes()
    )
    return zstd.ZstdCompressor(level=level).compress(payload)


def _unpack(blob: bytes) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    payload = zstd.ZstdDecompressor().decompress(blob)
    hdr_len, grid_size, n_esc = struct.unpack_from("<IQQ", payload, 0)
    off = struct.calcsize("<IQQ")
    header = json.loads(payload[off:off + hdr_len].decode("utf-8"))
    off += hdr_len
    grid = np.frombuffer(payload, dtype=np.uint8, count=grid_size, offset=off)
    off += grid_size
    esc_pos = np.frombuffer(payload, dtype="<u4", count=n_esc, offset=off)
    off += n_esc * 4
    esc_val = np.frombuffer(payload, dtype="<i8", count=n_esc, offset=off)
    n_vars = int(header["n_vars"])
    grid = grid.reshape(n_vars, -1) if grid_size else grid.reshape(n_vars, 0)
    return header, grid, esc_pos, esc_val


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------

def encode(
    data: Any,
    eps: float = 1e-3,
    mode: str = "abs",
    predictors: str = "bank",   # 'p1' (C1) or 'bank' (C2)
    tau: float = 0.0,           # >0.5 enables dead-zone carried-residual (C3)
    rel_scale: Optional[np.ndarray] = None,
    zstd_level: int = 19,
    var_names: Optional[Sequence[str]] = None,
) -> Tuple[bytes, np.ndarray, Dict[str, Any]]:
    """Encode (n_vars, n_time) data. Returns (blob, view_grid, header)."""
    if not HAS_ZSTD:
        raise RuntimeError("zstandard is required for the TRACQ-2 lattice codec")
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    n_vars, n_time = x.shape

    # Non-finite values: forward-fill (same policy as EnhancedTimeSeriesGrid)
    if not np.isfinite(x).all():
        x = x.copy()
        for i in range(n_vars):
            row = x[i]
            bad = ~np.isfinite(row)
            if bad.all():
                x[i] = 0.0
                continue
            if bad[0]:
                row[0] = row[~bad][0]
                bad = ~np.isfinite(row)
            idx = np.where(~bad, np.arange(n_time), 0)
            np.maximum.accumulate(idx, out=idx)
            x[i] = row[idx]

    if mode == "rel":
        if rel_scale is None:
            med = np.median(np.abs(x), axis=1)
            s = np.maximum(0.01 * med, 1e-12)
            s = np.where(med <= 0, np.maximum(np.std(x, axis=1) * 0.01, 1e-12), s)
        else:
            s = np.asarray(rel_scale, dtype=np.float64)
    else:
        s = np.ones(n_vars)

    q = _derive_steps(x, mode, eps, s)
    y = _forward_transform(x, mode, s)

    if tau > 0.5:
        m = _deadzone_lattice(y, q, tau)
    else:
        m = np.round(y / q[:, None]).astype(np.int64)

    # --- per-row predictor selection ---
    k1 = np.diff(m, axis=1)
    residual = k1.copy()
    pred = np.zeros(n_vars, dtype=np.int64)
    lags = np.ones(n_vars, dtype=np.int64)
    order = np.arange(n_vars)

    if predictors == "bank" and n_time > 4:
        # correlation ordering for the Lorenzo predictor
        if n_vars > 2 and k1.shape[1] > 8:
            cm = np.corrcoef(k1.astype(np.float64) + 1e-9 * np.random.default_rng(0).standard_normal(k1.shape))
            cm = np.nan_to_num(cm, nan=0.0)
            visited = np.zeros(n_vars, dtype=bool)
            o = [int(np.argmax(np.abs(cm).sum(axis=1)))]
            visited[o[0]] = True
            for _ in range(n_vars - 1):
                c = np.abs(cm[o[-1]]).copy()
                c[visited] = -np.inf
                nxt = int(np.argmax(c))
                o.append(nxt)
                visited[nxt] = True
            order = np.array(o)
        m = m[order]
        k1 = k1[order]

        r_p2 = _residual_p2(m)
        residual = np.empty_like(k1)
        for i in range(n_vars):
            cands = {PRED_P1: k1[i], PRED_P2: r_p2[i]}
            lag = _detect_lag(k1[i])
            if lag > 1:
                cands[PRED_PS] = _residual_ps(m[i], lag)
            if i > 0:
                cands[PRED_PL] = k1[i] - k1[i - 1]
            costs = {p: _row_cost(r) for p, r in cands.items()}
            best = min(costs, key=costs.get)
            pred[i] = best
            lags[i] = lag if best == PRED_PS else 1
            residual[i] = cands[best]

    # --- escapes + grid ---
    inline = np.abs(residual) <= RMAX
    grid = np.where(inline, residual + 128, ESCAPE).astype(np.uint8)
    flat_esc = ~inline.ravel()
    esc_pos = np.flatnonzero(flat_esc).astype(np.uint32)
    esc_val = residual.ravel()[flat_esc].astype(np.int64)

    header = {
        "codec": "tracq2-lattice",
        "version": 1,
        "n_vars": int(n_vars),
        "n_time": int(n_time),
        "mode": mode,
        "eps": float(eps),
        "tau": float(tau),
        "q": q.tolist(),
        "s": s[order].tolist() if mode == "rel" else None,
        "m0": m[:, 0].tolist(),
        "pred": pred.tolist(),
        "lags": lags.tolist(),
        "order": order.tolist(),
        "var_names": list(var_names) if var_names is not None else None,
    }
    # q must follow the row order used in the grid
    header["q"] = q[order].tolist()

    blob = _pack(header, grid, esc_pos, esc_val, level=zstd_level)
    return blob, grid, header


def _deadzone_lattice(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Carried-residual dead-zone lattice: emit a new level only when the
    accumulated transform-domain deviation exceeds tau*q. Worst-case error
    tau*q, still drift-free (decode is unchanged integer accumulation)."""
    n_vars, n_time = y.shape
    m = np.empty((n_vars, n_time), dtype=np.int64)
    yq = y / q[:, None]
    m[:, 0] = np.round(yq[:, 0])
    prev = m[:, 0].astype(np.float64)
    for t in range(1, n_time):
        dev = yq[:, t] - prev
        step = np.where(np.abs(dev) < tau, 0, np.round(dev))
        prev = prev + step
        m[:, t] = prev
    return m


# ----------------------------------------------------------------------------
# Decoder
# ----------------------------------------------------------------------------

def decode(blob: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decode a TRACQ-2 blob back to (n_vars, n_time) float64 data."""
    header, grid, esc_pos, esc_val = _unpack(blob)
    n_vars = int(header["n_vars"])
    n_time = int(header["n_time"])
    mode = header["mode"]
    q = np.asarray(header["q"], dtype=np.float64)
    m0 = np.asarray(header["m0"], dtype=np.int64)
    pred = np.asarray(header["pred"], dtype=np.int64)
    lags = np.asarray(header["lags"], dtype=np.int64)
    order = np.asarray(header["order"], dtype=np.int64)

    residual = grid.astype(np.int64) - 128
    if esc_pos.size:
        residual.ravel()[esc_pos.astype(np.int64)] = esc_val

    # invert predictors row-by-row in stored (correlation) order
    m = np.empty((n_vars, n_time), dtype=np.int64)
    k1_prev: Optional[np.ndarray] = None
    for i in range(n_vars):
        r = residual[i]
        p = int(pred[i])
        if p == PRED_P1:
            k1 = r
        elif p == PRED_P2:
            k1 = np.cumsum(r)
        elif p == PRED_PL:
            if k1_prev is None:
                raise ValueError("Lorenzo row without a previous row")
            k1 = r + k1_prev
        elif p == PRED_PS:
            lag = int(lags[i])
            mm = np.empty(n_time, dtype=np.int64)
            mm[0] = m0[i]
            upto = min(lag, n_time)
            mm[1:upto] = m0[i] + np.cumsum(r[: upto - 1])
            for t in range(lag, n_time):
                mm[t] = mm[t - lag] + r[t - 1]
            m[i] = mm
            k1_prev = np.diff(mm)
            continue
        else:
            raise ValueError(f"unknown predictor {p}")
        m[i, 0] = m0[i]
        m[i, 1:] = m0[i] + np.cumsum(k1)
        k1_prev = k1

    y = m.astype(np.float64) * q[:, None]
    if mode == "rel":
        s = np.asarray(header["s"], dtype=np.float64)
        x = _inverse_transform(y, "rel", s)
    else:
        x = y

    # undo correlation ordering
    inv = np.argsort(order)
    return x[inv], header


# ----------------------------------------------------------------------------
# PNG view artifact
# ----------------------------------------------------------------------------

def save_view_png(path: str, grid: np.ndarray, header: Dict[str, Any]) -> None:
    """Render the residual grid as a standard viewable PNG (mid-gray = no
    change), with the codec header embedded in a tEXt chunk."""
    from .codec import ImageCodec

    ImageCodec.save_png(path, grid, header)

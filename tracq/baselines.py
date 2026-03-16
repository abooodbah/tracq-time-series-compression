import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import tempfile
import os

# Simple baselines for quick lossy comparisons.

# ============================================================================
# Feature flags for optional dependencies
# ============================================================================
try:
    import zfpy
    HAS_ZFP = True
except ImportError:
    HAS_ZFP = False

try:
    import h5py
    import hdf5plugin
    HAS_HDF5PLUGIN = True
except ImportError:
    HAS_HDF5PLUGIN = False


# ============================================================================
# Error-Bounded Lossy Compressors (SZ3/ZFP) - HPC Gold Standards
# ============================================================================

def zfp_compress(
    x: np.ndarray,
    mode: str = "tolerance",
    tolerance: Optional[float] = None,
    rate: Optional[float] = None,
    precision: Optional[int] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Compress using ZFP (error-bounded lossy compressor).

    Args:
        x: 2D array (vars x time) of float64/float32 data
        mode: One of "tolerance" (absolute error bound), "rate" (bits per value),
              "precision" (bit planes), or "reversible" (lossless)
        tolerance: Absolute error tolerance (for mode="tolerance")
        rate: Bits per value (for mode="rate")
        precision: Number of bit planes (for mode="precision")

    Returns:
        (compressed_bytes, metadata)
    """
    if not HAS_ZFP:
        raise ImportError("zfpy not installed. Install with: pip install zfpy")

    # Ensure contiguous float64 array for ZFP
    arr = np.ascontiguousarray(x, dtype=np.float64)

    if mode == "tolerance":
        if tolerance is None:
            tolerance = 1e-3
        compressed = zfpy.compress_numpy(arr, tolerance=tolerance)
        meta = {"mode": "tolerance", "tolerance": float(tolerance)}
    elif mode == "rate":
        if rate is None:
            rate = 16.0
        compressed = zfpy.compress_numpy(arr, rate=rate)
        meta = {"mode": "rate", "rate": float(rate)}
    elif mode == "precision":
        if precision is None:
            precision = 32
        compressed = zfpy.compress_numpy(arr, precision=precision)
        meta = {"mode": "precision", "precision": int(precision)}
    elif mode == "reversible":
        compressed = zfpy.compress_numpy(arr)
        meta = {"mode": "reversible"}
    else:
        raise ValueError(f"Unknown ZFP mode: {mode}")

    meta["shape"] = list(arr.shape)
    meta["dtype"] = str(arr.dtype)
    meta["compressor"] = "zfp"

    return bytes(compressed), meta


def zfp_decompress(buf: bytes, meta: Dict[str, Any]) -> np.ndarray:
    """Decompress ZFP-compressed data."""
    if not HAS_ZFP:
        raise ImportError("zfpy not installed. Install with: pip install zfpy")

    return zfpy.decompress_numpy(buf)


def sz3_compress(
    x: np.ndarray,
    mode: str = "abs",
    abs_error: Optional[float] = None,
    rel_error: Optional[float] = None,
    psnr: Optional[float] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Compress using SZ3 via hdf5plugin (error-bounded lossy compressor).

    Args:
        x: 2D array (vars x time) of float64/float32 data
        mode: One of "abs" (absolute error), "rel" (relative error), "psnr"
        abs_error: Absolute error bound (for mode="abs")
        rel_error: Relative error bound (for mode="rel")
        psnr: Target PSNR (for mode="psnr")

    Returns:
        (compressed_bytes, metadata)
    """
    if not HAS_HDF5PLUGIN:
        raise ImportError("hdf5plugin not installed. Install with: pip install hdf5plugin h5py")

    arr = np.ascontiguousarray(x, dtype=np.float64)

    # Use a temporary HDF5 file to leverage SZ3's HDF5 filter
    fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)

    try:
        with h5py.File(tmp_path, 'w') as f:
            if mode == "abs":
                if abs_error is None:
                    abs_error = 1e-3
                f.create_dataset('data', data=arr,
                               compression=hdf5plugin.SZ3(absolute=abs_error))
                meta = {"mode": "abs", "abs_error": float(abs_error)}
            elif mode == "rel":
                if rel_error is None:
                    rel_error = 1e-3
                f.create_dataset('data', data=arr,
                               compression=hdf5plugin.SZ3(relative=rel_error))
                meta = {"mode": "rel", "rel_error": float(rel_error)}
            elif mode == "psnr":
                if psnr is None:
                    psnr = 60.0
                f.create_dataset('data', data=arr,
                               compression=hdf5plugin.SZ3(peak_signal_to_noise_ratio=psnr))
                meta = {"mode": "psnr", "psnr": float(psnr)}
            else:
                raise ValueError(f"Unknown SZ3 mode: {mode}")

        # Read the compressed HDF5 file as bytes
        with open(tmp_path, 'rb') as f:
            compressed = f.read()

        meta["shape"] = list(arr.shape)
        meta["dtype"] = str(arr.dtype)
        meta["compressor"] = "sz3"

        return compressed, meta
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def sz3_decompress(buf: bytes, meta: Dict[str, Any]) -> np.ndarray:
    """Decompress SZ3-compressed data stored in HDF5 format."""
    if not HAS_HDF5PLUGIN:
        raise ImportError("hdf5plugin not installed. Install with: pip install hdf5plugin h5py")

    fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)

    try:
        with open(tmp_path, 'wb') as f:
            f.write(buf)

        with h5py.File(tmp_path, 'r') as f:
            return f['data'][:]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def zfp_sweep(
    x: np.ndarray,
    tolerances: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
    rates: Tuple[float, ...] = (4, 8, 12, 16, 24, 32),
) -> List[Dict[str, Any]]:
    """
    Sweep ZFP configurations for rate-distortion analysis.

    Returns list of dicts with: {mode, param, bytes, rmse, ratio, ...}
    """
    if not HAS_ZFP:
        return []

    from .metrics import rmse as calc_rmse, corr as calc_corr

    results = []
    orig_bytes = x.nbytes

    # Tolerance sweep (absolute error bound)
    for tol in tolerances:
        try:
            compressed, meta = zfp_compress(x, mode="tolerance", tolerance=tol)
            decompressed = zfp_decompress(compressed, meta)

            results.append({
                "compressor": "zfp",
                "mode": "tolerance",
                "param": tol,
                "param_name": "tolerance",
                "bytes": len(compressed),
                "ratio": len(compressed) / orig_bytes,
                "rmse": float(calc_rmse(x, decompressed)),
                "corr": float(calc_corr(x, decompressed)),
                "max_error": float(np.max(np.abs(x - decompressed))),
            })
        except Exception as e:
            results.append({
                "compressor": "zfp",
                "mode": "tolerance",
                "param": tol,
                "error": str(e),
            })

    # Rate sweep (bits per value)
    for r in rates:
        try:
            compressed, meta = zfp_compress(x, mode="rate", rate=r)
            decompressed = zfp_decompress(compressed, meta)

            results.append({
                "compressor": "zfp",
                "mode": "rate",
                "param": r,
                "param_name": "rate",
                "bytes": len(compressed),
                "ratio": len(compressed) / orig_bytes,
                "rmse": float(calc_rmse(x, decompressed)),
                "corr": float(calc_corr(x, decompressed)),
                "max_error": float(np.max(np.abs(x - decompressed))),
            })
        except Exception as e:
            results.append({
                "compressor": "zfp",
                "mode": "rate",
                "param": r,
                "error": str(e),
            })

    return results


def sz3_sweep(
    x: np.ndarray,
    abs_errors: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
    rel_errors: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4),
) -> List[Dict[str, Any]]:
    """
    Sweep SZ3 configurations for rate-distortion analysis.

    Returns list of dicts with: {mode, param, bytes, rmse, ratio, ...}
    """
    if not HAS_HDF5PLUGIN:
        return []

    from .metrics import rmse as calc_rmse, corr as calc_corr

    results = []
    orig_bytes = x.nbytes

    # Absolute error sweep
    for err in abs_errors:
        try:
            compressed, meta = sz3_compress(x, mode="abs", abs_error=err)
            decompressed = sz3_decompress(compressed, meta)

            results.append({
                "compressor": "sz3",
                "mode": "abs",
                "param": err,
                "param_name": "abs_error",
                "bytes": len(compressed),
                "ratio": len(compressed) / orig_bytes,
                "rmse": float(calc_rmse(x, decompressed)),
                "corr": float(calc_corr(x, decompressed)),
                "max_error": float(np.max(np.abs(x - decompressed))),
            })
        except Exception as e:
            results.append({
                "compressor": "sz3",
                "mode": "abs",
                "param": err,
                "error": str(e),
            })

    # Relative error sweep
    for err in rel_errors:
        try:
            compressed, meta = sz3_compress(x, mode="rel", rel_error=err)
            decompressed = sz3_decompress(compressed, meta)

            results.append({
                "compressor": "sz3",
                "mode": "rel",
                "param": err,
                "param_name": "rel_error",
                "bytes": len(compressed),
                "ratio": len(compressed) / orig_bytes,
                "rmse": float(calc_rmse(x, decompressed)),
                "corr": float(calc_corr(x, decompressed)),
                "max_error": float(np.max(np.abs(x - decompressed))),
            })
        except Exception as e:
            results.append({
                "compressor": "sz3",
                "mode": "rel",
                "param": err,
                "error": str(e),
            })

    return results


def get_available_hpc_compressors() -> Dict[str, bool]:
    """Return availability of HPC compressors."""
    return {
        "zfp": HAS_ZFP,
        "sz3": HAS_HDF5PLUGIN,
    }

def paa_compress(x: np.ndarray, segments: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    # x shape: (vars, time)
    v, t = x.shape
    seg = max(1, segments)
    seg_len = int(np.ceil(t / seg))
    out = []
    for i in range(seg):
        sl = slice(i * seg_len, min(t, (i + 1) * seg_len))
        if sl.stop - sl.start <= 0:
            break
        out.append(x[:, sl].mean(axis=1))
    coeffs = np.stack(out, axis=1)
    meta = {"segments": coeffs.shape[1], "seg_len": seg_len, "orig_len": t}
    return coeffs, meta


def paa_decompress(coeffs: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    v = coeffs.shape[0]
    seg_len = meta.get("seg_len", 1)
    t = meta.get("orig_len", coeffs.shape[1] * seg_len)
    recon = np.zeros((v, t))
    idx = 0
    for j in range(coeffs.shape[1]):
        length = min(seg_len, t - idx)
        if length <= 0:
            break
        recon[:, idx:idx+length] = coeffs[:, [j]]
        idx += length
    return recon


def pla_compress(x: np.ndarray, segments: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    v, t = x.shape
    seg = max(1, segments)
    seg_len = int(np.ceil(t / seg))
    slopes = []
    intercepts = []
    for i in range(seg):
        sl = slice(i * seg_len, min(t, (i + 1) * seg_len))
        tt = np.arange(sl.stop - sl.start)
        y = x[:, sl]
        if y.shape[1] == 0:
            break
        # simple OLS per variable (closed form)
        denom = np.dot(tt, tt) if tt.size > 0 else 1.0
        if denom == 0:
            m = np.zeros((v,))
        else:
            m = (y @ tt) / denom
        b = y[:, 0]
        slopes.append(m)
        intercepts.append(b)
    slopes = np.stack(slopes, axis=1) if slopes else np.zeros((v, 0))
    intercepts = np.stack(intercepts, axis=1) if intercepts else np.zeros((v, 0))
    meta = {"segments": intercepts.shape[1], "seg_len": seg_len, "orig_len": t}
    coeffs = np.stack([intercepts, slopes], axis=0)  # shape (2, vars, segments)
    return coeffs, meta


def pla_decompress(coeffs: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    intercepts = coeffs[0]
    slopes = coeffs[1]
    v, s = intercepts.shape
    seg_len = meta.get("seg_len", 1)
    t = meta.get("orig_len", seg_len * s)
    recon = np.zeros((v, t))
    idx = 0
    for j in range(s):
        length = min(seg_len, t - idx)
        tt = np.arange(length)
        recon[:, idx:idx+length] = intercepts[:, [j]] + slopes[:, [j]] * tt
        idx += length
    return recon


def sax_compress(x: np.ndarray, segments: int, alphabet: int = 8) -> Tuple[np.ndarray, Dict[str, Any]]:
    # z-normalize per variable, PAA then quantize to alphabet
    v, t = x.shape
    xn = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)
    paa, meta = paa_compress(xn, segments)
    # breakpoints for Gaussian
    from scipy.stats import norm  # type: ignore
    bps = norm.ppf(np.linspace(0, 1, alphabet + 1)[1:-1])
    symbols = np.digitize(paa, bps).astype(np.int32)
    meta.update({"alphabet": alphabet, "bps": bps.tolist()})
    return symbols, meta


def sax_decompress(symbols: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    alphabet = meta.get("alphabet", 8)
    bps = np.array(meta.get("bps", []))
    # Use finite bin centers (avoid +/-inf from open-ended edges)
    if bps.size:
        centers = np.empty((alphabet,), dtype=float)
        if alphabet == 1:
            centers[0] = 0.0
        else:
            centers[0] = float(bps[0]) - 1.0
            centers[-1] = float(bps[-1]) + 1.0
            if alphabet > 2:
                centers[1:-1] = 0.5 * (bps[:-1] + bps[1:])
    else:
        centers = np.linspace(-1.0, 1.0, int(alphabet), dtype=float)

    paa = centers[symbols]
    v, s = paa.shape
    seg_len = meta.get("seg_len", 1)
    t = meta.get("orig_len", seg_len * s)
    recon = np.zeros((v, t))
    idx = 0
    for j in range(s):
        length = min(seg_len, t - idx)
        recon[:, idx:idx+length] = paa[:, [j]]
        idx += length
    return recon


def delta_zstd_compress(x: np.ndarray, level: int = 3) -> Tuple[bytes, Dict[str, Any]]:
    """
    Delta encoding + Zstandard compression (near-lossless baseline).

    First-order delta encoding converts the signal to differences, which
    are then compressed with Zstandard.  The only loss comes from
    float64 -> float64 delta precision, so the method is effectively
    lossless for float64 inputs.

    Args:
        x: 2D array (vars x time) of float64 data
        level: Zstandard compression level (1-22)

    Returns:
        (compressed_bytes, metadata)
    """
    import zstandard as zstd

    arr = np.ascontiguousarray(x, dtype=np.float64)
    # First-order delta along time axis
    first_col = arr[:, :1].copy()
    deltas = np.diff(arr, axis=1)
    # Pack first column + deltas
    payload = first_col.tobytes() + deltas.tobytes()
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(payload)
    meta = {
        "compressor": "delta_zstd",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "level": int(level),
    }
    return bytes(compressed), meta


def delta_zstd_decompress(buf: bytes, meta: Dict[str, Any]) -> np.ndarray:
    """Decompress delta+Zstd data back to the original array."""
    import zstandard as zstd

    shape = tuple(meta.get("shape", ()))
    n_vars, n_time = shape
    dctx = zstd.ZstdDecompressor()
    payload = dctx.decompress(buf)
    # Unpack first column and deltas
    first_col_bytes = n_vars * 8  # float64
    first_col = np.frombuffer(payload[:first_col_bytes], dtype=np.float64).reshape(n_vars, 1)
    deltas = np.frombuffer(payload[first_col_bytes:], dtype=np.float64).reshape(n_vars, n_time - 1)
    # Reconstruct via cumulative sum
    arr = np.concatenate([first_col, deltas], axis=1)
    arr = np.cumsum(arr, axis=1)
    return arr


def gorilla_like_compress(x: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
    # Simplified Gorilla-style: round to int32, delta-of-delta varint-coded, then zlib.
    import zlib
    xi = np.round(x).astype(np.int64)
    deltas = np.diff(xi, axis=1, prepend=xi[:, :1])
    dod = np.diff(deltas, axis=1, prepend=deltas[:, :1])
    flat = dod.ravel()
    encoded = bytearray()
    for v in flat:
        # zigzag
        zz = (v << 1) ^ (v >> 63)
        while True:
            byte = zz & 0x7F
            zz >>= 7
            if zz:
                encoded.append(byte | 0x80)
            else:
                encoded.append(byte)
                break
    comp = zlib.compress(bytes(encoded), level=3)
    meta = {"shape": x.shape, "dtype": "int64"}
    return comp, meta


def gorilla_like_decompress(buf: bytes, meta: Dict[str, Any]) -> np.ndarray:
    import zlib
    shape = tuple(meta.get("shape", ()))
    data = zlib.decompress(buf)
    vals = []
    n = len(data)
    i = 0
    while i < n:
        shift = 0
        zz = 0
        while True:
            b = data[i]
            i += 1
            zz |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        v = (zz >> 1) ^ -(zz & 1)
        vals.append(v)
    flat = np.array(vals, dtype=np.int64)
    # invert delta-of-delta
    dod = flat.reshape(shape[0], -1)
    deltas = np.cumsum(dod, axis=1)
    recon = np.cumsum(deltas, axis=1)
    return recon.astype(float)

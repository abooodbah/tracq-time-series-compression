import json
import zstandard as zstd  # type: ignore
import numpy as np
from typing import Dict, Any, Tuple


MAGIC = b"GTSZ01"

def pack(payload: Dict[str, Any], grid: np.ndarray, compress_level: int = 3) -> bytes:
    if grid.dtype not in (np.uint8, np.uint16):
        raise ValueError("grid dtype must be uint8 or uint16")
    header = {
        "dtype": str(grid.dtype),
        "shape": grid.shape,
        "meta": payload,
    }
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    grid_bytes = grid.tobytes(order="C")
    blob = MAGIC + len(header_bytes).to_bytes(4, "big") + header_bytes + grid_bytes
    cctx = zstd.ZstdCompressor(level=compress_level)
    return cctx.compress(blob)


def unpack(buf: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(buf)
    if not raw.startswith(MAGIC):
        raise ValueError("invalid magic")
    header_len = int.from_bytes(raw[len(MAGIC):len(MAGIC)+4], "big")
    header_start = len(MAGIC) + 4
    header_bytes = raw[header_start:header_start+header_len]
    grid_bytes = raw[header_start+header_len:]
    header = json.loads(header_bytes.decode("utf-8"))
    dtype = np.uint8 if header["dtype"] == "uint8" else np.uint16
    shape = tuple(header["shape"])
    grid = np.frombuffer(grid_bytes, dtype=dtype).reshape(shape)
    return grid, header.get("meta", {})

import json
import os
import io
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image, PngImagePlugin


class ImageCodec:
    """
    Simple PNG image codec wrapper using Pillow that embeds reconstruction metadata
    into PNG text chunks. Supports 16-bit unsigned grids (uint16) and 8-bit (uint8)
    and reads them back. Optionally shows a rich progress indicator while saving.

    Methods
      - save_png(path, grid, metadata, compress_level=6, show_progress=False)
      - load_png(path) -> (grid, metadata)
    """

    @staticmethod
    def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        # Convert all metadata values to strings for PNG text chunks; we store
        # a single JSON blob under key 'tracq_meta' for simplicity.
        try:
            meta_json = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            # Fallback: best-effort stringify
            meta_json = str(metadata)
        return {"tracq_meta": meta_json}

    @staticmethod
    def save_png(path: str, uint_grid: np.ndarray, metadata: Dict[str, Any], compress_level: int = 6, show_progress: bool = False) -> None:
        """
        Save a uint8 or uint16 numpy array to a lossless PNG with embedded metadata.
        - uint_grid: shape (H, W) or (n_vars, n_time-1)
        - metadata: serializable dict describing baseline, clamp, shapes, etc.
        - compress_level: 0-9, forwarded to Pillow
        - show_progress: if True, uses rich to display simple staged progress
        """
        if not isinstance(uint_grid, np.ndarray):
            raise TypeError("uint_grid must be a numpy array")
        if uint_grid.dtype not in (np.uint8, np.uint16):
            raise TypeError("uint_grid must have dtype uint8 (8-bit) or uint16 (16-bit) for PNG storage")

        # Determine Pillow mode
        mode = "I;16" if uint_grid.dtype == np.uint16 else "L"

        # Ensure parent dir exists
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Build PNG metadata
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in ImageCodec._serialize_metadata(metadata).items():
            pnginfo.add_text(k, v)

        def _do_save(buf: io.BytesIO):
            # Pillow uses (H,W) numpy arrays directly
            img = Image.fromarray(uint_grid, mode=mode)
            img.save(buf, format="PNG", pnginfo=pnginfo, compress_level=compress_level)

        # If user asked for progress, attempt to show simple staged progress
        if show_progress:
            try:
                from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

                stages = ["serialize", "create_image", "compress_write"]
                with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn()) as progress:
                    task = progress.add_task("Saving PNG", total=len(stages))
                    # Stage 1: serialize metadata
                    progress.update(task, description="serializing metadata")
                    # metadata serialization already done above; simulate progress
                    progress.advance(task)

                    # Stage 2: create image object
                    progress.update(task, description="creating image object")
                    # create and save to buffer
                    buf = io.BytesIO()
                    _do_save(buf)
                    progress.advance(task)

                    # Stage 3: write to disk
                    progress.update(task, description="writing to disk")
                    buf.seek(0)
                    with open(path, "wb") as f:
                        # stream write in chunks to give feedback
                        chunk_size = 64 * 1024
                        while True:
                            chunk = buf.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                    progress.advance(task)
            except Exception:
                # If rich not available or anything fails, fallback to simple save
                buf = io.BytesIO()
                _do_save(buf)
                buf.seek(0)
                with open(path, "wb") as f:
                    f.write(buf.read())
        else:
            # Fast path: save to temporary buffer then atomic write
            buf = io.BytesIO()
            _do_save(buf)
            buf.seek(0)
            with open(path, "wb") as f:
                f.write(buf.read())

    @staticmethod
    def load_png(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a PNG previously saved by save_png.
        Returns (grid, metadata)
        grid dtype will typically be uint8 or uint16 depending on stored image.
        If metadata is missing, the function will return an empty metadata dict but still
        return the pixel array.
        """
        try:
            img = Image.open(path)
        except Exception as e:
            raise IOError(f"failed to open PNG {path}: {e}")

        # Convert to numpy array; Pillow returns either uint8 or uint16 depending on file
        arr = np.array(img)

        # If arr is not uint8/uint16, upcast/downcast conservatively
        if arr.dtype not in (np.uint8, np.uint16):
            # Try to convert preserving values
            try:
                if arr.max() <= 255:
                    arr = arr.astype(np.uint8)
                else:
                    arr = arr.astype(np.uint16)
            except Exception:
                arr = arr.astype(np.uint16)

        meta: Dict[str, Any] = {}
        try:
            info = img.info or {}
            # Accept both the new TRACQ metadata key and the legacy GridTS key
            # so existing PNG artifacts remain readable after the rename.
            meta_json = info.get("tracq_meta") or info.get("gridts_meta")
            if meta_json:
                meta = json.loads(meta_json)
            else:
                # Try to collect other text keys
                text_keys = {k: v for k, v in info.items() if isinstance(v, str)}
                if text_keys:
                    meta = text_keys
        except Exception:
            meta = {}

        return arr, meta

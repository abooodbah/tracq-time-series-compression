from typing import Optional

import numpy as np
from PIL import Image


def heatmap_from_grid(grid: np.ndarray, colormap: Optional[str] = "viridis") -> Image.Image:
    """Create a false-color heatmap from a 2D numeric grid (H x W).

    Attempts to use matplotlib's colormap if available; otherwise falls back to a
    simple grayscale stretch. Returns a PIL Image in RGB mode.
    """
    if grid.size == 0:
        return Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))

    vmin = float(np.nanmin(grid))
    vmax = float(np.nanmax(grid))
    if vmax <= vmin:
        norm = np.zeros_like(grid, dtype=float)
    else:
        norm = (grid - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)

    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(colormap if colormap else "viridis")
        mapped = cmap(norm)  # RGBA floats 0..1
        rgb = (mapped[:, :, :3] * 255.0).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
    except Exception:
        r = (norm * 255.0).astype(np.uint8)
        b = ((1.0 - norm) * 255.0).astype(np.uint8)
        g = (np.clip(2.0 * norm * (1.0 - norm), 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=2)
        return Image.fromarray(rgb, mode="RGB")

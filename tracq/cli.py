from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from PIL import Image, PngImagePlugin
from rich.console import Console
from rich.progress import Progress

from .core import TimeSeriesGrid, rmse
from .container import pack as pack_zst, unpack as unpack_zst
from .codec import ImageCodec
from . import viewer

app = typer.Typer(help="TRACQ: CLI for grid-based time series compression")
console = Console()


def _read_csv(path: Path):
    """Read CSV into array shaped (n_vars, n_time).

    Heuristic: if pandas is available, treat each CSV column as a variable and
    rows as timesteps. Falls back to numpy.loadtxt.
    """
    # Fast path: try numpy first (avoids heavy pandas import on clean numeric CSVs)
    try:
        arr = np.loadtxt(str(path), delimiter=",")
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        else:
            if arr.shape[0] >= arr.shape[1]:
                arr = arr.T
        return arr.astype(float), None
    except Exception:
        pass

    # Fallback to pandas for messy CSVs
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(path, header=None, dtype=float, engine="c")
        arr = df.values.T.astype(float)
        return arr, None
    except Exception:
        # final fallback to numpy with default delimiter
        arr = np.loadtxt(str(path), delimiter=",")
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        else:
            if arr.shape[0] >= arr.shape[1]:
                arr = arr.T
        return arr.astype(float), None


def _quantize_with_levels(pct_grid: np.ndarray, clamp: float, levels: int) -> np.ndarray:
    """Quantize pct_grid into integer levels [0, levels-1]."""
    if pct_grid.size == 0:
        return np.zeros((pct_grid.shape[0], 0), dtype=np.int32)
    pct_clipped = np.clip(pct_grid, -clamp, clamp)
    scaled = (pct_clipped + clamp) / (2.0 * clamp) * (levels - 1)
    return np.round(scaled).astype(np.int32)


def _dequantize_from_levels(q_levels: np.ndarray, clamp: float, levels: int) -> np.ndarray:
    if q_levels.size == 0:
        return np.empty((q_levels.shape[0], 0), dtype=float)
    scaled = q_levels.astype(float) / (levels - 1)
    pct = scaled * (2.0 * clamp) - clamp
    return pct


def _dequantize_8bit(q: np.ndarray, clamp: float) -> np.ndarray:
    # wrapper to use core implementation
    return TimeSeriesGrid.dequantize_8bit(q, clamp)


def _serialize_png_to_bytes(img: Image.Image, metadata: dict, compress_level: int = 6) -> bytes:
    pnginfo = PngImagePlugin.PngInfo()
    meta_str = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)
    pnginfo.add_text("tracq_meta", meta_str)
    bio = io.BytesIO()
    img.save(bio, format="PNG", pnginfo=pnginfo, compress_level=compress_level)
    return bio.getvalue()


@app.command()
def compress(
    input: Path = typer.Argument(..., exists=True, readable=True, help="Input CSV file. Columns are treated as variables and rows as timesteps."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output PNG path. Defaults to <input>.tracq.png."),
    bits: Optional[int] = typer.Option(None, "-b", "--bits", help="Quantization bit depth: typically 8 or 16. If omitted, auto-tune is run."),
    clamp_pct: float = typer.Option(500.0, "--clamp", help="Clamp percent for delta values (positive)."),
    epsilon: float = typer.Option(1e-9, "--epsilon", help="Small epsilon to avoid division by zero)."),
    auto_tune: bool = typer.Option(False, "--auto-tune", help="Automatically pick bit-depth/levels based on RMSE and compressed size."),
    max_rmse: Optional[float] = typer.Option(None, "--max-rmse", help="If set, auto-tuner will prefer smaller bit-depth if RMSE <= this value."),
    force: bool = typer.Option(False, "-f", "--force", help="Overwrite existing output file without prompting."),
    rgb: bool = typer.Option(False, "--rgb", help="Experimental: map first three variables to R,G,B channels (8-bit only)."),
):
    """
    Compress a CSV time-series into a PNG grid using TRACQ encoding.

    If bits is omitted, auto-tuning will be performed when --auto-tune is provided.
    """
    if bits is not None and bits not in (8, 16):
        console.print("[red]Error:[/red] --bits must be 8 or 16 if provided")
        raise typer.Exit(code=2)

    out_path = output or (input.with_suffix(".tracq.png"))
    if out_path.exists() and not force:
        console.print(f"[yellow]Warning:[/yellow] output {out_path} already exists. Use --force to overwrite.")
        raise typer.Exit(code=3)

    try:
        data_arr, var_names = _read_csv(input)
    except Exception as e:
        console.print(f"[red]Failed to read input CSV:[/red] {e}")
        raise typer.Exit(code=4)

    grid = TimeSeriesGrid(data_arr, clamp_pct=clamp_pct, epsilon=epsilon, var_names=var_names)
    baseline, pct_grid = grid.compute_percent_grid()

    # If rgb experimental mode requested and there are fewer than 3 variables, error
    if rgb and grid.n_vars < 3:
        console.print("[red]Error:[/red] --rgb requires at least 3 variables (columns).")
        raise typer.Exit(code=7)

    chosen_bits = bits
    # Auto-tune if requested or bits omitted
    if auto_tune or chosen_bits is None:
        # binary search over bit-levels (4..16) using a data sample to find smallest bits that meets max_rmse
        sample_cols = min(512, max(1, grid.n_time))
        # build sample: use first sample_cols points
        sample_slice = slice(0, sample_cols)
        data_sample = data_arr[:, sample_slice]

        # default max_rmse is relative percent 1% of mean magnitude if not provided
        if max_rmse is None:
            mean_mag = np.mean(np.abs(data_sample)) if data_sample.size else 1.0
            default_abs_rmse = 0.01 * (mean_mag if mean_mag > 0 else 1.0)
            target_rmse = default_abs_rmse
        else:
            target_rmse = max_rmse

        low_bits = 4
        high_bits = 16
        best_bits: Optional[int] = None

        # Precompute percent grid for sample once
        sample_pct_grid = None
        if data_sample.shape[1] > 1:
            sample_grid = TimeSeriesGrid(data_sample, clamp_pct=clamp_pct, epsilon=epsilon)
            _, sample_pct_grid = sample_grid.compute_percent_grid()

        while low_bits <= high_bits:
            mid = (low_bits + high_bits) // 2
            levels = 2 ** mid
            # quantize sample
            # need pct_grid sample for quantization: take pct of first sample_len points
            if data_sample.shape[1] <= 1:
                # trivial case
                test_rmse = 0.0
            else:
                q_levels = _quantize_with_levels(sample_pct_grid, clamp_pct, levels)
                deq_pct = _dequantize_from_levels(q_levels, clamp_pct, levels)
                # reconstruct sample timeseries
                factors = 1.0 + deq_pct / 100.0
                cumprod = np.cumprod(factors, axis=1)
                recon = data_sample[:, :1] * np.concatenate([np.ones((data_sample.shape[0], 1)), cumprod], axis=1)
                test_rmse = rmse(data_sample, recon)

            # If rmse acceptable, try to lower bits (higher compression)
            if np.isfinite(test_rmse) and test_rmse <= target_rmse:
                best_bits = mid
                high_bits = mid - 1
            else:
                low_bits = mid + 1

        if best_bits is None:
            # nothing satisfied target; pick highest precision
            chosen_bits = 16
        else:
            # map best_bits to physical bit depth supported: <=8 ->8 else ->16
            chosen_bits = 8 if best_bits <= 8 else 16

        console.print(f"[green]Auto-tune:[/green] target_rmse={target_rmse:.6g}, selected bit depth={chosen_bits}")

    # Build both candidates but reuse serialized bytes to avoid extra encodes
    candidates = []

    try:
        q16, meta16 = grid.quantize_16bit()
        metadata16 = dict(meta16)
        metadata16.update({"bit_depth": 16})
        img16 = Image.fromarray(q16, mode="I;16") if q16.size > 0 else Image.fromarray(np.zeros((q16.shape[0], 1), dtype=np.uint16), mode="I;16")
        bytes16 = _serialize_png_to_bytes(img16, metadata16)
        try:
            recon16, _ = TimeSeriesGrid.reconstruct_from_quantized(q16, metadata16)
            rmse16 = rmse(data_arr, recon16)
        except Exception:
            rmse16 = float("inf")
        candidates.append({"bits": 16, "bytes": len(bytes16), "rmse": rmse16, "q": q16, "metadata": metadata16, "png_bytes": bytes16})
    except Exception:
        candidates.append({"bits": 16, "bytes": float("inf"), "rmse": float("inf"), "q": None, "metadata": None, "png_bytes": None})

    try:
        # percent grid computed earlier
        if pct_grid.size == 0:
            q8 = np.zeros((pct_grid.shape[0], 0), dtype=np.uint8)
        else:
            pct_clipped = np.clip(pct_grid, -clamp_pct, clamp_pct)
            scaled = (pct_clipped + clamp_pct) / (2.0 * clamp_pct) * 255.0
            q8 = np.round(scaled).astype(np.uint8)
        metadata8 = {
            "n_vars": int(grid.n_vars),
            "n_time": int(grid.n_time),
            "clamp_pct": float(clamp_pct),
            "epsilon": float(epsilon),
            "dtype": "uint8",
            "baseline": baseline.tolist(),
            "var_names": var_names if var_names is not None else None,
            "bit_depth": 8,
        }
        if rgb and chosen_bits == 8 and grid.n_vars >= 3:
            # stack first 3 vars into RGB image
            stacked = np.stack([q8[0], q8[1], q8[2]], axis=2)
            img8 = Image.fromarray(stacked, mode="RGB")
        else:
            img8 = Image.fromarray(q8, mode="L") if q8.size > 0 else Image.fromarray(np.zeros((q8.shape[0], 1), dtype=np.uint8), mode="L")
        bytes8 = _serialize_png_to_bytes(img8, metadata8)
        # Reconstruct 8-bit
        try:
            recon8, _ = TimeSeriesGrid.reconstruct_from_quantized(q8, metadata8)
            rmse8 = rmse(data_arr, recon8)
        except Exception:
            rmse8 = float("inf")
        candidates.append({"bits": 8, "bytes": len(bytes8), "rmse": rmse8, "q": q8, "metadata": metadata8, "png_bytes": bytes8})
    except Exception:
        candidates.append({"bits": 8, "bytes": float("inf"), "rmse": float("inf"), "q": None, "metadata": None, "png_bytes": None})

    # Choose final candidate
    chosen = None
    if auto_tune or bits is None:
        # prefer smaller bytes among candidates that meet max_rmse if provided
        viable = [c for c in candidates if c["q"] is not None and c["rmse"] != float("inf")]
        if max_rmse is not None:
            by_rmse = [c for c in viable if c["rmse"] <= max_rmse]
            if by_rmse:
                chosen = min(by_rmse, key=lambda c: c["bytes"])
        if chosen is None and viable:
            chosen = min(viable, key=lambda c: c["bytes"])
        if chosen is None:
            # fallback to candidate matching chosen_bits
            chosen = next((c for c in candidates if c["bits"] == chosen_bits), candidates[0])
    else:
        chosen = next((c for c in candidates if c["bits"] == bits), None)

    if chosen is None or chosen.get("q") is None:
        console.print("[red]Error:[/red] failed to select a valid quantization candidate")
        raise typer.Exit(code=5)

    # Present summary to user
    console.print("[bold]Compression Summary[/bold]")
    console.print(f"Input: {input}")
    console.print(f"Output (pending): {out_path}")
    console.print(f"Chosen bit depth: [green]{chosen['bits']}[/green]")
    console.print(f"Estimated compressed bytes (in-memory): [cyan]{chosen['bytes']}[/cyan]")
    console.print(f"Reconstruction RMSE (approx): [magenta]{chosen['rmse']:.6g}[/magenta]")

    # Ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save chosen image to disk with metadata
    try:
        if "png_bytes" in chosen and chosen["png_bytes"] is not None:
            with open(out_path, "wb") as f:
                f.write(chosen["png_bytes"])
        else:
            pnginfo = PngImagePlugin.PngInfo()
            meta_blob = json.dumps(chosen.get("metadata", {}), separators=(",", ":"), ensure_ascii=False)
            pnginfo.add_text("tracq_meta", meta_blob)
            img_mode = "I;16" if chosen.get("bits", 16) == 16 else "L"
            img = Image.fromarray(chosen["q"], mode=img_mode)
            img.save(str(out_path), format="PNG", pnginfo=pnginfo, compress_level=6)
    except Exception as e:
        console.print(f"[red]Failed to write output PNG:[/red] {e}")
        raise typer.Exit(code=6)

    console.print(f"[bold green]Wrote[/bold green] {out_path}")


@app.command()
def decompress(
    input_png: Path = typer.Argument(..., exists=True, readable=True, help="Input TRACQ PNG file."),
    output_csv: Optional[Path] = typer.Option(None, "-o", "--output", help="Output CSV path. Defaults to <input>.recon.csv."),
    force: bool = typer.Option(False, "-f", "--force", help="Overwrite existing output file without prompting."),
    original: Optional[Path] = typer.Option(None, "--original", help="Optional original CSV to compute RMSE against."),
    assume_baseline_zero: bool = typer.Option(False, "--assume-baseline-zero", help="If metadata missing, assume baseline zeros to allow reconstruction (unsafe)."),
    manual_baseline: Optional[str] = typer.Option(None, "--manual-baseline", help="Comma-separated baseline values to use if metadata stripped, e.g., '100.0,100.0'"),
):
    """Decompress a TRACQ PNG into a reconstructed CSV timeseries.

    If PNG metadata has been stripped, use --assume-baseline-zero to reconstruct with
    zeros baseline (may be incorrect), or supply --manual-baseline with comma-separated
    baseline values matching the number of variables.
    """
    out_path = output_csv or input_png.with_suffix(".recon.csv")
    if out_path.exists() and not force:
        console.print(f"[yellow]Warning:[/yellow] output {out_path} already exists. Use --force to overwrite.")
        raise typer.Exit(code=3)

    try:
        arr, meta = ImageCodec.load_png(str(input_png))
    except Exception as e:
        console.print(f"[red]Failed to load PNG:[/red] {e}")
        raise typer.Exit(code=4)

    # If metadata missing, handle gracefully: allow manual baseline or assumption flag
    if not meta or "baseline" not in meta:
        console.print("[yellow]Warning:[/yellow] PNG metadata missing or stripped; cannot find reconstruction metadata.")
        # infer dtype if possible
        inferred_dtype = None
        if arr.dtype == np.uint8:
            inferred_dtype = "uint8"
        elif arr.dtype == np.uint16:
            inferred_dtype = "uint16"
        else:
            # fallback
            inferred_dtype = "uint16"
        meta = meta or {}
        meta.setdefault("dtype", inferred_dtype)

        if manual_baseline:
            try:
                vals = [float(x.strip()) for x in manual_baseline.split(",") if x.strip() != ""]
                meta["baseline"] = vals
                meta["n_vars"] = len(vals)
                console.print(f"[green]Info:[/green] Using manual baseline of length {len(vals)}")
            except Exception as e:
                console.print(f"[red]Error:[/red] failed to parse --manual-baseline: {e}")
                raise typer.Exit(code=8)
        elif assume_baseline_zero:
            # attempt to infer number of variables from image shape
            if arr.ndim == 2:
                n_vars = arr.shape[0]
            elif arr.ndim == 3 and arr.shape[2] == 3:
                console.print("[red]Error:[/red] image appears to be RGB; cannot safely assume baseline length. Provide --manual-baseline.")
                raise typer.Exit(code=8)
            else:
                n_vars = arr.shape[0]
            meta["baseline"] = [0.0] * int(n_vars)
            meta.setdefault("n_vars", int(n_vars))
            meta.setdefault("n_time", int(arr.shape[1] + 1 if arr.ndim == 2 else 1))
            console.print("[yellow]Warning:[/yellow] No metadata found; reconstructing using zero baseline (this may be incorrect).")
        else:
            console.print("[red]Error:[/red] PNG is missing reconstruction metadata. Provide --manual-baseline or use --assume-baseline-zero to proceed.")
            console.print("Example: --manual-baseline '100.0,100.0' or --assume-baseline-zero")
            raise typer.Exit(code=8)

    # Determine dtype and dequantize
    try:
        dtype = meta.get("dtype", None)
        if dtype == "uint8" or arr.dtype == np.uint8:
            q = arr.astype(np.uint8)
        else:
            # treat as uint16 by default
            q = arr.astype(np.uint16)

        # If RGB image (3 channels) - not a quantized percent grid
        if q.ndim == 3 and q.shape[2] == 3:
            console.print("[red]Error:[/red] input PNG appears to be an RGB visualization, not a TRACQ quantized grid.")
            raise typer.Exit(code=9)

        recon, info = TimeSeriesGrid.reconstruct_from_quantized(q, meta)
    except Exception as e:
        console.print(f"[red]Failed to reconstruct timeseries:[/red] {e}")
        raise typer.Exit(code=5)

    # Write CSV: rows=time, cols=variables to be friendly with CSV consumers
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # transpose to time x variables
        np.savetxt(str(out_path), recon.T, delimiter=",", fmt="%.10g")
    except Exception as e:
        console.print(f"[red]Failed to write CSV:[/red] {e}")
        raise typer.Exit(code=6)

    console.print(f"[bold green]Wrote reconstructed CSV[/bold green] {out_path}")

    # Optionally compute RMSE if original provided
    if original is not None and original.exists():
        try:
            orig_arr, _ = _read_csv(original)
            error = rmse(orig_arr, recon)
            console.print(f"Reconstruction RMSE vs original: [magenta]{error:.6g}[/magenta]")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] failed to compute RMSE vs original: {e}")


@app.command()
def compress_zst(
    input: Path = typer.Argument(..., exists=True, readable=True, help="Input CSV file."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output .tracq.zst path. Defaults to <input>.tracq.zst."),
    bits: int = typer.Option(8, "--bits", help="Quantization bit depth (8 or 16)."),
    clamp_pct: float = typer.Option(500.0, "--clamp", help="Clamp percent for delta values."),
    epsilon: float = typer.Option(1e-9, "--epsilon", help="Small epsilon to avoid division by zero."),
    compress_level: int = typer.Option(3, "--zstd-level", help="zstd compression level"),
    force: bool = typer.Option(False, "-f", "--force", help="Overwrite existing output"),
    original: Optional[Path] = typer.Option(None, "--original", help="Optional original CSV to report RMSE."),
):
    if bits not in (8, 16):
        console.print("[red]Error:[/red] bits must be 8 or 16")
        raise typer.Exit(code=2)

    out_path = output or input.with_suffix(".tracq.zst")
    if out_path.exists() and not force:
        console.print(f"[yellow]Warning:[/yellow] {out_path} exists; use --force to overwrite")
        raise typer.Exit(code=3)

    data_arr, var_names = _read_csv(input)
    grid = TimeSeriesGrid(data_arr, clamp_pct=clamp_pct, epsilon=epsilon, var_names=var_names)
    if bits == 8:
        q, meta = grid.quantize_8bit()
    else:
        q, meta = grid.quantize_16bit()
    meta = dict(meta)
    meta["bit_depth"] = bits
    meta["var_names"] = var_names if var_names is not None else None
    blob = pack_zst(meta, q, compress_level=compress_level)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(blob)
    console.print(f"[bold green]Wrote[/bold green] {out_path} ({len(blob)} bytes)")

    if original is not None and original.exists():
        recon, _ = TimeSeriesGrid.reconstruct_from_quantized(q, meta)
        err = rmse(data_arr, recon)
        console.print(f"RMSE vs original: {err:.6g}")


@app.command()
def decompress_zst(
    input_zst: Path = typer.Argument(..., exists=True, readable=True, help="Input .tracq.zst file."),
    output_csv: Optional[Path] = typer.Option(None, "-o", "--output", help="Output CSV path."),
    force: bool = typer.Option(False, "-f", "--force", help="Overwrite output"),
    original: Optional[Path] = typer.Option(None, "--original", help="Optional original CSV to compute RMSE."),
):
    out_path = output_csv or input_zst.with_suffix(".recon.csv")
    if out_path.exists() and not force:
        console.print(f"[yellow]Warning:[/yellow] {out_path} exists; use --force to overwrite")
        raise typer.Exit(code=3)

    with open(input_zst, "rb") as f:
        buf = f.read()
    grid, meta = unpack_zst(buf)
    recon, _ = TimeSeriesGrid.reconstruct_from_quantized(grid, meta)
    np.savetxt(str(out_path), recon.T, delimiter=",", fmt="%.10g")
    console.print(f"[bold green]Wrote reconstructed CSV[/bold green] {out_path}")

    if original is not None and original.exists():
        orig_arr, _ = _read_csv(original)
        err = rmse(orig_arr, recon)
        console.print(f"RMSE vs original: {err:.6g}")


@app.command()
def view(
    input_png: Path = typer.Argument(..., exists=True, readable=True, help="Input TRACQ PNG file."),
    output_img: Optional[Path] = typer.Option(None, "-o", "--output", help="Output visualization PNG path. Defaults to <input>.view.png."),
    colormap: str = typer.Option("viridis", "--colormap", help="Matplotlib colormap name to render heatmap; falls back to grayscale if unavailable."),
    force: bool = typer.Option(False, "-f", "--force", help="Overwrite existing output file without prompting."),
    rgb: bool = typer.Option(False, "--rgb", help="If the PNG encodes 3 channels (experimental), render them as color RGB."),
):
    """Generate a false-color heatmap of the percent-change noise grid to visualize volatility."""
    out_path = output_img or input_png.with_suffix(".view.png")
    if out_path.exists() and not force:
        console.print(f"[yellow]Warning:[/yellow] output {out_path} already exists. Use --force to overwrite.")
        raise typer.Exit(code=3)

    try:
        arr, meta = ImageCodec.load_png(str(input_png))
    except Exception as e:
        console.print(f"[red]Failed to load PNG:[/red] {e}")
        raise typer.Exit(code=4)

    # If RGB experimental file (3-channel), handle separately
    try:
        if rgb and arr.ndim == 3 and arr.shape[2] == 3:
            # assume it's already an RGB image; save directly (or copy)
            img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        else:
            # extract pct_grid from metadata and pixel grid
            dtype = meta.get("dtype", None)
            if dtype == "uint8" or arr.dtype == np.uint8:
                q = arr.astype(np.uint8)
                pct_grid = _dequantize_8bit(q, float(meta.get("clamp_pct", 500.0))) if q.size > 0 else np.empty((0, 0), dtype=float)
            else:
                q = arr.astype(np.uint16)
                pct_grid = TimeSeriesGrid.dequantize_16bit(q, float(meta.get("clamp_pct", 500.0))) if q.size > 0 else np.empty((0, 0), dtype=float)

            # noise grid = absolute percent changes
            noise = np.abs(pct_grid)
            img = viewer.heatmap_from_grid(noise, colormap=colormap)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path), format="PNG")
    except Exception as e:
        console.print(f"[red]Failed to render view:[/red] {e}")
        raise typer.Exit(code=5)

    console.print(f"[bold green]Wrote visualization[/bold green] {out_path}")


if __name__ == "__main__":
    app()

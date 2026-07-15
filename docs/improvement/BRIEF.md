# TRACQ Improvement Brief (for design-proposal agents)

## Mission
The paper "Efficient Analysis of Multivariate Time Series with Adaptive Compression and Quantization" (TRACQ) was **rejected from IEEE TBD because compression performance did not beat competing algorithms**. Your job: propose design improvements that let TRACQ **genuinely surpass** the competition in rate-distortion (compression ratio vs reconstruction error), while **keeping the core idea**.

## The core idea (MUST be preserved)
1. Represent a multivariate time series (N variables × T steps) as a **2D grid of relative changes** (one pixel per variable per timestep).
2. Apply **adaptive quantization** to that grid.
3. Store the quantized grid as a **standard image (PNG) or Zstd blob** with reconstruction metadata — the compressed artifact remains a *viewable image* supporting visual inspection and compressed-domain anomaly detection.

Improvements may change *how* relative changes are computed, predicted, quantized, and entropy-coded — but the artifact must remain a per-timestep 2D grid interpretable as an image (per-timestep temporal resolution preserved, no aggressive downsampling like PAA).

## Current pipeline (as implemented in `tracq/core_enhanced.py`)
- Percentage change: P[i,t] = 100*(X[i,t]-X[i,t-1])/max(|X[i,t-1]|,eps)  — computed OPEN-LOOP from original data.
- Optional per-variable baseline offset for near-zero channels.
- Per-variable adaptive clamp C_i (percentile/max blend), normalize to [-1,1].
- Mu-law companding (mu=255 fixed), uniform quantization to 8 or 16 bits.
- Decode: dequantize → inverse mu-law → cumulative product: X̂[t] = X̂[t-1]*(1+P̂[t]/100). Errors compound multiplicatively.
- Optional anchors: exact float64 values every A steps to reset drift.
- Container: grayscale PNG (DEFLATE) with JSON metadata in tEXt chunks, or Zstd blob.

## Measured failure modes (from the submitted paper)
1. **ZFP and SZ3 dominate RMSE by 3–5 orders of magnitude at comparable ratios.** E.g. UCI Air Quality: ZFP tol=1e-1 gets RMSE 0.004 at ratio 0.273; TRACQ Enh 8b+Anchors gets RMSE 108.3 at ratio 0.114. TRACQ never wins the rate-distortion comparison.
2. **MetroPT-3 (1.5M steps): unanchored TRACQ diverges to RMSE ~3.8e95.** Anchored every 100 steps: RMSE 1.71 at ratio 0.0297 — but PAA-1024 gets RMSE 1.55 at ratio 0.0007. TRACQ is not Pareto-optimal.
3. **Lossy TRACQ ratios (0.05–0.22) barely beat lossless Delta+Zstd (0.12–0.24).** A lossy method that isn't much smaller than lossless is hard to justify.
4. Near-zero channels are numerically unstable (percent change blows up), patched with offset heuristics.
5. No error bound: users cannot specify a max error, unlike SZ3/ZFP.

## Root-cause observations (verified in code)
- **Open-loop encoding**: quantized percent-changes are applied by cumprod at decode; per-step quantization errors compound without bound. Classic fix in DPCM literature: closed-loop encoding (encoder maintains the decoder's reconstruction and encodes the change relative to the *reconstruction*, not the original), which bounds per-step error to one quantization step and eliminates drift entirely — anchors become unnecessary.
- Only an order-1 "previous value" predictor is used. No delta-of-delta, no linear extrapolation, no cross-channel exploitation.
- Entropy stage is PNG DEFLATE on the raw quantized grid; mu-law output uses the full [0,255] range so DEFLATE has little skew to exploit. No explicit residual sparsity is created or exploited.
- mu is fixed at 255 regardless of data distribution; clamp is percentile-based, not rate-distortion-optimized.
- 8/16-bit are the only operating points; no per-variable bit allocation.

## Competitors to beat (implementations available in repo / pip)
- ZFP (zfpy, tolerance mode), SZ3 (prediction + error-bounded quantization + entropy coding)
- PAA / SAX (aggressive downsampling), simplified Gorilla (delta-of-delta varint + zlib)
- Lossless: Gzip, Delta+Zstd, Parquet
- Metrics used: RMSE, MAE, SMAPE, Pearson rho, compression ratio (encoded/original bytes), encode/decode MB/s.

## Repo layout (clone at C:\Users\Abdulfatah\personal\research\tracq\tracq-time-series-compression)
- tracq/core.py – base TRACQ; tracq/core_enhanced.py – enhanced variant (read this first)
- tracq/codec.py, container.py – PNG/Zstd packaging; tracq/baselines.py – PAA/SAX/Gorilla/etc.
- scripts/realworld_benchmark.py, rate_distortion.py, metropt3_rate_distortion.py – paper benchmarks
- Python 3.12, numpy 2.4, zfpy, zstandard, pillow available on this machine.

## What a winning proposal looks like
- Keeps the image-grid artifact and per-timestep resolution.
- Closes the RMSE gap to ZFP/SZ3 at equal or better ratios, or wins ratio at equal RMSE — on the real UCI datasets, not just synthetic.
- Ideally provides a user-specifiable error bound (max abs/rel error per point).
- Simple enough to implement in numpy in this repo and benchmark honestly.
- State expected gains and WHY (information-theoretic or empirical reasoning), implementation risk, and how it preserves visual inspectability.

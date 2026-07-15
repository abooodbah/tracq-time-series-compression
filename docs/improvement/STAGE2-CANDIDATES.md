# Stage 2 — Refined Candidate Algorithms

Synthesis of the six Stage-1 proposal sets (docs/improvement/stage1/*.json), refined
into implementable candidates. All six lenses independently converged on the same
root-cause diagnosis and the same top fix; the candidates below combine their
mechanisms into coherent codecs sharing one substrate.

## Diagnosis recap (unanimous across lenses)
1. **Open-loop cumprod decode** turns per-step quantization error into an unbounded
   multiplicative random walk (with bias => exponential divergence). Anchors are an
   expensive bandage. This is most of the 3–5 order RMSE gap and all of the
   MetroPT-3 1e95 blowup.
2. **mu=255 companding maximizes symbol entropy** (~7.3–8 bits/symbol across the full
   byte range), so PNG-DEFLATE/Zstd compress almost nothing: measured ~0.9 bytes out
   of 1 per sample. The entropy stage is dead weight by design.
3. **No user error bound** — SZ3/ZFP's headline capability is absent.
4. Percent-change domain is singular near zero => epsilon/auto_offset heuristics.

## Shared substrate: the TRACQ-2 lattice core
Keeps the core TRACQ idea (per-timestep 2D relative-change grid, adaptive
per-variable quantization, image-viewable artifact) while being drift-free and
error-bounded *by construction*:

- **Transform (per variable):** y_i[t] = asinh(x_i[t] / s_i). Native handling of
  zeros and negatives — deletes epsilon, auto_offset, near_zero_std_factor. The
  scale s_i sets the abs/rel crossover: |x| << s_i behaves like x/s_i (absolute
  error regime), |x| >> s_i like ln|2x/s_i| (relative error regime).
  - `mode=abs`: s_i large (≈ sigma_i or range-derived) — bound is ~absolute; best RMSE.
  - `mode=rel`: s_i small — bound is ~relative; best SMAPE / multi-scale story vs ZFP.
- **Lattice quantization (this IS the adaptive quantization):**
  m_i[t] = round(y_i[t] / q_i), per-variable step q_i derived from a
  **user-specified error bound** eps: q_i = 2*asinh-domain-step such that transform-
  domain error ≤ q_i/2 pointwise, which maps to |x̂−x| ≤ s_i·sinh-propagated bound.
- **Grid (the image):** k_i[t] = m_i[t] − m_i[t−1], stored as uint8 pixel k+128.
  Mid-gray = no change; bright/dark = up/down moves — same visual semantics as
  TRACQ today, but the histogram is a zero-centered Laplacian => actually
  compressible, and quiet stretches are literal mid-gray.
- **No drift, no anchors:** decode is m = m0 + cumsum(k) (exact integer arithmetic),
  x̂ = s·sinh(q·m). cumsum(diff(m)) == m identically => |ŷ−y| ≤ q/2 for ALL t.
  (Verified by a Stage-1 agent at 1.5M steps: max err 0.00200 with q/2 = 0.002.)
  Equivalent to closed-loop DPCM but fully vectorized — no sequential encode loop.
- **Escape sidecar (exactness for outliers):** codes are clipped to [-126,127];
  pixel 255 marks escapes; exact residuals go to a sparse Zstd sidecar
  (delta-coded positions + int values), patched before cumsum at decode. Replaces
  the clamp's unbounded outlier destruction. Sidecar bytes count in every ratio.
- **Dual artifact:** PRIMARY = Zstd-19 blob (residual bytes + sidecar + metadata)
  — the honest measured size. VIEW = deterministic PNG render of the same grid
  (free to regenerate from the blob). Report both; benchmark uses primary.

## Candidates for the Stage-3 bake-off
- **C1 = TRACQ-CL** ("closed loop, minimal"): substrate as-is, order-1 predictor
  (k = diff(m)). The pure root-cause fix; the ablation baseline for the paper.
- **C2 = TRACQ-EB** ("error-bounded, predictor bank"): C1 + per-row integer
  predictor selection among
  {P0: identity/constant, P1: diff, P2: delta-of-delta, PL: 2D Lorenzo on
  correlation-reordered rows, PS: seasonal lag-L diff (L from autocorrelation)}
  chosen by per-row residual byte-entropy (vectorized), 3-bit choice per row in
  metadata. All integer-linear => exact integer decode (nested cumsums).
- **C3 = TRACQ-DZ** ("dead-zone rate knob"): C2 + carried-residual dead-zone
  encoder (emit 0 while accumulated transform residual < tau*q; worst-case error
  tau*q, still bounded, decode unchanged). Sequential encode (numba if present,
  chunked numpy fallback). tau sweep traces the high-ratio Pareto end.
- Each candidate runs in `abs` and `rel` modes, sweeping eps to trace full R-D curves.

## Predicted outcomes (to be validated honestly in Stage 3)
- C1 alone: RMSE drops 2–5 orders at unchanged ratio; MetroPT-3 divergence gone.
- C2: additional ~1.3–2x ratio improvement from peaky residuals + predictor bank.
- Target vs ZFP on UCI: comparable RMSE at ~2–3x smaller ratio (ZFP pays 0.27–0.47
  ratios for its RMSE; the lattice+Zstd path should reach similar RMSE near 0.10).
- vs PAA on MetroPT-3: match/beat RMSE 1.55 at competitive ratio while keeping
  per-timestep resolution (PAA cannot).
- Differentiators ZFP structurally cannot match: guaranteed pointwise mixed
  abs/rel bound on multi-scale data (ZFP's absolute-error SMAPE plateau), viewable
  artifact, compressed-domain analytics.

## What gets retired
mu-law companding, percentile clamp blending, anchors + their JSON overhead,
auto_offset/epsilon heuristics — all subsumed. They remain in the repo for the
paper's ablation/comparison tables.

## Honest-benchmark rules (Stage 3)
- All ratios include metadata + sidecar bytes; primary-artifact bytes measured.
- ZFP swept over tolerances (existing baseline path); include SZ3 REL-mode via WSL
  if available; PAA/PLA/SAX segment sweeps; Delta+Zstd, Gzip lossless floor.
- Same datasets as the paper: 3 UCI (5000 steps) + synthetic suite + MetroPT-3
  full-length for the divergence/scale story.
- Decode-side verification test: decode(encode(x)) must satisfy the stated bound
  for every point (unit test), including escapes.

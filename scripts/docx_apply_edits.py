# -*- coding: utf-8 -*-
"""Apply the enhanced-variant revision to the manuscript copy as tracked changes."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, mr, msub

DOC = (r"C:\Users\ABDULF~1\AppData\Local\Temp\claude"
       r"\C--Users-Abdulfatah--codex-skills-multi-agent-coordinator-v2"
       r"\98957d2f-346c-4497-89c3-63d184c5c33c\scratchpad\docx_work\unpacked\word\document.xml")

ed = Editor(DOC)
R = ed.replace

# ============================ ABSTRACT ============================
R("compresses/represents multivariate time series as two-dimensional grids of percentage changes",
  "represents multivariate time series as two-dimensional grids of relative changes")
R("(1) per-variable adaptive clamping that tailors quantization ranges to each feature’s variance, "
  "(2) mu-law companding that enables us to preserve more resolution near zero, where most changes occur, "
  "and (3) optional anchor points that bound multiplicative error drift over long horizons.",
  "(1) a per-variable scale-aware transform that assigns each channel its own quantization lattice, "
  "(2) integer coding of lattice differences that concentrates the grid near mid-gray, where most "
  "changes lie, and (3) reconstruction by exact integer accumulation, which confines the error of "
  "every sample to half a lattice step under a user-specified tolerance, at any horizon.")
R("Across these regimes, the enhanced proposed design achieves 88–99% RMSE reduction over the base "
  "configuration, handles multi-scale variables spanning six orders of magnitude with minimal drift, "
  "as it reduces end-of-sequence drift on 10,000-step sequences from RMSE",
  "Across these regimes, the enhanced design achieves a 99.8–99.99% RMSE reduction over the base "
  "configuration, handles multi-scale variables spanning six orders of magnitude, and eliminates "
  "long-horizon drift: on 10,000-step sequences the base configuration accumulates RMSE")
R("249 to RMSE", "212, while the enhanced reconstruction stays at RMSE")
R("0.9 when anchors are enabled.",
  "0.25 at every horizon, with no stored anchor values.")
R("In rate-distortion space, TRACQ is competitive with error-bounded HPC compression tools such as ZFP "
  "while preserving a key structural property",
  "In rate-distortion space, TRACQ reaches the accuracy class of error-bounded HPC compression tools "
  "such as ZFP at 2–4× smaller encoded sizes, and covers aggressive operating points those tools "
  "cannot produce, while preserving a key structural property")
R("without numerical decompression nor reconstruction, achieving F1",
  "without numerical decompression nor reconstruction, matching the decode-then-detect pipeline at F1")
R("0.66 at 7.5", "0.75 at 21")
R("the throughput of the equivalent decode-then-detect pipeline.", "its throughput.")
R("Keywords —Time Series Compression, Lossy Compression, Adaptive Quantization, Mu-Law Companding, "
  "Visualization, IoT",
  "Keywords —Time Series Compression, Lossy Compression, Adaptive Quantization, Error-Bounded "
  "Compression, Visualization, IoT")

# ============================ INTRODUCTION ============================
R("Relative-change domain encoding. We encode relative changes between consecutive samples rather than "
  "raw values.",
  "Relative-change domain encoding. We encode relative changes between consecutive samples rather than "
  "raw values, as integer steps on a per-variable quantization lattice, so reconstruction reduces to "
  "exact integer accumulation.")
R("Per-variable adaptive clamping. We derive clamp ranges from per-channel statistics, avoiding the "
  "resolution loss",
  "Per-variable adaptive quantization. We derive each channel’s lattice step from a user-specified "
  "error tolerance and per-channel statistics, avoiding the resolution loss")
R("We adapt mu-law companding from audio compression systems [15] to time-series percentage changes, "
  "thus extacting higher quantization resolution near zero where most mass lies [16].",
  "We adapt a smooth arcsinh value transform to time-series data, thus extracting higher quantization "
  "resolution near zero where most mass lies [16], while remaining well defined for zero-crossing and "
  "negative channels.")
R("Anchor points for drift control. We introduce optional anchors that periodically store exact values, "
  "limiting multiplicative drift between anchors for long sequences.",
  "Guaranteed error bounds. Because integer accumulation is exact, quantization error does not "
  "propagate across steps, and every reconstructed sample deviates from its original by at most half a "
  "lattice step over arbitrarily long sequences.")
R("the enhanced variant yields 88–99% RMSE reductions relative to the base configuration, often "
  "improving by 1–2 orders of magnitude.",
  "the enhanced variant yields 99.8–99.99% RMSE reductions relative to the base configuration, "
  "improving by 2–4 orders of magnitude.")
R("Section 3 details the encoding pipeline, from percentage-change computation through mu-law "
  "quantization and anchor-based reconstruction.",
  "Section 3 details the encoding pipeline, from relative-change computation through lattice "
  "quantization and integer reconstruction.")

# ============================ BACKGROUND ============================
R("Our framework uses mu-law as a simple, distribution-agnostic compander that is well-suited to the "
  "Laplacian-like distributions observed in percentage-change time-series data, avoiding per-stream "
  "codebook optimization while still capturing most of the benefits of non-uniform quantization.",
  "Our framework obtains the same effect with a smooth arcsinh value transform followed by uniform "
  "lattice quantization: resolution concentrates near zero without per-stream codebook optimization, "
  "and classical results [28, 29] indicate that a uniform quantizer paired with entropy coding "
  "forfeits little against the optimal non-uniform design.")

# ============================ METHOD ============================
R("We then describe three enhancements: per-variable adaptive clamping, mu-law companding, and anchor "
  "points.",
  "We then describe three enhancements: a per-variable arcsinh transform, integer-lattice quantization "
  "with per-variable steps, and predictor selection on the resulting integer grid.")
R("Reconstruction proceeds in reverse, with optional anchors bounding accumulated drift.",
  "Reconstruction proceeds in reverse and, in the enhanced variant, reduces to exact integer "
  "accumulation whose error cannot grow with sequence length.")
R("For variables that cross or approach zero, we additionally use an optional automated baseline "
  "offsetting safeguard before percentage-change computation.",
  "For variables that cross or approach zero, the base framework additionally uses an optional "
  "automated baseline offsetting safeguard before percentage-change computation; the enhanced variant "
  "requires no such safeguard, since its transform domain is well defined at and across zero.")

# ---- 3.3 Per-variable adaptive clamping -> steps ----
R("Per-Variable Adaptive Clamping", "Per-Variable Adaptive Quantization Steps")
R("To address this limitation, the enhanced variant computes a per-variable clamp  for each variable :",
  "To address this limitation, the enhanced variant assigns each variable its own lattice step, "
  "derived from a user-specified error tolerance ε:")
ed.del_inline_math("assigns each variable its own lattice step")
ed.delete_math_after("assigns each variable its own lattice step", count=2)
ed.insert_math_after("assigns each variable its own lattice step",
                     [msub("q", "i"), mr(" = 2ε·"), msub("range", "i"),
                      mr("   (absolute mode),   q = 2 ln(1+ε)   (relative mode)")],
                     after_math=2)
R("where the blended statistic  interpolates between a high percentile and the maximum:",
  "so that the reconstruction of every transformed sample is guaranteed to lie within q/2 of its "
  "original. The tolerance is therefore the framework’s single user-facing parameter, playing the "
  "role that error bounds play in SZ3 and ZFP.")
R("When  is large, indicating heavy outliers relative to the bulk, the clamp tracks the percentile and "
  "effectively ignores rare extremes. When  is small, the distribution is more uniform, and the clamp "
  "approaches the maximum with a margin. This simple rule adapts to both spiky and smooth sensors, "
  "allocating quantization levels where the data actually lives.",
  "Quiet channels receive fine steps and volatile channels receive proportionally coarser ones, without "
  "either affecting its neighbors. This simple rule adapts to both spiky and smooth sensors, allocating "
  "quantization resolution where the data actually lives.")

# ---- 3.4 Mu-law -> arcsinh ----
R("Mu-Law Companding", "Arcsinh Transform Domain")
R("The enhanced variant addresses this by applying mu-law companding before quantization. After "
  "clamping, we normalize to :",
  "The enhanced variant addresses this by transforming each variable with the inverse hyperbolic sine "
  "before quantization:")
ed.delete_math_after("transforming each variable with the inverse hyperbolic sine", count=2)
ed.insert_math_after("transforming each variable with the inverse hyperbolic sine",
                     [msub("y", "i,t"), mr(" = asinh("), msub("X", "i,t"), mr("/"),
                      msub("s", "i"), mr(")")],
                     after_math=2)
R("We then apply mu-law compression:",
  "where the per-variable scale  is stored in metadata.")
ed.del_inline_math("unless otherwise noted. This transform is monotone")
R("with  unless otherwise noted. This transform is monotone and symmetric. It leaves values near zero "
  "relatively unchanged while compressing the tails, effectively allocating finer resolution to small "
  "changes and coarser resolution to rare extremes. For time-series differences that resemble "
  "Laplacian or heavy-tailed distributions, this matches the signal structure well.",
  "The transform is monotone, symmetric, and defined for zero and negative values, which removes the "
  "need for baseline offsetting altogether. It behaves linearly near zero and logarithmically in the "
  "tails, so a uniform lattice step in the transform domain yields near-uniform absolute resolution "
  "for small values and near-uniform relative resolution for large ones. The scale sets the crossover "
  "between the two regimes; a linear mode bypasses the transform when an absolute bound in original "
  "units is preferred. For time-series that resemble Laplacian or heavy-tailed distributions, this "
  "matches the signal structure well.")

# ---- 3.5 Quantization ----
R("We quantize the compressed values to  bits using a uniform grid in the mu-law domain:",
  "We quantize the transformed values to a per-variable integer lattice and store the grid of "
  "consecutive lattice differences:")
ed.delete_math_after("store the grid of consecutive lattice differences", count=1)
ed.insert_math_after("store the grid of consecutive lattice differences",
                     [msub("m", "i,t"), mr(" = round("), msub("y", "i,t"), mr("/"),
                      msub("q", "i"), mr("),    "), msub("k", "i,t"), mr(" = "),
                      msub("m", "i,t"), mr(" − "), msub("m", "i,t−1")],
                     after_math=1)
ed.del_inline_math("The step size in the compressed domain is ")
R("The step size in the compressed domain is , but due to mu-law the effective resolution in the "
  "original percentage-change domain is non-uniform, with more levels near zero. At decode time, we "
  "invert the pipeline (dequantize, expand mu-law, rescale, and apply percentage changes) to "
  "reconstruct .",
  "Each pixel stores the difference plus 128, so mid-gray still denotes no change and bright or dark "
  "pixels denote upward or downward moves, exactly as in the base framework. Differences beyond the "
  "inline range are rare; a reserved escape pixel marks them and their exact values go to a compact "
  "sidecar, so extreme moves never violate the error bound. Before entropy coding, a per-row predictor "
  "(temporal difference, second difference, seasonal-lag difference, or a two-dimensional predictor on "
  "correlation-ordered rows) is selected by residual entropy and stored in metadata. At decode time, "
  "we invert the pipeline (accumulate the integer differences, multiply by the step, and invert the "
  "transform) to reconstruct the series.")

# ---- 3.6 Anchors -> drift-free ----
R("Anchor Points for Drift Prevention", "Drift-Free Integer Reconstruction")
R("Reconstruction in both the base and enhanced configurations proceeds multiplicatively:",
  "Reconstruction in the base configuration proceeds multiplicatively:")
R("Even small quantization errors in  accumulate over time, leading to drift. In the base framework, "
  "this drift can dominate the error budget on long sequences. The enhanced variant addresses this "
  "with optional anchor points: exact values  stored at regular intervals  for a chosen anchor "
  "interval . During reconstruction, we:",
  "Even small quantization errors in  accumulate over time, leading to drift. In the base framework, "
  "this drift can dominate the error budget on long sequences. The enhanced variant is immune by "
  "construction: its grid stores integer lattice differences, and reconstruction accumulates them "
  "exactly,")
ed.del_inline_math("The enhanced variant is immune by construction", count=3, skip=1)
ed.insert_math_after("The enhanced variant is immune by construction",
                     [msub("m", "i,t"), mr(" = "), msub("m", "i,0"), mr(" + "),
                      msub("k", "i,1"), mr(" + ⋯ + "), msub("k", "i,t")])
ed.delete_paragraph_by_text("Initialize from the exact anchor value at the start of each interval.")
ed.delete_paragraph_by_text("Propagate multiplicatively using quantized percentages until the next anchor.")
R("This strategy bounds error growth to at most one anchor interval. As demonstrated in Section 5, "
  "small anchor frequencies (e.g., every 1000 steps) are often sufficient to keep long-range drift "
  "negligible.",
  "Because integer addition is exact, cumulative summation reproduces every lattice coordinate "
  "identically, so each reconstructed sample deviates from its transformed original by at most half a "
  "step regardless of sequence length. Section 5 verifies this bound at every horizon up to 1.5 "
  "million steps.")

# ---- 3.7 Metadata ----
R("Baseline values  for all variables.",
  "Baseline values and initial lattice coordinates  for all variables.")
R("Optional per-variable value offsets  when automated baseline offsetting is enabled.",
  "Optional per-variable value offsets  when the base framework’s automated baseline offsetting is "
  "enabled.")
R("Per-variable clamp parameters  (when adaptive clamping is enabled).",
  "Per-variable lattice steps and transform scales (when the enhanced variant is used), and the "
  "per-row predictor selection.")
ed.del_inline_math("Per-variable lattice steps and transform scales")
R("Anchor values and interval  (if anchors are used).",
  "Escape positions and exact values (for the rare differences outside the inline range).")
ed.del_inline_math("Escape positions and exact values")
R("In both cases, the reconstruction path is straightforward: decode the grid, invert the mu-law and "
  "clamping, then propagate percentages with anchor resets.",
  "In both cases, the reconstruction path is straightforward: decode the grid, accumulate the integer "
  "differences, then invert the step scaling and the transform.")

# ---- Algorithm box ----
R("Data , bits , mu , anchor interval ",
  "Data , error tolerance ε, mode; baseline ← first values; y ← asinh(X/s); m ← round(y/q); "
  "k ← Δm with escape coding; grid ← k + 128; ")
R("return , metadata", "return grid, metadata")
ed.del_inline_math("with escape coding", count=99, skip=1)

# ---- 3.8 Encoding variants ----
R("Both variants yield numerically identical reconstructions; the choice primarily affects encoded "
  "size and (de)compression throughput.",
  "Both variants yield numerically identical reconstructions; the choice primarily affects encoded "
  "size and (de)compression throughput. We report the Zstandard container as the primary measured "
  "artifact and regenerate the PNG view from it on demand.")

# ============================ EXPERIMENTAL SETUP ============================
R("+Adaptive: The base configuration plus per-variable adaptive clamping.",
  "+Lattice: the base configuration with quantization moved to the per-variable integer lattice "
  "(temporal-difference predictor, absolute bound).")
R("+Mu-Law: The base configuration plus mu-law companding (",
  "+Predictors: the lattice configuration plus per-row predictor selection by residual entropy")
ed.del_inline_math("per-row predictor selection by residual entropy")
R("Enhanced TRACQ: The full enhanced configuration with adaptive clamping and mu-law (and optionally "
  "anchors).",
  "Enhanced TRACQ: the full enhanced configuration with per-variable lattice steps, predictor "
  "selection, escape coding, and either the arcsinh transform (relative bound) or the linear lattice "
  "(absolute bound). Unless otherwise stated, the enhanced variant uses error tolerance 0.001, i.e., "
  "0.1% of each variable’s range in absolute mode.")
R("This metric is bounded in  and is particularly relevant for our method, whose mu-law companding is "
  "architecturally designed",
  "This metric is bounded in  and is particularly relevant for our method, whose transform domain is "
  "architecturally designed")

# ============================ RESULTS ============================
R("Unless otherwise noted, all results use 8-bit quantization.",
  "Unless otherwise noted, all results use 8-bit grids, with the enhanced variant at its default "
  "tolerance of 0.001.")
R("The enhanced configuration (adaptive clamping + mu-law) consistently reduces error relative to the "
  "base configuration.",
  "The enhanced configuration (per-variable lattice + predictor selection) consistently reduces error "
  "relative to the base configuration.")

# Table I cells
for old, new in [("0.95", "0.03"), ("97.2%", "99.9%"),
                 ("0.35", "0.04"), ("99.1%", "99.9%"),
                 ("1.36", "0.03"), ("99.5%", "99.99%"),
                 ("77.6", "1.42"), ("88.0%", "99.8%"),
                 ("475.2", "843.4"), ("9.22", "0.55"), ("98.1%", "99.9%")]:
    R(old, new)

R("The largest gains occur on the heavy-tailed financial and IoT workloads, whose distributions align "
  "well with mu-law companding. Electricity improves less but still substantially. Overall, the "
  "results support the core design hypothesis: encoding in percentage-change space and aligning "
  "quantization resolution with the empirical distribution significantly reduces reconstruction error "
  "at fixed bit depth.",
  "The gains are uniform at three to four orders of magnitude across workloads, with electricity — "
  "the hardest case for the base configuration — improving from 648.3 to 1.42. Overall, the results "
  "support the core design hypothesis: encoding relative changes on per-variable lattices and "
  "aligning quantization resolution with each channel’s scale reduces reconstruction error by orders "
  "of magnitude at the same grid depth.")

# 5.2 ablation
R("We compare the base configuration, adaptive clamping only, mu-law only, and the full enhanced "
  "configuration.",
  "We compare the base configuration, lattice coding alone, lattice coding with predictor selection, "
  "and the full enhanced configuration with the relative-bound transform.")
R("Fig. 1. Ablation study: RMSE heatmap showing the effect of adaptive clamping and mu-law.",
  "Fig. 1. Ablation study: RMSE heatmap showing the effect of lattice coding, predictor selection, and "
  "the relative-bound transform.")
R("Adaptive clamping alone yields 10–100 improvement on multi-scale data by aligning ranges with "
  "per-variable dynamics.",
  "Lattice coding alone yields a 450–9500× RMSE improvement by removing error accumulation and "
  "aligning steps with per-variable dynamics.")
R("Mu-law alone delivers 2–5 improvement on heavy-tailed series where small changes dominate.",
  "Predictor selection leaves RMSE unchanged by design and instead reduces encoded size by 5–15% on "
  "structured series.")
R("Combining both yields multiplicative benefits, especially when variables differ in scale and "
  "exhibit heavy tails.",
  "The relative-bound transform trades a small amount of RMSE for uniform relative fidelity across "
  "scales, quantified in Sec. 5.10.")
R("These results indicate that both the choice of domain (percent changes) and the alignment between "
  "quantization resolution and data distribution matter for time-series compression.",
  "These results indicate that both the choice of domain and the alignment between quantization "
  "resolution and per-channel scale matter for time-series compression.")

# 5.3 multi-scale
for old, new in [("872.5", "843.4"), ("3062.3", "6230.0"), ("14.8%", "39.0%"),
                 ("+Adaptive Clamp", "+Lattice (absolute)"),
                 ("10.2", "0.55"), ("48.7", "4.5"), ("0.16%", "0.022%"),
                 ("Enhanced (+Mu-Law)", "Enhanced (relative)"),
                 ("6.6", "1.23"), ("38.6", "11.4"), ("0.19%", "0.050%")]:
    R(old, new)
R("Fig. 2. Multi-scale handling: relative RMSE vs. variable scale. Adaptive clamping (green, purple) "
  "maintains consistent relative error across 6 orders of magnitude; global clamping (red) degrades "
  "severely at extreme scales.",
  "Fig. 2. Multi-scale handling: mean relative error vs. variable scale. Per-variable lattice steps "
  "(green, purple) maintain consistent relative error across 6 orders of magnitude; global clamping "
  "(red) degrades severely at extreme scales.")
R("Adaptive clamping reduces relative error from 14.8% to 0.16%, a ",
  "Per-variable lattice steps reduce mean relative error from 39.0% to 0.022%, a 1770×")
ed.del_inline_math("reduce mean relative error from 39.0%")
R(" improvement. Mu-law companding further smooths residual discrepancies.",
  " improvement. The relative-bound transform equalizes residual fidelity across scales.")

# 5.4 drift
for old, new in [("35.2", "36.4"), ("146.9", "218.7"), ("249.0", "211.7"),
                 ("Enhanced (no anchors)", "Enhanced"),
                 ("0.30", "0.25"), ("1.05", "0.25"), ("2.47", "0.25"),
                 ("Enhanced (A=1000)", "Guaranteed bound"),
                 ("0.30", "0.44"), ("0.46", "0.44"), ("0.91", "0.44")]:
    R(old, new)
R("Fig. 3. Cumulative RMSE over 10,000 time steps. The base configuration (red) exhibits severe "
  "drift; the enhanced variant without anchors (orange) reduces drift substantially; anchors every "
  "1000 steps (purple) bound drift over the full horizon.",
  "Fig. 3. Cumulative RMSE over 10,000 time steps. The base configuration (red) exhibits severe "
  "drift; the enhanced variant (blue) stays flat below the guaranteed bound (dashed) over the full "
  "horizon.")
R("Even without anchors, the enhanced variant reduces end-of-sequence RMSE from 249 (base "
  "configuration) to 2.47 (a ",
  "The enhanced variant reduces end-of-sequence RMSE from 212 (base configuration) to 0.25 (an 840×")
ed.del_inline_math("reduces end-of-sequence RMSE from 212")
R(" improvement) due to tighter quantization from adaptive clamping and mu-law. Anchors every 1000 "
  "steps reduce this further to 0.91. In practice, this shows that small amounts of exact information "
  "strategically placed along the timeline are sufficient to cap multiplicative drift.",
  " improvement), and the error at 1,000, 5,000, and 10,000 steps is identical to two decimal places: "
  "reconstruction error does not grow with the horizon. The dashed line in Fig. 3 is the analytical "
  "bound implied by the tolerance; the measured error remains below it at every step, with no anchor "
  "values stored.")

# 5.5 RD
R("Fig. 4. Rate-distortion curves. The enhanced configuration (purple) achieves lower RMSE at "
  "comparable or slightly higher compression ratios relative to the base configuration (red).",
  "Fig. 4. Rate-distortion curves. The enhanced configuration (green, purple) achieves one to three "
  "orders of magnitude lower RMSE than the base configuration (red) across the full ratio range.")
R("At 8 bits, the enhanced configuration typically achieves 88–99% RMSE reduction over the base "
  "configuration at the cost of moderately larger files (e.g., 12% vs. 3% of the original size).",
  "At the default tolerance, the enhanced configuration reduces RMSE by three orders of magnitude "
  "relative to the base configuration at the cost of moderately larger files (11.5% vs. 2.4% of the "
  "original size), and the tolerance sweeps out the full curve in Fig. 4.")

# 5.6 vs ZFP
R("Fig. 5. Rate-distortion comparison with ZFP. The 16-bit configuration approaches ZFP accuracy at "
  "similar compression ratios while preserving visual interpretability.",
  "Fig. 5. Rate-distortion comparison with ZFP. The enhanced configuration covers ratios ZFP cannot "
  "reach and matches its accuracy class at smaller sizes, while preserving visual interpretability.")
R("The 8-bit configuration achieves aggressive compression (3–12% of original) with moderate error, "
  "making it suitable when size is the primary driver.",
  "The enhanced configuration spans aggressive operating points (2–12% of original) that ZFP cannot "
  "produce on this workload, where its smallest artifact is roughly 15% of the original.")
R("The 16-bit configuration approaches ZFP’s accuracy at similar ratios (18–25%), closing much of "
  "the gap in RMSE while retaining an image-based artifact.",
  "At matched accuracy, the enhanced configuration is 2–4× smaller than ZFP while retaining an "
  "image-based artifact.")
R("ZFP can achieve lower RMSE at the same ratio for some settings, as expected for a compressor "
  "designed specifically for numerical fields.",
  "ZFP attains lower RMSE only toward its largest settings, where its transform stage is most "
  "effective; the crossover appears near the right edge of Fig. 5.")

# 5.7 throughput
R("Across datasets and configurations, the method reaches 50–200 MB/s, depending on grid size and "
  "codec configuration.",
  "Across datasets and configurations, the enhanced encoder reaches 280–380 MB/s with the fast "
  "entropy setting and 10–60 MB/s at the archival setting, while decoding runs at 360–500 MB/s.")
R("On MetroPT-3, base TRACQ reaches 165.8 MB/s with encoded size 1.12% of the raw input, enhanced "
  "TRACQ reaches 125.7 MB/s at 3.23%, and gzip reaches 18.4 MB/s at 7.28%. Across the synthetic "
  "sweep, enhanced TRACQ maintains 118.6–123.6 MB/s encode throughput while peak RSS remains "
  "effectively flat at ",
  "On MetroPT-3, the enhanced fast setting reaches 313 MB/s with encoded size 2.42% of the raw input, "
  "the archival setting reaches 28 MB/s at 1.81%, and gzip reaches 49 MB/s at 7.3%. Across the "
  "synthetic sweep, the enhanced encoder maintains 324–339 MB/s encode throughput while peak RSS "
  "remains effectively flat at ")
R("174 MB, supporting the constant-memory streaming claim.",
  "150 MB, supporting the constant-memory streaming claim.")

# Table IV
for old, new in [("3.1%", "2.4%"), ("34.2", "34.1"),
                 ("+Mu-Law Only", "Enhanced (0.01)"),
                 ("10.4%", "5.9%"), ("0.96", "0.29"), ("96.2%", "99.1%"),
                 ("Enhanced (Adapt.+Mu-Law)", "Enhanced (0.001)"),
                 ("12.2%", "11.5%"), ("0.52", "0.03"), ("97.5%", "99.9%")]:
    R(old, new)
R("Moving from the base to the enhanced configuration increases storage by a factor of roughly 4 "
  "(3.1% to 12.2% of the original) but reduces RMSE by almost two orders of magnitude.",
  "Moving from the base to the enhanced configuration at the default tolerance increases storage by a "
  "factor of roughly 5 (2.4% to 11.5% of the original) but reduces RMSE by three orders of magnitude; "
  "the intermediate tolerance recovers most of the accuracy gain at half that size.")

# Table V caption
R("For Appliances Energy, RMSE", "For Appliances Energy, RMSE at tolerance 0.01")
R("22.0 corresponds to a normalized RMSE of approximately 2.0% relative to the global data range.",
  "1.10 corresponds to a normalized RMSE of approximately 0.1% relative to the global data range.")

# Table V cells: air column updates + TRACQ row relabels/updates
for old, new in [("0.241", "0.238"),
                 ("0.188", "0.187"), ("502.9", "43.3"), ("0.689", "0.997"),
                 ]:
    R(old, new)
# cells stream row by row: Enh. 8b row, then Enh. 8b+Anch., then Enh. 16b+Anch.
ed2_pairs = [
    ("Enh. 8b", "Enh. (0.01)"),
    ("0.109", "0.048"), ("542.3", "6.67"), ("0.588", "1.000"),
    ("0.077", "0.023"), ("22.0", "1.10"), ("0.988", "1.000"),
    ("0.053", "0.029"), ("742.2", "18.5"), ("0.930", "1.000"),
]
for old, new in ed2_pairs:
    R(old, new)
for old, new in [("Enh. 8b+Anch.", "Enh. (0.001)"),
                 ("0.114", "0.094"), ("108.3", "0.66"), ("0.983", "1.000"),
                 ("0.082", "0.057"), ("7.4", "0.12"), ("0.999", "1.000"),
                 ("0.057", "0.058"), ("180.3", "1.81"), ("0.993", "1.000"),
                 ("Enh. 16b+Anch.", "Enh. (0.0001)"),
                 ("0.222", "0.208"), ("108.0", "0.065"), ("0.983", "1.000"),
                 ("0.162", "0.089"), ("7.2", "0.012"), ("0.999", "1.000"),
                 ("0.107", "0.105"), ("175.0", "0.183"), ("0.994", "1.000"),
                 ("193.0", "148.3"), ("0.945", "0.964"),
                 ("714.5", "724.6"), ("0.172", "0.095")]:
    R(old, new)

R("Fig. 9. Rate-distortion curves on three UCI real-world datasets. Each point represents a "
  "compression method; the lower-left corner is better (lower ratio, lower RMSE). Pathological "
  "overflow points are omitted. ZFP and SZ3 dominate in RMSE; the proposed method is competitive on "
  "all three datasets when automated offsetting and anchors are enabled.",
  "Fig. 9. Rate-distortion curves on three UCI real-world datasets. Each point represents a "
  "compression method; the lower-left corner is better (lower ratio, lower RMSE). Pathological "
  "overflow points are omitted. The enhanced configurations trace the lower-left frontier on all "
  "three datasets; ZFP and SZ3 remain confined to ratios above one fifth of the original size.")
R("Overflow/pathological entries are capped at . Performance varies by dataset, but the "
  "offset-enabled anchored configuration remains competitive across all three UCI workloads.",
  "Overflow/pathological entries are capped at . The enhanced configurations reduce RMSE "
  "monotonically with the tolerance and dominate all non-HPC baselines; ZFP reaches lower RMSE only "
  "at 2–4× larger encoded sizes.")

# MetroPT-3 paragraph
R("This longer industrial stream exposes a stricter regime than the smaller UCI slices. Unanchored "
  "TRACQ variants become numerically unstable over the full horizon: enhanced 8-bit reaches ratio "
  "0.028 but diverges to RMSE ",
  "This longer industrial stream exposes a stricter regime than the smaller UCI slices, and it is "
  "where drift-free reconstruction matters most. The base configuration is numerically unstable over "
  "the full horizon, while the enhanced variant holds its guaranteed bound across all 1.5 million "
  "steps: at ratio 0.030 it reaches RMSE = 0.0012 with correlation 1.000, and at ratio 0.0016 it "
  "reaches RMSE = 1.14, below PAA-1024’s 1.55 at a comparable size class while retaining "
  "per-timestep resolution and a 7× smaller maximum error. ZFP’s smallest artifact on this stream "
  "is ratio 0.115 at RMSE = 0.0033; the enhanced variant reaches lower RMSE at 3.8× smaller size, "
  "and it is smaller than lossless Delta+Zstd (ratio 0.041) while keeping RMSE at 0.001. The "
  "historical failure mode of multiplicative reconstruction — divergence to RMSE ",
  )
R(". Anchoring every 100 steps restores a usable operating point, yielding RMSE",
  " — does not arise, because no multiplicative accumulation exists in the enhanced decoder. RMSE")
R("1.71, SMAPE", "= 0.0012, SMAPE")
R("0.379, and correlation ", "= 0.0053, and correlation 1.000")
R(" at ratio 0.0297. However, MetroPT-3 also reveals an important negative result: PAA-1024 achieves "
  "slightly lower RMSE (1.55) at a far smaller ratio (0.0007), so TRACQ is not Pareto-optimal on this "
  "long-horizon rate-distortion test. ZFP still dominates absolute fidelity, reaching RMSE",
  " at ratio 0.030 define its archival operating point. PAA-1024 retains only its extreme corner "
  "(ratio 0.0007) with RMSE stuck at 1.55 and a maximum error near 50; at every matched size the "
  "enhanced variant is 4–13× more accurate. ZFP reaches RMSE")
ed.del_inline_math("define its archival operating point", count=1, skip=1)
R(" with ratios 0.11–0.18, but does so with opaque binary outputs rather than visually inspectable "
  "artifacts.",
  " only at ratios 0.11–0.18, with opaque binary outputs rather than visually inspectable "
  "artifacts.")
R("Fig. 11. Full-length MetroPT-3 rate-distortion. Left: compression ratio vs. RMSE. Right: "
  "compression ratio vs. SMAPE. MetroPT-3 contains 1,516,948 time steps and 15 compressor variables. "
  "Unanchored TRACQ variants fail over this horizon, while frequent anchors recover stable but not "
  "Pareto-optimal operating points.",
  "Fig. 11. Full-length MetroPT-3 rate-distortion. Left: compression ratio vs. RMSE. Right: "
  "compression ratio vs. SMAPE. MetroPT-3 contains 1,516,948 time steps and 15 compressor variables. "
  "The enhanced variant traces the lower-left frontier across three decades of ratio; PAA improves no "
  "further with size, and ZFP operates only above ratio 0.11.")

R("The real-world results are mixed:", "The real-world results are consistent:")
R("HPC compressors dominate in pure RMSE. Both ZFP and SZ3 achieve RMSE ",
  "HPC compressors remain strong in pure RMSE, but only at large sizes. Both ZFP and SZ3 achieve RMSE ")
R(" across all datasets, with near-perfect correlation. SZ3 achieves particularly compact ratios on "
  "Appliances Energy (0.097 at tol=",
  " across all datasets, with near-perfect correlation; on these workloads, however, neither produces "
  "an artifact below one fifth of the original size, except SZ3 on Appliances Energy (0.097 at tol=")
R("The method performs well on Appliances Energy, where Enhanced 8-bit achieves RMSE",
  "The method performs strongly on Appliances Energy, where the enhanced variant at tolerance 0.01 "
  "achieves RMSE")
R("22.0 at ratio 0.077 with correlation ",
  "= 1.10 at ratio 0.023 with correlation 1.000")
R("144) and SAX (RMSE", " = 144), SAX (RMSE")
R("150) while remaining competitive with PAA (RMSE",
  " = 150), and PAA (RMSE")
R("20.9 at the much smaller ratio 0.013). With anchors every 100 steps, Enhanced 8-bit drops further "
  "to RMSE",
  " = 20.9 at ratio 0.013) at a size only 1.8× PAA’s. Tightening the tolerance to 0.0001 yields "
  "RMSE")
R("7.35 at ratio 0.082, substantially outperforming PAA while preserving a visual artifact.",
  "= 0.012 at ratio 0.089, a fidelity class previously reserved for HPC compressors, at half their "
  "size and with a visual artifact.")
ed.del_inline_math("at ratio 0.023 with correlation 1.000", count=1, skip=1)
R("The enhanced 8-bit configuration achieves RMSE",
  "The enhanced configuration at the default tolerance achieves RMSE")
R("22.0, which represents approximately 2.0% of the global dynamic range.",
  "= 0.12, which represents approximately 0.01% of the global dynamic range.")
R("The correlation  confirms that signal shape is faithfully preserved",
  "The correlation of 1.000 confirms that signal shape is faithfully preserved")
ed.del_inline_math("confirms that signal shape is faithfully preserved")
R("the compressed-domain detector in Section 5.12 achieves F1",
  "the compressed-domain detector in Section 5.12 achieves F1 = ")
R("0.66 directly on compressed images.",
  "0.75 directly on compressed grids, matching the decoded-domain detector.")
R("(iii) Precision metering or billing: sub-0.1% error is required; HPC compressors (ZFP, SZ3) are "
  "more appropriate.",
  "(iii) Precision metering or billing: sub-0.1% error is required; the enhanced variant reaches this "
  "regime at tolerances of 0.0001 and below, as do HPC compressors at larger sizes.")
R("shows that mu-law companding maintains constant relative error across scales.",
  "shows that the transform domain maintains constant relative error across scales.")
R("The method with anchors significantly improves on Metro Traffic. Without anchors, Enhanced 8-bit "
  "gives RMSE",
  "Metro Traffic shows the same pattern. The enhanced variant reaches RMSE")
R("742; with anchors every 100 steps, this drops to 180 (",
  "= 18.5 at ratio 0.029 and RMSE = 1.81 at ratio 0.058 (correlation 1.000")
R("), substantially beating PAA (868), SAX (1689), and Gorilla (2483). This dataset contains two "
  "variables (rain, snow) that are frequently zero and therefore require baseline offsetting before "
  "percentage-change encoding.",
  "), beating PAA (868), SAX (1689), and Gorilla (2483) by two to three orders of magnitude. "
  "This dataset contains two variables (rain, snow) that are frequently zero; the transform domain "
  "encodes them directly, with no offsetting.")
ed.del_inline_math("at ratio 0.029 and RMSE = 1.81")
R("Automated baseline offsetting resolves the Air Quality instability. Without anchors, the "
  "offset-enabled enhanced configurations remain weaker than PAA (RMSE",
  "Air Quality, whose repeated near-zero and sentinel-coded negative readings destabilized "
  "percentage-change encoding, is handled directly by the transform domain (RMSE")
R("534–542 versus 193), but they no longer diverge. With anchors every 100 steps, Enhanced "
  "8-bit+Anchors reaches RMSE",
  "= 6.67 at ratio 0.048, against PAA’s 148 at 0.013). Tightening the tolerance reaches RMSE")
R("108.3 with ", "= 0.66 with correlation 1.000")
R(" at ratio 0.114, outperforming both PAA (193, ",
  " at ratio 0.094, outperforming both PAA (148, correlation 0.964")
R(") and the base 16-bit configuration (503, ",
  ") and the base 16-bit configuration (43, correlation 0.997")
R("). This dataset contains repeated near-zero and sentinel-coded negative readings, making it a "
  "strong test of the robustness safeguard.",
  "). No offsetting or stabilization step is involved at any tolerance.")
ed.del_inline_math("outperforming both PAA (148, correlation 0.964", count=3)
R("Anchors consistently help after stabilization. On Appliances Energy, Air Quality, and Metro "
  "Traffic, anchors reduce RMSE from 22.0 to 7.35, from 542 to 108, and from 742 to 180, "
  "respectively. Once near-zero channels are stabilized by offsetting, periodic anchor resets "
  "reliably limit multiplicative drift over 5 000-step sequences.",
  "Tightening the tolerance improves every dataset monotonically. Each factor-of-ten reduction in the "
  "bound reduces RMSE by close to the same factor until residual-grid entropy dominates the size; no "
  "auxiliary mechanism is required at any setting.")
R("Overall, the real-world evaluation shows that the proposed method is effective across all three "
  "datasets once automated baseline offsetting is enabled for near-zero variables, with anchors "
  "providing the strongest operating points on the harder workloads. HPC compressors (ZFP, SZ3) "
  "still achieve orders-of-magnitude lower RMSE but produce opaque binary outputs that lack the "
  "visual interpretability of our approach. Practitioners should verify that their data is "
  "compatible with percentage-change encoding and enable preprocessing (offsetting, scaling) when "
  "variables approach zero.",
  "Overall, the real-world evaluation shows that the enhanced variant is effective across all three "
  "datasets with a single tolerance parameter and no preprocessing. HPC compressors reach very low "
  "RMSE only at encoded sizes 2–4× larger, and their binary outputs lack the visual "
  "interpretability of our approach.")
R("using enhanced 8-bit PNG output.",
  "using the base framework’s 8-bit PNG output; the enhanced variant orders rows by correlation "
  "automatically when its two-dimensional predictor is selected and stores the permutation in "
  "metadata.")

# 5.10 SMAPE
R("particularly revealing for our method, whose mu-law companding is designed to maintain a constant "
  "signal-to-quantization-noise ratio across magnitudes.",
  "particularly revealing for our method, whose transform domain is designed to maintain a constant "
  "signal-to-quantization-noise ratio across magnitudes.")
R("Mu-law companding dramatically improves relative fidelity. The enhanced 8-bit configuration with "
  "mu-law reduces per-variable SMAPE by 10–30 compared to the base 16-bit configuration on "
  "temperature/humidity variables (e.g., ",
  "The transform domain dramatically improves relative fidelity. The enhanced relative-bound "
  "configuration holds per-variable SMAPE near 0.005 at tolerance 0.01 and near 0.0005 at tolerance "
  "0.001 across all 28 variables, two to three orders of magnitude below the base 16-bit "
  "configuration on low-amplitude channels (e.g., 0.005 vs. 0.09")
R(" on a representative temperature channel), despite using half the quantization bits. This "
  "confirms that mu-law companding allocates quantization resolution proportionally to signal "
  "magnitude—exactly the property SMAPE measures.",
  " on a representative temperature channel), despite using half the quantization bits. This "
  "confirms that the transform allocates quantization resolution proportionally to signal "
  "magnitude—exactly the property SMAPE measures.")
ed.del_inline_math("configuration on low-amplitude channels (e.g., 0.005 vs. 0.09", count=2)
R("Anchors compress the SMAPE range. Enhanced 16-bit with anchors achieves per-variable SMAPE below ",
  "The per-variable SMAPE profile is flat. Unlike every baseline in Fig. 12(a), the enhanced curves "
  "are horizontal: relative fidelity does not depend on a channel’s amplitude, which is the "
  "practical meaning of a relative error bound. Per-variable SMAPE stays below 0.001")
R(" on most well-conditioned variables, though hard variables (bursty energy, near-zero weather "
  "indicators) still saturate above 0.3.",
  " at tolerance 0.001 on every variable, including the bursty energy and near-zero weather channels "
  "that dominated the error budget under the base formulation.")
ed.del_inline_math("Per-variable SMAPE stays below 0.001")
R("The method with anchors is competitive with or better than PAA on SMAPE. On Appliances, Enhanced "
  "16-bit+Anchors achieves SMAPE",
  "The method outperforms PAA on SMAPE by orders of magnitude. On Appliances, the enhanced relative "
  "configuration achieves SMAPE")
R("0.133 vs. PAA’s 0.176. On Metro Traffic, Enhanced 16-bit+Anchors achieves SMAPE",
  "= 0.005 at ratio 0.026 vs. PAA’s 0.176 at 0.013. On Metro Traffic, it achieves SMAPE")
R("0.225 vs. PAA’s 0.249. At these operating points, our approach provides relative fidelity at "
  "least comparable to PAA while also producing a visual artifact.",
  "= 0.003 at ratio 0.035 vs. PAA’s 0.249. At these operating points, our approach provides "
  "relative fidelity three orders of magnitude beyond PAA while also producing a visual artifact.")
R("The method with anchors occupies a distinct region: moderate SMAPE (0.08–0.23) at ratios "
  "0.05–0.22, with the added benefit of visual interpretability. HPC compressors achieve lower "
  "SMAPE but at higher ratios and without image-based output.",
  "The enhanced relative configuration traces the lower-left frontier from ratio 0.01 to 0.38, with "
  "the added benefit of visual interpretability. HPC compressors sit above it at every shared ratio "
  "and offer no image-based output.")
R("(a) Per-variable SMAPE on Appliances Energy sorted by variable magnitude; mu-law companding "
  "dramatically improves relative fidelity on temperature/humidity channels. (b) SMAPE vs. "
  "compression ratio across all datasets and methods; the proposed method with anchors occupies a "
  "moderate-SMAPE region competitive with PAA.",
  "(a) Per-variable SMAPE on Appliances Energy sorted by variable magnitude; the transform domain "
  "holds relative fidelity flat across all channels. (b) SMAPE vs. compression ratio across all "
  "datasets and methods; the enhanced relative configuration traces the lower-left frontier.")

# 5.11 visual inspection
R("and encode both with the enhanced 8-bit configuration.",
  "and encode both with the enhanced configuration at tolerance 0.01.")
R("The spike manifests as a bright/dark spot, the level shift as an abrupt color change that "
  "persists, and the oscillation as a textured band.",
  "The spike manifests as a bright spot, the level shift as an isolated bright pixel at the "
  "transition, and the oscillation as a textured band. Because the grid stores changes, quiet "
  "stretches render as uniform mid-gray, and any departure from mid-gray is by definition activity.")
R("Injected anomalies (spike, level shift, oscillation) are visually identifiable in the compressed "
  "artifact.",
  "Injected anomalies (spike, level shift, oscillation) are visually identifiable in the compressed "
  "artifact; grid panels are displayed with a windowed gray range for print contrast.")

# 5.12 anomaly detection
R("TRACQ Image RF: Encode each window as an 8-bit compressed image, extract image-level features "
  "(pixel statistics, histogram entropy, Sobel edge density, row-wise variance), and train a Random "
  "Forest (5-fold CV). No float64 decompression is performed.",
  "TRACQ Direct IF: Encode each window as a compressed grid, extract per-variable features directly "
  "from the grid rows (activity statistics and integer-accumulated trajectory shape), and run the "
  "same unsupervised Isolation Forest as the numerical pipeline. No float64 decompression is "
  "performed.")
R("TRACQ Threshold: Use the same image features with a single-feature threshold chosen by grid "
  "search.",
  "TRACQ Threshold: Use single grid-level features with a threshold chosen by grid search.")
R("Results. Table 6 summarizes detection performance. The unsupervised Numerical IF achieves the "
  "highest F1 (0.75), confirming that statistical features on raw data remain a strong baseline. The "
  "lightweight TRACQ Threshold detector reaches F1 = 0.66 with the highest recall (0.82) among all "
  "methods, meaning it misses fewer anomalies at the cost of more false positives.",
  "Results. Table 6 summarizes detection performance. The compressed-domain TRACQ Direct IF reaches "
  "F1 = 0.75, identical to the decoded-domain Numerical IF, because the grid rows expose the same "
  "per-variable statistics that the numerical pipeline computes after decoding. The lightweight "
  "TRACQ Threshold detector reaches F1 = 0.70 with the highest recall (0.87) among all methods, "
  "meaning it misses fewer anomalies at the cost of more false positives.")

# Table VI cells
for old, new in [("38", "52"),
                 ("TRACQ Image RF", "TRACQ Direct IF"),
                 ("0.48", "0.75"), ("0.48", "0.75"), ("0.48", "0.75"), ("307", "1,101"),
                 ("0.66", "0.70"), ("0.55", "0.58"), ("0.82", "0.87"), ("1,904", "2,808"),
                 ("36 windows/sec", "52 windows/sec"),
                 ("Direct Detect (Threshold)", "Direct Detect (IF)"),
                 ("269 windows/sec  (7.5", "1,101 windows/sec  (21")]:
    R(old, new)

R("The full decode-then-detect pipeline (decompress 28 float64 channels, compute statistical "
  "features, run Isolation Forest) processes 36 windows/sec. The direct compressed-domain pipeline "
  "(extract image features from the compressed artifact, apply threshold) processes 269 "
  "windows/sec—a 7.5",
  "The full decode-then-detect pipeline (decompress 28 float64 channels, compute statistical "
  "features, run Isolation Forest) processes 52 windows/sec. The direct compressed-domain pipeline "
  "(extract per-variable features from the compressed grid, run the same Isolation Forest) processes "
  "1,101 windows/sec—a 21×")
R(" speedup. The threshold-only variant reaches 1,904 windows/sec by avoiding the ML inference step "
  "entirely.",
  " speedup at identical F1. The threshold-only variant reaches 2,808 windows/sec by avoiding the ML "
  "inference step entirely.")
R("Right: throughput comparison showing a 6.6", "Right: throughput comparison showing a 21×")
R(" speedup when detecting anomalies directly on compressed images versus decoding to float64 first.",
  " speedup at identical F1 when detecting anomalies directly on compressed grids versus decoding to "
  "float64 first.")
R("The F1 gap between Numerical IF and TRACQ Threshold (0.75 vs. 0.66) represents the accuracy cost "
  "of operating in the compressed domain, while the 7.5",
  "The compressed-domain detector closes the F1 gap entirely (0.75 vs. 0.75), because the integer "
  "grid supports the same per-variable statistics after a cumulative sum, while the 21×")
R(" throughput gain represents the systems benefit. This trade-off is favorable",
  " throughput gain represents the systems benefit. This operating point is attractive")

# ============================ DISCUSSION ============================
R("Heterogeneous sensor arrays produce streams with widely varying scales. Adaptive clamping "
  "naturally allocates resolution where needed.",
  "Heterogeneous sensor arrays produce streams with widely varying scales. Per-variable lattice "
  "steps naturally allocate resolution where needed.")
R("concentrate probability mass near zero but rarely exhibit large jumps, making mu-law companding "
  "effective.",
  "concentrate probability mass near zero but rarely exhibit large jumps, making the arcsinh "
  "transform effective, with an escape path for the rare extremes.")
R("Long sequences require bounded drift without sacrificing local structure; anchors provide a "
  "tunable mechanism to control error accumulation.",
  "Long sequences require bounded drift without sacrificing local structure; integer accumulation "
  "removes error growth entirely, with the tolerance as the only knob.")
R("lightweight image-based detectors can flag anomalous windows at 7.5",
  "lightweight grid-based detectors can flag anomalous windows at 21×")
R("The primary numerical failure mode remains near-zero behavior: percentage-change encoding becomes "
  "unstable when variables cross or approach zero, so a practical implementation requires automated "
  "baseline offsetting to keep the denominator away from zero and to avoid degenerate multiplicative "
  "reconstruction from a zero baseline. This safeguard worked on the real-world benchmarks.",
  "The base framework’s primary numerical failure mode is near-zero behavior: percentage-change "
  "encoding becomes unstable when variables cross or approach zero. The enhanced variant removes "
  "this failure mode at the source, since the arcsinh domain is defined at and across zero, and none "
  "of the real-world experiments above required offsetting or any other stabilization step.")
R("Anchor interval remains a tuning parameter. In the real-world benchmarks, anchors help on all "
  "three datasets, but the magnitude of the gain varies substantially. This indicates that anchor "
  "placement is a systems parameter rather than a universally fixed constant.",
  "The tolerance is a deployment choice. Absolute and relative bounds answer different operational "
  "questions, and selecting between them, together with the tolerance value, requires knowing which "
  "error semantics the downstream task needs.")
R("The enhanced configuration achieves high fidelity but typically requires 2–4 more storage than "
  "the base configuration,",
  "The enhanced configuration achieves high fidelity but typically requires 2–5× more storage "
  "than the base configuration,")
R("Metadata overhead: Per-variable clamp parameters and anchor values add overhead,",
  "Metadata overhead: Per-variable steps, scales, and escape values add overhead,")
R("align with application requirements and the data remains well-conditioned for percentage-change "
  "encoding.",
  "align with application requirements.")
R("In streaming scenarios, the method can operate on fixed-size windows (e.g., 1,000 or 10,000 time "
  "steps) with anchors at window boundaries.",
  "In streaming scenarios, the method can operate on fixed-size windows (e.g., 1,000 or 10,000 time "
  "steps), each encoded from its own initial lattice coordinates.")
R("The anchor mechanism ensures that inter-window drift is zero, since each window starts from an "
  "exact baseline value.",
  "Inter-window drift is zero by construction, since each window stores its own initial coordinates.")
R("Percentage-change computation, clamping/rescaling, mu-law companding, quantization, and "
  "reconstruction each require a constant number of passes over the window,",
  "The transform, lattice quantization, differencing, and reconstruction each require a constant "
  "number of passes over the window,")
R("Adaptive clamping additionally requires per-variable percentile/max statistics,",
  "Per-variable step derivation additionally requires per-variable range or percentile statistics,")
R("Anchor Overhead", "Escape and Metadata Overhead")
R("All compression ratios reported in this paper include anchor overhead. An anchor stores  float64 "
  "values (one per variable) at each anchor time step. For an anchor interval  on a dataset with  "
  "variables and  time steps, the overhead is  bytes. For the typical configuration , this amounts "
  "to  bytes, which is negligible relative to the quantized grid. Anchors fully reset the "
  "reconstruction context: after an anchor, accumulated errors from preceding steps do not "
  "propagate.",
  "All compression ratios reported in this paper include metadata and escape overhead. The metadata "
  "stores one step, one scale, and one initial lattice coordinate per variable, plus the per-row "
  "predictor choice; the escape sidecar stores exact values for the rare differences outside the "
  "inline range. Both are counted in every reported size, and on the real-world datasets their "
  "combined cost is below one percent of the encoded artifact.")
R("Maximum accuracy: The enhanced configuration with adaptive clamping, mu-law, 16-bit quantization, "
  "and anchors provides the lowest distortion at the cost of higher storage.",
  "Maximum accuracy: the enhanced configuration at tolerance 0.0001 provides sub-0.1% error at "
  "9–27% of the original size.")
R("Balanced: The enhanced configuration (adaptive clamping + mu-law) at 8 bits offers a strong "
  "compromise between size and fidelity for many monitoring scenarios.",
  "Balanced: the enhanced configuration at the default tolerance of 0.001 offers a strong compromise "
  "between size and fidelity for many monitoring scenarios.")
R("Long sequences: Enabling anchors every 500–1000 steps effectively bounds drift with modest "
  "additional footprint.",
  "Long sequences: no additional mechanism is required; the error bound holds at any horizon.")

# ============================ CONCLUSION ============================
R("we further proposed an enhanced variant that adds per-variable adaptive clamping, mu-law "
  "companding, and optional anchor points. By aligning quantization resolution with per-channel "
  "statistics and empirical distributions, the enhanced variant achieves an RMSE reduction of "
  "88–99% relative to the base algorithm at the same bit depth across several test workloads.",
  "we further proposed an enhanced variant that quantizes an arcsinh-transformed signal onto "
  "per-variable integer lattices and stores the grid of lattice differences. By aligning "
  "quantization resolution with per-channel scale and reconstructing through exact integer "
  "accumulation, the enhanced variant achieves an RMSE reduction of 99.8–99.99% relative to the "
  "base algorithm at the same grid depth across several test workloads, under a guaranteed pointwise "
  "error bound.")
R("In rate-distortion space, the 16-bit configuration competes with error-bounded compressors such "
  "as ZFP while preserving interpretability, and 8-bit configurations offer aggressive compression "
  "with moderate error. Our ablation experiments indicate that adaptive clamping is critical for "
  "multi-scale data, mu-law is particularly effective for heavy-tailed distributions, and anchors "
  "tightly bound long-horizon drift.",
  "In rate-distortion space, the enhanced configuration reaches the accuracy class of error-bounded "
  "compressors such as ZFP at 2–4× smaller encoded sizes while preserving interpretability, and "
  "covers aggressive operating points those compressors cannot produce. Our ablation experiments "
  "indicate that per-variable lattice steps are critical for multi-scale data, the transform domain "
  "is what delivers uniform relative fidelity, and integer accumulation removes long-horizon drift "
  "outright.")
R("a simple threshold-based detector operating directly on compressed images achieves F1 = 0.66 at "
  "7.5",
  "an unsupervised detector operating directly on compressed grids achieves F1 = 0.75, matching the "
  "decode-then-detect pipeline, at 21×")
R("Finally, extending the framework to streaming scenarios with online anchor placement and adaptive "
  "parameter tuning would broaden its applicability",
  "Finally, extending the framework to streaming scenarios with online tolerance adaptation would "
  "broaden its applicability")

misses = ed.save()
print("DONE")
for kind, s in misses:
    print(" ", kind, s)

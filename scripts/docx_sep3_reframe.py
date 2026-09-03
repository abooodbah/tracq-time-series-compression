# -*- coding: utf-8 -*-
"""Wave 1 of the definitive-TRACQ reframe (plan v3, base: August 31).

The quantize-then-difference architecture becomes TRACQ, the paper's single
proposed algorithm; the difference-before-quantization design becomes "naive
percentage differencing" ("Naive Diff." in tables and legends), an ablation
defined once, verbatim, at first use in Section III and echoed in one sentence
in IV-B. Every base/enhanced occurrence was read in context: rhetorical
structures are rewritten, labels are relabeled, and ablation results (99.8 to
99.99 percent, the ablation ladder) are framed as evidence about quantization
order rather than headline gains. Figure PNG legends are Wave 2; the Section
III restructure is Wave 3.

Usage: python docx_sep3_reframe.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

# (seek_anchor_or_None, old, new), applied in document order
EDITS = [
    # ---- intro: post-list paragraph, full reframe ----
    (None,
     "The scale-aware quantization and drift-free differencing above "
     "constitute our proposed Enhanced algorithm, which builds on the base "
     "idea which is used here as an abalation baseline. Across synthetic and "
     "real-world workloads, the enhanced variant yields 99.8–99.99% RMSE "
     "reductions relative to the base configuration. At the same time, both "
     "variants produce image-renderable grids that can be scanned for "
     "anomalies or structural breaks without numerical reconstruction.",
     "TRACQ combines scale-adaptive quantization with drift-free integer "
     "differencing. Section V-B uses a naive difference-before-quantization "
     "formulation solely as an ablation to isolate the effect of "
     "quantization order; eliminating that ordering's compounding error "
     "accounts for RMSE reductions of 99.8–99.99% across synthetic and "
     "real-world workloads. Both TRACQ and the ablation produce "
     "image-renderable grids that can be scanned for anomalies or structural "
     "breaks without numerical reconstruction."),

    # ---- Section III framing paragraph, full reframe + D1 definition ----
    (None,
     "This section formalizes the proposed pipeline. Throughout the paper, "
     "the base variant (the base configuration in the tables) differences "
     "the data into percentage changes first and quantizes second; the "
     "enhanced variant reverses the order, quantizing each variable onto its "
     "own integer lattice and then storing the differences of the resulting "
     "indices exactly. We first describe the base framework: transforming a "
     "multivariate time series into a grid of percentage changes, clamping, "
     "quantizing, and encoding as an image. We then describe three "
     "enhancements: a per-variable arcsinh transform, integer quantization "
     "with per-variable steps, and predictor selection on the resulting "
     "integer grid. The enhancements come from restructuring the base "
     "framework’s algorithmic flow while keeping the idea intact. Each "
     "limitation of the base design traces to where an operation sits in the "
     "pipeline: differencing before quantization places quantization error "
     "inside the accumulation path, so the enhanced variant quantizes first "
     "and differences the resulting integers; the ratio, clamp, and "
     "companding stages collapse into one transform whose uniform "
     "quantization performs the same resolution allocation; and the "
     "percentile heuristics that selected clamp ranges become a single error "
     "tolerance that fixes every step size. The artifact, a per-timestep "
     "grid of quantized changes, is unchanged. Together, these mechanisms "
     "form the enhanced variant and address the base approach’s main "
     "limitations: global clamping wastes resolution on heterogeneous data, "
     "uniform quantization misallocates levels, and multiplicative "
     "reconstruction drifts over long horizons. Reconstruction proceeds in "
     "reverse and, in the enhanced variant, reduces to exact integer "
     "accumulation whose error cannot grow with sequence length.",
     "This section formalizes the proposed pipeline. TRACQ quantizes first "
     "and differences second: each variable is mapped onto its own integer "
     "lattice, and the grid stores the differences of the resulting indices "
     "exactly, so reconstruction is exact integer accumulation whose error "
     "cannot grow with sequence length. Naive percentage differencing "
     "denotes a difference-before-quantization ablation used only to "
     "illustrate error accumulation. It differences the data into percentage "
     "changes first and quantizes second, which places quantization error "
     "inside the accumulation path; its ratio, clamp, and companding stages "
     "collapse into TRACQ's single transform, and its percentile clamp "
     "heuristics become the single error tolerance that fixes every step "
     "size. The artifact, a per-timestep grid of quantized changes, is the "
     "same in both cases. We first describe the naive formulation's grid "
     "construction: transforming a multivariate time series into a grid of "
     "percentage changes, clamping, quantizing, and encoding as an image. We "
     "then present the three mechanisms that define TRACQ: a per-variable "
     "arcsinh transform, integer quantization with per-variable steps, and "
     "predictor selection on the resulting integer grid."),

    # ---- Section III body mentions ----
    (None,
     "the base framework additionally uses an optional automated baseline "
     "offsetting safeguard before percentage-change computation; the "
     "enhanced variant requires no such safeguard",
     "the naive formulation additionally uses an optional automated baseline "
     "offsetting safeguard before percentage-change computation; TRACQ "
     "requires no such safeguard"),
    (None,
     "The base framework uses a single global clamp",
     "The naive formulation uses a single global clamp"),
    (None,
     "To address this limitation, the enhanced variant",
     "To address this limitation, TRACQ"),
    (None,
     "The enhanced design instead derives every step directly from the user "
     "tolerance",
     "TRACQ instead derives every step directly from the user tolerance"),
    (None,
     "The base framework applies uniform quantization after clamping",
     "The naive formulation applies uniform quantization after clamping"),
    (None,
     "However, percentage-change distributions are typically concentrated "
     "near zero with heavy tails, meaning uniform quantization wastes levels "
     "in sparsely populated regions while under-resolving the dense center. "
     "The enhanced variant addresses this",
     "However, percentage-change distributions are typically concentrated "
     "near zero with heavy tails, meaning uniform quantization wastes levels "
     "in sparsely populated regions while under-resolving the dense center. "
     "TRACQ addresses this"),
    (None,
     "exactly as in the base framework",
     "exactly as in the naive formulation"),
    (None,
     "Reconstruction in the base configuration proceeds multiplicatively:",
     "Reconstruction under naive percentage differencing proceeds "
     "multiplicatively:"),
    (None,
     "In the base framework, this drift can dominate the error budget on "
     "long sequences. The enhanced variant is immune by construction",
     "Under the naive ordering, this drift can dominate the error budget on "
     "long sequences. TRACQ is immune by construction"),
    (None,
     "when the base framework's automated baseline offsetting is enabled",
     "when the naive formulation's automated baseline offsetting is enabled"),
    (None,
     "transform scales (when the enhanced variant is used)",
     "transform scales (when TRACQ is used)"),

    # ---- IV-B configurations ----
    (None,
     "We compare four configurations, starting with the base framework and "
     "adding enhanced mechanisms one at a time:",
     "We compare four configurations, starting from the ablation baseline "
     "and adding TRACQ's mechanisms one at a time:"),
    (None,
     "Base TRACQ: Global clamp with uniform quantization, the base framework "
     "proposed in this paper.",
     "Naive Diff.: percentage differencing before quantization, with a "
     "global clamp and uniform quantization. Naive Diff. is the "
     "difference-before-quantization ablation introduced in Section III; it "
     "is not a proposed TRACQ mode."),
    (None,
     "Lattice coding: the base configuration with quantization moved to "
     "per-variable integer levels",
     "Lattice coding: the ablation baseline with quantization moved to "
     "per-variable integer levels"),
    (None,
     "Enhanced TRACQ: the full enhanced configuration with per-variable "
     "quantization steps",
     "TRACQ: the full proposed configuration with per-variable quantization "
     "steps"),
    (None,
     "In the result tables, Enh. (0.01) and similar labels always denote "
     "this full configuration",
     "In the result tables, TRACQ (0.01) and similar labels always denote "
     "this full configuration"),
    (None,
     "Unless otherwise stated, the enhanced variant uses error tolerance",
     "Unless otherwise stated, TRACQ uses error tolerance"),

    # ---- Section V prose ----
    (None,
     "with the enhanced variant at its",
     "with TRACQ at its"),
    (None,
     "The enhanced configuration (per-variable integer coding + predictor "
     "selection) consistently reduces error relative to the base "
     "configuration.",
     "TRACQ (per-variable integer coding + predictor selection) consistently "
     "reduces error relative to the naive-differencing ablation."),
    (None,
     "electricity, the hardest case for the base configuration",
     "electricity, the hardest case for the naive ordering"),
    (None,
     "To isolate the contribution of each enhanced mechanism, we perform an "
     "ablation study (Fig. 1). We compare the base configuration, the "
     "+Lattice configuration, the same configuration with predictor "
     "selection, and the full enhanced configuration with the relative-bound "
     "transform.",
     "To isolate the contribution of each mechanism, we perform an ablation "
     "study (Fig. 1). We compare the naive-differencing baseline, the "
     "+Lattice configuration, the same configuration with predictor "
     "selection, and full TRACQ with the relative-bound transform."),
    (None,
     "Base TRACQ (Global Clamp)",
     "Naive Diff. (Global Clamp)"),
    ("Error Drift on 10,000-Step",
     "Base TRACQ",
     "Naive Diff."),
    (None,
     "the enhanced error at 1,000, 5,000, and 10,000 steps is identical",
     "TRACQ's error at 1,000, 5,000, and 10,000 steps is identical"),
    (None,
     "The base error is not monotone",
     "The naive baseline's error is not monotone"),
    (None,
     "Fig. 4. Rate-distortion comparison of the base and enhanced "
     "configurations.",
     "Fig. 4. Rate-distortion comparison of TRACQ and the naive-differencing "
     "ablation."),
    (None,
     "At the default tolerance, the enhanced configuration reduces RMSE by "
     "three orders of magnitude relative to the base configuration",
     "At the default tolerance, TRACQ reduces RMSE by three orders of "
     "magnitude relative to the naive ablation"),
    (None,
     "the enhanced variant's RMSE is 1.2–19× lower",
     "TRACQ's RMSE is 1.2–19× lower"),
    (None,
     "At matched accuracy, the enhanced configuration typically halves",
     "At matched accuracy, TRACQ typically halves"),
    (None,
     "Across datasets and configurations, the enhanced encoder reaches",
     "Across datasets and configurations, the TRACQ encoder reaches"),
    (None,
     "On MetroPT-3, the enhanced fast setting reaches",
     "On MetroPT-3, the fast setting reaches"),
    (None,
     "Across the synthetic sweep, the enhanced encoder maintains",
     "Across the synthetic sweep, TRACQ maintains"),
    ("8-bit Compression vs",
     "Base TRACQ",
     "Naive Diff."),
    (None,
     "Moving from the base to the enhanced configuration at the default "
     "tolerance increases storage",
     "Moving from the naive ablation to TRACQ at the default tolerance "
     "increases storage"),
    (None,
     "the enhanced configurations trace the lower-right frontier on all "
     "three datasets",
     "the TRACQ configurations trace the lower-right frontier on all three "
     "datasets"),
    (None,
     "In Fig. 10 the enhanced configurations reduce RMSE monotonically",
     "In Fig. 10 the TRACQ configurations reduce RMSE monotonically"),
    (None,
     "Base 16b",
     "Naive Diff. (16b)"),
    (None,
     "Enh. (0.01)",
     "TRACQ (0.01)"),
    (None,
     "Enh. (0.001)",
     "TRACQ (0.001)"),
    (None,
     "Enh. (0.0001)",
     "TRACQ (0.0001)"),
    # ---- Fig. 11 paragraph ----
    (None,
     "The base configuration is numerically unstable over the full horizon, "
     "while the enhanced variant holds its guaranteed bound",
     "Naive percentage differencing is numerically unstable over the full "
     "horizon, while TRACQ holds its guaranteed bound"),
    (None,
     "the enhanced variant reaches lower RMSE at 3.8× smaller size",
     "TRACQ reaches lower RMSE at 3.8× smaller size"),
    (None,
     "The initial algorithm's design fails due to multiplicative "
     "reconstruction, with high divergence to RMSE ",
     "The catastrophic divergence of the naive ablation, RMSE near "),
    (None,
     ", does not arise, because no multiplicative accumulation exists in the "
     "enhanced decoder.",
     ", does not arise, because no multiplicative accumulation exists in the "
     "TRACQ decoder."),
    (None,
     "at every matched size the enhanced variant is 4–13× more "
     "accurate",
     "at every matched size TRACQ is 4–13× more accurate"),
    (None,
     "its artifacts are a median of 1.9× larger than the enhanced "
     "variant's on this stream",
     "its artifacts are a median of 1.9× larger than TRACQ's on this "
     "stream"),
    # ---- real-world prose ----
    (None,
     "where the enhanced variant at tolerance 0.01 achieves",
     "where TRACQ at tolerance 0.01 achieves"),
    (None,
     "The enhanced configuration at the default tolerance achieves",
     "TRACQ at the default tolerance achieves"),
    (None,
     "the enhanced variant reaches this regime at tolerances of 0.0001",
     "TRACQ reaches this regime at tolerances of 0.0001"),
    (None,
     "The enhanced variant reaches RMSE= 18.5",
     "TRACQ reaches RMSE= 18.5"),
    (None,
     "and the base 16-bit configuration (503, correlation 0.689)",
     "and the naive 16-bit ablation (503, correlation 0.689)"),
    (None,
     "the real-world evaluation shows that the enhanced variant is effective",
     "the real-world evaluation shows that TRACQ is effective"),
    (None,
     "using the base framework's 8-bit PNG output",
     "using the naive formulation's 8-bit PNG output"),
    (None,
     "The enhanced relative-bound configuration holds per-variable SMAPE",
     "The relative-bound configuration holds per-variable SMAPE"),
    (None,
     "below the base 16-bit configuration on low-amplitude channels",
     "below the naive 16-bit ablation on low-amplitude channels"),
    (None,
     "On Appliances, the enhanced relative configuration achieves",
     "On Appliances, the relative-bound configuration achieves"),
    (None,
     "The enhanced relative configuration traces the lower-right frontier",
     "The relative-bound configuration traces the lower-right frontier"),
    (None,
     "encode both with the enhanced configuration at tolerance 0.0",
     "encode both with TRACQ at tolerance 0.0"),
    (None,
     "values above one indicate that the enhanced variant's artifact is "
     "smaller",
     "values above one indicate that TRACQ's artifact is smaller"),
    (None,
     "At equal RMSE, the enhanced variant produces the smaller artifact",
     "At equal RMSE, TRACQ produces the smaller artifact"),
    (None,
     "yet the enhanced variant still leads it at matched RMSE",
     "yet TRACQ still leads it at matched RMSE"),
    (None,
     "Maximum accuracy: the enhanced configuration at tolerance 0.0001 "
     "provides",
     "Maximum accuracy: TRACQ at tolerance 0.0001 provides"),
    (None,
     "Balanced: the enhanced configuration at the default tolerance of "
     "0.001 offers",
     "Balanced: TRACQ at the default tolerance of 0.001 offers"),

    # ---- conclusion, full reframe ----
    (None,
     "This paper presented TRACQ, a visual-first lossy compression framework "
     "for multivariate time series. The base framework introduces "
     "percentage-change domain encoding with image-based storage, producing "
     "compressed outputs that are valid, viewable images and support direct "
     "visual inspection without decoding. To address the limitations of "
     "global clamping and uniform quantization in the base framework, we "
     "further proposed an enhanced variant that quantizes an "
     "arcsinh-transformed signal onto per-variable integer levels and stores "
     "the grid of their differences. By aligning quantization resolution "
     "with per-channel scale and reconstructing through exact integer "
     "accumulation, the enhanced variant keeps every reconstructed sample "
     "within a guaranteed pointwise error bound at any horizon.",
     "This paper presented TRACQ, a visual-first lossy compression framework "
     "for multivariate time series. TRACQ quantizes an arcsinh-transformed "
     "signal onto per-variable integer levels and stores the grid of their "
     "differences, so the compressed output is a valid, viewable image that "
     "supports direct visual inspection without decoding. By aligning "
     "quantization resolution with per-channel scale and reconstructing "
     "through exact integer accumulation, TRACQ keeps every reconstructed "
     "sample within a guaranteed pointwise error bound at any horizon: "
     "quantizing before differencing separates quantization error from "
     "temporal accumulation."),
    # curly-apostrophe originals (the lib logs, not raises, on a miss, so
    # these are listed with their exact document form)
    (None,
     "when the base framework’s automated baseline offsetting is enabled",
     "when the naive formulation's automated baseline offsetting is enabled"),
    (None,
     "The initial algorithm’s design fails due to multiplicative "
     "reconstruction, with high divergence to RMSE ",
     "The catastrophic divergence of the naive ablation, RMSE near "),
    (None,
     "using the base framework’s 8-bit PNG output",
     "using the naive formulation's 8-bit PNG output"),
    (None,
     "the enhanced variant orders rows by correlation automatically when "
     "its two-dimensional predictor is selected",
     "TRACQ orders rows by correlation automatically when its "
     "two-dimensional predictor is selected"),
]


def variants(s):
    yield s
    if "'" in s:
        yield s.replace("'", "’")


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    n = 0
    for anchor, old, new in EDITS:
        if anchor:
            ed.seek(anchor)
        done = False
        for v in variants(old):
            try:
                ed.replace(v, new)
                done = True
                break
            except Exception:
                continue
        if not done:
            print(f"MISSING ({n} applied before failure): {old[:70]}")
            sys.exit(1)
        n += 1

    # safety net: a deleted range must never hold a live field instruction
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    converted = 0
    for d in ed.root.iter(ns + "del"):
        for it in d.iter(ns + "instrText"):
            it.tag = ns + "delInstrText"
            converted += 1
    if converted:
        print(f"instrText inside w:del converted: {converted}")
    ed.save()
    print(f"applied {n} edits")
    print("SEP-3 WAVE 1 DONE")


if __name__ == "__main__":
    main(sys.argv[1])

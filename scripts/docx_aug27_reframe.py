# -*- coding: utf-8 -*-
"""August 27 reframe (supervisor's meeting minutes, 2026-08-26).

The paper positions itself as purpose-built for time series rather than
general-purpose compression, with chronological-order preservation stated as
the headline differentiator. The abstract trims its motivation to one sentence
and reaches the method immediately; the contributions gain a connective
narrative (time-series characteristics -> three operational needs) and each
bullet names the characteristic it exploits and the need it serves; the
compressed-domain speedup is attributed causally to the order-preserving grid.

The chronology claim is scoped as a conjunction (order-preserving AND
error-bounded) so no baseline in Table IV falsifies it: PAA/SAX keep order but
cannot reconstruct within a bound; ZFP/SZ3 reconstruct within bounds but expose
no temporal structure until decompressed.

Usage: python docx_aug27_reframe.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

EDITS = [
    # ---- abstract: one-sentence motivation, then domain + differentiator ----
    ("The rapid growth of multivariate time series data strains storage and "
     "transmission systems. Compression is a direct way to reduce this burden. "
     "Plain compression methods often exhibit a known trade-off in the field: "
     "lossless algorithms typically achieve only modest reductions in storage "
     "size, whereas aggressive lossy approaches downsample valuable local "
     "temporal structures, causing errors to accumulate. This paper introduces "
     "TRACQ (Time-series Relative Adaptive Compression and Quantization), a "
     "compression framework that represents multivariate time series as "
     "two-dimensional grids of quantized temporal changes and applies adaptive "
     "quantization followed by image-based encoding, so that one stored "
     "artifact is simultaneously error-bounded, renderable as a standard "
     "image, and analyzable without floating-point reconstruction.",
     "Multivariate time series from dense sensor and IoT deployments pose a "
     "conflicting demand: aggressive lossy compression to cut storage and "
     "bandwidth, and stored data that remains open to rapid visual triage and "
     "real-time anomaly screening. This paper introduces TRACQ (Time-series "
     "Relative Adaptive Compression and Quantization), an error-bounded lossy "
     "compression framework purpose-built for multivariate time series: it "
     "maps multi-channel streams onto two-dimensional integer grids that "
     "preserve the chronological order of the samples in the compressed "
     "domain, so that one stored artifact is simultaneously error-bounded, "
     "renderable as a standard image, and analyzable in integer form without "
     "floating-point reconstruction. Error-bounded compressors such as ZFP "
     "and SZ3 expose no temporal structure until fully decompressed, while "
     "order-preserving representations such as PAA and SAX cannot reconstruct "
     "the signal within a bound; TRACQ provides both properties in one "
     "artifact."),
    # typo while we are in the sentence
    ("compressed using image codes (e.g., PNG)",
     "compressed using image codecs (e.g., PNG)"),
    # anomaly sentence: chronology stated as the enabler (text before the F1
    # math and after the times math; the math runs themselves stay untouched)
    ("Our tests also demonstrate that anomaly detection can be directly "
     "applied on our resulting compressed images without needing "
     "reconstruction of the time-series, matching the decode-then-detect "
     "pipeline accuracy at ",
     "Because the grid preserves chronological order, anomaly detection runs "
     "directly on the stored integer grids, matching the decode-then-detect "
     "pipeline on injected anomalies at "),
    (" the throughput.",
     " its throughput; on documented industrial failures it retains most of "
     "the decoded pipeline's ranking quality at 27× the throughput."),

    # ---- contributions: narrative lead-in, then per-bullet ties ----
    ("This paper makes five concrete contributions:",
     "The design follows from the physical characteristics of time series "
     "data: channels span heterogeneous scales, consecutive changes are small "
     "most of the time but heavy-tailed, horizons run to millions of steps, "
     "and the data is monitored as it arrives. These characteristics "
     "translate into three operational needs: storage efficiency, "
     "transmission stability, and downstream analysis on the stored form. "
     "This paper makes five concrete contributions, each stated with the "
     "characteristic it exploits and the need it serves:"),
    ("Visual change-domain encoding: We encode a multivariate time series as "
     "a two-dimensional grid of quantized temporal changes that renders as a "
     "grayscale image and supports integer-domain analysis, so neither "
     "inspection nor screening requires floating-point reconstruction.",
     "Visual change-domain encoding: Exploiting the temporal smoothness of "
     "real streams, we encode a multivariate time series as a two-dimensional "
     "grid of quantized temporal changes whose columns preserve chronological "
     "order; the grid renders as a grayscale image and supports "
     "integer-domain analysis, so neither inspection nor screening requires "
     "floating-point reconstruction (downstream analysis)."),
    ("Drift-free integer reconstruction: We eliminate cumulative drift from "
     "our initial conceptual design by quantizing each variable onto its own "
     "integer lattice before differencing, so reconstruction becomes exact "
     "integer accumulation and every transformed sample stays within half a "
     "quantization step of its transformed original at any horizon.",
     "Drift-free integer reconstruction: Because monitoring streams run to "
     "millions of steps without a reset point, accumulated error is fatal; we "
     "eliminate the cumulative drift of our initial conceptual design by "
     "quantizing each variable onto its own integer lattice before "
     "differencing, so reconstruction becomes exact integer accumulation and "
     "every transformed sample stays within half a quantization step of its "
     "transformed original at any horizon (transmission stability)."),
    ("Scale-aware adaptive quantization: We derive each channel's lattice "
     "step from a user-specified tolerance and apply a symmetric arcsinh "
     "transform to compress the variable’s range without baseline offsetting.",
     "Scale-aware adaptive quantization: Multivariate streams mix channels "
     "whose scales differ by orders of magnitude; we derive each channel's "
     "lattice step from a user-specified tolerance and apply a symmetric "
     "arcsinh transform to compress the variable's range without baseline "
     "offsetting, so quantization resolution goes where each channel's data "
     "actually lies (storage efficiency)."),
    ("Compressed-domain anomaly screening: we show that an unsupervised "
     "Isolation Forest operates directly on the encoded integer grids, "
     "matching the decoded floating-point pipeline on injected anomalies (F1 "
     "= 0.75) and retaining most of its quality on real-time documented "
     "industrial timeseries, at 21× to 27× the throughput..",
     "Compressed-domain anomaly screening: Because the grid preserves "
     "chronological order, temporal features can be computed directly on the "
     "stored integers; an unsupervised Isolation Forest matches the decoded "
     "floating-point pipeline on injected anomalies (F1 = 0.75) and retains "
     "most of its ranking quality on documented industrial failures, at "
     "21× to 27× the throughput (downstream analysis)."),
    ("-memory windowed encoder that scales seamlessly across modern "
     "multi-core architectures, sustaining",
     "-memory windowed encoder that processes unbounded streams as fixed "
     "independent windows, sustaining"),
    ("a 73× speedup to serial execution",
     "a 73× speedup over serial execution (transmission stability and "
     "storage at scale)"),

    # ---- Section V: the speedup attributed causally to chronology ----
    ("Crucially, the compressed-domain path achieves a 27× throughput "
     "speedup by bypassing floating-point decompression entirely.",
     "Because the grid preserves chronological order, trajectory shapes and "
     "per-timestep activity are computed directly on the stored integers; "
     "the compressed-domain path therefore never reconstructs the "
     "floating-point series, which is the source of the 27× throughput "
     "gain."),
]


def variants(s):
    yield s
    if "'" in s:
        yield s.replace("'", "’")


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    full = "\n".join(ed.para_text(p) for p in ed.paras)
    claimed, resolved, missing = set(), [], []
    for old, new in EDITS:
        hit = None
        for v in variants(old):
            start = 0
            while True:
                pos = full.find(v, start)
                if pos < 0:
                    break
                if pos not in claimed:
                    hit = (pos, v, new)
                    break
                start = pos + 1
            if hit:
                break
        if hit:
            claimed.add(hit[0])
            resolved.append(hit)
        else:
            missing.append(old[:70])
    if missing:
        print("DRY-RUN MISSING (aborting, nothing applied):")
        for m in missing:
            print("   !!", m)
        sys.exit(1)
    for pos, v, new in sorted(resolved):
        ed.replace(v, new)
    ed.save()
    print(f"applied {len(resolved)} edits")
    print("AUG-27 REFRAME DONE")


if __name__ == "__main__":
    main(sys.argv[1])

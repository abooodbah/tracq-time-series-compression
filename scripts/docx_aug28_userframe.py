# -*- coding: utf-8 -*-
"""August 28: the author's final abstract and contributions, applied tracked.

The abstract becomes the author's rewritten version (hybrid opening, plain-text
throughout, no result numerals); the leftover inline math of the old closing
sentence (F1, times sign) is tracked-deleted since the new text carries none.
The contribution list becomes the author's five Title-Case bullets, with
scale-aware quantization moved ahead of drift-free differencing, and the
lead-in naming the structural properties and three operational requirements.
The O(1) math in the streaming bullet is kept in place and the new text is
stitched around it.

Usage: python docx_aug28_userframe.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

NEW_ABSTRACT = (
    "The rapid growth of multivariate time series data strains storage and "
    "transmission systems. Compression is a direct way to reduce this burden. "
    "However, compression methods face a fundamental trade-off: lossless "
    "algorithms achieve only modest size reductions on floating-point data, "
    "whereas lossy approaches often downsample local patterns or cause errors "
    "to accumulate over long sequences. Furthermore, existing error-bounded "
    "scientific compressors (such as ZFP and SZ3) produce binary formats that "
    "require full decompression before data can be inspected or analyzed. "
    "This paper introduces TRACQ (Time-series Relative Adaptive Compression "
    "and Quantization), an error-bounded lossy compression framework "
    "purpose-built for multivariate time series. TRACQ converts multi-channel "
    "data into two-dimensional integer grids that preserve strict, "
    "sample-by-sample chronological order directly within the compressed "
    "domain. The proposed method combines three mechanisms: (1) an adaptive "
    "scaling transform that balances variables across different scales, (2) "
    "quantizing each variable onto an integer grid using step sizes derived "
    "from a user-specified error tolerance, and (3) storing consecutive "
    "integer differences so that reconstruction becomes an integer addition "
    "problem, ensuring errors never accumulate over time. The resulting "
    "compressed grid can be viewed directly as a standard grayscale image "
    "(such as a PNG) for instant visual inspection or stored as a compact "
    "binary file (via Zstandard) for efficient transmission. Evaluations "
    "across synthetic workloads, real-world benchmarks, and million-step "
    "industrial streams show that TRACQ consistently produces smaller files "
    "than state-of-the-art scientific compressors at matched accuracy while "
    "scaling favorably due to a constant working memory footprint. Finally, "
    "because the grid preserves chronological order, downstream tasks such as "
    "anomaly detection can run directly on the stored compressed domain data "
    "without the need for full decompression, closely approaching the "
    "accuracy of raw floating-point pipelines while running more than an "
    "order of magnitude faster."
)

OLD_ABSTRACT_HEAD = (
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
    "artifact. The method combines three mechanisms: (1) a per-variable "
    "scale-aware transform that gives each variable its own quantization "
    "lattice, (2) integer coding of the differences between consecutive "
    "lattice indices, and (3) reconstruction by exact integer accumulation, "
    "which bounds the error of every sample to half a quantization step "
    "given a user-specified tolerance. The resulting grids can be further "
    "compressed using image codecs (e.g., PNG) or general-purpose "
    "compressors, such as Zstandard. We evaluate a base and an enhanced "
    "variant of our pipeline on synthetic sensor, financial, IoT, and "
    "electricity workloads, as well as real-world UCI datasets. The enhanced "
    "pipeline holds a user-specified tolerance bound across time series "
    "variables; reconstruction error stays inside it regardless of horizon "
    "length. TRACQ reaches the accuracy of error-bounded HPC compression "
    "tools such as ZFP and SZ3 with smaller final artifacts (1.2–1.9× "
    "smaller against SZ3, 1.5–6.9× against ZFP, and up to 22× on the "
    "longest stream) and compresses to sizes their accuracy-targeted modes "
    "do not reach. Because the grid preserves chronological order, anomaly "
    "detection runs directly on the stored integer grids, matching the "
    "decode-then-detect pipeline on injected anomalies at "
)

EDITS = [
    (OLD_ABSTRACT_HEAD, NEW_ABSTRACT),
    (" 0.75 while running at ", ""),
    (" its throughput; on documented industrial failures it retains most of "
     "the decoded pipeline's ranking quality at 27× the throughput.", ""),

    # ---- contributions lead-in ----
    ("The design follows from the physical characteristics of time series "
     "data: channels span heterogeneous scales, consecutive changes are small "
     "most of the time but heavy-tailed, horizons run to millions of steps, "
     "and the data is monitored as it arrives. These characteristics "
     "translate into three operational needs: storage efficiency, "
     "transmission stability, and downstream analysis on the stored form. "
     "This paper makes five concrete contributions, each stated with the "
     "characteristic it exploits and the need it serves:",
     "TRACQ exploits key structural properties of multivariate time series: "
     "continuous temporal trends, scales that differ by orders of magnitude "
     "across channels, inter-variable correlations, and long horizons "
     "spanning millions of steps. These properties translate into three "
     "operational requirements: storage efficiency, transmission "
     "scalability, and downstream analysis directly on the stored form. This "
     "paper makes five concrete contributions:"),

    # ---- bullet 1 ----
    ("Visual change-domain encoding: Exploiting the temporal smoothness of "
     "real streams, we encode a multivariate time series as a two-dimensional "
     "grid of quantized temporal changes whose columns preserve chronological "
     "order; the grid renders as a grayscale image and supports "
     "integer-domain analysis, so neither inspection nor screening requires "
     "floating-point reconstruction (downstream analysis).",
     "Time-Preserving 2D Grid Representation: We encode multivariate time "
     "series as a two-dimensional grid of integer differences that preserves "
     "strict, sample-by-sample chronological order across every variable. "
     "Whereas transform-based scientific compressors (such as ZFP) map data "
     "into spectral coefficient blocks and symbolic methods (such as PAA and "
     "SAX) collapse temporal resolution through windowed averaging, TRACQ "
     "produces an artifact that renders directly as a grayscale image for "
     "human triage or packs into a Zstandard container for transmission "
     "(downstream analysis and storage)."),

    # ---- doc position 2 becomes the author's scale-aware bullet ----
    ("Drift-free integer reconstruction: Because monitoring streams run to "
     "millions of steps without a reset point, accumulated error is fatal; we "
     "eliminate the cumulative drift of our initial conceptual design by "
     "quantizing each variable onto its own integer lattice before "
     "differencing, so reconstruction becomes exact integer accumulation and "
     "every transformed sample stays within half a quantization step of its "
     "transformed original at any horizon (transmission stability).",
     "Scale-Aware Adaptive Quantization: Channels spanning vastly different "
     "scales share a single user-specified tolerance. Each variable's "
     "lattice step is derived directly from this tolerance, and a "
     "channel-adaptive arsinh transform compresses extreme values "
     "logarithmically while remaining linear near zero. This formulation "
     "accommodates positive, negative, and zero-crossing signals without "
     "artificial baseline offsets (storage efficiency)."),

    # ---- doc position 3 becomes the author's drift-free bullet ----
    ("Scale-aware adaptive quantization: Multivariate streams mix channels "
     "whose scales differ by orders of magnitude; we derive each channel's "
     "lattice step from a user-specified tolerance and apply a symmetric "
     "arcsinh transform to compress the variable's range without baseline "
     "offsetting, so quantization resolution goes where each channel's data "
     "actually lies (storage efficiency).",
     "Drift-Free Differencing with Guaranteed Error Bounds: In "
     "floating-point delta encoding, quantization errors compound across "
     "time steps and cause severe drift. We eliminate this by quantizing "
     "each transformed variable onto an integer lattice before differencing. "
     "Reconstruction reduces to exact integer summation, ensuring every "
     "sample stays within half a quantization step (≤ q/2) of the "
     "original across arbitrary horizons (reconstruction fidelity)."),

    # ---- bullet 4 ----
    ("Compressed-domain anomaly screening: Because the grid preserves "
     "chronological order, temporal features can be computed directly on the "
     "stored integers; an unsupervised Isolation Forest matches the decoded "
     "floating-point pipeline on injected anomalies (F1 = 0.75) and retains "
     "most of its ranking quality on documented industrial failures, at "
     "21× to 27× the throughput (downstream analysis).",
     "Direct Anomaly Detection on Stored Integers: Because the compressed "
     "grid preserves sample-level chronological order, machine learning "
     "models can slide windows and extract sequential features directly on "
     "the compressed representation. An unsupervised Isolation Forest "
     "operating on the raw integer grids reaches near-parity with decoded "
     "float64 pipelines (F1 = 0.75 on synthetic benchmarks, and ROC-AUC = "
     "0.943 vs. 0.948 on documented industrial failures) at 21×–"
     "27× the throughput (downstream analysis)."),

    # ---- bullet 5: stitched around the O(1) math ----
    ("Constant-memory streaming and multi-core scaling: We design an ",
     "Constant-Memory Streaming and Multi-Core Scaling: We design a "
     "windowed encoder that keeps working memory independent of sequence "
     "length ("),
    ("-memory windowed encoder that processes unbounded streams as fixed "
     "independent windows, sustaining ",
     " auxiliary overhead). Across 112 cores, the implementation sustains "),
    ("end-to-end throughput across 112 cores on a 1 TB stream, a 73× "
     "speedup over serial execution (transmission stability and storage at "
     "scale)",
     "end-to-end compression throughput on a 1 TB stream (a 73× speedup "
     "over single-core execution) while keeping per-worker resident memory "
     "strictly below 118 MB (transmission scalability)."),
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
    # the new abstract is plain text: tracked-delete the old inline math and
    # the plain-text "F1"/"21" runs that sat against the deleted math signs
    n = ed.del_inline_math("an integer addition problem", count=99)
    ed.replace("F1 21", "")
    ed.save()
    print(f"applied {len(resolved)} edits; abstract inline math deleted: {n}")
    print("AUG-28 USER FRAME DONE")


if __name__ == "__main__":
    main(sys.argv[1])

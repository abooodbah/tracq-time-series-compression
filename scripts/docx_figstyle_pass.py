# -*- coding: utf-8 -*-
"""Apply the advisor's figure-style feedback to a document tree: the new
title, short name-only figure captions (panel letters now live inside the
figures), and migration of caption explanations into the citing body text
where the body does not already state them.

Usage: python docx_figstyle_pass.py <tree_root> <iso_fig_n> <scaling_fig_n> [--highdim]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor


def build_edits(iso_n, sc_n, highdim):
    E = [
        ("Efficient Analysis of Multivariate Time Series with Adaptive Compression and "
         "Quantization",
         "A Novel Approach to Multivariate Time Series Analysis Using Adaptive Compression "
         "and Quantization"),
        # ---- caption shortening ----
        ("Fig. 1. Ablation study: RMSE heatmap showing the effect of lattice coding, "
         "predictor selection, and the relative-bound transform. Darker colors indicate "
         "lower (better) RMSE.",
         "Fig. 1. Ablation study: RMSE by configuration and data type."),
        ("Fig. 2. Multi-scale handling: mean relative error vs. variable scale. Per-variable "
         "quantization steps (green, purple) maintain consistent relative error across 6 "
         "orders of magnitude; global clamping (red) degrades severely at extreme scales.",
         "Fig. 2. Mean relative error vs. variable scale."),
        ("Fig. 3. Cumulative RMSE over 10,000 time steps. The base configuration (red) "
         "exhibits severe drift; the enhanced variant (blue) stays flat below the guaranteed "
         "bound (dashed) over the full horizon.",
         "Fig. 3. Cumulative RMSE over 10,000 time steps on (a) linear and (b) logarithmic "
         "scales."),
        ("Fig. 4. Rate-distortion curves. The enhanced configuration (green, purple) "
         "achieves one to three orders of magnitude lower RMSE than the base configuration "
         "(red) across the full ratio range.",
         "Fig. 4. Rate-distortion comparison of the base and enhanced configurations."),
        ("Fig. 5. Rate-distortion comparison with ZFP. The enhanced configuration covers "
         "ratios ZFP cannot reach and matches its accuracy class at smaller sizes, while "
         "preserving visual interpretability.",
         "Fig. 5. Rate-distortion comparison with ZFP."),
        ("Fig. 6. Encoding and decoding throughput (MB/s). The proposed method achieves "
         "throughput comparable to ZFP while adding visual interpretability.",
         "Fig. 6. (a) Encoding and (b) decoding throughput on synthetic workloads."),
        ("Fig. 7. Encoding throughput on real-world UCI datasets. The proposed method's "
         "throughput is competitive with general-purpose lossless compressors.",
         "Fig. 7. Encoding throughput on the real-world UCI datasets."),
        ("Fig. 8. Streaming scalability on large workloads. Left: enhanced TRACQ throughput "
         "and peak RSS on synthetic 16-variable sensor streams from  to  rows using "
         "100,000-step windows. Right: MetroPT-3 streaming benchmark on 1,516,948 rows and "
         "15 compressor sensors using 10,000-step windows.",
         "Fig. 8. Streaming scalability: (a) synthetic streams; (b) MetroPT-3 benchmark."),
        ("Fig. 9. Rate-distortion curves on three UCI real-world datasets. Each point "
         "represents a compression method; the lower-left corner is better (lower ratio, "
         "lower RMSE). Pathological overflow points are omitted. The enhanced configurations "
         "trace the lower-left frontier on all three datasets.",
         "Fig. 9. Rate-distortion curves on the three UCI datasets."),
        ("Fig. 10. Grouped bar comparison of RMSE across methods and real-world datasets "
         "(log scale). Overflow/pathological entries are capped at . The enhanced "
         "configurations reduce RMSE monotonically with the tolerance and dominate all "
         "non-HPC baselines; ZFP reaches lower RMSE only at 1.5–7× larger encoded sizes on "
         "these datasets (Fig. %d)." % iso_n,
         "Fig. 10. RMSE by method across the real-world datasets."),
        ("Fig. 11. Full-length MetroPT-3 rate-distortion. Left: compression ratio vs. RMSE. "
         "Right: compression ratio vs. SMAPE. MetroPT-3 contains 1,516,948 time steps and 15 "
         "compressor variables. The enhanced variant traces the lower-left frontier across "
         "three decades of ratio; PAA improves no further with size, and ZFP operates only "
         "above ratio 0.11.",
         "Fig. 11. Full-length MetroPT-3 rate-distortion: (a) RMSE; (b) SMAPE."),
        ("Fig. 12. SMAPE analysis. (a) Per-variable SMAPE on Appliances Energy sorted by "
         "variable magnitude; the transform domain holds relative fidelity flat across all "
         "channels. (b) SMAPE vs. compression ratio across all datasets and methods; the "
         "enhanced relative configuration traces the lower-left frontier.",
         "Fig. 12. SMAPE analysis: (a) per-variable SMAPE on Appliances Energy; (b) SMAPE "
         "vs. compression ratio."),
        ("Fig. 13. Visual inspection demonstration: normal vs. anomalous signal encoded as "
         "8-bit compressed heatmaps. Injected anomalies (spike, level shift, oscillation) "
         "are visually identifiable in the compressed artifact; grid panels are displayed "
         "with a windowed gray range for print contrast.",
         "Fig. 13. Visual inspection demonstration: (a) raw anomalous signal; (b) compressed "
         "grid, normal; (c) compressed grid, anomalous."),
        ("Fig. 14. Compressed-domain anomaly detection. Left: F1, precision, and recall "
         "across four detection pipelines. Right: throughput comparison showing a 21× "
         "speedup at identical F1 when detecting anomalies directly on compressed grids "
         "versus decoding to float64 first.",
         "Fig. 14. Compressed-domain anomaly detection: (a) detection quality; (b) "
         "throughput."),
        ("Fig. %d. Encoded size at matched RMSE. (a) MetroPT-3: measured size of all three "
         "codecs at equal RMSE. (b) Size advantage at the same RMSE against the stronger of "
         "ZFP and SZ3 at each point; values above one indicate that the enhanced variant's "
         "artifact is smaller." % iso_n,
         "Fig. %d. Encoded size at matched RMSE: (a) MetroPT-3; (b) advantage over the "
         "stronger of ZFP and SZ3." % iso_n),
        ("Fig. %d. Node-parallel scaling on one 112-core Sapphire Rapids node, median of "
         "three runs (bars span min–max). (a) Strong scaling of a fixed 50 GB stream: "
         "measured speedup against the ideal line. (b) Weak scaling at 8 GB per worker: "
         "parallel efficiency (left axis) and encode-stage per-core throughput (right "
         "axis)." % sc_n,
         "Fig. %d. Node-parallel scaling on one Stampede3 node: (a) strong scaling; (b) "
         "weak scaling." % sc_n),
        # ---- migrations into citing body text ----
        ("the full enhanced configuration with the relative-bound transform.",
         "the full enhanced configuration with the relative-bound transform. Darker cells "
         "in the heatmap indicate lower RMSE."),
        ("Rate-distortion curves are plotted in Fig. 9, and a grouped bar comparison "
         "appears in Fig. 10.",
         "Rate-distortion curves are plotted in Fig. 9, and a grouped bar comparison "
         "appears in Fig. 10. In Fig. 9 the lower-left corner is better and pathological "
         "overflow points are omitted; the enhanced configurations trace the lower-left "
         "frontier on all three datasets. In Fig. 10 overflow entries are capped for "
         "display; the enhanced configurations reduce RMSE monotonically with the tolerance "
         "and dominate all non-HPC baselines, while ZFP reaches lower RMSE only at 1.5–7× "
         "larger encoded sizes (Fig. %d)." % iso_n),
        ("and encode both with the enhanced configuration at tolerance 0.01.",
         "and encode both with the enhanced configuration at tolerance 0.01. The grid "
         "panels are displayed with a windowed gray range for print contrast."),
        ("reports the comparison on the three UCI datasets and the full-length MetroPT-3 "
         "stream.",
         "reports the comparison on the three UCI datasets and the full-length MetroPT-3 "
         "stream; values above one in its advantage panel indicate that the enhanced "
         "variant's artifact is smaller."),
        ("Each point is the median of three runs.",
         "Each point is the median of three runs, and the error bars span the minimum to "
         "maximum of those runs."),
    ]
    if highdim:
        E.append(
            ("Fig. 15. High-dimensional scaling on grouped synthetic workloads (2,000 "
             "steps, 16 correlated groups). (a) The fast encoding path and the decoder are "
             "insensitive to width; predictor selection pays a quadratic correlation cost. "
             "(b) Encoded size improves with width while metadata remains a small share of "
             "the artifact.",
             "Fig. 15. High-dimensional scaling: (a) throughput vs. width; (b) encoded size "
             "and metadata share vs. width."))
    return E


def variants(s):
    yield s
    if "'" in s:
        yield s.replace("'", "’")
    if "'" in s and "–" in s:
        yield s.replace("'", "’").replace("–", "-")


def main():
    tree = sys.argv[1]
    iso_n, sc_n = int(sys.argv[2]), int(sys.argv[3])
    highdim = "--highdim" in sys.argv
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    full = "\n".join(ed.para_text(p) for p in ed.paras)
    claimed, resolved, missing = set(), [], []
    for old, new in build_edits(iso_n, sc_n, highdim):
        hit = None
        for v in variants(old):
            start = 0
            while True:
                p = full.find(v, start)
                if p < 0:
                    break
                if p not in claimed:
                    hit = (p, v, new)
                    break
                start = p + 1
            if hit:
                break
        if hit:
            claimed.add(hit[0])
            resolved.append(hit)
        else:
            missing.append(old[:80])
    if missing:
        print("DRY-RUN MISSING (aborting, no edits applied):")
        for m in missing:
            print("   !!", m)
        sys.exit(1)
    for p, v, new in sorted(resolved):
        ed.replace(v, new)
    # the shortened captions for Figs 8 and 10 drop the sentences that carried
    # inline math (row-count range, display cap): delete those math elements
    ed.del_inline_math("Fig. 8. Streaming scalability", count=9)
    ed.del_inline_math("Fig. 10. RMSE by method", count=9)
    ed.save()
    print("FIGSTYLE PASS DONE")


if __name__ == "__main__":
    main()

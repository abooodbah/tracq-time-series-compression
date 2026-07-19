# -*- coding: utf-8 -*-
"""Fresh-submission polish pass: align claims with what was measured and make
cross-references match the rendered heading style.

- TB paragraph: separate end-to-end node throughput from encode-stage per-core
  throughput so the two figures are arithmetically consistent.
- Scope ZFP/SZ3 envelope statements to the evaluated tolerance settings.
- Name the rounded-delta baseline for what it is (lossy; Gorilla-inspired).
- Replace the zero-copy coinage with a precise description.
- Define the transform scale and abs-mode step operationally.
- Arabic section/table references -> rendered Roman/letter style.
- Fig. 7 caption: the plot shows encoding throughput only.
- Drop the speculative compiled-throughput multiplier.
- Restyle two body paragraphs mistakenly carrying Heading2 style.

Usage: python docx_review_polish_pass.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q

EDITS = [
    # ---- abstract ----
    ("as two-dimensional grids of relative changes and applies adaptive quantization",
     "as two-dimensional grids of quantized temporal changes and applies adaptive quantization"),
    ("at every horizon, with no stored anchor values",
     "at every horizon, without periodic anchor corrections"),
    ("covers aggressive operating points those tools cannot produce, while preserving a key "
     "structural property: compressed artifacts remain valid images, supporting",
     "covers aggressive operating points their accuracy-targeted modes do not reach, while "
     "preserving a key structural property: the quantized grid renders as a standard image, "
     "supporting"),
    ("without numerical decompression nor reconstruction",
     "without reconstructing the floating-point series"),
    # ---- introduction ----
    ("into two-dimensional grids of relative changes, adaptively quantizes them",
     "into two-dimensional grids of temporal changes, adaptively quantizes them"),
    ("Relative-change domain encoding. We encode relative changes between consecutive samples "
     "rather than raw values",
     "Change-domain integer encoding. We encode changes between consecutive samples rather "
     "than raw values"),
    ("produce valid images that can be scanned for anomalies",
     "produce image-renderable grids that can be scanned for anomalies"),
    ("Section 3 details the encoding pipeline", "Section III details the encoding pipeline"),
    ("Section 4 describes the synthetic workloads", "Section IV describes the synthetic workloads"),
    ("Section 5 quantifies", "Section V quantifies"),
    # ---- method ----
    ("a per-timestep grid of relative changes",
     "a per-timestep grid of quantized changes"),
    ("The scale sets the crossover between the two regimes; a linear mode bypasses the "
     "transform when an absolute bound in original units is preferred.",
     "The scale sets the crossover between the two regimes and is set per variable to one "
     "percent of the channel's median absolute value, with the same fraction of its standard "
     "deviation as a fallback when the median is zero; a linear mode bypasses the transform "
     "when an absolute bound in original units is preferred, with the step derived from the "
     "variable's observed minimum-to-maximum span in the encoded window."),
    # ---- experimental setup ----
    ("A simplified Gorilla-style [27] encoder using delta-of-delta varint coding followed by "
     "zlib.",
     "A rounded-delta encoder that quantizes samples to integers and applies Gorilla-style "
     "[27] delta-of-delta varint coding followed by zlib; the rounding step makes it lossy, "
     "unlike Gorilla itself."),
    # ---- results ----
    ("encoding relative changes as per-variable integer steps",
     "encoding temporal changes as per-variable integer steps"),
    ("Table 2 and Fig. 2 examine", "Table II and Fig. 2 examine"),
    ("Table 3 and Fig. 3 examine", "Table III and Fig. 3 examine"),
    ("with no anchor values stored", "with no periodic anchor corrections required"),
    ("quantified in Sec. 5.10", "quantified in Sec. V-J"),
    ("Fig. 7. Encoding and decoding throughput on real-world UCI datasets",
     "Fig. 7. Encoding throughput on real-world UCI datasets"),
    ("in 51 s, sustaining 19.5 GB/s in aggregate (360 MB/s per core) while the peak",
     "in 51 s of wall time, 19.5 GB/s end to end including in-worker data generation; the "
     "encode stage itself sustained 360 MB/s per core, in line with the single-core sweep, "
     "while the peak"),
    ("A compiled (C/Rust) implementation would likely achieve 2–5",
     "A compiled implementation of the same pipeline is left as engineering work."),
    (" higher throughput, which we leave as straightforward engineering work.", " "),
    ("Table 4 summarizes the 8-bit", "Table IV summarizes the 8-bit"),
    ("that ZFP cannot produce on this workload",
     "that ZFP's tolerance mode does not reach on this workload"),
    ("three UCI datasets (Section 4)", "three UCI datasets (Section IV)"),
    ("Table 5 summarizes the key metrics", "Table V summarizes the key metrics"),
    ("computed against the original float64 data.",
     "computed against the original float64 data. Correlations printed as 1.000 round from "
     "values of at least 0.9995."),
    ("analyzed separately in Section 5.10", "analyzed separately in Section V-J"),
    ("outperforming Gorilla (RMSE", "outperforming the rounded-delta baseline (RMSE"),
    ("and Gorilla (2483)", "and the rounded-delta baseline (2483)"),
    ("Gorilla-like encoding produces", "The rounded-delta baseline produces"),
    ("; on these workloads, however, neither produces an artifact below one fifth",
     "; at the evaluated tolerance settings on these workloads, however, neither produces an "
     "artifact below one fifth"),
    ("frontier on all three datasets; ZFP and SZ3 remain confined to ratios above one fifth "
     "of the original size.",
     "frontier on all three datasets."),
    ("(Section 5.10) reveals per-variable", "(Section V-J) reveals per-variable"),
    ("computationally without numerical decompression, which we call zero-copy analytics.",
     "computationally without reconstructing the floating-point series, operating directly on "
     "the stored integer representation."),
    ("Table 6 summarizes detection performance", "Table VI summarizes detection performance"),
    ("because ZFP's size floor sits near one fifth of the original on these workloads",
     "because ZFP's tolerance mode saturates near one fifth of the original size on these "
     "workloads"),
    # ---- discussion / conclusion ----
    ("the compressed-domain detector in Section 5.12",
     "the compressed-domain detector in Section V-L"),
    ("decode-then-detect pipelines (Section 5.12)", "decode-then-detect pipelines (Section V-L)"),
    ("and covers aggressive operating points those compressors cannot produce.",
     "and covers aggressive operating points their accuracy-targeted modes do not reach."),
]


def variants(s):
    yield s
    if "'" in s:
        yield s.replace("'", "’")


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    full = "\n".join(ed.para_text(p) for p in ed.paras)

    claimed = set()
    resolved, missing = [], []
    for old, new in EDITS:
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
            missing.append(old[:70])

    if missing:
        print("DRY-RUN MISSING (aborting, no edits applied):")
        for m in missing:
            print("   !!", m)
        sys.exit(1)

    for p, v, new in sorted(resolved):
        ed.replace(v, new)

    # drop the multiplier's math symbol left between the two rewritten segments
    for para in ed.paras:
        t = ed.para_text(para)
        if "vectorized encoding path" in t and "left as engineering work" in t:
            oms = para.findall(q("m:oMath"))
            if len(oms) == 1:
                ed.del_inline_math("vectorized encoding path", count=1, skip=0)
            else:
                print("  !! compiled-claim para has %d oMath, skipped" % len(oms))
            break

    # Table V row label: the cell paragraph containing exactly "Gorilla"
    for pi, para in enumerate(ed.paras):
        if ed.para_text(para).strip() == "Gorilla":
            ed.cursor_p, ed.cursor_c = pi, 0
            ed.replace("Gorilla", "Rounded delta")
            break

    # restyle the two body paragraphs that carry Heading2
    fixed = 0
    for para in ed.paras:
        ppr = para.find(q("w:pPr"))
        if ppr is None:
            continue
        st = ppr.find(q("w:pStyle"))
        if st is None or st.get(q("w:val")) != "Heading2":
            continue
        t = ed.para_text(para).strip()
        if t.startswith("The scalar offset") or ("error tolerance" in t and "baseline" in t):
            ppr.remove(st)
            fixed += 1
    print("restyled headings:", fixed)

    ed.save()
    print("REVIEW POLISH DONE")


if __name__ == "__main__":
    main(sys.argv[1])

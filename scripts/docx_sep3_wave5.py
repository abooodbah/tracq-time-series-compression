# -*- coding: utf-8 -*-
"""Wave 5: reviewer follow-ups on the definitive-TRACQ build.

Table IV gives the ablation its own category (Naive Diff. no longer sits under
TRACQ, honoring "it is not a proposed TRACQ mode"); the related-work image
paragraph describes the definitive representation rather than the naive one;
the Section III-A naive motivation shrinks by another third; five of the
repeated no-decompression claims in Sections V-K through V-M are removed or
merged, keeping one statement per section role; Fig. 8's caption names the
plotted series; and a subject-verb slip in the intro is corrected.

The deletions shorten the flow, so two frame anchors move to keep every
column filled: Fig. 14's block re-anchors after the matched-operating-points
sentence (its old anchor beached page 14), and Fig. 2's block re-anchors at
the top of Section V-E (its old anchor in V-C forced the V-D text onto the
figure page and beached page 6; the residual gap there is set by Table II's
keep-together unit). Fig. 3 stays at its original anchor before the
verification paragraph.

Usage: python docx_sep3_wave5.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q

EDITS = [
    ("We proposes a novel approach", "We propose a novel approach"),
    # related work: describe the definitive representation
    ("each pixel corresponds to a quantized percentage change",
     "each pixel represents the change between two consecutive quantized "
     "values"),
    # III framing: naive motivation cut by a third
    ("It differences the data into percentage changes first and quantizes "
     "second, which places quantization error inside the accumulation path; "
     "its ratio, clamp, and companding stages collapse into TRACQ's single "
     "transform, and its percentile clamp heuristics become the single error "
     "tolerance that fixes every step size. The artifact, a per-timestep "
     "grid of quantized changes, is the same in both cases. We first "
     "describe the naive formulation's grid construction: transforming a "
     "multivariate time series into a grid of percentage changes, clamping, "
     "quantizing, and encoding as an image. We then present the three "
     "mechanisms that define TRACQ:",
     "It differences the data into percentage changes first and quantizes "
     "second, which places quantization error inside the accumulation path. "
     "The artifact, a per-timestep grid of quantized changes, is the same in "
     "both cases. The rest of this section presents the three mechanisms "
     "that define TRACQ:"),
    ("For variables that cross or approach zero, the naive formulation "
     "additionally uses an optional automated baseline offsetting safeguard "
     "before percentage-change computation; TRACQ requires no such "
     "safeguard, since its transform domain is well defined at and across "
     "zero.",
     "The naive formulation also needs a baseline offsetting safeguard for "
     "variables that cross or approach zero; TRACQ does not, since its "
     "transform is defined at and across zero."),
    # V-K/L/M: dedupe the no-decompression claims
    ("All three anomalies are visually distinguishable in the compressed "
     "heatmap without numerical decompression.",
     "All three anomalies are visually distinguishable in the compressed "
     "heatmap."),
    ("existing image viewers and monitoring tools can display them without "
     "specialized decoders",
     "existing image viewers and monitoring tools can display them directly"),
    ("run the same unsupervised Isolation Forest as the numerical pipeline. "
     "No float64 decompression is performed.",
     "run the same unsupervised Isolation Forest as the numerical pipeline."),
    ("This experiment shows that the compressed images can support both "
     "human inspection and machine detection. Error-bounded compressors "
     "such as ZFP and SZ3 optimize for archival fidelity; analyzing the "
     "stored data requires full decompression back to floating point. TRACQ "
     "instead produces image files from which features can be extracted "
     "directly, allowing compressed-domain analysis without reconstruction. "
     "This property is useful for active monitoring pipelines when data "
     "volumes are too large for full real-time decompression.",
     "This experiment shows that the compressed grids support both human "
     "inspection and machine detection, a combination the archival-focused "
     "formats of ZFP and SZ3 do not offer."),
    # Fig. 8 caption names its series
    ("Fig. 8. Streaming scalability: (a) synthetic streams; (b) MetroPT-3 "
     "benchmark.",
     "Fig. 8. Streaming scalability: (a) synthetic streams; (b) MetroPT-3 "
     "benchmark throughput and compression factor for TRACQ-Fast, "
     "TRACQ-Archival, and Gzip."),
]


def variants(s):
    yield s
    if "'" in s:
        yield s.replace("'", "\u2019")


def fix_table_iv(ed):
    """Ablation gets its own category cell; TRACQ keeps only its three rows."""
    ns = q("w:")[:-1] + "}"  # namespace brace form via helper

    for tbl in ed.root.iter(q("w:tbl")):
        if "Naive Diff. (16b)" not in "".join(tbl.itertext()):
            continue
        rows = list(tbl.iter(q("w:tr")))
        naive_row = next(r for r in rows
                         if "Naive Diff. (16b)" in "".join(r.itertext()))
        first_tc = naive_row.find(q("w:tc"))
        # rename the merged category head on the naive row
        for t in first_tc.iter(q("w:t")):
            if t.text and "TRACQ" in t.text:
                t.text = "Ablation"
        # the next row restarts the TRACQ merge and carries the label
        nxt = naive_row.getnext()
        tc2 = nxt.find(q("w:tc"))
        tcpr = tc2.find(q("w:tcPr"))
        vm = tcpr.find(q("w:vMerge"))
        vm.set(q("w:val"), "restart")
        # clone run formatting from the naive row's category run
        src_run = first_tc.find(q("w:p") + "/" + q("w:r"))
        para2 = tc2.find(q("w:p"))
        import copy
        if src_run is not None:
            run = copy.deepcopy(src_run)
            for t in run.iter(q("w:t")):
                t.text = "TRACQ"
        else:
            run = para2.makeelement(q("w:r"), {})
            t = run.makeelement(q("w:t"), {})
            t.text = "TRACQ"
            run.append(t)
        para2.append(run)
        return True
    return False


def frame_block(ed, prefix):
    cap = next(p for p in ed.paras
               if ed.para_text(p).strip().startswith(prefix))
    blk = [cap]
    cur = cap
    while True:
        sib = cur.getprevious()
        if sib is None or sib.tag != q("w:p") or \
                sib.find(q("w:pPr") + "/" + q("w:framePr")) is None or \
                ed.para_text(sib).strip().startswith("Fig."):
            break
        blk.insert(0, sib)
        cur = sib
    return blk


def move_after(ed, paras, anchor_text):
    target = next(p for p in ed.paras if anchor_text in ed.para_text(p))
    for p in reversed(paras):
        p.getparent().remove(p)
        target.addnext(p)


def reflow(ed):
    """Re-anchor Figs. 14 and 2 for the post-deletion flow."""
    move_after(ed, frame_block(ed, "Fig. 14. Compressed-domain"),
               "Rate-distortion claims are easiest to judge at matched "
               "operating")
    blk2 = frame_block(ed, "Fig. 2. Mean relative error")
    for p in blk2:
        p.find(q("w:pPr") + "/" + q("w:framePr")).set(q("w:yAlign"), "top")
    move_after(ed, blk2, "Fig. 4 plots the rate-distortion trade-off")


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    for old, new in EDITS:
        done = False
        for v in variants(old):
            try:
                ed.replace(v, new)
                done = True
                break
            except Exception:
                continue
    ok = fix_table_iv(ed)
    reflow(ed)
    ed.save()
    print(f"text edits attempted: {len(EDITS)}; Table IV category split: {ok}; "
          f"Figs. 14 and 2 re-anchored")


if __name__ == "__main__":
    main(sys.argv[1])

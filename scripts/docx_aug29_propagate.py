# -*- coding: utf-8 -*-
"""August 29: carry the time-series framing through the body of the paper.

The abstract and contribution list stay untouched. The frame propagates in
plain language: the intro names the purpose-built positioning and the
chronological-order property, related work states both halves of the
differentiator where each family is discussed, the problem formulation pins
down that columns are time steps in original order, the discussion and
conclusion echo the three operational requirements, and the follow-up sentence
after the bullets points at the right mechanisms after the list reorder.

Formatting repairs in Section V: the MetroPT-3 anomaly table gets its own
Table VI caption (it carried a copy of Table V's Appliances caption), the
"(0.148 vs. 0.180, a )" fragment and the missing period are fixed, and the two
Node-Parallel Scaling body paragraphs styled as figure captions return to
body style.

Usage: python docx_aug29_propagate.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q

# (seek_anchor_or_None, old, new) applied in document order
EDITS = [
    (None,
     "Error-Bounded Compression, Visualization, IoT.",
     "Error-Bounded Compression, Compressed-Domain Analytics, Visualization, "
     "IoT."),
    (None,
     "We focus on a complementary regime: lossy compression that maintains "
     "high numeric fidelity while producing compressed outputs that can be "
     "inspected directly.",
     "We focus on a complementary regime: lossy compression purpose-built "
     "for time series, keeping high numeric fidelity while producing "
     "compressed outputs that can be inspected directly. Throughout, the "
     "stored grid keeps samples in chronological order, and that single "
     "property is what lets the same file be viewed as an image and screened "
     "by detectors without reconstruction."),
    (None,
     "The last three mechanisms constitute our proposed Enhanced algorithm, "
     "which builds on the base idea.",
     "The scale-aware quantization and drift-free differencing above "
     "constitute our proposed Enhanced algorithm, which builds on the base "
     "idea."),
    (None,
     "and enable fast similarity search and indexing.",
     "and enable fast similarity search and indexing. Their outputs keep "
     "their time order, but the averaging is one-way: the original signal "
     "cannot be recovered from them within any error bound."),
    (None,
     "These compressors produce compact binary formats that require "
     "specialized decoders and inspection tools.",
     "These compressors produce compact binary formats that require "
     "specialized decoders and inspection tools, and their outputs reveal no "
     "time structure until they are decompressed back to floating point."),
    (None,
     "The goal is to reduce storage and transmission cost while keeping "
     "reconstruction error small and maintaining an interpretable grid "
     "structure.",
     "The goal is to reduce storage and transmission cost while keeping "
     "reconstruction error small and maintaining an interpretable grid "
     "structure. Each row of the grid is one variable and each column is one "
     "time step, in the original order; nothing in the encoding rearranges "
     "time."),
    # the MetroPT-3 table carried a copy of the Appliances caption; the seek
    # jumps past the legitimate Table V so the duplicate is the one renamed
    ("To evaluate whether compressed-domain analytics transfer",
     "TABLE V. Anomaly Detection on Appliances Energy.",
     "TABLE VI. Anomaly Detection on Real MetroPT-3 Compressor Failures."),
    (None,
     "(0.148 vs. 0.180, a )",
     "(0.148 vs. 0.180)"),
    (None,
     "flagged by the first-stage filter",
     "flagged by the first-stage filter."),
    # ends before "pipelines" so the deletion span stays clear of the
    # field-based hyperlink that follows ("(Section V-L)")
    (None,
     "flag anomalous windows at 21× the throughput of decode-then-detect",
     "flag anomalous windows at 21× to 27× the throughput of "
     "decode-then-detect"),
    (None,
     "producing artifacts that are 1.2 ~ 6.9x smaller at matched accuracy "
     "and up to 22x smaller on a million-step data stream, reaching "
     "compression ratios that alternatives fail to converge on while "
     "maintain the same degree of accuracy.",
     "producing artifacts with median sizes 1.2–1.9× smaller than SZ3 and "
     "1.5–6.9× smaller than ZFP at matched accuracy, up to 22× smaller on a "
     "million-step data stream, and reaching compressed sizes that the "
     "alternatives' accuracy-targeted modes do not reach."),
    (None,
     "Our compressed-domain anomaly detection experiment further shows that "
     "the compressed test sets are not merely interpretable but "
     "machine-actionable: unsupervised anomaly detection operating directly "
     "on the resulting compressed grids achieves F1 = 0.75, matching the "
     "decode-then-detect pipeline while running at ",
     "Because the stored grids keep chronological order, they are not merely "
     "interpretable but machine-actionable: unsupervised anomaly detection "
     "running directly on the grids matches the decode-then-detect pipeline "
     "at F1 = 0.75 on injected anomalies, retains most of its ranking "
     "quality on real compressor failures, and runs at "),
    (None,
     "can substantially improve time-series compression while retaining "
     "outputs that support both human inspection and enable direct machine "
     "analysis.",
     "can substantially improve time-series compression while retaining "
     "outputs that stay in time order and serve all three needs at once: "
     "storage efficiency, transmission scalability, and analysis directly on "
     "the stored form."),
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
            print(f"MISSING: {old[:70]}")
            sys.exit(1)
        n += 1

    # stray doubled multiplication sign in the conclusion, if one survives as
    # inline math next to the text sign
    ed.del_inline_math("not merely interpretable but machine-actionable",
                       count=99)

    # the two scaling body paragraphs styled as figure captions
    fixed = 0
    for p in ed.paras:
        t = ed.para_text(p)
        if t.startswith(("The terabyte run above uses all 112 cores",
                         "At 112 workers the fixed 50 GB input")):
            st = p.find(q("w:pPr") + "/" + q("w:pStyle"))
            if st is not None and st.get(q("w:val")) == "FigureCaptionFullWidth":
                st.set(q("w:val"), "BodyText")
                fixed += 1
    # Fig. 14's bottom frame collides with Fig. 13's bottom panel once the
    # acceptance fix restores the true flow; anchor its block one page later,
    # where it sat in the v7.2 layout
    def ptext(p):
        return ed.para_text(p).strip()
    cap = next(p for p in ed.paras
               if ptext(p).startswith("Fig. 14. Compressed-domain"))
    blk = [cap]
    for step in ("prev", "next"):
        cur = cap
        while True:
            sib = cur.getprevious() if step == "prev" else cur.getnext()
            if sib is None or sib.tag != q("w:p") or                     sib.find(q("w:pPr") + "/" + q("w:framePr")) is None or                     ptext(sib).startswith("Fig."):
                break
            blk.insert(0, sib) if step == "prev" else blk.append(sib)
            cur = sib
    target = next(p for p in ed.paras
                  if "This experiment shows that the compressed images"
                  in ptext(p))
    for p in reversed(blk):
        p.getparent().remove(p)
        target.addnext(p)
    print(f"Fig. 14 block relocated ({len(blk)} paras)")

    # safety net: a deleted range must never hold a live field instruction
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    converted = 0
    for d in ed.root.iter(ns + "del"):
        for it in d.iter(ns + "instrText"):
            it.tag = ns + "delInstrText"
            converted += 1
    if converted:
        print(f"instrText inside w:del converted to delInstrText: {converted}")

    ed.save()
    print(f"applied {n} text edits; caption-styled body paragraphs fixed: {fixed}")
    print("AUG-29 PROPAGATE DONE")


if __name__ == "__main__":
    main(sys.argv[1])

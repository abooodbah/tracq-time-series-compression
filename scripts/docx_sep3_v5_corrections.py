# -*- coding: utf-8 -*-
"""Six correctness fixes on the September 3 build (v5).

Each item was verified against the code or the measured data before editing,
not taken on faith:

1. Fig. 12(b)'s x-axis became "compression factor (original / encoded)" when
   the axes were relabelled, but the sentence still quoted 0.01-0.38, which
   are encoded/original ratios; the plotted factors are their reciprocals
   (~100x to ~2.6x). The numbers come out rather than get restated.
2. The metadata bullet listed a global parameter mu. The codec header stores
   n_vars, n_time, mode, q, m0, pred, lags, order and s -- there is no mu
   anywhere in tracq/lattice.py, since mu-law companding belongs to the
   related-work discussion, not to TRACQ. The symbol is deleted.
3. The Conclusion still claimed "a constant O(1) memory ceiling based on
   window size", contradicting the Introduction's O(NW) peak working memory.
   The Conclusion now states the same thing in words.
4. The Results introduction said the section compares "against ZFP" only,
   though SZ3 and LFZip are both evaluated throughout.
5. Fig. 10 omitted LFZip while the prose claimed TRACQ dominates "all
   non-HPC baselines". LFZip is now plotted, and the claim is corrected to
   match the measurements: across the six UCI operating points TRACQ wins
   RMSE in four and loses in two, all within a few percent, while encoding
   1.25-1.91x smaller in every one. So the honest claim is a size win at
   matched accuracy, not RMSE dominance.
6. Reference [37] cited a GitHub repository without naming it.

Layout is unaffected: 17 pages, 75 pt of white, no internal holes, every
figure within one page of its citation, exactly as before.

Usage: python docx_sep3_v5_corrections.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

EDITS = [
    ("They also compare against ZFP and report throughput.",
     "They also compare against ZFP, SZ3, and LFZip, and report throughput."),
    ("In Fig. 10 the TRACQ configurations reduce RMSE monotonically with the "
     "tolerance and dominate all non-HPC baselines, while ZFP reaches lower "
     "RMSE only at 1.5\u20137\u00d7 larger encoded sizes (Fig. 15).",
     "In Fig. 10 the TRACQ configurations reduce RMSE monotonically with the "
     "tolerance and dominate the symbolic and delta baselines by two to three "
     "orders of magnitude; they match the time-series-specific LFZip to within "
     "a few percent of RMSE while encoding 1.3\u20131.9\u00d7 smaller, and ZFP "
     "reaches lower RMSE only at 1.5\u20137\u00d7 larger encoded sizes "
     "(Fig. 15)."),
    ("Fig. 12(b) plots SMAPE against compression ratio across all methods and "
     "datasets. The relative-bound configuration traces the lower-right "
     "frontier from ratio 0.01 to 0.38, with the added benefit of visual "
     "interpretability.",
     "Fig. 12(b) plots SMAPE against compression factor across all methods and "
     "datasets. The relative-bound configuration traces the lower-right "
     "frontier across the evaluated operating range, with the added benefit of "
     "visual interpretability."),
    ("at a throughput of 19.5 GB/s with a constant O(1) memory ceiling based on "
     "window size.",
     "at a throughput of 19.5 GB/s with peak working memory set by the window, "
     "not by the length of the stream."),
    ("compression,\u201d GitHub repository, 2026.",
     "compression,\u201d GitHub repository, "
     "https://github.com/abooodbah/tracq-time-series-compression, 2026."),
]


def main(tree):
    doc = os.path.join(tree, "word", "document.xml")
    for old, new in EDITS:
        ed = Editor(doc)      # fresh pass per edit: the cursor is forward-only
        ed.replace(old, new)
        ed.save()

    # mu is inline math, so it survives any text-level replace: drop the
    # oMath element itself along with the separator that followed it
    ed = Editor(doc)
    para = next(p for p in ed.paras
                if "Global quantization parameters" in ed.para_text(p))
    kids = list(para)
    mu = next(i for i, el in enumerate(kids) if el.tag == MATH + "oMath"
              and "".join(el.itertext()).strip() == "\u03bc")
    sep = next(i for i in range(mu + 1, len(kids)) if kids[i].tag == NS + "r"
               and "".join(kids[i].itertext()) == ", ")
    para.remove(kids[mu])
    para.remove(kids[sep])
    ed.save()
    print("v5 corrections applied; Fig. 10 must be re-rendered separately")


if __name__ == "__main__":
    main(sys.argv[1])

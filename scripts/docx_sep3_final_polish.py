# -*- coding: utf-8 -*-
"""Final polish on the author's 17-page build.

Three fixes, each verified page-neutral (the 17-page fit has under one line
of slack, so every change was bisected against the page count):

- The plain "References:" body line becomes an IEEE-style unnumbered
  heading: Heading1 with numbering suppressed (numId 0), keepNext disabled
  and spacing zeroed - the style's keepNext binds the heading to entry [1]
  across the column break and costs a page otherwise.
- The conclusion's damaged closing phrase "analysis directly on the form"
  is repaired as "analysis on the stored form"; restoring the original
  "directly on the stored form" wraps one more line and spills the last
  reference onto an 18th page, so the shorter equivalent is used.
- Fig. 14(a) is regenerated to mirror Table V: the same four pipelines in
  the same order, with the Numerical RF row (F1 0.237, precision 0.253,
  recall 0.222) reproduced by rerunning scripts/anomaly_detection_experiment.py
  (the rerun also reconfirms the table's rounded 0.24/0.25/0.22). The
  legend moves to a horizontal row in the headroom so no bar label is
  covered. The rendered image14.png replaces word/media/image15.png
  (this lineage stores Fig. 14 there).

Usage: python docx_sep3_final_polish.py <tree_root>
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))

    refs = next(p for p in ed.paras
                if ed.para_text(p).strip() == "References:")
    ppr = refs.find(NS + "pPr")
    st = ppr.find(NS + "pStyle")
    st.set(NS + "val", "Heading1")
    kn = ppr.makeelement(NS + "keepNext", {NS + "val": "0"})
    st.addnext(kn)
    numpr = ppr.makeelement(NS + "numPr", {})
    numpr.append(numpr.makeelement(NS + "ilvl", {NS + "val": "0"}))
    numpr.append(numpr.makeelement(NS + "numId", {NS + "val": "0"}))
    kn.addnext(numpr)
    sp = ppr.makeelement(NS + "spacing", {NS + "before": "0", NS + "after": "0"})
    numpr.addnext(sp)
    ind = ppr.makeelement(NS + "ind", {NS + "left": "0", NS + "firstLine": "0"})
    sp.addnext(ind)
    for t in refs.iter(NS + "t"):
        if t.text and "References" in t.text:
            t.text = "References"

    ed.replace("transmission scalability, and analysis directly on the form.",
               "transmission scalability, and analysis on the stored form.")
    ed.save()

    fig = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper_results", "lattice", "figs_v2", "image14.png")
    shutil.copy(fig, os.path.join(tree, "word", "media", "image15.png"))
    print("references heading, stored-form repair, Fig. 14 swap done")


if __name__ == "__main__":
    main(sys.argv[1])

# -*- coding: utf-8 -*-
"""Wave 3: Section III restructured around the definitive algorithm.

Target structure: A. Problem Formulation and Design Motivation (the naive
formulation condensed to its equation pair and one safeguard sentence) ->
B. Adaptive Arcsinh Transform -> C. Error-Bounded Lattice Quantization (step
derivation and quantization merged) -> D. Drift-Free Integer Differencing
(with the integer accumulation identity stated inline) -> E. Predictors,
Metadata, and Packaging. The three-effects list and the baseline-offsetting
equation block are tracked-deleted; the paper shrinks rather than shuffles.

Usage: python docx_sep3_wave3.py <tree_root>
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q

AUTHOR = "Abdulfatah Bahbouh"
DATE = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [9700]

RENAMES = [
    ("Problem Formulation", "Problem Formulation and Design Motivation"),
    ("Per-Variable Adaptive Quantization Steps",
     "Error-Bounded Lattice Quantization"),
    ("Arcsinh Transform Domain", "Adaptive Arcsinh Transform"),
    ("Drift-Free Integer Reconstruction", "Drift-Free Integer Differencing"),
    ("Metadata and Reconstruction", "Predictors, Metadata, and Packaging"),
]

TRIMS = [
    ("prevents division by zero. This transformation has three useful "
     "effects:",
     "prevents division by zero."),
    ("well defined at and across zero. Let",
     "well defined at and across zero."),
    ("reproduces every quantized coordinate identically, so",
     "reproduces every quantized coordinate identically "
     "(mi,t = mi,0 + ki,1 + ⋯ + ki,t), so"),
]

# subsection headings whose content folds into the surviving section
DEL_HEADINGS = ["Percentage Change Computation", "Quantization",
                "Encoding Variants"]


def del_paragraph_tracked(p):
    """Mark a whole paragraph deleted: content wrapped in w:del, mark del'd."""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    _id[0] += 1
    d = p.makeelement(ns + "del", {ns + "id": str(_id[0]), ns + "author": AUTHOR,
                                   ns + "date": DATE})
    kids = [c for c in p if c.tag != ns + "pPr"]
    if kids:
        kids[0].addprevious(d)
        for c in kids:
            d.append(c)
        for t in d.iter(ns + "t"):
            t.tag = ns + "delText"
        for it in d.iter(ns + "instrText"):
            it.tag = ns + "delInstrText"
    ppr = p.find(ns + "pPr")
    if ppr is None:
        ppr = p.makeelement(ns + "pPr", {})
        p.insert(0, ppr)
    rpr = ppr.find(ns + "rPr")
    if rpr is None:
        rpr = ppr.makeelement(ns + "rPr", {})
        ppr.append(rpr)
    _id[0] += 1
    rpr.insert(0, rpr.makeelement(ns + "del", {ns + "id": str(_id[0]),
                                               ns + "author": AUTHOR,
                                               ns + "date": DATE}))


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))

    def heads():
        out = {}
        for p in ed.paras:
            ppr = p.find(q("w:pPr"))
            st = (ppr.find(q("w:pStyle")).get(q("w:val"))
                  if ppr is not None and ppr.find(q("w:pStyle")) is not None
                  else "")
            if st.startswith("Heading"):
                out.setdefault(ed.para_text(p).strip(), p)
        return out

    # locate the condensation targets by content BEFORE any mutation
    idx = {i: p for i, p in enumerate(ed.paras)}
    kill = []
    hd = heads()
    # three-effects list: the ListParagraph1 items after the P-equation plus
    # the closing sentence
    started = False
    for i, p in enumerate(ed.paras):
        t = ed.para_text(p).strip()
        if t.startswith("It normalizes across heterogeneous variables"):
            started = True
        if started:
            kill.append(p)
            if t.startswith("These properties make percentage changes"):
                break
    # offsetting equation block: from the bare equation after the safeguard
    # sentence through the shifted-channel sentence
    seen_safeguard = False
    for i, p in enumerate(ed.paras):
        t = ed.para_text(p).strip()
        if "automated baseline offsetting safeguard" in t:
            seen_safeguard = True
            continue
        if seen_safeguard:
            kill.append(p)
            if t.startswith("We encode the shifted channel"):
                break

    for name in DEL_HEADINGS:
        kill.append(hd[name])

    # 1) renames and trims (tracked replaces)
    for old, new in RENAMES + TRIMS:
        ed.replace(old, new)

    # 2) tracked deletions
    for p in kill:
        del_paragraph_tracked(p)

    # 3) move the arcsinh block before the (renamed) lattice section
    arc_head = heads()["Adaptive Arcsinh Transform"]
    block = [arc_head]
    cur = arc_head
    while True:
        nxt = cur.getnext()
        if nxt is None or nxt.tag != q("w:p"):
            break
        ppr = nxt.find(q("w:pPr"))
        st = (ppr.find(q("w:pStyle")).get(q("w:val"))
              if ppr is not None and ppr.find(q("w:pStyle")) is not None
              else "")
        if st.startswith("Heading"):
            break
        block.append(nxt)
        cur = nxt
    target = heads()["Error-Bounded Lattice Quantization"]
    for p in block:
        p.getparent().remove(p)
    for p in block:
        target.addprevious(p)
    ed.save()
    print(f"renames+trims done; {len(kill)} paragraphs tracked-deleted; "
          f"arcsinh block ({len(block)} paras) moved")


if __name__ == "__main__":
    main(sys.argv[1])

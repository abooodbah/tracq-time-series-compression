# -*- coding: utf-8 -*-
"""Layout-review fixes on the September 3 build.

An independent layout review (docx_layout_audit.py plus a page-by-page
visual pass) found three defects after the figure-coherence reflow: Fig. 16's
caption was orphaned on the page after its panels, Table III split across a
page break, and a two-inch void sat under Table IV. The fixes, applied in
document order:

- Every in-column numbered table (I, II, III, V, VI) gets a keepNext chain:
  caption plus every table paragraph except the last row's, so no table can
  split across a column or page again. Table IV is exempt (full-width span).
- Table III moves after its discussion paragraph, so the throughput page
  fills and the table lands whole at the top of the next page.
- Fig. 16's display extent shrinks to 0.88 and its image binds to its
  caption, which reunites the caption with the panels beside Fig. 14's
  bottom frame.
- Figs. 9 and 10 re-anchor after the post-Table IV paragraph, pulling the
  full-length MetroPT-3 text up to fill the void under the table; Fig. 11
  then re-anchors after the Metro Traffic bullet so its frame stays one
  page later instead of colliding with the Fig. 9/10 stack.
- Fig. 13's panel (a) re-anchors after the Section V-L intro so the
  operational-settings paragraph and the L opening fill its old page's
  right column; the panel still lands at the bottom of the experimental
  setup page, one column turn after its discussion.

Usage: python docx_sep3_review_fixes.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q


def add_keepnext(p):
    ppr = p.find(q("w:pPr"))
    if ppr is None:
        ppr = p.makeelement(q("w:pPr"), {})
        p.insert(0, ppr)
    if ppr.find(q("w:keepNext")) is None:
        kn = ppr.makeelement(q("w:keepNext"), {})
        st = ppr.find(q("w:pStyle"))
        st.addnext(kn) if st is not None else ppr.insert(0, kn)


def frame_block(ed, prefix):
    cap = next(p for p in ed.paras
               if ed.para_text(p).strip().startswith(prefix))
    blk = [cap]
    prev = cap.getprevious()
    while prev is not None and prev.tag == q("w:p") and \
            prev.find(q("w:pPr") + "/" + q("w:framePr")) is not None and \
            not ed.para_text(prev).strip().startswith("Fig."):
        blk.insert(0, prev)
        prev = prev.getprevious()
    return blk


def move_after(ed, paras, anchor_text):
    target = next(p for p in ed.paras if anchor_text in ed.para_text(p))
    assert target.find(q("w:pPr") + "/" + q("w:sectPr")) is None
    for p in paras:
        p.getparent().remove(p)
    for p in reversed(paras):
        target.addnext(p)


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))

    chained = 0
    for cap in ed.paras:
        t = ed.para_text(cap).strip()
        if not (t.startswith("TABLE ") and t[6] in "IV"):
            continue
        tbl = cap.getnext()
        if tbl is None or tbl.tag != q("w:tbl"):
            continue
        if "Naive Diff. (16b)" in "".join(tbl.itertext()):
            continue
        rows = list(tbl.iter(q("w:tr")))
        add_keepnext(cap)
        for r in rows[:-1]:
            for p in r.iter(q("w:p")):
                add_keepnext(p)
        chained += 1

    cap3 = next(p for p in ed.paras if ed.para_text(p).strip()
                .startswith("TABLE III. 8-bit Compression"))
    tbl3 = cap3.getnext()
    target = next(p for p in ed.paras
                  if "Moving from the naive ablation to TRACQ at the default"
                  in ed.para_text(p))
    for el in (cap3, tbl3):
        el.getparent().remove(el)
    target.addnext(tbl3)
    target.addnext(cap3)

    cap16 = next(p for p in ed.paras
                 if ed.para_text(p).strip().startswith("Fig. 16. Node-parallel"))
    img16 = cap16.getprevious()
    dr = img16.find(".//" + q("w:drawing"))
    ns_wp = ("{http://schemas.openxmlformats.org/drawingml/2006/"
             "wordprocessingDrawing}")
    ns_a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    for ext in list(dr.iter(ns_wp + "extent")) + list(dr.iter(ns_a + "ext")):
        if ext.get("cx") is None or ext.get("cy") is None:
            continue  # extension-list a:ext carries no size
        for att in ("cx", "cy"):
            ext.set(att, str(int(int(ext.get(att)) * 0.88)))
    add_keepnext(img16)

    blk9 = frame_block(ed, "Fig. 9. Rate-distortion curves")
    blk10 = frame_block(ed, "Fig. 10. RMSE by method")
    for blk in (blk10, blk9):
        move_after(ed, blk,
                   "To complement the 5,000-step UCI comparisons above")
    move_after(ed, frame_block(ed, "Fig. 11. Full-length MetroPT-3"),
               "Metro Traffic shows the same pattern")

    show = next(p for p in ed.paras
                if "Fig. 13 shows the resulting heatmaps" in ed.para_text(p))
    raw = show.getnext()
    assert raw.find(q("w:pPr") + "/" + q("w:framePr")) is not None
    move_after(ed, [raw],
               "Here we test whether they can be detected computationally")

    ed.save()
    print(f"tables chained: {chained}; Table III relocated; Fig. 16 scaled "
          f"and bound; Figs. 9-11 and 13(a) re-anchored")


if __name__ == "__main__":
    main(sys.argv[1])

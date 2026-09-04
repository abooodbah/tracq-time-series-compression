# -*- coding: utf-8 -*-
"""Close the residual white gaps in the September 3 LFZip build.

The starting layout had three gaps: ~99 pt on p6-L and ~98 pt on p7-L, each a
white band above a bottom-anchored full-width figure, plus ~77 pt trailing on
p13-R. All three are keep-block quantization: an in-column figure block or a
keepNext-chained table could not fit in the space left at the foot of a
column, jumped, and left the remainder empty.

Search result over eleven measured candidates (each exported through Word and
measured on the rendered PDF): moving a figure EARLIER always made things
worse, because the block then displaced the text that had been filling the
column. Moving it later, so prose fills the column first, is what works.
Deleting Fig. 10 was also tested and rejected: the last page is full, so
removing ~235 pt cannot save a page and only relocates the white.

The four surviving moves:

- Fig. 1's image is anchored after the second ablation bullet, so the bullets
  fill p6's left column before the figure floats to the right column.
- Fig. 3's frame becomes top-anchored. A bottom-anchored full-width frame
  forces every column above it to end early whenever the next block does not
  fit; anchoring it to the top of the page removes that trap entirely, and
  the figure still lands directly after the citation that ends the previous
  page.
- Figs. 4 and 5 move together (order preserved) to after the first
  "Three results stand out" bullet, and that lead-in gets keepNext so it can
  never be left dangling above a column break with its colon and no list.
- Fig. 14's frame becomes top-anchored for the same reason as Fig. 3.

An adversarial review of the first attempt caught a defect no measurement of
white space would find: with the figures moved, Table IV's full-width caption
came to rest against the column text above it (0.1 pt clearance), reading as
though the caption belonged to that paragraph. The caption therefore carries
an explicit 8 pt space-before. docx_layout_audit.py now checks that clearance.

Result: 125 pt of white against 274 pt before, no internal holes at all, no
caption collisions, no split tables, figure order and adjacency unchanged.
The one remaining gap, 75 pt at the foot of p7's left column, is the floor
for that page: Fig. 4's image and caption form a 181 pt block that cannot fit
there, and every anchor further down the page pushes it onto p8, which
measured far worse (a 234 pt hole and a frame collision).

Usage: python docx_sep3_gapfill_layout.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q

NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def image_and_caption(ed, caption_prefix):
    cap = next(p for p in ed.paras
               if ed.para_text(p).strip().startswith(caption_prefix))
    img = cap.getprevious()
    assert img.find(".//" + NS + "drawing") is not None, caption_prefix
    return img, cap


def frame_block(ed, caption_prefix):
    """A framed figure's paragraphs: the caption plus its silent neighbours."""
    cap = next(p for p in ed.paras
               if ed.para_text(p).strip().startswith(caption_prefix))
    blk = [cap]
    for step in ("prev", "next"):
        cur = cap
        while True:
            sib = cur.getprevious() if step == "prev" else cur.getnext()
            if (sib is None or sib.tag != NS + "p"
                    or sib.find(NS + "pPr/" + NS + "framePr") is None
                    or ed.para_text(sib).strip().startswith("Fig.")):
                break
            blk.insert(0, sib) if step == "prev" else blk.append(sib)
            cur = sib
    return blk


def move_after(ed, paras, anchor_text):
    target = next(p for p in ed.paras if anchor_text in ed.para_text(p))
    assert target.find(NS + "pPr/" + NS + "sectPr") is None
    for p in paras:
        p.getparent().remove(p)
    for p in reversed(paras):
        target.addnext(p)


def set_prop(para, tag, attrs=None):
    ppr = para.find(NS + "pPr")
    if ppr is None:
        ppr = para.makeelement(NS + "pPr", {})
        para.insert(0, ppr)
    el = ppr.find(NS + tag)
    if el is None:
        el = ppr.makeelement(NS + tag, {})
        style = ppr.find(NS + "pStyle")
        style.addnext(el) if style is not None else ppr.insert(0, el)
    for k, v in (attrs or {}).items():
        el.set(NS + k, v)
    return el


def top_anchor(ed, caption_prefix):
    for p in frame_block(ed, caption_prefix):
        p.find(NS + "pPr/" + NS + "framePr").set(NS + "yAlign", "top")


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))

    move_after(ed, list(image_and_caption(ed, "Fig. 1. Ablation study")),
               "The relative-bound transform trades a small amount of RMSE")
    top_anchor(ed, "Fig. 3. Cumulative RMSE")
    top_anchor(ed, "Fig. 14. Compressed-domain")

    pairs = [image_and_caption(ed, "Fig. 4. Rate-distortion comparison"),
             image_and_caption(ed, "Fig. 5. Rate-distortion comparison with ZFP")]
    flat = [p for pair in pairs for p in pair]
    move_after(ed, flat,
               "At matched sizes below roughly one fifth of the original")

    lead_in = next(p for p in ed.paras
                   if ed.para_text(p).strip() == "Three results stand out:")
    set_prop(lead_in, "keepNext")

    caption = next(p for p in ed.paras
                   if ed.para_text(p).strip().startswith("TABLE IV. Real-World"))
    set_prop(caption, "spacing", {"before": "160"})

    # Table V follows the throughput paragraph rather than preceding it: its
    # keepNext chain is ~140 pt and could not fit at the foot of p13, which
    # left the whole column short. Both paragraphs discuss the table, so the
    # reading order survives the move.
    tbl_cap = next(p for p in ed.paras if ed.para_text(p).strip()
                   .startswith("TABLE V. Anomaly Detection on Appliances"))
    tbl = tbl_cap.getnext()
    assert tbl.tag == NS + "tbl"
    after = next(p for p in ed.paras
                 if "The critical systems-level result is throughput"
                 in ed.para_text(p))
    for el in (tbl_cap, tbl):
        el.getparent().remove(el)
    after.addnext(tbl)
    after.addnext(tbl_cap)

    ed.save()
    print("Figs. 1/4/5 re-anchored, Figs. 3/14 top-anchored, lead-in kept, "
          "Table IV caption cleared")


if __name__ == "__main__":
    main(sys.argv[1])

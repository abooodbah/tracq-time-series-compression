# -*- coding: utf-8 -*-
"""Fig. 13 consolidated-stack variant of the author's gap-free layout.

Built from the author's own edited clean file (his second edit round, which
already achieved zero white gaps but left Fig. 15 on the page before
Fig. 14 and Fig. 13's caption two pages after its discussion). Two moves:

- Fig. 13's panel (a) and the (b)/(c)+caption block merge into one
  full-width top-of-page stack (all frame paragraphs yAlign=top,
  contiguous), anchored after pipeline list item 3 - the last paragraph
  that renders on the discussion page - so the whole figure tops the very
  next page after the Section V-K discussion.
- Fig. 14's frame block re-anchors after its citing paragraph ("Fig. 14
  visualizes both the detection quality..."), landing it at the bottom of
  its citation page and restoring caption number order 1-16 (the flow it
  displaces travels to the next page together with Figs. 15/16).

Usage: python docx_sep3_fig13_stack_variant.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q


def frame_block(ed, prefix, both=False):
    cap = next(p for p in ed.paras
               if ed.para_text(p).strip().startswith(prefix))
    blk = [cap]
    prev = cap.getprevious()
    while prev is not None and prev.tag == q("w:p") and \
            prev.find(q("w:pPr") + "/" + q("w:framePr")) is not None and \
            not ed.para_text(prev).strip().startswith("Fig."):
        blk.insert(0, prev)
        prev = prev.getprevious()
    if both:
        nxt = cap.getnext()
        while nxt is not None and nxt.tag == q("w:p") and \
                nxt.find(q("w:pPr") + "/" + q("w:framePr")) is not None and \
                not ed.para_text(nxt).strip().startswith("Fig."):
            blk.append(nxt)
            nxt = nxt.getnext()
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

    # panel (a): the bottom-frame image pair wherever the author left it
    # (between the "Fig. 13 shows" paragraph and "Fig. 14 visualizes")
    i_show = next(i for i, p in enumerate(ed.paras)
                  if "Fig. 13 shows the resulting heatmaps" in ed.para_text(p))
    i_f14 = next(i for i, p in enumerate(ed.paras)
                 if "Fig. 14 visualizes" in ed.para_text(p))
    pa = []
    for p in ed.paras[i_show:i_f14]:
        fr = p.find(q("w:pPr") + "/" + q("w:framePr"))
        if fr is not None and fr.get(q("w:yAlign")) == "bottom":
            pa.append(p)
    assert pa and any(p.find(".//" + q("w:drawing")) is not None for p in pa)

    stack = pa + frame_block(ed, "Fig. 13. Visual inspection")
    for p in stack:
        p.find(q("w:pPr") + "/" + q("w:framePr")).set(q("w:yAlign"), "top")
    move_after(ed, stack,
               "integer-accumulated trajectory shape), and run the same")

    move_after(ed, frame_block(ed, "Fig. 14. Compressed-domain", both=True),
               "Fig. 14 visualizes both the detection quality")

    ed.save()
    print("Fig. 13 stacked after its discussion; Fig. 14 on its citing page")


if __name__ == "__main__":
    main(sys.argv[1])

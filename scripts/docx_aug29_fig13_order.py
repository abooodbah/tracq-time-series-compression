# -*- coding: utf-8 -*-
"""Fig. 13 and Fig. 14 placed with their own text, in reading order.

Panel (a) sits at the bottom of the page whose text introduces it ("panel (a)
only sets the reference"), the grids (b)/(c) with the caption crown the next
page, and Fig. 14 takes the bottom of that same page, directly after its own
discussion. Reading order is (a), then (b)/(c) plus caption, then Fig. 14;
figure numbers stay in page order; every anchor has abundant text before and
after it, so no column is left stranded.

Usage: python docx_aug29_fig13_order.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q


def frame_block(ed, prefix):
    cap = next(p for p in ed.paras
               if ed.para_text(p).strip().startswith(prefix))
    blk = [cap]
    for step in ("prev", "next"):
        cur = cap
        while True:
            sib = cur.getprevious() if step == "prev" else cur.getnext()
            if sib is None or sib.tag != q("w:p") or \
                    sib.find(q("w:pPr") + "/" + q("w:framePr")) is None or \
                    ed.para_text(sib).strip().startswith("Fig."):
                break
            blk.insert(0, sib) if step == "prev" else blk.append(sib)
            cur = sib
    return blk


def move_after(ed, paras, anchor_text):
    target = next(p for p in ed.paras if anchor_text in ed.para_text(p))
    for p in reversed(paras):
        p.getparent().remove(p)
        target.addnext(p)
    return target


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    blk13 = frame_block(ed, "Fig. 13. Visual inspection")
    raw, grids, caption = blk13

    # panel (a): bottom of the page that introduces it
    raw.find(q("w:pPr") + "/" + q("w:framePr")).set(q("w:yAlign"), "bottom")
    move_after(ed, [raw], "Fig. 13 shows the resulting heatmaps side by side")

    # grids + caption: top of the following page
    for p in (grids, caption):
        p.find(q("w:pPr") + "/" + q("w:framePr")).set(q("w:yAlign"), "top")
    move_after(ed, [grids, caption],
               "This experiment shows that the compressed images")

    # Fig. 14: bottom-anchored in the next section, so the section's text
    # fills the grid page's remaining columns and the frame settles one page
    # after its citation
    blk14 = frame_block(ed, "Fig. 14. Compressed-domain")
    for p in blk14:
        p.find(q("w:pPr") + "/" + q("w:framePr")).set(q("w:yAlign"), "bottom")
    move_after(ed, blk14, "Table VI compares direct compressed-domain screening")

    ed.save()
    print("(a) -> bottom of its discussion page; (b)/(c)+caption -> next page "
          "top; Fig. 14 -> same page bottom")


if __name__ == "__main__":
    main(sys.argv[1])

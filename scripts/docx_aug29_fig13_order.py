# -*- coding: utf-8 -*-
"""Fig. 13 as one clean full-page stack, panels in reading order.

The split arrangement (panel (a) on one anchor, grids on another) interleaved
unrelated Fig. 14 discussion text between the panels and beached the previous
page's right column at half height, because the anchors could not fit on the
page where the text ran out. The block becomes a single top-anchored stack,
(a) then (b)/(c) then the caption, anchored after the last of the anomaly
discussion paragraphs: the preceding text then fills both columns of its own
page, the stack occupies the top of the next page with a text band below it,
and Fig. 14's bottom-anchored frame settles on the page after.

Usage: python docx_aug29_fig13_order.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    cap = next(p for p in ed.paras
               if ed.para_text(p).strip().startswith("Fig. 13. Visual inspection"))
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

    # one contiguous top stack: (a), (b)/(c), caption
    for p in blk:
        p.find(q("w:pPr") + "/" + q("w:framePr")).set(q("w:yAlign"), "top")

    # anchor after the last discussion paragraph so the text before it can
    # finish its own page; Fig. 14's block, anchored at the same spot, stays
    # after this one in document order
    target = next(p for p in ed.paras
                  if "This experiment shows that the compressed images"
                  in ed.para_text(p))
    for p in reversed(blk):
        p.getparent().remove(p)
        target.addnext(p)

    # Fig. 14 cannot share that page with the full stack: anchor its block in
    # the following section so its bottom frame settles on the next page
    cap14 = next(p for p in ed.paras
                 if ed.para_text(p).strip().startswith("Fig. 14. Compressed-domain"))
    blk14 = [cap14]
    for step in ("prev", "next"):
        cur = cap14
        while True:
            sib = cur.getprevious() if step == "prev" else cur.getnext()
            if sib is None or sib.tag != q("w:p") or                     sib.find(q("w:pPr") + "/" + q("w:framePr")) is None or                     ed.para_text(sib).strip().startswith("Fig."):
                break
            blk14.insert(0, sib) if step == "prev" else blk14.append(sib)
            cur = sib
    target14 = next(p for p in ed.paras
                    if "To evaluate whether compressed-domain analytics"
                    in ed.para_text(p))
    for p in reversed(blk14):
        p.getparent().remove(p)
        target14.addnext(p)
    print(f"Fig. 14 block ({len(blk14)} paras) anchored in the next section")
    ed.save()
    print("Fig. 13 stacked (a)->(b)/(c)->caption at a page top, "
          "anchored after the discussion text")


if __name__ == "__main__":
    main(sys.argv[1])

# -*- coding: utf-8 -*-
"""v7.2: fill the mostly-empty page 13 in the July-14 lineage.

Word left a 246pt hole between the V-K text and the bottom-anchored Fig. 13
raw panel: the following flow paragraphs sat after the figure anchors, so
nothing could occupy the band above the frame. The Fig. 13 block therefore
anchors deeper into V-L (just before the Table V caption), letting the
anomaly-detection prose fill the page above the panel. That reflow pulled
Fig. 14's top frame onto the Fig. 13 grids page where the two overlapped, so
Fig. 14's block becomes bottom-anchored and settles on the following page.

July-14 lineage only; the July-31 lineage paginates densely without either
change. Verified against Word's own pagination (COM export), using per-page
y-coverage (union of content intervals), not max-content-bottom, which a
bottom frame touching the margin can satisfy on a mostly-empty page.

Usage: python docx_v72_layout_fix.py <tree_root>
"""

import os
import sys

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(tag):
    return "{%s}%s" % (W, tag.split(":")[-1])


def ptext(p):
    return "".join(t.text or "" for t in p.iter(q("w:t"))).strip()


def frame_block(root, prefix):
    """The caption paragraph plus contiguous frame paragraphs on both sides."""
    cap = next(p for p in root.iter(q("w:p")) if ptext(p).startswith(prefix))
    blk = [cap]
    for step in ("prev", "next"):
        cur = cap
        while True:
            sib = cur.getprevious() if step == "prev" else cur.getnext()
            if sib is None or sib.tag != q("w:p") or \
                    sib.find(q("w:pPr") + "/" + q("w:framePr")) is None or \
                    ptext(sib).startswith("Fig."):
                break
            blk.insert(0, sib) if step == "prev" else blk.append(sib)
            cur = sib
    return blk


def main(tree):
    path = os.path.join(tree, "word", "document.xml")
    doc = etree.parse(path)
    root = doc.getroot()

    blk13 = frame_block(root, "Fig. 13. Visual inspection")
    target = next(p for p in root.iter(q("w:p"))
                  if "Table V summarizes detection performance" in ptext(p))
    for p in reversed(blk13):
        p.getparent().remove(p)
        target.addnext(p)
    print(f"Fig. 13 block ({len(blk13)} paras) anchored before the Table V caption")

    blk14 = frame_block(root, "Fig. 14. Compressed-domain")
    for p in blk14:
        p.find(q("w:pPr") + "/" + q("w:framePr")).set(q("w:yAlign"), "bottom")
    print(f"Fig. 14 block ({len(blk14)} paras) set to bottom anchor")

    doc.write(path, xml_declaration=True, encoding="UTF-8", standalone=True)
    print("V7.2 LAYOUT FIX DONE")


if __name__ == "__main__":
    main(sys.argv[1])

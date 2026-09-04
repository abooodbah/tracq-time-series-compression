# -*- coding: utf-8 -*-
"""Put Fig. 11 on the page that discusses it, without adding a page.

In the previous build Fig. 11 was captioned on p11 while its only citation sat
on p9, and p11 held nothing but UCI bullet discussion: the figure was not
merely two pages away, it was stranded among text about a different dataset.

The cause was paragraph order, not figure placement. The MetroPT-3 paragraph
that cites Fig. 11 sat immediately after Table IV, wedged between the table
and its own "The real-world results are consistent:" bullet list. Moving that
paragraph to after the UCI conclusion fixes the adjacency and reads better,
since Table IV's discussion is no longer interrupted by a different dataset.

Moving it costs a page on its own, because the relocated text pushes the tail
of the document past the last page by about 57 pt. Two cheap savings pay for
it: the Figs. 9/10 stack anchors one paragraph later, which fills the ~105 pt
that was stranded at the foot of p9's right column, and the "vol./no."
metadata comes out of the 14 reference entries that still carried it, which
also makes the reference list uniform (the author had already trimmed the
others by hand).

Verified after the change: 17 pages, 75 pt of white (unchanged, all of it the
known floor at the foot of p7's left column), no internal holes, no overlaps,
figure numbering monotonic, every figure and table within one page of its
introducing citation, and references [1]-[43] intact.

Usage: python docx_sep3_v4_layout.py <tree_root>
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def find(ed, needle):
    return next(p for p in ed.paras if needle in ed.para_text(p))


def frame_block(ed, caption_prefix):
    cap = next(p for p in ed.paras
               if ed.para_text(p).strip().startswith(caption_prefix))
    blk = [cap]
    prev = cap.getprevious()
    while (prev is not None and prev.tag == NS + "p"
           and prev.find(NS + "pPr/" + NS + "framePr") is not None
           and not ed.para_text(prev).strip().startswith("Fig.")):
        blk.insert(0, prev)
        prev = prev.getprevious()
    return blk


def move_after(paras, target):
    for p in paras:
        p.getparent().remove(p)
    for p in reversed(paras):
        target.addnext(p)


def main(tree):
    doc = os.path.join(tree, "word", "document.xml")

    ed = Editor(doc)
    para = find(ed, "To complement the 5,000-step UCI comparisons above")
    move_after([para],
               find(ed, "Overall, the real-world evaluation shows that TRACQ"))
    move_after(frame_block(ed, "Fig. 11. Full-length"), para)
    move_after([p for pref in ("Fig. 9. Rate-distortion curves",
                               "Fig. 10. RMSE by method")
                for p in frame_block(ed, pref)],
               find(ed, "The Appliances Energy dataset spans 28 variables"))
    ed.save()

    # uniform reference style: drop the volume/issue metadata that survived
    ed = Editor(doc)
    frags, seen = [], False
    for p in ed.paras:
        t = re.sub(r"\s+", " ", ed.para_text(p)).strip()
        if t == "References":
            seen = True
            continue
        if seen and t.startswith("["):
            m = re.search(r"vol\.\s*\d+(?:,\s*no\.\s*\d+)?,\s*", t)
            if m:
                frags.append(m.group(0))
    for frag in frags:
        e = Editor(doc)          # fresh pass: the editor cursor is forward-only
        e.replace(frag, "")
        e.save()
    print(f"Fig. 11 paragraph and frame relocated; Figs. 9/10 anchored later; "
          f"{len(frags)} reference entries trimmed")


if __name__ == "__main__":
    main(sys.argv[1])

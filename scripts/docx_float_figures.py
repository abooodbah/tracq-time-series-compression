# -*- coding: utf-8 -*-
"""Convert the full-width figure spans to page-anchored floating frames.

Each one-column figure span becomes a frame anchored to the top (or bottom)
of the page, and only the span's own single-column section marker is removed,
so the full-width table keeps its span and text runs in two uninterrupted
columns. Contiguous spans sharing identical frame properties merge into one
frame; spans that would otherwise collide on the same page are sent to the
bottom instead.

Usage: python docx_float_figures.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q

DRAW = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

# spans anchored to the page bottom instead of the top, keyed by the media
# file of their first drawing (resolved per tree via the rels table)
BOTTOM_IMAGES = {"fig3", "fig13raw"}


def frame_attrs(align):
    return {q("w:w"): "10450", q("w:hAnchor"): "margin", q("w:xAlign"): "center",
            q("w:vAnchor"): "margin", q("w:yAlign"): align, q("w:wrap"): "notBeside"}


def set_frame(p, align):
    ppr = p.find(q("w:pPr"))
    if ppr is None:
        ppr = p.makeelement(q("w:pPr"), {})
        p.insert(0, ppr)
    old = ppr.find(q("w:framePr"))
    if old is not None:
        ppr.remove(old)
    st = ppr.find(q("w:pStyle"))
    ppr.insert(list(ppr).index(st) + 1 if st is not None else 0,
               ppr.makeelement(q("w:framePr"), frame_attrs(align)))


def main(tree):
    import re
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    rels = open(os.path.join(tree, "word", "_rels", "document.xml.rels"),
                encoding="utf-8").read()
    id2img = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="media/(image\d+)\.png"', rels))

    def para_images(p):
        return [id2img.get(b.get(R + "embed"), "?") for b in p.iter(A + "blip")]

    def wide(p):
        for ext in p.iter(DRAW):
            if int(ext.get("cx", "0")) > 4100000:  # > ~4.5 in
                return True
        return False

    body = ed.root.find(q("w:body"))
    children = list(body)
    spans, cur = [], []
    for el in children:
        if el.tag == q("w:p") and el.find(q("w:pPr") + "/" + q("w:sectPr")) is not None:
            sp = el.find(q("w:pPr") + "/" + q("w:sectPr"))
            cols = sp.find(q("w:cols"))
            ncols = cols.get(q("w:num")) if cols is not None else None
            has_wide = any(k.tag == q("w:p") and wide(k) for k in cur)
            if ncols is None and has_wide:
                spans.append({"content": list(cur), "close": el})
            cur = []
        elif el.tag == q("w:p"):
            cur.append(el)
        elif el.tag in (q("w:bookmarkStart"), q("w:bookmarkEnd")):
            continue  # cross-reference anchors are transparent
        else:
            cur = []  # a table resets the run: never treat table spans as figures

    n_top, n_bottom = 0, 0
    for k, sp in enumerate(spans):
        figs = [p for p in sp["content"] if wide(p)]
        cap_text = " ".join(ed.para_text(p) for p in sp["content"])
        # Fig 3 goes to a page bottom so it never fights the span ahead of it
        default = "bottom" if "Cumulative RMSE over 10,000" in cap_text else "top"
        split13 = len(figs) >= 2 and "Visual inspection" in cap_text
        for p in sp["content"]:
            if split13 and p is figs[0]:
                set_frame(p, "bottom")  # Fig 13 raw panel to the page bottom
                n_bottom += 1
            else:
                set_frame(p, default)
                if wide(p):
                    if default == "top":
                        n_top += 1
                    else:
                        n_bottom += 1
        sp["close"].getparent().remove(sp["close"])

    # Fig 11 anchors on the page figs 9+10 already occupy; frames do not defer
    # to the next page, so move its block a few paragraphs later in the flow
    for sp in spans:
        cap_text = " ".join(ed.para_text(p) for p in sp["content"])
        if "Full-length MetroPT-3 rate-distortion" not in cap_text:
            continue
        target = None
        for p in ed.root.iter(q("w:p")):
            if "Air Quality, whose repeated near-zero" in ed.para_text(p):
                target = p
                break
        if target is not None:
            for blockp in reversed(sp["content"]):
                blockp.getparent().remove(blockp)
                target.addnext(blockp)
            print("Fig 11 block relocated after the Air Quality bullet")
        break
    print(f"framed {len(spans)} spans ({n_top} top figures, {n_bottom} bottom)")
    ed.save()
    print("FLOAT PASS DONE")


if __name__ == "__main__":
    main(sys.argv[1])

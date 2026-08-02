# -*- coding: utf-8 -*-
"""Font/whitespace pass: uniform table and caption sizes, no half-empty pages.

Every numbered table's cell text gets an explicit 8pt size and every Fig./TABLE
caption paragraph an explicit 9pt size, replacing the mix of explicit sizes and
style defaults. Figure frame blocks whose anchors produced page gaps or frame
collisions in Word's pagination are relocated later in the flow, keyed by their
caption text: Fig. 13 (both lineages) moves after the dashboard-screening
paragraph, and Figs. 7-8 (July-14 lineage only, --july14) move after the
paragraph following Table III so they anchor on the page after Fig. 6.

Verified against Word's own pagination (COM PDF export), not just LibreOffice:
the two renderers paginate frames differently and only Word's view matters.

Usage: python docx_font_gap_pass.py <tree_root> [--july14]
"""

import os
import sys

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(tag):
    return "{%s}%s" % (W, tag.split(":")[-1])


# rPr children that must precede w:sz in the schema order
PRE_SZ = [q(t) for t in
          ("rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline "
           "shadow emboss imprint noProof snapToGrid vanish webHidden color "
           "spacing w kern position").split()]


def set_size(r, half_points):
    rpr = r.find(q("w:rPr"))
    if rpr is None:
        rpr = r.makeelement(q("w:rPr"), {})
        r.insert(0, rpr)
    for tag in (q("w:sz"), q("w:szCs")):
        old = rpr.find(tag)
        if old is not None:
            rpr.remove(old)
    pos = 0
    for i, child in enumerate(rpr):
        if child.tag in PRE_SZ:
            pos = i + 1
    rpr.insert(pos, rpr.makeelement(q("w:sz"), {q("w:val"): str(half_points)}))
    rpr.insert(pos + 1, rpr.makeelement(q("w:szCs"), {q("w:val"): str(half_points)}))


def ptext(p):
    return "".join(t.text or "" for t in p.iter(q("w:t"))).strip()


def block_for(root, prefix):
    """A caption paragraph plus the contiguous frame paragraphs above it."""
    cap = next(p for p in root.iter(q("w:p")) if ptext(p).startswith(prefix))
    blk = [cap]
    prev = cap.getprevious()
    while prev is not None and prev.tag == q("w:p") and \
            prev.find(q("w:pPr") + "/" + q("w:framePr")) is not None and \
            not ptext(prev).startswith("Fig."):
        blk.insert(0, prev)
        prev = prev.getprevious()
    return blk


def move_after(root, blocks, target_substring):
    target = next(p for p in root.iter(q("w:p"))
                  if target_substring in ptext(p))
    moved = [p for blk in blocks for p in blk]
    for p in reversed(moved):
        p.getparent().remove(p)
        target.addnext(p)
    return len(moved)


def main(tree, july14):
    path = os.path.join(tree, "word", "document.xml")
    doc = etree.parse(path)
    root = doc.getroot()

    n_tbl = 0
    for i, tbl in enumerate(root.iter(q("w:tbl"))):
        if i == 0:
            continue  # the author block keeps its own size
        for r in tbl.iter(q("w:r")):
            if "".join(t.text or ""
                       for t in r.iter(q("w:t"), q("w:delText"))).strip():
                set_size(r, 16)
                n_tbl += 1

    n_cap = 0
    for p in root.iter(q("w:p")):
        if ptext(p).startswith(("Fig. ", "TABLE ")):
            for r in p.iter(q("w:r")):
                set_size(r, 18)
                n_cap += 1

    n13 = move_after(root, [block_for(root, "Fig. 13. Visual inspection")],
                     "support automated anomaly screening at the image level")
    print(f"table runs -> 8pt: {n_tbl}; caption runs -> 9pt: {n_cap}; "
          f"Fig. 13 block moved ({n13} paras)")

    if july14:
        n78 = move_after(root,
                         [block_for(root, "Fig. 7. Encoding throughput"),
                          block_for(root, "Fig. 8. Streaming scalability")],
                         "Moving from the base to the enhanced configuration")
        print(f"Figs. 7-8 blocks moved ({n78} paras)")

    doc.write(path, xml_declaration=True, encoding="UTF-8", standalone=True)
    print("FONT/GAP PASS DONE")


if __name__ == "__main__":
    main(sys.argv[1], "--july14" in sys.argv[2:])

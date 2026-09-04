# -*- coding: utf-8 -*-
"""Layout audit for the Word-exported manuscript PDF.

Checks, in order of severity:
  1. Overlapping wide figures (frames stacked on the same page space).
  2. White gaps: per-column content intervals. An internal hole (white band
     between two pieces of content, typically where a figure or table
     keep-block could not fit) reads far worse than a short column, so it is
     flagged at 45 pt while a trailing gap is allowed 110 pt. Full-width
     content (frames and one-column spans such as Table IV) counts toward
     both columns.
  2b. Caption collision: a full-width caption must clear the column text
     above it by at least 6 pt. A caption welded to the preceding paragraph
     is more visible to a reviewer than any amount of white space.
  3. Figure caption order: true captions ("Fig. N. Text" at line start) must
     appear on non-decreasing pages.
  4. Figure-discussion adjacency: a caption must not appear before the first
     page that cites the figure, nor more than one page after it; a caption
     sitting on a page with no citation of its own figure is reported for
     eyeballing even at distance one, since that is how the Fig. 2/3
     disjointed-page defect looked.
  5. Split tables (optional, needs the unzipped tree): every in-column
     numbered table should keep its caption and rows in one block, which in
     this manuscript is done with keepNext on the caption and on every table
     paragraph except those of the last row. Tables missing that protection
     are reported.

Usage: python docx_layout_audit.py <exported.pdf> [tree_root]
Exit code 0 = no hard failures (reports may still list eyeball items).
"""

import re
import sys

import fitz

TOP, BOT = 54, 780
MID = 300


def column_intervals(page):
    iv = {"L": [], "R": []}
    for b in page.get_text("blocks"):
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        if x0 < MID and x1 > MID + 20:
            iv["L"].append((y0, y1))
            iv["R"].append((y0, y1))
        else:
            iv["L" if x0 < MID else "R"].append((y0, y1))
    for img in page.get_image_info():
        r = fitz.Rect(img["bbox"])
        if r.width > 400:
            iv["L"].append((r.y0, r.y1))
            iv["R"].append((r.y0, r.y1))
        else:
            iv["L" if r.x0 < MID else "R"].append((r.y0, r.y1))
    return iv


def audit_pdf(path):
    doc = fitz.open(path)
    hard, soft = [], []

    for pno in range(len(doc)):
        page = doc[pno]
        wides = [fitz.Rect(i["bbox"]) for i in page.get_image_info()
                 if fitz.Rect(i["bbox"]).width > 400]
        for i in range(len(wides)):
            for j in range(i + 1, len(wides)):
                inter = wides[i] & wides[j]
                if not inter.is_empty and inter.get_area() > 100:
                    hard.append(f"overlap: two wide figures collide on "
                                f"p{pno + 1}")
        if pno == len(doc) - 1:
            continue
        iv = column_intervals(page)
        blocks = [(b[0], b[1], b[2], b[3], b[4])
                  for b in page.get_text("blocks")]
        for side in ("L", "R"):
            cur = TOP
            for y0, y1 in sorted(iv[side]):
                # an internal hole reads far worse than a short column, so it
                # is flagged at a much tighter threshold
                if y0 - cur > 45:
                    hard.append(f"gap: p{pno + 1}-{side} INTERNAL hole of "
                                f"{round(y0 - cur)}pt at y={round(cur)}")
                cur = max(cur, y1)
            if BOT - cur > 110:
                hard.append(f"gap: p{pno + 1}-{side} trailing "
                            f"{round(BOT - cur)}pt")

        # a full-width caption must not touch the column text above it
        for x0, y0, x1, y1, txt in blocks:
            if x0 < MID and x1 > MID + 20 and txt.strip().startswith(
                    ("TABLE", "Fig.")):
                above = [b for b in blocks if b[3] <= y0 + 1
                         and not (b[0] < MID and b[2] > MID + 20)]
                if above:
                    clear = y0 - max(b[3] for b in above)
                    if clear < 6:
                        hard.append(f"collision: p{pno + 1} caption "
                                    f"'{txt.strip()[:24]}' has {clear:.1f}pt "
                                    f"clearance above it")

    cap_page, cite_pages = {}, {}
    for pno in range(len(doc)):
        for line in doc[pno].get_text().splitlines():
            m = re.match(r"\s*Fig\.\s*(\d+)\.\s+[A-Z(]", line)
            if m:
                cap_page.setdefault(int(m.group(1)), pno + 1)
                continue  # a caption line is not a citation
            for m in re.finditer(r"Fig\.\s*(\d+)", line):
                cite_pages.setdefault(int(m.group(1)), set()).add(pno + 1)

    order = sorted(cap_page.items())
    for i in range(1, len(order)):
        if order[i][1] < order[i - 1][1]:
            hard.append(f"order: Fig. {order[i][0]} (p{order[i][1]}) captions "
                        f"before Fig. {order[i - 1][0]} (p{order[i - 1][1]})")

    for n, cpage in sorted(cap_page.items()):
        cites = sorted(cite_pages.get(n, set()))
        if not cites:
            soft.append(f"eyeball: Fig. {n} (p{cpage}) is never cited in text")
            continue
        if cpage < min(cites):
            hard.append(f"adjacency: Fig. {n} shown p{cpage} before its "
                        f"first citation p{min(cites)}")
            continue
        nearest = min(abs(cpage - c) for c in cites)
        if nearest > 2:
            hard.append(f"adjacency: Fig. {n} shown p{cpage}, nearest "
                        f"citation {nearest} pages away (cited on "
                        f"{sorted(cites)})")
        elif nearest == 2:
            soft.append(f"eyeball: Fig. {n} on p{cpage} is 2 pages from its "
                        f"nearest citation (cited on {sorted(cites)})")
        elif cpage not in cites:
            soft.append(f"eyeball: Fig. {n} on p{cpage} has no citation on "
                        f"its own page; check it does not strand after its "
                        f"discussion beside unrelated sections")
    return hard, soft, len(doc)


def audit_tables(tree):
    sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]
                    if "/" in __file__ or "\\" in __file__ else ".")
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from docx_edit_lib import Editor, q
    ed = Editor(f"{tree}/word/document.xml")
    soft = []
    tables = list(ed.root.iter(q("w:tbl")))
    for ti, tbl in enumerate(tables):
        txt = "".join(tbl.itertext())
        if ti == 0 or "Naive Diff. (16b)" in txt:
            continue  # author table; full-width Table IV spans by design
        rows = list(tbl.iter(q("w:tr")))
        unkept = 0
        for r in rows[:-1]:
            for p in r.iter(q("w:p")):
                ppr = p.find(q("w:pPr"))
                if ppr is None or ppr.find(q("w:keepNext")) is None:
                    unkept += 1
        if unkept:
            label = txt.strip().split("\n")[0][:40] or f"table {ti}"
            soft.append(f"table-keep: in-column table '{label}...' has "
                        f"{unkept} paragraphs without keepNext; it can split "
                        f"across a column or page break")
    return soft


def main():
    pdf = sys.argv[1]
    hard, soft, pages = audit_pdf(pdf)
    if len(sys.argv) > 2:
        soft += audit_tables(sys.argv[2])
    print(f"pages={pages}")
    for h in hard:
        print(f"FAIL {h}")
    for s in soft:
        print(f"NOTE {s}")
    if not hard and not soft:
        print("clean: no overlaps, gaps, order or adjacency issues")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()

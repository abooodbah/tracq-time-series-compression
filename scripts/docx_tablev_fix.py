# -*- coding: utf-8 -*-
"""Repair Table V: remove the LaTeX cmidrule artifact, rebalance column widths
so values no longer wrap, bold the best sub-10%-size RMSE per dataset, and
state the bolding rule in the caption.

Usage: python docx_tablev_fix.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q
from lxml import etree

TREE = sys.argv[1]
DOC = os.path.join(TREE, "word", "document.xml")

# tracked text edits first
ed = Editor(DOC)
ed.replace("3-5(lr)6-8(lr)9-11 Category", "Category")
ed.replace("corresponds to a normalized RMSE of approximately 0.1% relative to the global data "
           "range.",
           "corresponds to a normalized RMSE of approximately 0.1% relative to the global data "
           "range. Boldface marks the lowest RMSE among methods encoding at or below one tenth of "
           "the original size.")
ed.save()

# layout + formatting (non-text): widths and bold values
parser = etree.XMLParser(remove_blank_text=False)
tree = etree.parse(DOC, parser)
root = tree.getroot()

NEW_W = [950, 1450, 670, 910, 585, 670, 910, 585, 670, 910, 583]

target = None
for tbl in root.iter(q("w:tbl")):
    txt = "".join(t.text or "" for t in tbl.iter(q("w:t")))
    if "Air Quality" in txt and "Gorilla" in txt:
        target = tbl
        break
assert target is not None, "Table V not found"

grid = target.find(q("w:tblGrid"))
cols = grid.findall(q("w:gridCol"))
assert len(cols) == len(NEW_W), f"expected 11 columns, got {len(cols)}"
for c, w in zip(cols, NEW_W):
    c.set(q("w:w"), str(w))

for tr in target.findall(q("w:tr")):
    cursor = 0
    for tc in tr.findall(q("w:tc")):
        tcpr = tc.find(q("w:tcPr"))
        if tcpr is None:
            tcpr = etree.Element(q("w:tcPr"))
            tc.insert(0, tcpr)
        span_el = tcpr.find(q("w:gridSpan"))
        span = int(span_el.get(q("w:val"))) if span_el is not None else 1
        width = sum(NEW_W[cursor:cursor + span])
        tcw = tcpr.find(q("w:tcW"))
        if tcw is None:
            tcw = etree.SubElement(tcpr, q("w:tcW"))
        tcw.set(q("w:w"), str(width))
        tcw.set(q("w:type"), "dxa")
        cursor += span

# bold the winning RMSE values (rule stated in the caption)
BOLD = {"0.90", "0.012", "1.81"}
bolded = 0
for r in target.iter(q("w:r")):
    ts = r.findall(q("w:t"))
    if len(ts) == 1 and (ts[0].text or "").strip() in BOLD:
        rpr = r.find(q("w:rPr"))
        if rpr is None:
            rpr = etree.Element(q("w:rPr"))
            r.insert(0, rpr)
        if rpr.find(q("w:b")) is None:
            rpr.insert(0, etree.Element(q("w:b")))
        bolded += 1

tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
print(f"Table V fixed: widths rebalanced, artifact removed, {bolded} values bolded")

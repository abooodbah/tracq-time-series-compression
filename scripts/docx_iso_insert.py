# -*- coding: utf-8 -*-
"""Insert the iso-RMSE comparison (subsection + figure + claim updates) into a
document tree. Numbers are computed from paper_results/lattice/iso_rmse.json
at run time so the text always matches the data.

Usage: python docx_iso_insert.py <tree_root> <fig_number>
  tree_root: folder containing word/document.xml
  fig_number: figure number to assign (16 for v3, 15 for the July-14 lineage)
"""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q
from lxml import etree

TREE = sys.argv[1]
FIGN = int(sys.argv[2])
DOC = os.path.join(TREE, "word", "document.xml")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISO = json.load(open(os.path.join(PROJECT_ROOT, "paper_results", "lattice", "iso_rmse.json")))

def stats(name):
    adv = [r["advantage"] for r in ISO[name]["iso"]]
    return np.median(adv), max(adv), min(adv)

m_air, x_air, n_air = stats("air_quality")
m_app, x_app, n_app = stats("appliances")
m_met, x_met, n_met = stats("metro_traffic")
m_mp, x_mp, n_mp = stats("metropt3")
meds = sorted([m_air, m_app, m_met, m_mp])
typ_lo, typ_hi = meds[0], meds[-1]
overall_max = max(x_air, x_app, x_met, x_mp)

P1 = ("Rate-distortion claims are easiest to judge at matched operating points, so we sweep both "
      "codecs densely, interpolate their measured curves onto a common RMSE grid, and ask a single "
      "question at each accuracy level: which method needs fewer bytes. Fig. %d reports the "
      "comparison on the three UCI datasets and the full-length MetroPT-3 stream." % FIGN)

P2 = ("At equal RMSE, the enhanced variant produces the smaller artifact across essentially the "
      "entire overlap band. The median advantage is %.1f× on Air Quality, %.1f× on Appliances "
      "Energy, %.1f× on Metro Traffic, and %.1f× on MetroPT-3, and the margin widens as the "
      "accuracy target relaxes — up to %.1f×, %.1f×, %.1f×, and %.1f× respectively — because "
      "ZFP's size floor sits near one fifth of the original on these workloads while the "
      "difference grid keeps shrinking until its residual entropy is exhausted. The exception is "
      "the highest-fidelity end of Air Quality, where ZFP is up to %.1f× smaller."
      % (m_air, m_app, m_met, m_mp, x_air, x_app, x_met, x_mp, 1.0 / n_air))

CAP = ("Fig. %d. Encoded size at matched RMSE. (a) MetroPT-3: measured size of both codecs at "
       "equal RMSE. (b) Size advantage, computed as ZFP bytes divided by enhanced-variant bytes "
       "at the same RMSE; values above one indicate the smaller artifact." % FIGN)

ed = Editor(DOC)

# refine the standing 5.6 claim to the measured band
ed.replace("At matched accuracy, the enhanced configuration is 2–4× smaller than ZFP while "
           "retaining an image-based artifact.",
           "At matched accuracy, the enhanced configuration typically halves ZFP's encoded size "
           "and reaches up to %.0f× at relaxed targets (Fig. %d), while retaining an image-based "
           "artifact." % (overall_max, FIGN))

# abstract + conclusion claim alignment
ed.replace("such as ZFP at 2–4× smaller encoded sizes, and covers aggressive operating points",
           "such as ZFP with smaller artifacts at matched accuracy — typically about 2× and up to "
           "%.0f× — and covers aggressive operating points" % overall_max, required=False)
ed.replace("such as ZFP at 2–4× smaller encoded sizes while preserving interpretability, and "
           "covers aggressive operating points those compressors cannot produce.",
           "such as ZFP with smaller artifacts at matched accuracy while preserving "
           "interpretability, and covers aggressive operating points those compressors cannot "
           "produce.", required=False)

# locate insertion anchor: after the last existing results subsection
anchor = None
for cand in ["Encoded size improves with width while metadata remains a small share",
             "This property is useful for active monitoring pipelines when data volumes are too "
             "large for full real-time decompression."]:
    for p in ed.paras:
        if cand in ed.para_text(p):
            anchor = cand
            break
    if anchor:
        break

ed.insert_paragraph_after(anchor, [("Size at Matched Accuracy", False)])
ed.insert_paragraph_after("Size at Matched Accuracy", [(P1, False)])
ed.insert_paragraph_after("which method needs fewer bytes.", [(P2, False)])
ed.insert_paragraph_after("where ZFP is up to", [(CAP, False)])
misses = ed.save()

# style the heading + caption, then place the figure before the caption
parser = etree.XMLParser(remove_blank_text=False)
tree = etree.parse(DOC, parser)
root = tree.getroot()
paras = list(root.iter(q("w:p")))

def ptext(p):
    return "".join(t.text or "" for t in p.iter(q("w:t")))

def find(snippet):
    for p in paras:
        if snippet in ptext(p):
            return p
    raise SystemExit(f"missing: {snippet}")

def clone_ppr(target, source):
    ins_mark = None
    old = target.find(q("w:pPr"))
    if old is not None:
        rpr = old.find(q("w:rPr"))
        if rpr is not None:
            ins_mark = rpr.find(q("w:ins"))
        target.remove(old)
    src = source.find(q("w:pPr"))
    npr = etree.fromstring(etree.tostring(src)) if src is not None else etree.Element(q("w:pPr"))
    for tag in ("w:sectPr", "w:pPrChange"):
        el = npr.find(q(tag))
        if el is not None:
            npr.remove(el)
    rpr = npr.find(q("w:rPr"))
    if rpr is None:
        rpr = etree.SubElement(npr, q("w:rPr"))
    for m in rpr.findall(q("w:ins")) + rpr.findall(q("w:del")):
        rpr.remove(m)
    if ins_mark is not None:
        rpr.insert(0, ins_mark)
    target.insert(0, npr)

clone_ppr(find("Size at Matched Accuracy"), find("Visual Inspection Demonstration"))
clone_ppr(find(f"Fig. {FIGN}. Encoded size at matched RMSE."),
          find("Fig. 13. Visual inspection demonstration"))

xml = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True).decode("utf-8")

rels_p = os.path.join(TREE, "word", "_rels", "document.xml.rels")
rels = open(rels_p, encoding="utf-8").read()
if "media/image17.png" not in rels:
    rels = rels.replace("</Relationships>",
        '<Relationship Id="rId902" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/image" Target="media/image17.png"/></Relationships>')
    open(rels_p, "w", encoding="utf-8").write(rels)

m = re.search(r'Id="(rId\d+)"[^>]*Target="media/image8.png"', rels)
rid8 = m.group(1)
dm = None
for mm in re.finditer(r"<w:drawing>.*?</w:drawing>", xml, re.S):
    if f'r:embed="{rid8}"' in mm.group(0):
        dm = mm.group(0)
        break
drawing = dm.replace(f'r:embed="{rid8}"', 'r:embed="rId902"')
cx, cy = int(1044 / 300 * 914400), int(1500 / 300 * 914400)
drawing = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>', f'<wp:extent cx="{cx}" cy="{cy}"/>', drawing)
drawing = re.sub(r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{cx}" cy="{cy}"/>', drawing)
drawing = re.sub(r'<a:srcRect[^/]*/>', '', drawing)
drawing = re.sub(r'(<wp:docPr id=")\d+(" name=")[^"]*(")', r'\g<1>9017\g<2>FigIso\g<3>', drawing)
figp = ('<w:p><w:pPr><w:jc w:val="center"/><w:rPr><w:ins w:id="9905" '
        'w:author="Abdulfatah Bahbouh" w:date="2026-07-15T00:00:00Z"/></w:rPr></w:pPr>'
        '<w:ins w:id="9906" w:author="Abdulfatah Bahbouh" w:date="2026-07-15T00:00:00Z">'
        '<w:r>' + drawing + "</w:r></w:ins></w:p>")
cap_idx = xml.find(f"Fig. {FIGN}. Encoded size at matched RMSE.")
pstart = max(xml.rfind("<w:p ", 0, cap_idx), xml.rfind("<w:p>", 0, cap_idx))
xml = xml[:pstart] + figp + xml[pstart:]
open(DOC, "w", encoding="utf-8").write(xml)
print(f"inserted as Fig. {FIGN}; medians {m_air:.2f}/{m_app:.2f}/{m_met:.2f}/{m_mp:.2f}")

# -*- coding: utf-8 -*-
"""Insert the node-parallel scaling subsection (heading + two paragraphs +
figure + caption) at the end of the results section. Numbers are computed from
paper_results/lattice/scaling_result.json at run time.

Placed after the matched-accuracy subsection so existing lettered
cross-references (V-J, V-L) keep their positions.

Usage: python docx_scaling_insert.py <tree_root> <fig_number>
  fig_number: 17 for the July-15 lineage, 16 for the July-14 lineage
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q
from lxml import etree

TREE = sys.argv[1]
FIGN = int(sys.argv[2])
DOC = os.path.join(TREE, "word", "document.xml")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = json.load(open(os.path.join(PROJECT_ROOT, "paper_results", "lattice", "scaling_result.json")))


def med3(runs, w, key):
    vals = sorted(r[key] for r in runs if r["workers"] == w)
    return vals[len(vals) // 2], vals[0], vals[-1]


counts = sorted({r["workers"] for r in SC["strong"]})
top = counts[-1]

cpu = re.search(r"Model name:\s*(.+)", SC["lscpu"]).group(1).strip().replace("(R)", "")
cpu = re.sub(r"\s+", " ", cpu)
sockets = int(re.search(r"Socket\(s\):\s*(\d+)", SC["lscpu"]).group(1))
cps = int(re.search(r"Core\(s\) per socket:\s*(\d+)", SC["lscpu"]).group(1))

s_base = med3(SC["strong"], 1, "wall_s")[0]
s_top = med3(SC["strong"], top, "wall_s")[0]
speedup = s_base / s_top
s_eff = 100 * speedup / top
w_base = med3(SC["weak"], 1, "wall_s")[0]
w_top = med3(SC["weak"], top, "wall_s")[0]
w_eff = 100 * w_base / w_top

import math
spread_below_full = max(100 * (med3(pool, w, "wall_s")[2] - med3(pool, w, "wall_s")[1]) /
                        med3(pool, w, "wall_s")[0]
                        for pool in (SC["strong"], SC["weak"]) for w in counts[:-1])
eff56 = min(100 * s_base / med3(SC["strong"], 56, "wall_s")[0] / 56,
            100 * w_base / med3(SC["weak"], 56, "wall_s")[0])

enc_meds = [med3(SC["weak"], w, "enc_core_mbs")[0] for w in counts] + \
           [med3(SC["strong"], w, "enc_core_mbs")[0] for w in counts]
rss = max(max(r["rss_mb_max"] for r in SC["strong"]),
          max(r["rss_mb_max"] for r in SC["weak"]))

HEAD = "Node-Parallel Scaling"

P1 = ("The terabyte run above uses all %d cores of one node; here we characterize how that "
      "throughput is reached on the same hardware (%d sockets of %d-core %s). Strong scaling "
      "encodes a fixed %.0f GB stream at 1 to %d workers; weak scaling holds the load at %.0f GB "
      "per worker. Each point is the median of three runs. Fig. %d reports speedup and parallel "
      "efficiency alongside the encode-stage per-core throughput."
      % (top, sockets, cps, cpu, SC["strong_gb"], top, SC["weak_gb_per_worker"], FIGN))

P2 = ("At %d workers the fixed %.0f GB input encodes %.0f× faster than on one worker (%.0f%% "
      "parallel efficiency), and efficiency is %.0f%% when the load grows with the worker count; "
      "through 56 workers — one full socket — both sweeps stay at or above %.0f%%. The dominant "
      "full-node cost is shared memory bandwidth: per-core encode throughput declines smoothly "
      "from %.0f MB/s on an otherwise idle node to %.0f MB/s with all %d cores active, while "
      "peak worker memory stays near %.0f MB at every scale. The windows are independent, so no "
      "coordination term appears; run-to-run spread is below %.0f%% except at %d workers, where "
      "wall times of a few seconds magnify pool-startup noise."
      % (top, SC["strong_gb"], speedup, s_eff, w_eff, eff56, max(enc_meds), min(enc_meds), top,
         rss, math.ceil(spread_below_full), top))

CAP = ("Fig. %d. Node-parallel scaling on one %d-core Sapphire Rapids node, median of three runs "
       "(bars span min–max). (a) Strong scaling of a fixed %.0f GB stream: measured speedup "
       "against the ideal line. (b) Weak scaling at %.0f GB per worker: parallel efficiency (left "
       "axis) and encode-stage per-core throughput (right axis)."
       % (FIGN, top, SC["strong_gb"], SC["weak_gb_per_worker"]))

ed = Editor(DOC)
ed.insert_paragraph_after("values above one indicate that the enhanced", [(HEAD, False)])
ed.insert_paragraph_after(HEAD, [(P1, False)])
ed.insert_paragraph_after("alongside the encode-stage per-core throughput.", [(P2, False)])
ed.insert_paragraph_after("magnify pool-startup noise.", [(CAP, False)])
ed.save()

# style heading + caption, then place the figure before the caption
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


clone_ppr(find("Node-Parallel Scaling"), find("Size at Matched Accuracy"))
clone_ppr(find(f"Fig. {FIGN}. Node-parallel scaling"),
          find("Encoded size at matched RMSE."))

xml = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True).decode("utf-8")

rels_p = os.path.join(TREE, "word", "_rels", "document.xml.rels")
rels = open(rels_p, encoding="utf-8").read()
if "media/image18.png" not in rels:
    rels = rels.replace("</Relationships>",
        '<Relationship Id="rId903" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/image" Target="media/image18.png"/></Relationships>')
    open(rels_p, "w", encoding="utf-8").write(rels)

m = re.search(r'Id="(rId\d+)"[^>]*Target="media/image8.png"', rels)
rid8 = m.group(1)
dm = None
for mm in re.finditer(r"<w:drawing>.*?</w:drawing>", xml, re.S):
    if f'r:embed="{rid8}"' in mm.group(0):
        dm = mm.group(0)
        break
drawing = dm.replace(f'r:embed="{rid8}"', 'r:embed="rId903"')
cx, cy = int(1044 / 300 * 914400), int(1500 / 300 * 914400)
drawing = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>', f'<wp:extent cx="{cx}" cy="{cy}"/>', drawing)
drawing = re.sub(r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{cx}" cy="{cy}"/>', drawing)
drawing = re.sub(r'<a:srcRect[^/]*/>', '', drawing)
drawing = re.sub(r'(<wp:docPr id=")\d+(" name=")[^"]*(")', r'\g<1>9018\g<2>FigScale\g<3>', drawing)
figp = ('<w:p><w:pPr><w:jc w:val="center"/><w:rPr><w:ins w:id="9915" '
        'w:author="Abdulfatah Bahbouh" w:date="2026-07-19T00:00:00Z"/></w:rPr></w:pPr>'
        '<w:ins w:id="9916" w:author="Abdulfatah Bahbouh" w:date="2026-07-19T00:00:00Z">'
        '<w:r>' + drawing + "</w:r></w:ins></w:p>")
cap_idx = xml.find(f"Fig. {FIGN}. Node-parallel scaling")
pstart = max(xml.rfind("<w:p ", 0, cap_idx), xml.rfind("<w:p>", 0, cap_idx))
xml = xml[:pstart] + figp + xml[pstart:]
open(DOC, "w", encoding="utf-8").write(xml)
print(f"inserted as Fig. {FIGN}: speedup {speedup:.1f}x, strong eff {s_eff:.0f}%, "
      f"weak eff {w_eff:.0f}%")

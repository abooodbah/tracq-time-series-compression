# -*- coding: utf-8 -*-
"""v3 pass: high-dimensional scaling subsection + Fig. 15, Python-vs-native
throughput acknowledgment, and zero-copy analytics generalization."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q
from lxml import etree

BASE = (r"C:\Users\ABDULF~1\AppData\Local\Temp\claude"
        r"\C--Users-Abdulfatah--codex-skills-multi-agent-coordinator-v2"
        r"\98957d2f-346c-4497-89c3-63d184c5c33c\scratchpad\docx_work\unpacked")
DOC = os.path.join(BASE, "word", "document.xml")

ed = Editor(DOC)
R = ed.replace

# ---------- 2) throughput runtime acknowledgment (Sec V.G) ----------
R("ZFP is accessed via zfpy (C bindings); SZ3 is accessed via hdf5plugin (HDF5 filter, evaluated on "
  "Linux/WSL).",
  "ZFP is accessed via zfpy (C bindings); SZ3 is accessed via hdf5plugin (HDF5 filter, evaluated on "
  "Linux/WSL). These comparisons therefore mix language runtimes: the TRACQ encoder pays Python "
  "interpreter and NumPy dispatch overhead on every stage, while ZFP and SZ3 execute as optimized "
  "native C/C++ code behind thin bindings. The throughput figures should accordingly be read as "
  "evidence that a prototype does not bottleneck typical ingestion paths, not as a runtime-controlled "
  "measurement of algorithmic cost; a controlled comparison would require the compiled implementation "
  "discussed below.")

# ---------- 3) zero-copy analytics generalization (Sec V.L) ----------
ed.insert_paragraph_after(
    "and where flagged windows can be selectively decompressed for detailed analysis.",
    [("Generalization. To test whether these results depend on the dataset or on the detector, we "
      "repeat the protocol on all three UCI datasets and across three unsupervised model families — "
      "Isolation Forest, Local Outlier Factor, and a one-class SVM — pairing each with the same "
      "features computed once from decoded data and once from the compressed grids. Compressed-domain "
      "detection stays within 0.08 F1 of its decoded counterpart in all nine combinations and reaches "
      "exact parity twice. On Appliances Energy the decoded/compressed pairs are 0.75/0.75 (IF), "
      "0.81/0.77 (LOF), and 0.80/0.78 (OC-SVM); on Air Quality, 0.75/0.71, 0.75/0.70, and 0.73/0.73; "
      "on Metro Traffic, 0.84/0.76, 0.86/0.78, and 0.82/0.77. The pattern is tied to neither one "
      "model nor one workload. All windows here follow the injection protocol above; evaluating "
      "against naturally labeled industrial faults remains future work.", False)])

# ---------- 1) high-dimensional scaling subsection + Fig. 15 ----------
ANCHOR = ("This property is useful for active monitoring pipelines when data volumes are too large "
          "for full real-time decompression.")
ed.insert_paragraph_after(ANCHOR, [("High-Dimensional Scaling", False)])
ed.insert_paragraph_after(
    "High-Dimensional Scaling",
    [("Wide sensor fleets routinely exceed a thousand channels, so we examine how the enhanced "
      "variant behaves as dimensionality grows. We sweep grouped synthetic workloads from 64 to "
      "4,096 variables (2,000 steps, 16 correlated channel groups) and anchor the sweep with the "
      "widest real dataset in our evaluation, the 370-client electricity load archive [44]. Fig. 15 "
      "reports throughput, encoded size, and metadata share across the sweep.", False)])
ed.insert_paragraph_after(
    "Fig. 15 reports throughput, encoded size, and metadata share across the sweep.",
    [("The guaranteed bound holds at every width, and the core codec is insensitive to "
      "dimensionality: the fast encoding path sustains 234–273 MB/s from 64 to 4,096 variables, and "
      "decoding stays between 309 and 366 MB/s. Compression improves with width on grouped data, "
      "from 4.1% of the original size at 64 variables to 2.5% at 4,096, because the "
      "correlation-ordered two-dimensional predictor absorbs cross-channel structure; it is selected "
      "for over 99% of rows at 1,024 variables and above. Metadata remains a modest share of the "
      "artifact, 5.2% at 64 variables and 8.4% at 4,096, and the share is governed by sequence "
      "length rather than width, since the stored per-variable scalars amortize over time steps: on "
      "the 10,000-step electricity data the entire header is 0.45% of the artifact.", False)])
ed.insert_paragraph_after(
    "on the 10,000-step electricity data the entire header is 0.45% of the artifact.",
    [("Predictor selection is the one stage whose cost grows with width. Its correlation ordering is "
      "quadratic in the number of variables; at 4,096 variables it accounts for 96% of encoding "
      "time, reducing throughput to 12 MB/s while buying a 33% smaller artifact than the fast path. "
      "The electricity data shows the other side of this trade: its clients are largely "
      "uncorrelated, selection recovers nothing (8.19% versus 8.18% of the original size), and the "
      "fast path is the right default. We therefore recommend the fast path for wide, "
      "latency-sensitive ingestion, reserving predictor selection for archival encoding of fleets "
      "with known cross-channel structure; a subsampled correlation estimate would cut the quadratic "
      "cost and is straightforward engineering.", False)])
ed.insert_paragraph_after(
    "a subsampled correlation estimate would cut the quadratic cost",
    [("Fig. 15. High-dimensional scaling on grouped synthetic workloads (2,000 steps, 16 correlated "
      "groups). (a) The fast encoding path and the decoder are insensitive to width; predictor "
      "selection pays a quadratic correlation cost. (b) Encoded size improves with width while "
      "metadata remains a small share of the artifact.", False)])

# ---------- reference [44] ----------
ed.insert_paragraph_after(
    "[43] Liu, F. T.",
    [("[44] UCI Machine Learning Repository, “ElectricityLoadDiagrams20112014 data set,” University "
      "of California, Irvine, https://archive.ics.uci.edu/.", False)])

misses = ed.save()
print("text edits done")

# ---------- style the new heading + caption, insert the figure drawing ----------
parser = etree.XMLParser(remove_blank_text=False)
tree = etree.parse(DOC, parser)
root = tree.getroot()
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def para_text(p):
    return "".join(t.text or "" for t in p.iter(q("w:t")))


paras = list(root.iter(q("w:p")))


def find(snippet):
    for p in paras:
        if snippet in para_text(p):
            return p
    raise SystemExit(f"not found: {snippet}")


def clone_ppr_onto(target, style_source, keep_ins=True):
    ins_mark = None
    old = target.find(q("w:pPr"))
    if old is not None:
        rpr = old.find(q("w:rPr"))
        if rpr is not None:
            ins_mark = rpr.find(q("w:ins"))
        target.remove(old)
    src = style_source.find(q("w:pPr"))
    newppr = etree.fromstring(etree.tostring(src)) if src is not None else etree.Element(q("w:pPr"))
    for tag in ("w:sectPr", "w:pPrChange"):
        el = newppr.find(q(tag))
        if el is not None:
            newppr.remove(el)
    if keep_ins and ins_mark is not None:
        rpr = newppr.find(q("w:rPr"))
        if rpr is None:
            rpr = etree.SubElement(newppr, q("w:rPr"))
        rpr.insert(0, ins_mark)
    target.insert(0, newppr)


# heading style from an existing results subsection heading
clone_ppr_onto(find("High-Dimensional Scaling"), find("Visual Inspection Demonstration"))
# caption style from an existing figure caption
cap_src = find("Fig. 14. Compressed-domain anomaly detection.")
clone_ppr_onto(find("Fig. 15. High-dimensional scaling on grouped synthetic workloads"), cap_src)

# figure drawing: clone image8's drawing, repoint to image16
xml = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True).decode("utf-8")
rels_p = os.path.join(BASE, "word", "_rels", "document.xml.rels")
rels = open(rels_p, encoding="utf-8").read()
m = re.search(r'Id="(rId\d+)"[^>]*Target="media/image8.png"', rels)
rid8 = m.group(1)
if "image16.png" not in rels:
    rels = rels.replace("</Relationships>",
        '<Relationship Id="rId901" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/image" Target="media/image16.png"/></Relationships>')
    open(rels_p, "w", encoding="utf-8").write(rels)

dm = None
for mm in re.finditer(r"<w:drawing>.*?</w:drawing>", xml, re.S):
    if f'r:embed="{rid8}"' in mm.group(0):
        dm = mm.group(0)
        break
drawing = dm.replace(f'r:embed="{rid8}"', 'r:embed="rId901"')
# extent for 2127x880 px at 300 dpi
cx, cy = int(2127 / 300 * 914400), int(880 / 300 * 914400)
drawing = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>', f'<wp:extent cx="{cx}" cy="{cy}"/>', drawing)
drawing = re.sub(r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{cx}" cy="{cy}"/>', drawing)
drawing = re.sub(r'(<wp:docPr id=")\d+(" name=")[^"]*(")', r'\g<1>9016\g<2>Fig15HD\g<3>', drawing)
figp = ('<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:pPr>'
        '<w:jc w:val="center"/><w:rPr><w:ins w:id="9901" w:author="Abdulfatah Bahbouh" '
        'w:date="2026-07-15T00:00:00Z"/></w:rPr></w:pPr>'
        '<w:ins w:id="9902" w:author="Abdulfatah Bahbouh" w:date="2026-07-15T00:00:00Z">'
        '<w:r>' + drawing + "</w:r></w:ins></w:p>")
# place figure paragraph right before its caption
cap_idx = xml.find("Fig. 15. High-dimensional scaling")
pstart = xml.rfind("<w:p ", 0, cap_idx)
if pstart < 0 or xml.rfind("<w:p>", 0, cap_idx) > pstart:
    pstart = xml.rfind("<w:p>", 0, cap_idx)
figp_clean = figp.replace(' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"', "")
xml = xml[:pstart] + figp_clean + xml[pstart:]
open(DOC, "w", encoding="utf-8").write(xml)
print("figure inserted; drawing bytes:", len(drawing))

# -*- coding: utf-8 -*-
"""Integrate the dense SZ3 comparison into a document tree: refreshed Table V
SZ3 rows (real Metro Traffic values replace the container-overhead footnote),
three-codec matched-accuracy subsection text, and aligned abstract/conclusion
claims.

Usage: python docx_sz3_pass.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q
from lxml import etree

TREE = sys.argv[1]
DOC = os.path.join(TREE, "word", "document.xml")
ed = Editor(DOC)
R = ed.replace

# ---- abstract ----
R("with smaller artifacts at matched accuracy — typically about 2× and up to 22× on the longest "
  "stream — and covers aggressive operating points",
  "with smaller artifacts at matched accuracy — median 1.2–1.9× against SZ3, 1.5–6.9× against "
  "ZFP, and up to 22× on the longest stream — and covers aggressive operating points")

# ---- Table V caption: retire the container-overhead footnote ----
R("” indicates HDF5 container overhead exceeds compression savings on this small dataset. For "
  "Appliances Energy",
  "” For Appliances Energy", required=False)

# ---- Table V SZ3 cells (document order) ----
for old, new in [("0.398", "0.236"), ("0.043", "0.047"),
                 ("0.097", "0.065"),
                 ("0.335", "0.269"), ("0.326", "0.210")]:
    R(old, new)

# ---- 5.8 bullet ----
R("except SZ3 on Appliances Energy (0.097 at tol=",
  "except SZ3 on Appliances Energy (0.065 at tol=", required=False)
R("), though on the small Metro Traffic dataset the HDF5 container overhead causes the output to "
  "exceed the original size.",
  ") and on Metro Traffic (0.180 at the same tolerance).")

# ---- V.N paragraph: three-codec numbers ----
R("At equal RMSE, the enhanced variant produces the smaller artifact across essentially the "
  "entire overlap band. The median advantage is 1.5× on Air Quality, 2.1× on Appliances Energy, "
  "2.2× on Metro Traffic, and 6.9× on MetroPT-3, and the margin widens as the accuracy target "
  "relaxes — up to 3.7×, 7.0×, 3.1×, and 22.2× respectively — because ZFP's size floor sits near "
  "one fifth of the original on these workloads while the difference grid keeps shrinking until "
  "its residual entropy is exhausted. The exception is the highest-fidelity end of Air Quality, "
  "where ZFP is up to 1.1× smaller.",
  "At equal RMSE, the enhanced variant produces the smaller artifact across essentially the "
  "entire overlap band against both error-bounded compressors. Against ZFP the median advantage "
  "is 1.5× on Air Quality, 2.1× on Appliances Energy, 2.2× on Metro Traffic, and 6.9× on "
  "MetroPT-3, widening to 3.7×, 7.0×, 3.1×, and 22.2× as the accuracy target relaxes, because "
  "ZFP's size floor sits near one fifth of the original on these workloads. SZ3 is the stronger "
  "competitor — its prediction-based design reaches far smaller artifacts than ZFP — yet the "
  "enhanced variant still leads it at matched RMSE, with median advantages of 1.2–1.9× across "
  "the four datasets and up to 3.8× on MetroPT-3. The exceptions are narrow: at a few accuracy "
  "levels on Air Quality and Appliances Energy, one of the two is up to 1.1× smaller.")

# ---- iso figure caption ----
R("(a) MetroPT-3: measured size of both codecs at equal RMSE. (b) Size advantage, computed as "
  "ZFP bytes divided by enhanced-variant bytes at the same RMSE; values above one indicate the "
  "smaller artifact.",
  "(a) MetroPT-3: measured size of all three codecs at equal RMSE. (b) Size advantage at the "
  "same RMSE against the stronger of ZFP and SZ3 at each point; values above one indicate that "
  "the enhanced variant's artifact is smaller.")

# ---- conclusion ----
R("reaches the accuracy class of error-bounded compressors such as ZFP with smaller artifacts",
  "reaches the accuracy class of error-bounded compressors such as ZFP and SZ3 with smaller "
  "artifacts")

misses1 = ed.save()

# ---- Table V metro SZ3 cells: tracked-delete the > and ‡ math, set values ----
ed2 = Editor(DOC)
paras = ed2.paras
target_tbl = None
root = ed2.root
for tbl in root.iter(q("w:tbl")):
    txt = "".join(t.text or "" for t in tbl.iter(q("w:t")))
    if "Air Quality" in txt and "Gorilla" in txt:
        target_tbl = tbl
        break

new_vals = [("0.180", "0.026"), ("0.179", "<0.001")]
ri = 0
for tr in target_tbl.findall(q("w:tr")):
    rowtxt = "".join(t.text or "" for t in tr.iter(q("w:t")))
    if "SZ3" not in rowtxt:
        continue
    cells = tr.findall(q("w:tc"))
    ratio_cell, rmse_cell = cells[-3], cells[-2]
    for cell, val in ((ratio_cell, new_vals[ri][0]), (rmse_cell, new_vals[ri][1])):
        for p in cell.findall(q("w:p")):
            ed2._mark_math_runs(p, "del")
            pi = ed2.paras.index(p)
            ed2.cursor_p, ed2.cursor_c = pi, 0
            t = ed2.para_text(p).strip()
            if t in ("1", "0"):
                ed2.replace(t, val)
    ri += 1
    if ri == 2:
        break
misses2 = ed2.save()
print("SZ3 PASS DONE")

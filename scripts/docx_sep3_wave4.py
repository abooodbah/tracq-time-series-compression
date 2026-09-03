# -*- coding: utf-8 -*-
"""Wave 4: consistency-audit fixes after the definitive-TRACQ reframe.

The hand-classified sweep (base, enhanced, framework, variant, percentage,
offset, multiplicative, drift, O(1), compression ratio, proposed, our method,
algorithm) found six residuals, all fixed here: three table rows spelled
"Enhanced (...)" rather than "Enh. (...)" and so missed by Wave 1's patterns,
Table II's bare "Enhanced" row, one prose "enhanced curves", and the "Toe
evaluate" typo. Fig. 2's frame block also anchors one paragraph deeper so the
column its section vacated (after Wave 3 shortened Section III) fills.

Usage: python docx_sep3_wave4.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    ed.replace("Enhanced (relative)", "TRACQ (rel)")
    ed.seek("Error Drift on 10,000-Step")
    ed.replace("Enhanced", "TRACQ")
    ed.seek("8-bit Compression vs")
    ed.replace("Enhanced (0.01)", "TRACQ (0.01)")
    ed.replace("Enhanced (0.001)", "TRACQ (0.001)")
    ed.replace("the enhanced curves are horizontal",
               "the TRACQ curves are horizontal")
    ed.save()
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    ed.replace("Toe evaluate our proposed method",
               "To evaluate our proposed method")

    cap = next(p for p in ed.paras
               if ed.para_text(p).strip().startswith("Fig. 2. Mean relative error"))
    blk = [cap]
    prev = cap.getprevious()
    while prev is not None and prev.tag == q("w:p") and \
            prev.find(q("w:pPr") + "/" + q("w:framePr")) is not None and \
            not ed.para_text(prev).strip().startswith("Fig."):
        blk.insert(0, prev)
        prev = prev.getprevious()
    target = next(p for p in ed.paras
                  if "Per-variable step selection reduces mean relative error"
                  in ed.para_text(p))
    for p in blk:
        p.getparent().remove(p)
    for p in reversed(blk):
        target.addnext(p)
    ed.save()
    print("wave-4 fixes applied; Fig. 2 re-anchored")


if __name__ == "__main__":
    main(sys.argv[1])

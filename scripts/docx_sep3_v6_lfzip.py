# -*- coding: utf-8 -*-
"""Dense LFZip sweep and its consequences for the September 3 build (v6).

Six requested items, plus the two prose corrections the new measurements
force:

1. "Section 5" in the error-bound discussion becomes "Section V".
2. Section V-M said "we sweep both codecs densely" while LFZip was only three
   matched-tolerance operating points. LFZip is now swept over 15 tolerances
   per UCI dataset and 8 on MetroPT-3 (scripts/lfzip_sweep.py, every pointwise
   bound verified), so the sentence becomes "all three codecs".
3. Figs. 9 and 10 drop the Rounded Delta series to declutter; it stays in
   Table IV. SAX stays: it is a single marker and contributes no clutter.
4. Legends gain framealpha 0.95. Fig. 11's legend then covered three TRACQ
   points, so the two panels now share one legend in panel (a), placed below
   the smallest measurement rather than on top of the data.
5. One LFZip sentence joins the MetroPT-3 rate-distortion paragraph, paid for
   by folding the archival operating point into the sentence that already
   states it and by trimming a repeated visual-interpretability claim, so the
   paragraph does not grow.
6. The Fig. 15 competitor curve now interpolates the dense LFZip sweep, which
   changes the measured result: LFZip is the smaller artifact at five of the
   twelve Air Quality accuracy levels (by up to 1.3x) and at two of eleven on
   Metro Traffic. Two sentences in Section V-M are corrected to say so. The
   ZFP and SZ3 medians quoted elsewhere are pairwise and are unchanged, as is
   the nine-operating-point comparison behind Table IV.

Usage: python docx_sep3_v6_lfzip.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

EDITS = [
    # 1. section reference style
    ("Section 5 verifies that every measured maximum error respects it.",
     "Section V verifies that every measured maximum error respects it."),

    # 2. LFZip is now swept as densely as ZFP and SZ3
    ("so we sweep both codecs densely, interpolate their measured curves onto "
     "a common RMSE grid",
     "so we sweep all three codecs densely, interpolate their measured curves "
     "onto a common RMSE grid"),

    # 5. bank the space for the LFZip sentence inside the same paragraph
    ("at ratio 0.030 it reaches RMSE = 0.0012 with correlation 1.000, and at "
     "ratio 0.0016",
     "at ratio 0.030 it reaches RMSE = 0.0012 (SMAPE 0.0053) with correlation "
     "1.000, and at ratio 0.0016"),
    (", with opaque binary outputs rather than visually inspectable artifacts.",
     ", with opaque binary outputs."),

    # 5. the LFZip sentence itself, parallel to the SZ3 sentence beside it
    ("its artifacts are a median of 1.9\u00d7 larger than TRACQ's on this "
     "stream.",
     "its artifacts are a median of 1.9\u00d7 larger than TRACQ's on this "
     "stream. The LFZip sweep sits further out still, a median of 3.7\u00d7 "
     "larger over the accuracy range it covers."),

    # 6. Fig. 15 now includes LFZip in the competitor envelope
    ("At equal RMSE, TRACQ produces the smaller artifact across essentially "
     "the entire overlap band against both error-bounded compressors.",
     "At equal RMSE, TRACQ produces the smaller artifact across essentially "
     "the entire overlap band against both scientific compressors."),
    ("LFZip provides the most direct time-series-specific comparison: across "
     "the three UCI datasets, TRACQ is smaller at eight of nine closely "
     "matched-error operating points, with advantages of 1.21\u00d7\u20131.91"
     "\u00d7; the tightest Air Quality setting is effectively a tie.",
     "LFZip provides the most direct time-series-specific comparison, and the "
     "dense sweep is where it competes best: TRACQ's median advantage is "
     "1.1\u00d7 on Air Quality, 1.2\u00d7 on Appliances Energy and Metro "
     "Traffic, and 3.7\u00d7 on MetroPT-3, but LFZip is the smaller artifact "
     "at five of the twelve Air Quality accuracy levels, by up to 1.3\u00d7."),
]

# the deleted sentence restated four numbers the paragraph already gives
DROP = ("RMSE= 0.0012, SMAPE= 0.0053, and correlation 1.000 at ratio 0.030 "
        "define its archival operating point. ")


def main(tree):
    doc = os.path.join(tree, "word", "document.xml")
    for old, new in EDITS + [(DROP, "")]:
        ed = Editor(doc)      # fresh pass per edit: the cursor is forward-only
        ed.replace(old, new)
        ed.save()
    print("v6 edits applied; Figs. 9, 10, 11 and 15 re-render separately")


if __name__ == "__main__":
    main(sys.argv[1])

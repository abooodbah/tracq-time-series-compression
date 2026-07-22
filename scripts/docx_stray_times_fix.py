# -*- coding: utf-8 -*-
"""Delete two stray multiplication-sign math remnants left behind by earlier
numeric edits in the Discussion bullets (rendered as doubled or leading x).

Usage: python docx_stray_times_fix.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

ed = Editor(os.path.join(sys.argv[1], "word", "document.xml"))
ed.del_inline_math("flag anomalous windows at 21", count=1)
ed.del_inline_math("Compression ratio trade-off", count=1)
ed.save()
print("STRAY TIMES FIX DONE")

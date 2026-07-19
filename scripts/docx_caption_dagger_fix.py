# -*- coding: utf-8 -*-
"""Remove the orphaned quoted dagger from the Table V caption: with the
container-overhead footnote retired and real SZ3 values in every cell, no
dagger remains in the table for the caption to define.

Usage: python docx_caption_dagger_fix.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

ed = Editor(os.path.join(sys.argv[1], "word", "document.xml"))
ed.del_inline_math("analyzed separately in Section", count=1, skip=1)
ed.replace("“” For Appliances", "For Appliances")
ed.save()
print("CAPTION DAGGER FIX DONE")

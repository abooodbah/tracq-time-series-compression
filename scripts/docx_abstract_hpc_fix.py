# -*- coding: utf-8 -*-
"""Align the abstract's lead-in with the three-codec results: the sentence
quotes SZ3 medians, so the exemplar list must name SZ3 alongside ZFP.

Usage: python docx_abstract_hpc_fix.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

ed = Editor(os.path.join(sys.argv[1], "word", "document.xml"))
ed.replace("tools such as ZFP with smaller artifacts",
           "tools such as ZFP and SZ3 with smaller artifacts")
ed.save()
print("ABSTRACT FIX DONE")

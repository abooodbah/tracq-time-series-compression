# -*- coding: utf-8 -*-
"""State the local benchmark platform once, ahead of the node-scale result, so
every throughput figure in the paper is tied to explicit hardware.

Usage: python docx_testbed_pass.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

TREE = sys.argv[1]
ed = Editor(os.path.join(TREE, "word", "document.xml"))
ed.replace(
    "streaming claim. The same windowed",
    "streaming claim. Single-node throughput figures in this section are measured in a "
    "single-threaded Python 3.12/NumPy 1.26 process on an Intel Core i7-13620H laptop. "
    "The same windowed")
ed.save()
print("TESTBED PASS DONE")

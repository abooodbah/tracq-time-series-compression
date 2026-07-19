# -*- coding: utf-8 -*-
"""Append the measured terabyte-scale node-parallel result (TACC Stampede3,
job on one 112-core Sapphire Rapids node) to the streaming paragraph.

Usage: python docx_tb_pass.py <tree_root>
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

TREE = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
TB = json.load(open(os.path.join(HERE, "..", "paper_results", "lattice", "tb_result.json")))

agg_gbs = TB["aggregate_mbs"] / 1000.0
sentence = (
    " The same windowed encoder also scales across cores without modification: on a "
    "112-core Sapphire Rapids node of TACC Stampede3, it processed a 1 TB synthetic "
    "stream as {nw:,} independent windows in {wall:.0f} s, sustaining {agg:.1f} GB/s in "
    "aggregate ({core:.0f} MB/s per core) while the peak resident set of every worker "
    "process stayed below {rss:.0f} MB."
).format(
    nw=TB["n_windows"],
    wall=TB["wall_s"],
    agg=agg_gbs,
    core=TB["per_core_mbs"],
    rss=TB["worker_rss_mb_max"] + 1,
)

ed = Editor(os.path.join(TREE, "word", "document.xml"))
ed.replace("supporting the constant-memory streaming claim.",
           "supporting the constant-memory streaming claim." + sentence)
ed.save()
print("TB PASS DONE")

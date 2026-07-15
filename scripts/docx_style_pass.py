# -*- coding: utf-8 -*-
"""Third tracked pass: style corrections on inserted prose."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

DOC = (r"C:\Users\ABDULF~1\AppData\Local\Temp\claude"
       r"\C--Users-Abdulfatah--codex-skills-multi-agent-coordinator-v2"
       r"\98957d2f-346c-4497-89c3-63d184c5c33c\scratchpad\docx_work\unpacked\word\document.xml")

ed = Editor(DOC)
R = ed.replace

# abstract: drop the dangling ", at any horizon" appendage
R("which confines the error of every sample to half a lattice step under a user-specified tolerance, "
  "at any horizon.",
  "which confines the error of every sample to half a lattice step under a user-specified tolerance "
  "at any horizon.")
# abstract: fix the double-"at" collision around the inline multiplication sign
R("0.75 at 21", "0.75 while running at 21")

# 3.3: "playing the role that ... play" repetition
R("The tolerance is therefore the framework’s single user-facing parameter, playing the role that "
  "error bounds play in SZ3 and ZFP.",
  "The tolerance is therefore the framework’s single user-facing parameter, and it serves the same "
  "purpose as the error bounds of SZ3 and ZFP.")
# 3.3: grammar ("without either affecting its neighbors")
R("Quiet channels receive fine steps and volatile channels receive proportionally coarser ones, "
  "without either affecting its neighbors.",
  "Quiet channels receive fine steps and volatile channels receive proportionally coarser ones, and "
  "neither choice affects the other channels.")

# 3.5: casual "go to"
R("a reserved escape pixel marks them and their exact values go to a compact sidecar",
  "a reserved escape pixel marks them and their exact values are stored in a compact sidecar")

# 5.8: grandiose "fidelity class previously reserved for"
R(", a fidelity class previously reserved for HPC compressors, at half their size and with a visual "
  "artifact.",
  ", an error level that ZFP and SZ3 reach only at more than twice the size, and the artifact "
  "remains a viewable image.")

# 5.10: editorializing aside
R("relative fidelity does not depend on a channel’s amplitude, which is the practical meaning of a "
  "relative error bound.",
  "relative fidelity does not depend on a channel’s amplitude, as a relative error bound requires.")

# conclusion: cleft construction
R("the transform domain is what delivers uniform relative fidelity",
  "the transform domain delivers uniform relative fidelity")

misses = ed.save()
print("STYLE PASS DONE")

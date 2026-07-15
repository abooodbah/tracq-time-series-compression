# -*- coding: utf-8 -*-
"""Remove defensive phrasing; assert the same facts plainly.
Usage: python docx_defensiveness_pass.py <path-to-document.xml> [v3]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

DOC = sys.argv[1]
IS_V3 = len(sys.argv) > 2 and sys.argv[2] == "v3"

ed = Editor(DOC)
R = ed.replace

# Sec 2.2.2 — old trade-off framing now contradicts the measured results
R("this trades some compression efficiency for interpretability and easier integration into "
  "monitoring pipelines.",
  "this preserves interpretability and eases integration into monitoring pipelines.")

# Sec 3 intro — drop the denial construction, keep the positive statement
R("The enhancements do not bolt new machinery onto the base framework; they come from restructuring "
  "its algorithmic flow while keeping the idea intact.",
  "The enhancements come from restructuring the base framework’s algorithmic flow while keeping the "
  "idea intact.")

# Sec 3.4 — state the zero-limit as a property, not a rebuttal
R("where a purely relative guarantee is unattainable for any method. The relative guarantee is "
  "therefore complete only above the per-variable scale, and we report it as a mixed bound "
  "throughout; Section 5 verifies that every measured maximum error respects it.",
  "since a relative tolerance loses meaning at zero. The guarantee is therefore relative above the "
  "per-variable scale and absolute below it, reported as a mixed bound throughout; Section 5 "
  "verifies that every measured maximum error respects it.")

# Limitations — drop the proof-of-a-negative tail
R("The enhanced variant removes this failure mode at the source, since the arcsinh domain is defined "
  "at and across zero, and none of the real-world experiments above required offsetting or any "
  "other stabilization step.",
  "The enhanced variant removes this failure mode at the source, since the arcsinh domain is defined "
  "at and across zero.")

# Sec 6.1 — positive scoping instead of pre-emptive disclaimer
R("The method is not intended as a universal replacement for all time-series compression methods. "
  "Its design is particularly well-matched to scenarios where:",
  "The method targets a specific operating regime. Its design is particularly well-matched to "
  "scenarios where:")

if IS_V3:
    # v3 Generalization paragraph — observation instead of assertion
    R("No single model or workload drives the result.",
      "The margins are consistent across models and workloads.")

misses = ed.save()
print("DEFENSIVENESS PASS DONE", "v3" if IS_V3 else "v2")

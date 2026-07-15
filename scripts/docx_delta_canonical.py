# -*- coding: utf-8 -*-
"""Second tracked pass: align Air Quality / Metro numbers with the canonical
data build from GTC-research (the build the paper and its SZ3 runs used)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

DOC = (r"C:\Users\ABDULF~1\AppData\Local\Temp\claude"
       r"\C--Users-Abdulfatah--codex-skills-multi-agent-coordinator-v2"
       r"\98957d2f-346c-4497-89c3-63d184c5c33c\scratchpad\docx_work\unpacked\word\document.xml")

ed = Editor(DOC)
R = ed.replace

# ---- Table V: air-quality column back to canonical-build values ----
ed.seek("Real-World Dataset Comparison")
for old, new in [
    ("0.238", "0.241"),                      # Delta+Zstd air ratio
    ("0.187", "0.188"), ("43.3", "502.9"), ("0.997", "0.689"),   # Base 16b air
    ("0.048", "0.036"), ("6.67", "9.14"),    # Enh (0.01) air
    ("0.094", "0.079"), ("0.66", "0.90"),    # Enh (0.001) air
    ("0.208", "0.160"), ("0.065", "0.091"),  # Enh (0.0001) air
    ("148.3", "193.0"), ("0.964", "0.945"),  # PAA air
    ("724.6", "714.5"), ("0.095", "0.172"),  # SAX air
]:
    R(old, new)

# ---- Air Quality prose bullet ----
R("= 6.67 at ratio 0.048, against PAA’s 148 at 0.013",
  "= 9.14 at ratio 0.036, against PAA’s 193 at 0.013")
R("= 0.66 with correlation 1.000 at ratio 0.094",
  "= 0.90 with correlation 1.000 at ratio 0.079")
R("PAA (148, correlation 0.964)", "PAA (193, correlation 0.945)")
R("configuration (43, correlation 0.997)", "configuration (503, correlation 0.689)")

misses = ed.save()
print("DELTA DONE")

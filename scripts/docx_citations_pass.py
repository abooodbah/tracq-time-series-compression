# -*- coding: utf-8 -*-
"""Fourth tracked pass: attribute component techniques ([40]-[43]) and add the
in-text citations."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

DOC = (r"C:\Users\ABDULF~1\AppData\Local\Temp\claude"
       r"\C--Users-Abdulfatah--codex-skills-multi-agent-coordinator-v2"
       r"\98957d2f-346c-4497-89c3-63d184c5c33c\scratchpad\docx_work\unpacked\word\document.xml")

ed = Editor(DOC)
R = ed.replace

# Sec 2.3 — the uniform+entropy-coding near-optimality result is Gish-Pierce,
# not Lloyd/Max; repoint the claim
R("and classical results [28, 29] indicate that a uniform quantizer paired with entropy coding "
  "forfeits little against the optimal non-uniform design.",
  "and classical analysis shows that a uniform quantizer paired with entropy coding forfeits little "
  "against the optimal non-uniform design [40].")

# Sec 3.4 — arcsinh lineage
R("The transform is monotone, symmetric, and defined for zero and negative values, which removes "
  "the need for baseline offsetting altogether.",
  "The transform is monotone, symmetric, and defined for zero and negative values [42], which "
  "removes the need for baseline offsetting altogether.")

# Sec 3.5 — delta-of-delta and Lorenzo attributions
R("a per-row predictor (temporal difference, second difference, seasonal-lag difference, or a "
  "two-dimensional predictor on correlation-ordered rows) is selected by residual entropy",
  "a per-row predictor (temporal difference, second difference [27], seasonal-lag difference, or a "
  "two-dimensional Lorenzo predictor [41] on correlation-ordered rows) is selected by residual "
  "entropy")

# Sec 3.6 — closed-loop predictive quantization lineage via existing [16]
R("regardless of sequence length. Section 5 verifies this bound",
  "regardless of sequence length; the construction is the integer-arithmetic analogue of closed-loop "
  "predictive quantization [16]. Section 5 verifies this bound")

# Sec 5.12 — Isolation Forest citation at first use
R("then run an unsupervised Isolation Forest.",
  "then run an unsupervised Isolation Forest [43].")

# References [40]-[43], inserted after [39] in the paper's reference format
ed.insert_paragraph_after(
    "[39] Alliance for Open Media",
    [("[40] Gish, H. and J. N. Pierce, “Asymptotically efficient quantizing,” ", False),
     ("IEEE Transactions on Information Theory", True),
     (", vol. 14, no. 5, 1968.", False)])
ed.insert_paragraph_after(
    "[40] Gish, H.",
    [("[41] Ibarria, L., P. Lindstrom, J. Rossignac, and A. Szymczak, “Out-of-core compression "
      "and decompression of large n-dimensional scalar fields,” ", False),
     ("Computer Graphics Forum", True),
     (", vol. 22, no. 3, 2003.", False)])
ed.insert_paragraph_after(
    "[41] Ibarria, L.",
    [("[42] Burbidge, J. B., L. Magee, and A. L. Robb, “Alternative transformations to handle "
      "extreme values of the dependent variable,” ", False),
     ("Journal of the American Statistical Association", True),
     (", vol. 83, no. 401, 1988.", False)])
ed.insert_paragraph_after(
    "[42] Burbidge, J. B.",
    [("[43] Liu, F. T., K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in ", False),
     ("IEEE international conference on data mining (ICDM)", True),
     (", 2008.", False)])

misses = ed.save()
print("CITATIONS PASS DONE")

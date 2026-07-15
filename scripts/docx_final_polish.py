# -*- coding: utf-8 -*-
"""Pass 6: reduce 'lattice' usage, state the exact mixed error guarantee, and
explain the enhanced variant as a restructuring of the base algorithmic flow."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, mr, msub

DOC = (r"C:\Users\ABDULF~1\AppData\Local\Temp\claude"
       r"\C--Users-Abdulfatah--codex-skills-multi-agent-coordinator-v2"
       r"\98957d2f-346c-4497-89c3-63d184c5c33c\scratchpad\docx_work\unpacked\word\document.xml")

ed = Editor(DOC)
R = ed.replace

# ---- abstract: 3 -> 1 ----
R("(2) integer coding of lattice differences that concentrates the grid near mid-gray",
  "(2) integer coding of the resulting differences that concentrates the grid near mid-gray")
R("which confines the error of every sample to half a lattice step under a user-specified tolerance",
  "which confines the error of every sample to half a quantization step under a user-specified "
  "tolerance")

# ---- contributions ----
R("as integer steps on a per-variable quantization lattice, so reconstruction reduces to exact "
  "integer accumulation.",
  "as integer multiples of a per-variable quantization step, so reconstruction reduces to exact "
  "integer accumulation.")
R("We derive each channel’s lattice step from a user-specified error tolerance",
  "We derive each channel’s quantization step from a user-specified error tolerance")
R("deviates from its original by at most half a lattice step over arbitrarily long sequences.",
  "deviates from its original by at most half a quantization step over arbitrarily long sequences.")
R("through lattice quantization and integer reconstruction.",
  "through integer quantization and reconstruction.")

# ---- related work ----
R("followed by uniform lattice quantization: resolution concentrates near zero",
  "followed by uniform quantization: resolution concentrates near zero")

# ---- section 3 intro: fewer lattices + genesis of the enhanced variant ----
R("We then describe three enhancements: a per-variable arcsinh transform, integer-lattice "
  "quantization with per-variable steps, and predictor selection on the resulting integer grid.",
  "We then describe three enhancements: a per-variable arcsinh transform, integer quantization with "
  "per-variable steps, and predictor selection on the resulting integer grid. The enhancements do "
  "not bolt new machinery onto the base framework; they come from restructuring its algorithmic "
  "flow while keeping the idea intact. Each limitation of the base design traces to where an "
  "operation sits in the pipeline rather than to the idea itself: differencing before quantization "
  "places quantization error inside the accumulation path, so the enhanced variant quantizes first "
  "and differences the resulting integers; the ratio, clamp, and companding stages collapse into "
  "one transform whose uniform quantization performs the same resolution allocation; and the "
  "percentile heuristics that selected clamp ranges become a single error tolerance that fixes "
  "every step size. The artifact — a per-timestep grid of relative changes — is unchanged.")

# ---- 3.3: qualify the relative-tolerance claim ----
R("while in relative mode a single step in the transform domain yields a uniform relative tolerance "
  "for every channel.",
  "while in relative mode a single step in the transform domain yields a uniform relative tolerance "
  "for every channel above its scale.")

# ---- 3.4: precise mixed guarantee ----
R("so a uniform lattice step in the transform domain yields near-uniform absolute resolution",
  "so a uniform quantization step in the transform domain yields near-uniform absolute resolution")
R("For time-series that resemble Laplacian or heavy-tailed distributions, this matches the signal "
  "structure well.",
  "For time-series that resemble Laplacian or heavy-tailed distributions, this matches the signal "
  "structure well. The resulting guarantee is stated exactly: a transform-domain error of at most "
  "q/2 bounds every reconstructed value by")
ed.insert_paragraph_after(
    "The resulting guarantee is stated exactly",
    [("For values well above the scale the bound is relative, approaching e", False),
     ("q/2", False),
     (" − 1 = ε; as the value approaches zero it degrades gracefully to the absolute floor ε·s", False),
     ("i", False),
     (", where a purely relative guarantee is unattainable for any method. The relative guarantee "
      "is therefore complete only above the per-variable scale, and we report it as a mixed bound "
      "throughout; Section 5 verifies that every measured maximum error respects it.", False)])
ed.insert_math_after(
    "The resulting guarantee is stated exactly",
    [mr("|x̂ − x| ≤ (cosh(q/2) − 1)·|x| + sinh(q/2)·"), mr("√(x² + s²)")])

# ---- 3.5 ----
R("and store the grid of consecutive lattice differences:",
  "and store the grid of consecutive integer differences:")

# ---- 3.6 ----
R("its grid stores integer lattice differences, and reconstruction accumulates them exactly,",
  "its grid stores exact integer differences, and reconstruction accumulates them exactly,")
R("cumulative summation reproduces every lattice coordinate identically",
  "cumulative summation reproduces every quantized coordinate identically")

# ---- 3.7 metadata ----
R("Baseline values and initial lattice coordinates  for all variables.",
  "Baseline values and initial integer coordinates  for all variables.")
R("Per-variable lattice steps and transform scales (when the enhanced variant is used)",
  "Per-variable quantization steps and transform scales (when the enhanced variant is used)")

# ---- 4.2 configurations ----
R("+Lattice: the base configuration with quantization moved to the per-variable integer lattice "
  "(temporal-difference predictor, absolute bound).",
  "+Lattice: the base configuration with quantization moved to per-variable integer levels "
  "(temporal-difference predictor, absolute bound).")
R("+Predictors: the lattice configuration plus per-row predictor selection by residual entropy",
  "+Predictors: the preceding configuration plus per-row predictor selection by residual entropy")
R("Enhanced TRACQ: the full enhanced configuration with per-variable lattice steps, predictor "
  "selection, escape coding, and either the arcsinh transform (relative bound) or the linear "
  "lattice (absolute bound).",
  "Enhanced TRACQ: the full enhanced configuration with per-variable quantization steps, predictor "
  "selection, escape coding, and either the arcsinh transform (relative bound) or the linear "
  "mapping (absolute bound).")

# ---- results ----
R("The enhanced configuration (per-variable lattice + predictor selection) consistently reduces",
  "The enhanced configuration (per-variable integer coding + predictor selection) consistently "
  "reduces")
R("encoding relative changes on per-variable lattices and aligning quantization resolution",
  "encoding relative changes as per-variable integer steps and aligning quantization resolution")
R("We compare the base configuration, lattice coding alone, lattice coding with predictor "
  "selection, and the full enhanced configuration",
  "We compare the base configuration, the +Lattice configuration, the same configuration with "
  "predictor selection, and the full enhanced configuration")
R("Fig. 2. Multi-scale handling: mean relative error vs. variable scale. Per-variable lattice "
  "steps (green, purple)",
  "Fig. 2. Multi-scale handling: mean relative error vs. variable scale. Per-variable quantization "
  "steps (green, purple)")
R("Per-variable lattice steps reduce mean relative error from 39.0% to 0.022%",
  "Per-variable step selection reduces mean relative error from 39.0% to 0.022%")

# ---- discussion ----
R("Per-variable lattice steps naturally allocate resolution where needed.",
  "Per-variable quantization steps naturally allocate resolution where needed.")
R("each encoded from its own initial lattice coordinates.",
  "each encoded from its own initial integer coordinates.")
R("The transform, lattice quantization, differencing, and reconstruction each require",
  "The transform, quantization, differencing, and reconstruction each require")
R("one step, one scale, and one initial lattice coordinate per variable",
  "one step, one scale, and one initial integer coordinate per variable")

# ---- conclusion ----
R("quantizes an arcsinh-transformed signal onto per-variable integer lattices and stores the grid "
  "of lattice differences.",
  "quantizes an arcsinh-transformed signal onto per-variable integer levels and stores the grid of "
  "their differences.")
R("per-variable lattice steps are critical for multi-scale data",
  "per-variable quantization steps are critical for multi-scale data")

misses = ed.save()
print("POLISH PASS DONE")

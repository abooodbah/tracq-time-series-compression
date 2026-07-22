# -*- coding: utf-8 -*-
"""Reposition the paper's thesis: the contribution is the composition — one
stored artifact that is error-bounded, drift-free, image-renderable, and
analyzable in integer form — with the rate-distortion results serving as
evidence that this interpretability carries no size premium, rather than as a
state-of-the-art claim.

Usage: python docx_framing_pass.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor

EDITS = [
    # abstract: plant the thesis in the framework sentence
    ("followed by image-based encoding. The method combines three core mechanisms:",
     "followed by image-based encoding, so that one stored artifact is simultaneously "
     "error-bounded, drift-free, renderable as a standard image, and analyzable without "
     "floating-point reconstruction. The method combines three core mechanisms:"),
    # abstract: numbers become evidence for the property, not the thesis
    ("In rate-distortion space, TRACQ reaches the accuracy class",
     "This interpretability carries no rate-distortion premium on the evaluated workloads: "
     "TRACQ reaches the accuracy class"),
    ("do not reach, while preserving a key structural property: the quantized grid renders "
     "as a standard image, supporting rapid visual inspection and anomaly screening.",
     "do not reach."),
    # intro: the composition sentence ahead of the contribution list
    ("This paper makes five concrete contributions:",
     "Each individual mechanism in the pipeline is classical — companding, uniform scalar "
     "quantization, differential coding of integer indices, and entropy coding; the "
     "contribution lies in their composition, which makes one stored artifact error-bounded, "
     "drift-free at any horizon, image-renderable, and analyzable in integer form. This "
     "paper makes five concrete contributions:"),
    # related work: concede the components where reviewers look for prior art
    ("By contrast, TRACQ produces images that can be inspected visually while still "
     "supporting numerical reconstruction",
     "Companding transforms for relative-error control and closed-loop predictive "
     "quantization are likewise established techniques within this family. TRACQ differs "
     "not in any one mechanism but in their composition: it produces grids that render as "
     "images and can be analyzed in integer form while still supporting numerical "
     "reconstruction"),
    # method genesis: drop the self-justifying clause
    ("traces to where an operation sits in the pipeline rather than to the idea itself:",
     "traces to where an operation sits in the pipeline:"),
    # implementation notes: one sentence of scope instead of three of rebuttal
    ("These comparisons therefore mix language runtimes: the TRACQ encoder pays Python "
     "interpreter and NumPy dispatch overhead on every stage, while ZFP and SZ3 execute as "
     "optimized native C/C++ code behind thin bindings. The throughput figures show that a "
     "prototype keeps pace with typical ingestion paths; they do not isolate algorithmic "
     "cost from runtime overhead, and a controlled comparison would require the compiled "
     "implementation discussed below.",
     "These comparisons mix language runtimes — the TRACQ encoder pays interpreter overhead "
     "while ZFP and SZ3 run as native code — so the throughput figures show that a "
     "prototype keeps pace with typical ingestion paths rather than isolating algorithmic "
     "cost."),
    # conclusion: scope the accuracy-class sentence like the evidence
    ("with smaller artifacts at matched accuracy while preserving interpretability",
     "with smaller artifacts at matched accuracy on the evaluated workloads while "
     "preserving interpretability"),
]

# present only in the July-15 lineage; skip silently where absent
OPTIONAL = {"These comparisons therefore mix language runtimes: the TRACQ encoder pays Python "}


def variants(s):
    yield s
    if "'" in s:
        yield s.replace("'", "’")


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    full = "\n".join(ed.para_text(p) for p in ed.paras)
    claimed, resolved, missing = set(), [], []
    for old, new in EDITS:
        hit = None
        for v in variants(old):
            start = 0
            while True:
                p = full.find(v, start)
                if p < 0:
                    break
                if p not in claimed:
                    hit = (p, v, new)
                    break
                start = p + 1
            if hit:
                break
        if hit:
            claimed.add(hit[0])
            resolved.append(hit)
        elif any(old.startswith(op) for op in OPTIONAL):
            print("  (optional edit absent in this lineage, skipped: %s...)" % old[:50])
        else:
            missing.append(old[:70])
    if missing:
        print("DRY-RUN MISSING (aborting, no edits applied):")
        for m in missing:
            print("   !!", m)
        sys.exit(1)
    for p, v, new in sorted(resolved):
        ed.replace(v, new)
    ed.save()
    print("FRAMING PASS DONE")


if __name__ == "__main__":
    main(sys.argv[1])

# -*- coding: utf-8 -*-
"""v7 review round: abstract drops internal-baseline numbers, Table I folds
into Fig. 1 (tables renumber), the drift table reads as bound verification
with the base non-monotonicity explained, the intro composition sentence and
the leaked editorial note go away, the vestigial blend/percentile equations
are removed, tolerance labels finish converting to decimals, the roadmap
references are corrected, the unsupported SZ3 plateau claim is dropped, and
every stray math remnant is tracked-deleted.

Usage: python docx_v7_pass.py <tree_root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q

M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def in_del(el):
    p = el.getparent()
    while p is not None:
        if p.tag == q("w:del"):
            return True
        p = p.getparent()
    return False


def wrap_del(p, om, n):
    d = p.makeelement(q("w:del"), {q("w:id"): str(9800 + n),
                                   q("w:author"): "Abdulfatah Bahbouh",
                                   q("w:date"): "2026-08-02T00:00:00Z"})
    om.addprevious(d)
    d.append(om)


def msig(om):
    return "".join(t.text or "" for t in om.iter(M + "t")).replace(" ", "").replace(" ", "")


# (anchor substring, list of sigs to kill in order of appearance)
KILL = [
    ("strains storage and transmission systems", ["=", "="]),
    ("Even small quantization errors in", ["P"]),
    ("SMAPE: Symmetric Mean Absolute Percentage Error", ["[0,2]"]),
    ("Lattice coding alone yields", ["×"]),
    ("Predictor selection leaves RMSE unchanged", ["×"]),
    ("constant-SMAPE floor", ["10-1", "10-1", "10-4"]),
    ("The critical systems-level result is throughput", ["×"]),
    ("throughput gain represents the systems benefit", ["×"]),
    ("not merely interpretable but machine-actionable", ["×"]),
    ("so that the reconstruction of every transformed sample", ["blendi"]),
    ("Quiet channels receive fine steps", ["ri", "ri"]),
    ("distributing quantization levels evenly", ["[-1,1]"]),
]

EDITS = [
    # abstract: no internal-baseline numbers
    ("achieves a 99.8–99.99% RMSE reduction over the base configuration, handles "
     "multi-scale variables",
     "handles multi-scale variables"),
    ("and eliminates long-horizon drift: on 10,000-step sequences the base configuration "
     "accumulates RMSE",
     "and eliminates long-horizon drift: reconstruction error stays within the "
     "user-specified tolerance at every horizon, without periodic anchor corrections."),
    ("212, while the enhanced reconstruction stays at RMSE", " "),
    ("0.25 at every horizon, without periodic anchor corrections. This interpretability",
     "This interpretability"),
    # intro: composition sentence out
    ("Each individual mechanism in the pipeline is classical companding, uniform scalar "
     "quantization, differential coding of integer indices, and entropy coding; the "
     "contribution lies in their composition, which makes one stored artifact "
     "error-bounded, drift-free at any horizon, image-renderable, and analyzable in "
     "integer form. This paper makes five concrete contributions:",
     "This paper makes five concrete contributions:"),
    # roadmap
    ("Section 2 situates", "Section II situates"),
    ("and compares against ZFP. Section 7 summarizes the main insights, discusses "
     "limitations, and outlines directions for extending the framework",
     "and compares against ZFP and SZ3. Section VI discusses when to use the framework, "
     "and Section VII concludes with directions for extending it"),
    # III-C: motivation gaps and a self-contained tolerance sentence
    ("so that the reconstruction of every transformed sample is guaranteed",
     "The enhanced design instead derives every step directly from the user tolerance, "
     "so that the reconstruction of every transformed sample is guaranteed"),
    # base-drift paragraph gap
    ("Even small quantization errors in  accumulate over time",
     "Even small quantization errors in the percentage grid accumulate over time"),
    # SMAPE metric range as text
    ("This metric is bounded in  and is", "This metric is bounded in [0, 2] and is"),
    # V-A now cites the figure instead of the deleted table
    ("Table 1 summarizes RMSE across data types for 8-bit quantization.",
     "Figure 1 summarizes RMSE across configurations and data types for 8-bit "
     "quantization."),
    # drift table reframed as verification
    ("The enhanced variant reduces end-of-sequence RMSE from 212 (base configuration) to "
     "0.25 (an 840× improvement), and the error at 1,000, 5,000, and 10,000 steps is "
     "identical to two decimal places: reconstruction error does not grow with the "
     "horizon. The dashed line in Fig. 3 is the analytical bound implied by the "
     "tolerance; the measured error remains below it at every step, with no periodic "
     "anchor corrections required.",
     "Table II and Fig. 3 read as a verification of the bound: the enhanced error at "
     "1,000, 5,000, and 10,000 steps is identical to two decimal places and stays below "
     "the dashed analytical bound implied by the tolerance at every step, with no "
     "periodic anchor corrections required. The base error is not monotone, since "
     "multiplicative drift can partially cancel when successive percentage changes "
     "reverse sign, but it remains three orders of magnitude above the bound. The "
     "full-length MetroPT-3 run in Section V-I extends the same verification to 1.5 "
     "million steps."),
    # ZFP SMAPE floor: decimals, and drop the unsupported SZ3 sentence
    ("ZFP (tol=) achieves global SMAPE", "ZFP (0.1) achieves a global SMAPE of"),
    ("across tolerance settings ( through )",
     "across tolerance settings (0.1 through 0.0001)"),
    (" SZ3 does not exhibit this plateau (SMAPE decreases with tighter tolerance), "
     "likely due to its prediction-based architecture.", " "),
    # IV-B: what Enh. labels denote
    ("or the linear mapping (absolute bound). Unless otherwise stated",
     "or the linear mapping (absolute bound). In the result tables, Enh. (0.01) and "
     "similar labels always denote this full configuration at the given tolerance in "
     "absolute mode. Unless otherwise stated"),
    # renumbering after Table I folds into Fig. 1
    ("Table II and Fig. 2 examine", "Table I and Fig. 2 examine"),
    ("TABLE II. Multi-Scale Data (scales", "TABLE I. Multi-Scale Data (scales"),
    ("Table III and Fig. 3 examine", "Table II and Fig. 3 examine"),
    ("TABLE III. Error Drift", "TABLE II. Error Drift"),
    ("Table IV summarizes the 8-bit", "Table III summarizes the 8-bit"),
    ("TABLE IV. 8-bit Compression", "TABLE III. 8-bit Compression"),
    ("Table V summarizes the key", "Table IV summarizes the key"),
    ("TABLE V. Real-World Dataset Comparison.", "TABLE IV. Real-World Dataset Comparison."),
    ("Table VI summarizes detection", "Table V summarizes detection"),
    ("TABLE VI. Anomaly Detection", "TABLE V. Anomaly Detection"),
]

OPTIONAL = {
    "whose percentage changes rarely exceed  receives the same resolution as one that "
    "frequently spikes to .",
}
EDITS.append(("whose percentage changes rarely exceed  receives the same resolution as "
              "one that frequently spikes to .",
              "whose percentage changes rarely exceed a few percent receives the same "
              "resolution as one that spikes by hundreds of percent."))


def variants(s):
    yield s
    if "'" in s:
        yield s.replace("'", "’")


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    n = 0

    # --- structural: fold Table I away (caption + the table itself) ---
    for p in list(ed.paras):
        if ed.para_text(p).strip().startswith("TABLE I. 8-bit Quantization RMSE"):
            sib = p.getnext()
            while sib is not None and sib.tag != q("w:tbl"):
                nxt = sib.getnext()
                sib = nxt
            if sib is not None:
                sib.getparent().remove(sib)
            p.getparent().remove(p)
            print("Table I removed")
            break

    # --- structural: strip the leaked note, keep its section break ---
    for p in list(ed.paras):
        if ed.para_text(p).strip().startswith("Must fix to plot compression ratio"):
            for child in list(p):
                if child.tag != q("w:pPr"):
                    p.remove(child)
            print("leaked editorial note stripped (section break kept)")
            break

    # --- structural: vestigial blend equations ---
    removed = 0
    for p in list(ed.paras):
        sigs = [msig(om) for om in p.iter(M + "oMath") if not in_del(om)]
        if any(s.startswith("pi&=percentile") for s in sigs) or \
           any(s.startswith("Ci=minCmax") for s in sigs):
            if not ed.para_text(p).strip():
                p.getparent().remove(p)
                removed += 1
    print(f"blend equation paragraphs removed: {removed}")

    # --- stray math kills (all targets are direct children of their w:p) ---
    for anchor, sigs in KILL:
        for p in ed.paras:
            if anchor not in ed.para_text(p):
                continue
            budget = list(sigs)
            for om in [k for k in list(p) if k.tag == M + "oMath" and not in_del(k)]:
                if budget and msig(om) == budget[0]:
                    wrap_del(p, om, n)
                    n += 1
                    budget.pop(0)
            break
    print(f"stray math elements tracked-deleted: {n}")

    # --- text edits, position-resolved ---
    full = "\n".join(ed.para_text(p) for p in ed.paras)
    claimed, resolved, missing = set(), [], []
    for old, new in EDITS:
        hit = None
        for v in variants(old):
            start = 0
            while True:
                pos = full.find(v, start)
                if pos < 0:
                    break
                if pos not in claimed:
                    hit = (pos, v, new)
                    break
                start = pos + 1
            if hit:
                break
        if hit:
            claimed.add(hit[0])
            resolved.append(hit)
        elif old in OPTIONAL:
            print("  (optional edit absent, skipped)")
        else:
            missing.append(old[:70])
    if missing:
        print("DRY-RUN MISSING (aborting, no text edits applied):")
        for m_ in missing:
            print("   !!", m_)
        sys.exit(1)
    for pos, v, new in sorted(resolved):
        ed.replace(v, new)

    ed.save()
    print("V7 PASS DONE")


if __name__ == "__main__":
    main(sys.argv[1])

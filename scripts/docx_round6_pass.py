# -*- coding: utf-8 -*-
"""Round-6 content pass, from the author's in-document notes: shrink the
Table V caption into the text, reduce the iso figure to its advantage panel,
note the SZ3 sweep now shown in Fig. 11, flip frontier language for the
compression-ratio axes, keep Table II on one page, resize the iso drawing,
and (July-15 lineage only) mirror the author's Discussion cuts.

Usage: python docx_round6_pass.py <tree_root> <iso_fig_n> [--mirror-user-edits]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q as wq

WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def build_edits(iso_n):
    return [
        ("TABLE V. Real-World Dataset Comparison. Ratio = encoded bytes / original bytes "
         "(lower is smaller). RMSE and Pearson correlation () are computed against the "
         "original float64 data. Correlations printed as 1.000 round from values of at "
         "least 0.9995. SMAPE (relative fidelity) is analyzed separately in Section V-J. "
         "For Appliances Energy, RMSE at tolerance 0.011.10 corresponds to a normalized "
         "RMSE of approximately 0.1% relative to the global data range. Boldface marks the "
         "lowest RMSE among methods encoding at or below one tenth of the original size.",
         "TABLE V. Real-World Dataset Comparison."),
        ("Table V summarizes the key metrics.",
         "Table V summarizes the key metrics. In the table, ratio is encoded bytes over "
         "original bytes, so lower is smaller; RMSE and Pearson correlation are computed "
         "against the original float64 data, and correlations printed as 1.000 round from "
         "values of at least 0.9995. SMAPE is analyzed separately in Section V-J. For "
         "Appliances Energy, the RMSE of 1.10 at tolerance 0.01 corresponds to a normalized "
         "RMSE of approximately 0.1% of the global data range. Boldface marks the lowest "
         "RMSE among methods encoding at or below one tenth of the original size."),
        ("Fig. %d. Encoded size at matched RMSE: (a) MetroPT-3; (b) advantage over the "
         "stronger of ZFP and SZ3." % iso_n,
         "Fig. %d. Size advantage at matched RMSE against the stronger of ZFP and "
         "SZ3." % iso_n),
        ("values above one in its advantage panel indicate", "values above one indicate"),
        ("with opaque binary outputs rather than visually inspectable artifacts.",
         "with opaque binary outputs rather than visually inspectable artifacts. The SZ3 "
         "sweep in panel (a) shows the same pattern: at matched RMSE, its artifacts are a "
         "median of 1.9× larger than the enhanced variant's on this stream."),
        ("In Fig. 9 the lower-left corner is better",
         "In Fig. 9 the lower-right corner is better (more compression at lower error)"),
        ("trace the lower-left frontier on all three datasets",
         "trace the lower-right frontier on all three datasets"),
        ("traces the lower-left frontier from ratio 0.01 to 0.38",
         "traces the lower-right frontier from ratio 0.01 to 0.38"),
    ]


MIRROR = [
    ("is classical — companding", "is classical companding"),
    ("These recommendations are starting points; further tuning should reflect "
     "application-specific error tolerance and storage constraints.",
     "The error bound holds at any horizon, and these recommendations are starting "
     "points; further tuning should reflect application-specific error tolerance and "
     "storage constraints."),
]

MIRROR_DELETE = [
    "Limitations",
    "The base framework's primary numerical failure mode",
    "Our evaluation highlights several concrete limitations:",
    "The tolerance is a deployment choice.",
    "Compression ratio trade-off: The enhanced configuration achieves high fidelity",
    "Metadata overhead: Per-variable steps",
    "No temporal downsampling:",
    "These limitations indicate that the method is best used",
    "Maximum compression: The base configuration at 8 bits",
    "Long sequences: no additional mechanism is required",
]


def variants(s):
    yield s
    if "'" in s:
        yield s.replace("'", "’")


def main():
    tree = sys.argv[1]
    iso_n = int(sys.argv[2])
    mirror = "--mirror-user-edits" in sys.argv
    ed = Editor(os.path.join(tree, "word", "document.xml"))

    edits = build_edits(iso_n) + (MIRROR if mirror else [])
    full = "\n".join(ed.para_text(p) for p in ed.paras)
    claimed, resolved, missing = set(), [], []
    for old, new in edits:
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
        else:
            missing.append(old[:80])
    if missing:
        print("DRY-RUN MISSING (aborting, no edits applied):")
        for m in missing:
            print("   !!", m)
        sys.exit(1)
    for p, v, new in sorted(resolved):
        ed.replace(v, new)

    # Table V caption carried inline math (rho, equals): retire it
    ed.del_inline_math("TABLE V. Real-World Dataset Comparison.", count=9)

    if mirror:
        for snippet in MIRROR_DELETE:
            ed.cursor_p, ed.cursor_c = 0, 0
            hits = [v for v in variants(snippet)
                    if any(v in ed.para_text(p) for p in ed.paras)]
            target = hits[0] if hits else snippet
            ed.del_inline_math(target, count=9)
            ed.delete_paragraph_by_text(target)

    # iso figure drawing: single advantage panel is 1044x780 now
    import re as _re
    rels = open(os.path.join(tree, "word", "_rels", "document.xml.rels"),
                encoding="utf-8").read()
    m = _re.search(r'Id="(rId\d+)"[^>]*Target="media/image17\.png"', rels)
    iso_rid = m.group(1) if m else None
    for drawing in ed.root.iter(wq("w:drawing")):
        blips = [b for b in drawing.iter(A + "blip") if b.get(R + "embed") == iso_rid]
        if not blips:
            continue
        for ext in list(drawing.iter(WP + "extent")) + list(drawing.iter(A + "ext")):
            if ext.get("cx") is None:
                continue
            cx = int(ext.get("cx"))
            ext.set("cy", str(round(cx * 780 / 1044)))
        print("iso drawing extent updated (%s)" % iso_rid)

    # Table II: keep caption and table on one page
    paras = ed.paras
    for i, p in enumerate(paras):
        if "Multi-Scale Data (scales" in ed.para_text(p):
            ppr = p.find(wq("w:pPr"))
            if ppr is None:
                ppr = p.makeelement(wq("w:pPr"), {})
                p.insert(0, ppr)
            if ppr.find(wq("w:keepNext")) is None:
                ppr.insert(0, p.makeelement(wq("w:keepNext"), {}))
            sib = p.getnext()
            while sib is not None and sib.tag != wq("w:tbl"):
                sib = sib.getnext()
            if sib is not None:
                rows = sib.findall(wq("w:tr"))
                for r_i, tr in enumerate(rows):
                    trpr = tr.find(wq("w:trPr"))
                    if trpr is None:
                        trpr = tr.makeelement(wq("w:trPr"), {})
                        tr.insert(0, trpr)
                    if trpr.find(wq("w:cantSplit")) is None:
                        trpr.append(tr.makeelement(wq("w:cantSplit"), {}))
                    if r_i < len(rows) - 1:
                        for cp in tr.iter(wq("w:p")):
                            cppr = cp.find(wq("w:pPr"))
                            if cppr is None:
                                cppr = cp.makeelement(wq("w:pPr"), {})
                                cp.insert(0, cppr)
                            if cppr.find(wq("w:keepNext")) is None:
                                cppr.insert(0, cp.makeelement(wq("w:keepNext"), {}))
                print("Table II keep-together applied")
            break

    ed.save()
    print("ROUND 6 PASS DONE")


if __name__ == "__main__":
    main()

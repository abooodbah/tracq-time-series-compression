# -*- coding: utf-8 -*-
"""Standardize the manuscript on compression factor (v7).

The text mixed two inverse conventions: an encoded/original "ratio" (0.023)
in the tables and prose, and an original/encoded "compression factor" (43.9x)
on every figure axis. A reader who carries one convention into the other
inverts every size claim. v7 adopts the factor everywhere.

This is a reciprocal conversion, not a new measurement. Every table cell is
recomputed as 1/ratio from the same result file that produced the printed
ratio, and the script asserts the printed ratio first, so a source that no
longer matches the manuscript stops the run instead of silently rewriting a
number. Table IV's TRACQ rows are C2_bank_abs, confirmed against the RMSE
column as well as the ratio.

Prose uses the compact "at factor 43.9" once the metric is defined, which
keeps every converted sentence about the same length as the ratio it
replaces, so the 17-page layout is undisturbed.

Two printed values could NOT be tied to a source and are converted as
printed, flagged here rather than silently corrected:
  - Section V-G gives gzip on MetroPT-3 as 7.3% at 49 MB/s; figdata.json,
    which draws Fig. 8(b), has 7.65% at 48 MB/s. 7.3% -> factor 13.7 is used;
    the figure bar stays at 13.1.
  - The "9-27% of the original size" range in the recommendations matches no
    single configuration (C2_bank_abs spans 8.9-16.0%, C2_bank_rel
    9.9-28.7%). Converted as printed, to factors 3.7-11.

Usage: python docx_sep3_v7_factor.py <tree_root>
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "paper_results")
DS = ["uci_air_quality", "uci_appliances_energy", "uci_metro_traffic"]
SHORT = dict(zip(DS, ["air_quality", "appliances", "metro_traffic"]))
AIR, APP, MET = DS


def _lat(name):
    return json.load(open(os.path.join(RES, "lattice", name)))


RW = {d: json.load(open(os.path.join(RES, "realworld", d + "_results.json")))
      ["results"] for d in DS}
LR = _lat("lattice_results.json")
LZ = _lat("lfzip_results.json")
SZ = _lat("sz3_exact_tols.json")
PT = _lat("paper_tables.json")
MP = _lat("metropt3_lattice.json")


def factor(ratio):
    """1/ratio, printed to one decimal below 100 and whole at or above it."""
    f = 1.0 / ratio
    return "%.0f" % f if f >= 100 else "%.1f" % f


# Table IV, in document order: row label -> per-dataset source ratio
TABLE4 = [
    ("Gzip", lambda d: RW[d]["gzip"]["ratio"]),
    ("Delta+Zstd", lambda d: RW[d]["delta_zstd"]["ratio"]),
    ("Naive Diff. (16b)", lambda d: RW[d]["tracq_orig_16bit"]["ratio"]),
    ("TRACQ (0.01)", lambda d: LR[d]["C2_bank_abs_eps0.01"]["ratio"]),
    ("TRACQ (0.001)", lambda d: LR[d]["C2_bank_abs_eps0.001"]["ratio"]),
    ("TRACQ (0.0001)", lambda d: LR[d]["C2_bank_abs_eps0.0001"]["ratio"]),
    ("PAA-64", lambda d: RW[d]["paa"]["ratio"]),
    ("SAX-64", lambda d: RW[d]["sax"]["ratio"]),
    ("Rounded delta", lambda d: RW[d]["gorilla_like"]["ratio"]),
    ("ZFP (0.1)", lambda d: RW[d]["zfp_tol_0.1"]["ratio"]),
    ("ZFP (0.001)", lambda d: RW[d]["zfp_tol_0.001"]["ratio"]),
    ("SZ3 (0.1)", lambda d: SZ[SHORT[d]]["0.1"]["ratio"]),
    ("SZ3 (0.001)", lambda d: SZ[SHORT[d]]["0.001"]["ratio"]),
    ("LFZip (0.01)", lambda d: LZ[SHORT[d]]["0.01"]["ratio"]),
    ("LFZip (0.001)", lambda d: LZ[SHORT[d]]["0.001"]["ratio"]),
]
T4 = dict(TABLE4)

# Table III reports the same quantity as a percentage
TABLE3 = [("Naive Diff.", "base"), ("TRACQ (0.01)", "enh_eps1e-2"),
          ("TRACQ (0.001)", "enh_eps1e-3")]


def cell_text(tc):
    return "".join(tc.itertext()).strip()


def set_cell(ed, tc, new):
    """Replace a table cell's visible text with `new`, as a tracked change."""
    runs = [r for p in tc.iter(q("w:p")) for r in ed._para_runs(p)]
    assert runs, "empty cell"
    first = runs[0]
    host = first.getparent()
    idx = host.index(first)
    ins = ed._make_ins(first, new)
    for r in runs:
        ed._wrap_del(r)
    host.insert(idx, ins)


def convert_tables(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    tables = list(ed.root.iter(q("w:tbl")))

    rows = list(tables[3].iter(q("w:tr")))
    hdr = list(rows[0].iter(q("w:tc")))
    assert cell_text(hdr[1]) == "Ratio", cell_text(hdr[1])
    set_cell(ed, hdr[1], "Factor")
    for (label, key), row in zip(TABLE3, rows[1:]):
        cells = list(row.iter(q("w:tc")))
        assert cell_text(cells[0]) == label, cell_text(cells[0])
        r = PT["T4"][key]["ratio"]
        shown = float(cell_text(cells[1]).rstrip("%")) / 100.0
        assert abs(shown - r) <= 0.0006, \
            "T3 %s: doc %s vs source %.4f" % (label, cell_text(cells[1]), r)
        set_cell(ed, cells[1], factor(r))

    rows = list(tables[4].iter(q("w:tr")))
    hdr = list(rows[1].iter(q("w:tc")))
    for ci in (2, 5, 8):
        assert cell_text(hdr[ci]) == "Ratio", cell_text(hdr[ci])
        set_cell(ed, hdr[ci], "Factor")
    for (label, get), row in zip(TABLE4, rows[2:]):
        cells = list(row.iter(q("w:tc")))
        assert cell_text(cells[1]) == label, \
            "%s != %s" % (cell_text(cells[1]), label)
        for ci, d in zip((2, 5, 8), DS):
            r = get(d)
            # the doc rounds to three decimals; allow the last-place tie
            assert abs(float(cell_text(cells[ci])) - r) <= 0.0006, \
                "T4 %s/%s: doc %s vs source %.4f" % (
                    label, d, cell_text(cells[ci]), r)
            set_cell(ed, cells[ci], factor(r))
    ed.save()
    print("tables converted: %d + %d cells" % (len(TABLE3), len(TABLE4) * 3))


def f4(label, d):
    return factor(T4[label](d))


def prose_edits():
    t3 = dict((k, factor(PT["T4"][k]["ratio"])) for _, k in TABLE3)
    dz = [factor(RW[d]["delta_zstd"]["ratio"]) for d in DS]
    return [
        # the metric definition itself
        ("Compression Ratio: Encoded size divided by the original "
         "(uncompressed) size.",
         "Compression Factor: Original size divided by encoded size; larger "
         "is smaller."),

        # related work
        ("classical time-series methods such as PAA and SAX [10–12] achieve "
         "extreme ratios by discarding temporal detail",
         "classical methods such as PAA and SAX [10–12] achieve extreme "
         "compression factors by discarding temporal detail"),
        ("can yield extreme compression ratios (often 0.02 of the original "
         "size)",
         "can yield extreme compression factors (often 50× smaller than "
         "the original)"),

        # synthetic rate-distortion
        ("showing RMSE versus compression ratio for different configurations",
         "showing RMSE versus compression factor for different "
         "configurations"),
        ("moderately larger files (11.5% vs. 2.4% of the original size)",
         "moderately larger files (compression factor %s vs. %s)"
         % (t3["enh_eps1e-3"], t3["base"])),
        ("so they cover the same range of compression ratios",
         "so they cover the same range of compression factors"),
        ("At matched sizes below roughly one fifth of the original, TRACQ's",
         "At every matched compression factor above roughly 5, TRACQ's"),
        ("increases storage by a factor of roughly 5 (2.4% to 11.5% of the "
         "original)",
         "increases storage roughly fivefold (compression factor %s down "
         "to %s)"
         % (t3["base"], t3["enh_eps1e-3"])),

        # streaming throughput: converted as printed, see the module docstring
        ("with encoded size 2.42% of the raw input, the archival setting "
         "reaches 28 MB/s at 1.81%, and gzip reaches 49 MB/s at 7.3%.",
         "at a compression factor of 41.3, the archival setting reaches "
         "28 MB/s at 55.2, and gzip reaches 49 MB/s at factor 13.7."),

        # Table IV's reading instructions
        ("In the table, ratio is encoded bytes over original bytes, so lower "
         "is smaller;",
         "In the table, factor is original bytes over encoded bytes, so "
         "higher is smaller;"),
        ("among methods encoding at or below one tenth of the original size.",
         "among methods reaching a compression factor of at least 10."),
        ("dense tolerance sweeps over a shared ratio range",
         "dense tolerance sweeps over a shared range of factors"),

        # per-dataset discussion
        ("at ratio 0.023 with correlation",
         "at factor %s with correlation" % f4("TRACQ (0.01)", APP)),
        ("PAA (RMSE = 20.9 at ratio 0.013)",
         "PAA (RMSE = 20.9 at factor %s)" % f4("PAA-64", APP)),
        ("at ratio 0.089, an error level",
         "at factor %s, an error level" % f4("TRACQ (0.0001)", APP)),
        ("at ratio 0.029 and",
         "at factor %s and" % f4("TRACQ (0.01)", MET)),
        ("at ratio 0.058 (correlation",
         "at factor %s (correlation" % f4("TRACQ (0.001)", MET)),
        ("at ratio 0.036, against",
         "at factor %s, against" % f4("TRACQ (0.01)", AIR)),
        ("193 at 0.013)", "193 at %s)" % f4("PAA-64", AIR)),
        ("at ratio 0.079, outperforming",
         "at factor %s, outperforming" % f4("TRACQ (0.001)", AIR)),
        ("lossless reconstruction at ratios 0.12–0.24",
         "lossless reconstruction at factors %s–%s" % (dz[0], dz[2])),
        ("neither produces an artifact below one fifth of the original size, "
         "except SZ3 on Appliances Energy (0.065 at tolerance 0.1) and on "
         "Metro Traffic (0.180 at the same tolerance)",
         "neither reaches a compression factor above five, except SZ3 on "
         "Appliances Energy (factor %s at tolerance 0.1) and on Metro "
         "Traffic (factor %s at the same tolerance)"
         % (f4("SZ3 (0.1)", APP), f4("SZ3 (0.1)", MET))),

        # MetroPT-3 rate-distortion
        ("at ratio 0.030 it reaches",
         "at factor %s it reaches" % factor(MP["C1_p1_abs_eps0.0001"]["ratio"])),
        ("and at ratio 0.0016 it reaches",
         "and at factor %s it reaches" % factor(MP["C1_p1_abs_eps0.1"]["ratio"])),
        ("on this stream is ratio 0.115 at",
         "on this stream is factor %s at" % factor(MP["zfp_tol_0.1"]["ratio"])),
        ("lossless Delta+Zstd (ratio 0.041)",
         "lossless Delta+Zstd (factor %s)" % factor(MP["delta_zstd"]["ratio"])),
        ("its extreme corner (ratio 0.0007)",
         "its extreme corner (factor %s)" % factor(MP["paa_1024"]["ratio"])),
        ("only at ratios 0.11–0.18",
         "only at factors %s–%s" % (factor(MP["zfp_tol_0.001"]["ratio"]),
                                         factor(MP["zfp_tol_0.1"]["ratio"]))),

        # SMAPE and matched-accuracy sections
        ("HPC compressors sit above it at every shared ratio",
         "HPC compressors sit above it at every shared factor"),
        ("because ZFP needs sizes near one fifth of the original to reach",
         "because ZFP needs a compression factor near five to reach"),

        # overheads, recommendations, roadmap
        ("All compression ratios reported across our evaluations",
         "All compression factors reported across our evaluations"),
        ("balance accuracy, compression ratio, and complexity",
         "balance accuracy, compression factor, and complexity"),
        ("sub-0.1% error at 9–27% of the original size.",
         "sub-0.1% error at compression factors of 3.7–11."),
        ("may further improve compression ratios",
         "may further improve compression factors"),
    ]


def main(tree):
    convert_tables(tree)
    doc = os.path.join(tree, "word", "document.xml")
    edits = prose_edits()
    for old, new in edits:
        ed = Editor(doc)      # fresh pass per edit: the cursor is forward-only
        ed.replace(old, new)
        ed.save()
    print("v7: %d prose edits applied" % len(edits))


if __name__ == "__main__":
    main(sys.argv[1])

# -*- coding: utf-8 -*-
"""September 3 author-edit round: mirror the author's clean-file edits and
re-tune the tail anchors.

The author edited the shipped CLEAN file directly (tracked, 32 revisions):
the "Experimental setup. / Results. / Throughput." lead-in labels go away,
the Section V-M opening (closing sentence, heading, intro) moves before the
Fig. 13 grid panels, the "At equal RMSE" paragraph moves directly after the
matched-operating-points paragraph, and the spacer between Fig. 9's caption
and Fig. 10 goes away. replay() applies those edits to the tracked lineage
as tracked changes; the author's own file is the source of the clean
lineage, so the two accepted texts are verified identical after replay.

The moves shorten Section V-L's page, so reflow() re-tunes the tail
anchors: panel (a) after the We-extract paragraph, the grids after the
Table VI discussion, Fig. 15 after the O intro, Fig. 14's frame after
Fig. 15's caption, and Fig. 16 after the first VI-A bullet. restyle()
returns the "Because the grid preserves" paragraph to BodyText (it lost
its style, and its first-line indent, in the author's move).

Replay gotcha recorded in flatten(): paragraphs copied or deleted whole may
already carry w:ins/w:del from earlier waves; nesting same-type revision
marks corrupts the file for Word. Inside a new w:ins, previously deleted
subtrees are dropped and previously inserted wrappers unwrapped; inside a
new w:del, both wrapper kinds are unwrapped and text converted to delText.

Usage: python docx_sep3_user_edit_round.py <tree_root>
"""

import copy
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_edit_lib import Editor, q

NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
AUTHOR = "Abdulfatah Bahbouh"
DATE = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [21000]


def nid():
    _id[0] += 1
    return str(_id[0])


def del_para_tracked(p):
    d = p.makeelement(NS + "del", {NS + "id": nid(), NS + "author": AUTHOR,
                                   NS + "date": DATE})
    kids = [c for c in p if c.tag != NS + "pPr"]
    if kids:
        kids[0].addprevious(d)
        for c in kids:
            d.append(c)
    ppr = p.find(NS + "pPr")
    if ppr is None:
        ppr = p.makeelement(NS + "pPr", {})
        p.insert(0, ppr)
    rpr = ppr.find(NS + "rPr")
    if rpr is None:
        rpr = ppr.makeelement(NS + "rPr", {})
        ppr.append(rpr)
    rpr.insert(0, rpr.makeelement(NS + "del", {NS + "id": nid(),
                                               NS + "author": AUTHOR,
                                               NS + "date": DATE}))
    return d


def ins_copy(p):
    c = copy.deepcopy(p)
    w = c.makeelement(NS + "ins", {NS + "id": nid(), NS + "author": AUTHOR,
                                   NS + "date": DATE})
    kids = [k for k in c if k.tag != NS + "pPr"]
    if kids:
        kids[0].addprevious(w)
        for k in kids:
            w.append(k)
    ppr = c.find(NS + "pPr")
    if ppr is None:
        ppr = c.makeelement(NS + "pPr", {})
        c.insert(0, ppr)
    rpr = ppr.find(NS + "rPr")
    if rpr is None:
        rpr = ppr.makeelement(NS + "rPr", {})
        ppr.append(rpr)
    rpr.insert(0, rpr.makeelement(NS + "ins", {NS + "id": nid(),
                                               NS + "author": AUTHOR,
                                               NS + "date": DATE}))
    return c, w


def flatten(outer):
    """No nested revision marks inside a freshly added w:ins/w:del."""
    def unwrap(el):
        for child in list(el):
            el.addprevious(child)
        el.getparent().remove(el)

    if outer.tag == NS + "ins":
        for d in list(outer.iter(NS + "del")):
            if d is not outer and d.getparent() is not None:
                d.getparent().remove(d)
        for i in list(outer.iter(NS + "ins")):
            if i is not outer:
                unwrap(i)
    else:
        for i in list(outer.iter(NS + "ins")):
            if i is not outer:
                unwrap(i)
        for d in list(outer.iter(NS + "del")):
            if d is not outer:
                unwrap(d)
        for t in outer.iter(NS + "t"):
            t.tag = NS + "delText"
        for it in outer.iter(NS + "instrText"):
            it.tag = NS + "delInstrText"


def tracked_move(ed, para, target):
    c, w = ins_copy(para)
    target.addnext(c)
    flatten(w)
    flatten(del_para_tracked(para))
    return c


def replay(ed):
    for old, new in [
        ("Experimental setup. We extract", "We extract"),
        ("Results. Table V summarizes", "Table V summarizes"),
        ("Throughput. The critical systems-level", "The critical systems-level"),
    ]:
        ed.replace(old, new)
    mhead = next(p for p in ed.paras
                 if ed.para_text(p).strip() == "Real-world Anomaly Screening")
    mintro = mhead.getnext()
    anchor = next(p for p in ed.paras
                  if "This experiment shows that the compressed grids"
                  in ed.para_text(p))
    c1 = tracked_move(ed, mhead, anchor)
    tracked_move(ed, mintro, c1)
    ateq = next(p for p in ed.paras
                if "At equal RMSE, TRACQ produces the smaller artifact"
                in ed.para_text(p))
    rdc = next(p for p in ed.paras
               if "Rate-distortion claims are easiest to judge"
               in ed.para_text(p))
    tracked_move(ed, ateq, rdc)
    cap9 = next(p for p in ed.paras
                if ed.para_text(p).strip().startswith("Fig. 9. Rate-distortion"))
    spacer = cap9.getnext()
    if spacer is not None and not ed.para_text(spacer).strip() and \
            spacer.find(NS + "pPr/" + NS + "framePr") is not None:
        flatten(del_para_tracked(spacer))


def frame_block(ed, prefix, both=False):
    cap = next(p for p in ed.paras
               if ed.para_text(p).strip().startswith(prefix))
    blk = [cap]
    prev = cap.getprevious()
    while prev is not None and prev.tag == q("w:p") and \
            prev.find(q("w:pPr") + "/" + q("w:framePr")) is not None and \
            not ed.para_text(prev).strip().startswith("Fig."):
        blk.insert(0, prev)
        prev = prev.getprevious()
    if both:
        nxt = cap.getnext()
        while nxt is not None and nxt.tag == q("w:p") and \
                nxt.find(q("w:pPr") + "/" + q("w:framePr")) is not None and \
                not ed.para_text(nxt).strip().startswith("Fig."):
            blk.append(nxt)
            nxt = nxt.getnext()
    return blk


def move_after(ed, paras, anchor_text):
    target = next(p for p in ed.paras if anchor_text in ed.para_text(p))
    assert target.find(q("w:pPr") + "/" + q("w:sectPr")) is None
    for p in paras:
        p.getparent().remove(p)
    for p in reversed(paras):
        target.addnext(p)


def keep_next(p):
    ppr = p.find(NS + "pPr")
    if ppr is None:
        ppr = p.makeelement(NS + "pPr", {})
        p.insert(0, ppr)
    if ppr.find(NS + "keepNext") is None:
        kn = ppr.makeelement(NS + "keepNext", {})
        st = ppr.find(NS + "pStyle")
        st.addnext(kn) if st is not None else ppr.insert(0, kn)


def reflow(ed):
    lintro = next(p for p in ed.paras
                  if "Here we test whether they can be detected computationally"
                  in ed.para_text(p))
    blk = []
    cur = lintro.getnext()
    while cur is not None and cur.tag == q("w:p") and \
            cur.find(q("w:pPr") + "/" + q("w:framePr")) is not None and \
            not ed.para_text(cur).strip().startswith("Fig."):
        blk.append(cur)
        cur = cur.getnext()
    move_after(ed, blk, "We extract 198 non-overlapping windows of length 100")
    move_after(ed, frame_block(ed, "Fig. 13. Visual inspection", both=True),
               "Table VI compares direct compressed-domain screening")
    cap15 = next(p for p in ed.paras
                 if ed.para_text(p).strip().startswith("Fig. 15. Size advantage"))
    img15 = cap15.getprevious()
    keep_next(img15)
    tgt = next(p for p in ed.paras
               if "Fig. 16 reports speedup and parallel efficiency alongside"
               in ed.para_text(p))
    for p in (img15, cap15):
        p.getparent().remove(p)
    tgt.addnext(cap15)
    tgt.addnext(img15)
    blk14 = frame_block(ed, "Fig. 14. Compressed-domain", both=True)
    tgt15 = next(p for p in ed.paras
                 if ed.para_text(p).strip().startswith("Fig. 15. Size advantage"))
    for p in blk14:
        p.getparent().remove(p)
    for p in reversed(blk14):
        tgt15.addnext(p)
    cap16 = next(p for p in ed.paras
                 if ed.para_text(p).strip().startswith("Fig. 16. Node-parallel"))
    img16 = cap16.getprevious()
    tgt = next(p for p in ed.paras
               if "Heterogeneous sensor arrays produce streams"
               in ed.para_text(p))
    for p in (img16, cap16):
        p.getparent().remove(p)
    tgt.addnext(cap16)
    tgt.addnext(img16)


def restyle(ed):
    tgt = next(p for p in ed.paras
               if "Because the grid preserves chronological order"
               in ed.para_text(p))
    ppr = tgt.find(NS + "pPr")
    if ppr is None:
        ppr = tgt.makeelement(NS + "pPr", {})
        tgt.insert(0, ppr)
    st = ppr.find(NS + "pStyle")
    if st is None:
        st = ppr.makeelement(NS + "pStyle", {})
        ppr.insert(0, st)
    st.set(NS + "val", "BodyText")


def main(tree):
    ed = Editor(os.path.join(tree, "word", "document.xml"))
    replay(ed)
    reflow(ed)
    restyle(ed)
    ed.save()
    print("author edits replayed; tail anchors re-tuned; paragraph restyled")


if __name__ == "__main__":
    main(sys.argv[1])

# -*- coding: utf-8 -*-
"""Produce an all-changes-accepted copy of an unpacked tracked-changes tree.

Acceptance semantics: drop w:del subtrees (and delText), splice w:ins children
into the parent, drop paragraphs whose paragraph mark is del-marked and that
have no surviving runs, merge non-empty ones into the following paragraph, and
strip revision marks / w:trackChanges so the output carries no revision state.

Usage: python docx_accept_changes.py <tree_root> <out_tree_root>
"""

import os
import shutil
import sys

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(tag):
    pre, local = tag.split(":")
    return "{%s}%s" % (W, local)


M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def accept(path):
    tree = etree.parse(path)
    root = tree.getroot()

    # Word re-encodes tracked-deleted math three ways; resolve them before the
    # generic pass would strip the markers and lose the deletion:
    #   1. a w:del marker directly inside an m:r deletes that math run
    for mr in list(root.iter(M + "r")):
        if any(c.tag == q("w:del") for c in mr) and mr.getparent() is not None:
            mr.getparent().remove(mr)
    #   2. a w:del inside an m:ctrlPr deletes the construct owning the ctrlPr
    for cp in list(root.iter(M + "ctrlPr")):
        if cp.find(q("w:del")) is not None:
            owner = cp.getparent()
            if owner is not None and owner.getparent() is not None:
                owner.getparent().remove(owner)

    # remove deletions (content deletions and paragraph-mark markers alike)
    for d in list(root.iter(q("w:del"))):
        d.getparent().remove(d)

    # unwrap insertions
    for ins in list(root.iter(q("w:ins"))):
        parent = ins.getparent()
        if parent.tag == q("w:rPr"):
            parent.remove(ins)
            continue
        idx = list(parent).index(ins)
        for child in list(ins):
            parent.insert(idx, child)
            idx += 1
        parent.remove(ins)

    # paragraphs whose mark was deleted: re-read the pre-acceptance file to
    # find them (no w:p is ever nested in w:ins/w:del, so indices align)
    orig = etree.parse(path)
    marked = set()
    for i, p in enumerate(orig.getroot().iter(q("w:p"))):
        d = p.find(q("w:pPr") + "/" + q("w:rPr") + "/" + q("w:del"))
        if d is not None:
            marked.add(i)
    paras = list(root.iter(q("w:p")))
    for i in sorted(marked, reverse=True):
        p = paras[i]
        runs = [r for r in p.iter(q("w:r")) if r.getparent().tag != q("w:pPr")]
        text = "".join(t.text or "" for t in p.iter(q("w:t")))
        parent = p.getparent()
        if not runs and not text.strip():
            if p.find(q("w:pPr") + "/" + q("w:sectPr")) is None:
                parent.remove(p)
            continue
        # merge into next paragraph
        nxt = p.getnext()
        while nxt is not None and nxt.tag != q("w:p"):
            nxt = nxt.getnext()
        if nxt is None:
            continue
        insert_at = 0
        for child in list(nxt):
            if child.tag == q("w:pPr"):
                insert_at = 1
                break
        pos = insert_at
        for child in list(p):
            if child.tag == q("w:pPr"):
                continue
            nxt.insert(pos, child)
            pos += 1
        parent.remove(p)

    # strip delText remnants and empty ins/del leftovers
    for dt in list(root.iter(q("w:delText"))):
        r = dt.getparent()
        if r is not None and r.getparent() is not None:
            r.getparent().remove(r)

    #   3. hollow math left behind (all content deleted) renders as an empty
    #      box: drop any oMath/oMathPara with no math text remaining
    for om in list(root.iter(M + "oMathPara")) + list(root.iter(M + "oMath")):
        if om.getparent() is None:
            continue
        if not "".join(t.text or "" for t in om.iter(M + "t")).strip():
            om.getparent().remove(om)

    tree.write(path, xml_declaration=True, encoding="UTF-8", standalone=True)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    accept(os.path.join(dst, "word", "document.xml"))
    # drop trackChanges flag
    spath = os.path.join(dst, "word", "settings.xml")
    if os.path.exists(spath):
        st = etree.parse(spath)
        for el in list(st.getroot().iter(q("w:trackChanges"))):
            el.getparent().remove(el)
        st.write(spath, xml_declaration=True, encoding="UTF-8", standalone=True)
    print("accepted ->", dst)


if __name__ == "__main__":
    main()

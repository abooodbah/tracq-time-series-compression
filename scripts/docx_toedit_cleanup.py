# -*- coding: utf-8 -*-
"""Strip the author's inline TOEDIT instruction notes from an annotated tree:
remove the note insertions inside caption paragraphs, drop note-only
paragraphs entirely, and clear empty revision stubs. Real content edits by
the author are left untouched.

Usage: python docx_toedit_cleanup.py <tree_root>
"""

import os
import sys

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(t):
    return "{%s}%s" % (W, t)


NOTE_AUTHOR = "Bahbouh, Abdulfatah"

tree_root = sys.argv[1]
path = os.path.join(tree_root, "word", "document.xml")
tree = etree.parse(path)
root = tree.getroot()


def visible_text(p):
    parts = []
    for t in p.iter(q("t")):
        anc, deleted = t.getparent(), False
        while anc is not None:
            if anc.tag == q("del"):
                deleted = True
                break
            anc = anc.getparent()
        if not deleted:
            parts.append(t.text or "")
    return "".join(parts)


removed_ins, removed_para = 0, 0
for p in list(root.iter(q("p"))):
    if "TOEDIT" not in visible_text(p).upper():
        continue
    for ins in list(p.iter(q("ins"))):
        if ins.get(q("author")) == NOTE_AUTHOR:
            ins.getparent().remove(ins)
            removed_ins += 1
    if not visible_text(p).strip():
        p.getparent().remove(p)
        removed_para += 1

# empty revision stubs left by Word (no delText content)
removed_stub = 0
for d in list(root.iter(q("del"))):
    if d.get(q("author")) == NOTE_AUTHOR:
        if not "".join(t.text or "" for t in d.iter(q("delText"))):
            d.getparent().remove(d)
            removed_stub += 1

tree.write(path, xml_declaration=True, encoding="UTF-8", standalone=True)
print(f"removed {removed_ins} note insertions, {removed_para} note paragraphs, "
      f"{removed_stub} empty stubs")

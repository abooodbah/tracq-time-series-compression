"""Tracked-changes editor for WordprocessingML documents.

Applies an ordered list of exact-string edits to word/document.xml as Word
tracked changes (w:del + w:ins), with support for tracked math deletion and
tracked insertion of new OMML equation paragraphs. Edits are consumed in
document order via a cursor, so short strings (table cells) resolve to the
correct occurrence.
"""

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W, "m": M}

AUTHOR = "Abdulfatah Bahbouh"
DATE = "2026-07-14T00:00:00Z"


def q(tag):
    pre, local = tag.split(":")
    return f"{{{W if pre == 'w' else M}}}{local}"


class Editor:
    def __init__(self, path):
        self.path = path
        parser = etree.XMLParser(remove_blank_text=False)
        self.tree = etree.parse(path, parser)
        self.root = self.tree.getroot()
        self.paras = list(self.root.iter(q("w:p")))
        self.cursor_p = 0
        self.cursor_c = 0
        self._id = 9000
        self.log = []

    # ------------------------------------------------------------------
    def nid(self):
        self._id += 1
        return str(self._id)

    @staticmethod
    def _run_text(r):
        out = []
        for t in r.findall(q("w:t")):
            out.append(t.text or "")
        return "".join(out)

    def _para_runs(self, p):
        """Live (non-deleted) runs anywhere under p, including runs nested in
        hyperlinks, fields, and smart tags."""
        runs = []
        for r in p.iter(q("w:r")):
            anc = r.getparent()
            skip = False
            while anc is not None and anc is not p:
                if anc.tag == q("w:del"):
                    skip = True
                    break
                anc = anc.getparent()
            if not skip:
                runs.append(r)
        return runs

    _SPACE_MAP = str.maketrans({c: " " for c in "\xa0      "})

    def para_text(self, p):
        # exotic spaces normalized 1:1 to plain space so offsets stay aligned
        return "".join(self._run_text(r) for r in self._para_runs(p)).translate(self._SPACE_MAP)

    # ------------------------------------------------------------------
    def _split_run(self, r, at):
        """Split run r at char offset `at`; returns (left, right) elements."""
        text = self._run_text(r)
        left_txt, right_txt = text[:at], text[at:]
        parent = r.getparent()
        idx = parent.index(r)
        right = etree.fromstring(etree.tostring(r))
        for el in (r, right):
            for t in el.findall(q("w:t")):
                el.remove(t)
        for el, txt in ((r, left_txt), (right, right_txt)):
            t = etree.SubElement(el, q("w:t"))
            t.text = txt
            t.set(f"{{{XML}}}space", "preserve")
        parent.insert(idx + 1, right)
        return r, right

    def _wrap_del(self, r):
        """Wrap run r in w:del, converting w:t to w:delText."""
        parent = r.getparent()
        idx = parent.index(r)
        d = etree.Element(q("w:del"))
        d.set(q("w:id"), self.nid())
        d.set(q("w:author"), AUTHOR)
        d.set(q("w:date"), DATE)
        parent.remove(r)
        for t in r.findall(q("w:t")):
            t.tag = q("w:delText")
            t.set(f"{{{XML}}}space", "preserve")
        d.append(r)
        parent.insert(idx, d)
        return d

    def _make_ins(self, rpr_src, text):
        ins = etree.Element(q("w:ins"))
        ins.set(q("w:id"), self.nid())
        ins.set(q("w:author"), AUTHOR)
        ins.set(q("w:date"), DATE)
        r = etree.SubElement(ins, q("w:r"))
        if rpr_src is not None:
            rpr = rpr_src.find(q("w:rPr"))
            if rpr is not None:
                r.append(etree.fromstring(etree.tostring(rpr)))
        t = etree.SubElement(r, q("w:t"))
        t.text = text
        t.set(f"{{{XML}}}space", "preserve")
        return ins

    # ------------------------------------------------------------------
    def seek(self, snippet):
        """Advance the cursor to the paragraph containing snippet (no edit)."""
        for pi in range(self.cursor_p, len(self.paras)):
            if snippet in self.para_text(pi if False else self.paras[pi]):
                self.cursor_p = pi
                self.cursor_c = 0
                return True
        print(f"  !! seek failed: {snippet[:60]!r}")
        return False

    def replace(self, old, new, required=True):
        """Tracked replacement of `old` with `new`, searching from cursor."""
        for pi in range(self.cursor_p, len(self.paras)):
            p = self.paras[pi]
            text = self.para_text(p)
            start = self.cursor_c if pi == self.cursor_p else 0
            a = text.find(old, start)
            if a < 0:
                continue
            b = a + len(old)
            runs = self._para_runs(p)
            # map char offsets to runs
            spans = []
            off = 0
            for r in runs:
                ln = len(self._run_text(r))
                spans.append((off, off + ln, r))
                off += ln
            affected = [(s, e, r) for s, e, r in spans if s < b and e > a]
            # trim first/last runs
            s0, e0, r0 = affected[0]
            if a > s0:
                _, right = self._split_run(r0, a - s0)
                affected[0] = (a, e0, right)
                if len(affected) == 1:
                    pass
            s1, e1, r1 = affected[-1]
            if b < e1:
                left, _ = self._split_run(r1, b - s1)
                affected[-1] = (s1, b, left)
            first_run = affected[0][2]
            last_del = None
            for _, _, r in affected:
                last_del = self._wrap_del(r)
            if new:
                ins = self._make_ins(first_run, new)
                parent = last_del.getparent()
                parent.insert(parent.index(last_del) + 1, ins)
            self.cursor_p = pi
            self.cursor_c = a + len(new)
            self.log.append(("ok", old[:50]))
            return True
        self.log.append(("MISS", old[:80]))
        if required:
            # diagnose: does the string exist anywhere, and where vs cursor?
            where = []
            for pj, p in enumerate(self.paras):
                if old in self.para_text(p):
                    where.append(pj)
            print(f"  !! NO MATCH: {old[:70]!r} cursor={self.cursor_p} found_at={where[:4]}")
        return False

    # ------------------------------------------------------------------
    def delete_paragraph_by_text(self, snippet):
        """Tracked-delete the whole paragraph containing snippet."""
        for pi in range(self.cursor_p, len(self.paras)):
            p = self.paras[pi]
            if snippet in self.para_text(p):
                for r in list(self._para_runs(p)):
                    self._wrap_del(r)
                self._mark_para_mark_deleted(p)
                self.cursor_p = pi + 1
                self.cursor_c = 0
                self.log.append(("ok-delpara", snippet[:50]))
                return True
        self.log.append(("MISS-delpara", snippet[:80]))
        print(f"  !! NO MATCH delete_paragraph: {snippet[:90]!r}")
        return False

    def _mark_para_mark_deleted(self, p):
        ppr = p.find(q("w:pPr"))
        if ppr is None:
            ppr = etree.Element(q("w:pPr"))
            p.insert(0, ppr)
        rpr = ppr.find(q("w:rPr"))
        if rpr is None:
            rpr = etree.SubElement(ppr, q("w:rPr"))
        d = etree.Element(q("w:del"))
        d.set(q("w:id"), self.nid())
        d.set(q("w:author"), AUTHOR)
        d.set(q("w:date"), DATE)
        # revision marks must be the first children of the paragraph-mark rPr
        rpr.insert(0, d)

    # ------------------------------------------------------------------
    def _wrap_tracked(self, el, kind):
        """Wrap element el (e.g. m:oMath) in a w:ins/w:del tracked container —
        CT_RunTrackChange explicitly admits math children."""
        parent = el.getparent()
        # lift a display equation out of its oMathPara shell so the tracked
        # container sits directly among the paragraph's children
        if parent is not None and parent.tag == q("m:oMathPara"):
            shell = parent
            parent = shell.getparent()
            idx = parent.index(shell)
            shell.remove(el)
            parent.remove(shell)
        else:
            idx = parent.index(el)
            parent.remove(el)
        box = etree.Element(q(f"w:{kind}"))
        box.set(q("w:id"), self.nid())
        box.set(q("w:author"), AUTHOR)
        box.set(q("w:date"), DATE)
        box.append(el)
        parent.insert(idx, box)
        return box

    def _mark_math_runs(self, container, kind):
        """Track-mark every m:oMath under (or at) container."""
        if container.tag == q("m:oMath"):
            self._wrap_tracked(container, kind)
            return 1
        n = 0
        for om in list(container.iter(q("m:oMath"))):
            self._wrap_tracked(om, kind)
            n += 1
        return n

    def del_inline_math(self, snippet, count=99, skip=0):
        """Tracked-delete inline m:oMath elements inside the paragraph that
        contains `snippet` (searched from the start, snippet must be unique)."""
        for pi in range(len(self.paras)):
            p = self.paras[pi]
            if snippet in self.para_text(p):
                oms = p.findall(q("m:oMath"))
                done = 0
                for om in oms[skip:skip + count]:
                    self._mark_math_runs(om, "del")
                    done += 1
                self.log.append(("ok-delinline", f"{snippet[:40]} x{done}"))
                return done
        self.log.append(("MISS-delinline", snippet[:80]))
        print(f"  !! NO MATCH del_inline_math: {snippet[:90]!r}")
        return 0

    def delete_math_after(self, anchor, count=1, within=6):
        """Tracked-delete the next `count` math paragraphs after the paragraph
        containing `anchor`."""
        for pi in range(len(self.paras)):
            if anchor in self.para_text(self.paras[pi]):
                done = 0
                for pj in range(pi + 1, min(pi + 1 + within, len(self.paras))):
                    pmath = self.paras[pj]
                    if pmath.find(q("m:oMathPara")) is not None or pmath.find(q("m:oMath")) is not None:
                        self._mark_math_runs(pmath, "del")
                        self._mark_para_mark_deleted(pmath)
                        done += 1
                        if done == count:
                            # do not advance the text cursor: prose between or
                            # after equations may still need edits
                            self.log.append(("ok-delmath", f"{anchor[:40]} x{count}"))
                            return True
                break
        self.log.append(("MISS-delmath", anchor[:80]))
        print(f"  !! NO MATCH delete_math_after: {anchor[:90]!r}")
        return False

    def insert_math_after(self, anchor, omath_children, after_math=0):
        """Insert a new tracked math paragraph after the paragraph containing
        `anchor` (skipping `after_math` following math paragraphs first)."""
        for pi in range(len(self.paras)):
            if anchor in self.para_text(self.paras[pi]):
                target = self.paras[pi]
                skipped = 0
                pj = pi
                while skipped < after_math and pj + 1 < len(self.paras):
                    pj += 1
                    pm = self.paras[pj]
                    if pm.find(q("m:oMathPara")) is not None or pm.find(q("m:oMath")) is not None:
                        target = pm
                        skipped += 1
                newp = etree.Element(q("w:p"))
                ppr = etree.SubElement(newp, q("w:pPr"))
                jc = etree.SubElement(ppr, q("w:jc"))
                jc.set(q("w:val"), "center")
                rpr = etree.SubElement(ppr, q("w:rPr"))
                ins = etree.SubElement(rpr, q("w:ins"))
                ins.set(q("w:id"), self.nid())
                ins.set(q("w:author"), AUTHOR)
                ins.set(q("w:date"), DATE)
                box = etree.SubElement(newp, q("w:ins"))
                box.set(q("w:id"), self.nid())
                box.set(q("w:author"), AUTHOR)
                box.set(q("w:date"), DATE)
                om = etree.SubElement(box, q("m:oMath"))
                for ch in omath_children:
                    om.append(ch)
                parent = target.getparent()
                parent.insert(parent.index(target) + 1, newp)
                self.paras.insert(pj + 1 if after_math else pi + 1, newp)
                self.log.append(("ok-insmath", anchor[:50]))
                return True
        self.log.append(("MISS-insmath", anchor[:80]))
        print(f"  !! NO MATCH insert_math_after: {anchor[:90]!r}")
        return False

    def insert_paragraph_after(self, anchor, segments):
        """Insert a tracked new paragraph after the paragraph containing
        `anchor`. segments = list of (text, italic) tuples; pPr is copied from
        the anchor paragraph so styling (e.g. reference list format) matches."""
        for pi in range(len(self.paras)):
            if anchor in self.para_text(self.paras[pi]):
                src = self.paras[pi]
                newp = etree.Element(q("w:p"))
                src_ppr = src.find(q("w:pPr"))
                if src_ppr is not None:
                    cp = etree.fromstring(etree.tostring(src_ppr))
                    for tag in ("w:sectPr", "w:pPrChange"):
                        el = cp.find(q(tag))
                        if el is not None:
                            cp.remove(el)
                    newp.append(cp)
                ppr = newp.find(q("w:pPr"))
                if ppr is None:
                    ppr = etree.SubElement(newp, q("w:pPr"))
                rpr = ppr.find(q("w:rPr"))
                if rpr is None:
                    rpr = etree.SubElement(ppr, q("w:rPr"))
                # a copied pPr may already carry revision marks — strip them
                for old_mark in rpr.findall(q("w:ins")) + rpr.findall(q("w:del")):
                    rpr.remove(old_mark)
                mark = etree.Element(q("w:ins"))
                mark.set(q("w:id"), self.nid())
                mark.set(q("w:author"), AUTHOR)
                mark.set(q("w:date"), DATE)
                rpr.insert(0, mark)
                box = etree.SubElement(newp, q("w:ins"))
                box.set(q("w:id"), self.nid())
                box.set(q("w:author"), AUTHOR)
                box.set(q("w:date"), DATE)
                for text, italic in segments:
                    r = etree.SubElement(box, q("w:r"))
                    if italic:
                        rr = etree.SubElement(r, q("w:rPr"))
                        etree.SubElement(rr, q("w:i"))
                    t = etree.SubElement(r, q("w:t"))
                    t.text = text
                    t.set(f"{{{XML}}}space", "preserve")
                src.addnext(newp)
                self.paras.insert(pi + 1, newp)
                self.log.append(("ok-inspara", anchor[:50]))
                return True
        self.log.append(("MISS-inspara", anchor[:80]))
        print(f"  !! NO MATCH insert_paragraph_after: {anchor[:90]!r}")
        return False

    # ------------------------------------------------------------------
    def save(self):
        self.tree.write(self.path, xml_declaration=True, encoding="UTF-8", standalone=True)
        misses = [e for e in self.log if e[0].startswith("MISS")]
        print(f"edits: {len(self.log)} total, {len(misses)} missed")
        return misses


# ----------------------------------------------------------------------------
# OMML builders
# ----------------------------------------------------------------------------

def mr(text):
    r = etree.Element(q("m:r"))
    t = etree.SubElement(r, q("m:t"))
    t.text = text
    t.set(f"{{{XML}}}space", "preserve")
    return r


def msub(base, sub):
    s = etree.Element(q("m:sSub"))
    e = etree.SubElement(s, q("m:e"))
    e.append(mr(base))
    sb = etree.SubElement(s, q("m:sub"))
    sb.append(mr(sub))
    return s

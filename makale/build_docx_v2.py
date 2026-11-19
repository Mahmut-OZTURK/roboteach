"""Builds makale/RoboTeach_TR_v2.docx from makale_tr_v2.md (Times New Roman, readable tables)."""
import os
import subprocess
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "makale_tr_v2.md")
OUT = os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "RoboTeach_TR_v2.docx")
REF = "/tmp/roboteach_ref.docx"

SIZES = {"Normal": 12, "Body Text": 12, "First Paragraph": 12, "Compact": 11, "Title": 16,
         "Heading 1": 14, "Heading 2": 12, "Heading 3": 12, "Caption": 11, "Image Caption": 11, "Table Caption": 11}


def build_reference():
    with open(REF, "wb") as fh:
        fh.write(subprocess.check_output(["pandoc", "--print-default-data-file", "reference.docx"]))
    d = Document(REF)
    for st in d.styles:
        size = SIZES.get(st.name)
        if size is None:
            continue
        try:
            st.font.name = "Times New Roman"
            st.font.size = Pt(size)
            rpr = st.element.get_or_add_rPr()
            rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = rpr.makeelement(qn("w:rFonts"), {})
                rpr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rf.set(qn(a), "Times New Roman")
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                if rf.get(qn(a)) is not None:
                    del rf.attrib[qn(a)]
            if st.name.startswith("Heading") or st.name == "Title":
                st.font.bold = True
                st.font.color.rgb = None
            if st.name in ("Body Text", "First Paragraph"):
                st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        except Exception:
            pass
    d.save(REF)


def fix_tables(path):
    d = Document(path)
    total = 9000
    for t in d.tables:
        tbl = t._tbl
        tblPr = tbl.tblPr
        ncol = max(len(r.findall(qn("w:tc"))) for r in tbl.findall(qn("w:tr")))
        if ncol == 3 and t.rows[0].cells[0].text in ("Görev", "Hata türü"):
            ws = [0.16, 0.42, 0.42]
        else:
            first = 0.34 if ncol > 3 else (0.2 if ncol == 3 else 0.5)
            ws = [first] + [(1 - first) / (ncol - 1)] * (ncol - 1)
        tw = [int(total * w) for w in ws]
        old = tbl.find(qn("w:tblGrid"))
        if old is not None:
            tbl.remove(old)
        grid = OxmlElement("w:tblGrid")
        for w in tw:
            gc = OxmlElement("w:gridCol")
            gc.set(qn("w:w"), str(w))
            grid.append(gc)
        tblPr.addnext(grid)
        tw_el = tblPr.find(qn("w:tblW"))
        if tw_el is None:
            tw_el = OxmlElement("w:tblW")
            tblPr.append(tw_el)
        tw_el.set(qn("w:w"), str(total))
        tw_el.set(qn("w:type"), "dxa")
        lay = tblPr.find(qn("w:tblLayout"))
        if lay is None:
            lay = OxmlElement("w:tblLayout")
            tblPr.append(lay)
        lay.set(qn("w:type"), "fixed")
        borders = OxmlElement("w:tblBorders")
        for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{e}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:color"), "808080")
            borders.append(el)
        old_b = tblPr.find(qn("w:tblBorders"))
        if old_b is not None:
            tblPr.remove(old_b)
        tblPr.append(borders)
        for r in tbl.findall(qn("w:tr")):
            for i, tc in enumerate(r.findall(qn("w:tc"))):
                tcPr = tc.find(qn("w:tcPr"))
                if tcPr is None:
                    tcPr = OxmlElement("w:tcPr")
                    tc.insert(0, tcPr)
                w = tcPr.find(qn("w:tcW"))
                if w is None:
                    w = OxmlElement("w:tcW")
                    tcPr.insert(0, w)
                w.set(qn("w:w"), str(tw[min(i, len(tw) - 1)]))
                w.set(qn("w:type"), "dxa")
        for row in t.rows:
            for c in row.cells:
                for para in c.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(11)
                        run.font.name = "Times New Roman"
    d.save(path)
    return len(d.tables), len(d.inline_shapes)


if __name__ == "__main__":
    build_reference()
    subprocess.check_call(["pandoc", SRC, "-o", OUT, f"--reference-doc={REF}", f"--resource-path={HERE}"])
    print("tables, images:", fix_tables(OUT))

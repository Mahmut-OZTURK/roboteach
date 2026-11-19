"""Builds RoboTeach_TR_v2.docx from makale_tr_v2.md with journal-style layout.

Usage: python build_docx_v2.py [input.md] [output.docx]
"""
import os
import re
import subprocess
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "makale_tr_v2.md")
OUT = os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "RoboTeach_TR_v2.docx")
REF = "/tmp/roboteach_ref.docx"
FONT = "Times New Roman"
TEXT_W = 9070  # A4 with 2.5 cm margins, twips

STYLE_SIZES = {"Normal": 12, "Body Text": 12, "First Paragraph": 12, "Compact": 12,
               "Title": 16, "Heading 1": 14, "Heading 2": 12, "Heading 3": 12,
               "Image Caption": 11, "Table Caption": 11, "Captioned Figure": 12, "Figure": 12}


def set_font(rpr_owner, name=FONT):
    rpr = rpr_owner.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), name)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]


def build_reference():
    with open(REF, "wb") as fh:
        fh.write(subprocess.check_output(["pandoc", "--print-default-data-file", "reference.docx"]))
    d = Document(REF)
    for st in d.styles:
        if st.type != 1:  # paragraph styles only
            continue
        try:
            set_font(st.element)
        except Exception:
            pass
        size = STYLE_SIZES.get(st.name)
        if size:
            st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0, 0, 0)
        pf = st.paragraph_format
        if st.name in ("Body Text", "First Paragraph", "Compact", "Normal"):
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.15
            pf.space_before = Pt(0)
            pf.space_after = Pt(6)
        if st.name.startswith("Heading") or st.name == "Title":
            st.font.bold = True
            st.font.italic = False
            pf.keep_with_next = True
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER if st.name == "Title" else WD_ALIGN_PARAGRAPH.LEFT
            pf.space_before = Pt(18 if st.name == "Heading 1" else 12)
            pf.space_after = Pt(6)
        if st.name in ("Image Caption", "Table Caption", "Captioned Figure", "Figure"):
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            st.font.italic = False
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.5))
    d.save(REF)


def border_el(tag, sz, color="000000"):
    el = OxmlElement(f"w:{tag}")
    el.set(qn("w:val"), "single" if sz else "nil")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), color)
    return el


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), fill)
    tcPr.append(sh)


def cell_borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    if top is not None:
        b.append(border_el("top", top[0], top[1]))
    if bottom is not None:
        b.append(border_el("bottom", bottom[0], bottom[1]))
    b.append(border_el("left", 0))
    b.append(border_el("right", 0))
    tcPr.append(b)


def column_widths(t):
    head = [c.text.strip() for c in t.rows[0].cells]
    n = len(head)
    if head[:3] == ["Görev", "Örnek komut", "Başarı koşulu"]:
        return [0.14, 0.46, 0.40], [0, 0, 0]
    if head[:1] == ["Hata türü"]:
        return [0.18, 0.38, 0.44], [0, 0, 0]
    if head[:1] == ["Kapatılan bileşen"]:
        return [0.46, 0.20, 0.20, 0.14], [0, 1, 1, 1]
    if head[:1] == ["Koşul"]:
        return [0.32, 0.17, 0.17, 0.17, 0.17], [0, 1, 1, 1, 1]
    first = 0.30 if n > 3 else 0.40
    return [first] + [(1 - first) / (n - 1)] * (n - 1), [0] + [1] * (n - 1)


def format_table(t):
    tbl = t._tbl
    tblPr = tbl.tblPr
    ws, centered = column_widths(t)
    tw = [int(TEXT_W * w) for w in ws]
    for tag in ("w:tblGrid",):
        old = tbl.find(qn(tag))
        if old is not None:
            tbl.remove(old)
    grid = OxmlElement("w:tblGrid")
    for w in tw:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(w))
        grid.append(gc)
    tblPr.addnext(grid)
    for tag in ("w:tblW", "w:tblLayout", "w:tblBorders", "w:tblStyle", "w:tblLook", "w:tblCellMar", "w:jc"):
        old = tblPr.find(qn(tag))
        if old is not None:
            tblPr.remove(old)
    w_el = OxmlElement("w:tblW")
    w_el.set(qn("w:w"), str(TEXT_W))
    w_el.set(qn("w:type"), "dxa")
    tblPr.append(w_el)
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    tblPr.append(jc)
    borders = OxmlElement("w:tblBorders")
    for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(border_el(e, 0))
    tblPr.append(borders)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    mar = OxmlElement("w:tblCellMar")
    for side, v in (("top", 60), ("bottom", 60), ("left", 100), ("right", 100)):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), str(v))
        m.set(qn("w:type"), "dxa")
        mar.append(m)
    tblPr.append(mar)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    nrows = len(t.rows)
    last_is_total = t.rows[-1].cells[0].text.strip().startswith(("Toplam", "Total"))
    for ri, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        if ri == 0:
            trPr.append(OxmlElement("w:tblHeader"))
        for ci, cell in enumerate(row.cells):
            tcPr = cell._tc.get_or_add_tcPr()
            w = tcPr.find(qn("w:tcW"))
            if w is None:
                w = OxmlElement("w:tcW")
                tcPr.insert(0, w)
            w.set(qn("w:w"), str(tw[min(ci, len(tw) - 1)]))
            w.set(qn("w:type"), "dxa")
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            top = bottom = None
            if ri == 0:
                top, bottom = (12, "000000"), (6, "000000")
                shade(cell, "F2F2F2")
            elif ri == nrows - 1:
                bottom = (12, "000000")
                if last_is_total:
                    top = (6, "000000")
                else:
                    top = (4, "BFBFBF")
            else:
                top = (4, "BFBFBF")
            cell_borders(cell, top, bottom)
            for para in cell.paragraphs:
                pf = para.paragraph_format
                pf.space_before = Pt(0)
                pf.space_after = Pt(0)
                pf.line_spacing = 1.0
                pf.keep_with_next = ri < nrows - 1
                pf.alignment = (WD_ALIGN_PARAGRAPH.CENTER if (ri == 0 and centered[min(ci, len(centered) - 1)])
                                or (ri > 0 and centered[min(ci, len(centered) - 1)]) else WD_ALIGN_PARAGRAPH.LEFT)
                for run in para.runs:
                    run.font.size = Pt(11)
                    run.font.name = FONT
                    set_font(run._r)
                    if ri == 0:
                        run.font.bold = True


def has_drawing(p):
    return bool(p._p.findall(".//" + qn("w:drawing")))


def bold_prefix(p, pattern):
    """Make the leading 'Şekil N.' / 'Tablo N.' label bold and the rest regular."""
    text = p.text
    m = re.match(pattern, text)
    if not m:
        return
    label = m.group(0)
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    r1 = p.add_run(label)
    r1.bold = True
    r2 = p.add_run(text[len(label):])
    r2.bold = False
    for r in (r1, r2):
        r.font.name = FONT
        set_font(r._r)
        r.font.size = Pt(11)


def format_paragraphs(d):
    paras = d.paragraphs
    in_front = True
    in_refs = False
    for i, p in enumerate(paras):
        style = p.style.name
        text = p.text.strip()
        pf = p.paragraph_format
        if text.startswith("Öz:") or text.startswith("Abstract:"):
            in_front = False
        if in_front and style != "Title":
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.space_after = Pt(4)
            if "Üniversitesi" in text or "University" in text:
                for r in p.runs:
                    r.italic = True
                    r.font.size = Pt(11)
            continue
        if style.startswith("Heading 1") and text in ("Kaynaklar", "References"):
            in_refs = True
            continue
        if has_drawing(p):
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.space_before = Pt(6)
            pf.space_after = Pt(6)
            pf.keep_with_next = True
            continue
        if style in ("Image Caption",) or text.startswith("Şekil ") and style.endswith("Caption"):
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.space_after = Pt(12)
            bold_prefix(p, r"^(Şekil|Fig\.) \d+\.")
            continue
        if re.match(r"^(Tablo|Table) \d+\.", text):
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.keep_with_next = True
            pf.space_before = Pt(12)
            pf.space_after = Pt(6)
            bold_prefix(p, r"^(Tablo|Table) \d+\.")
            continue
        if text.startswith(("Not:", "Note:")):
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf.space_after = Pt(12)
            for r in p.runs:
                r.font.size = Pt(10)
            continue
        if in_refs:
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.left_indent = Cm(0.8)
            pf.first_line_indent = Cm(-0.8)
            pf.space_after = Pt(3)
            pf.line_spacing = 1.0
            for r in p.runs:
                r.font.size = Pt(10)
            continue


def fix_sections(d):
    for sec in d.sections:
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Cm(2.5))


def postprocess(path):
    d = Document(path)
    fix_sections(d)
    for t in d.tables:
        format_table(t)
    format_paragraphs(d)
    body = d.element.body
    for el in body.iterchildren():
        if el.tag == qn("w:tbl"):
            nxt = el.getnext()
            if nxt is not None and nxt.tag == qn("w:p"):
                pPr = nxt.get_or_add_pPr()
                sp = pPr.find(qn("w:spacing"))
                if sp is None:
                    sp = OxmlElement("w:spacing")
                    pPr.append(sp)
                sp.set(qn("w:before"), "240")
    d.save(path)
    return len(d.tables), len(d.inline_shapes)


if __name__ == "__main__":
    build_reference()
    subprocess.check_call(["pandoc", SRC, "-o", OUT, f"--reference-doc={REF}", f"--resource-path={HERE}"])
    print("tables, images:", postprocess(OUT))

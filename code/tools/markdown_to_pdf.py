from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


FONT = "ArialUnicode"
CODE_FONT = "Courier"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(FONT, FONT_PATH))


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ChineseTitle",
            parent=base["Title"],
            fontName=FONT,
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "h2": ParagraphStyle(
            "ChineseH2",
            parent=base["Heading1"],
            fontName=FONT,
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=13,
            spaceAfter=7,
        ),
        "h3": ParagraphStyle(
            "ChineseH3",
            parent=base["Heading2"],
            fontName=FONT,
            fontSize=12.5,
            leading=17,
            textColor=colors.HexColor("#374151"),
            spaceBefore=9,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "ChineseBody",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10,
            leading=15,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "list": ParagraphStyle(
            "ChineseList",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10,
            leading=15,
            leftIndent=18,
            firstLineIndent=-13,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "ChineseTable",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.4,
            leading=9.5,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "ChineseTableHeader",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.6,
            leading=9.7,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "ChineseCode",
            parent=base["Code"],
            fontName=CODE_FONT,
            fontSize=8,
            leading=10,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f3f4f6"),
            borderColor=colors.HexColor("#e5e7eb"),
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=7,
        ),
    }
    return styles


def inline_markup(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    rendered = []
    for part in parts:
        if len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            code = html.escape(part[1:-1])
            rendered.append(f'<font face="{CODE_FONT}" color="#111827">{code}</font>')
        else:
            rendered.append(html.escape(part))
    return "".join(rendered)


def split_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def is_table_separator(line: str) -> bool:
    cells = split_table_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def is_table_start(lines: list[str], idx: int) -> bool:
    return idx + 1 < len(lines) and "|" in lines[idx] and is_table_separator(lines[idx + 1])


def collect_paragraph(lines: list[str], idx: int) -> tuple[str, int]:
    parts = []
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("#") or stripped.startswith("- ") or re.match(r"\d+\. ", stripped):
            break
        if stripped.startswith("|") and "|" in stripped:
            break
        if stripped.startswith("```"):
            break
        parts.append(stripped)
        idx += 1
    return " ".join(parts), idx


def build_table(table_lines: list[str], styles: dict[str, ParagraphStyle], page_width: float) -> Table:
    rows = [split_table_row(line) for line in table_lines if not is_table_separator(line)]
    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    data = []
    for r_idx, row in enumerate(normalized):
        style_name = "table_header" if r_idx == 0 else "table"
        data.append([Paragraph(inline_markup(cell), styles[style_name]) for cell in row])

    col_widths = [page_width / max_cols] * max_cols
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ]
        )
    )
    return table


def parse_markdown(markdown: str, styles: dict[str, ParagraphStyle], page_width: float):
    lines = markdown.splitlines()
    story = []
    idx = 0
    in_code = False
    code_lines: list[str] = []

    while idx < len(lines):
        raw = lines[idx]
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                text = "<br/>".join(html.escape(item) for item in code_lines) or " "
                story.append(Paragraph(text, styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            idx += 1
            continue

        if in_code:
            code_lines.append(line)
            idx += 1
            continue

        if not stripped:
            idx += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(inline_markup(stripped[2:].strip()), styles["title"]))
            idx += 1
            continue

        if stripped.startswith("## "):
            if story:
                story.append(Spacer(1, 3))
            story.append(Paragraph(inline_markup(stripped[3:].strip()), styles["h2"]))
            idx += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(inline_markup(stripped[4:].strip()), styles["h3"]))
            idx += 1
            continue

        if is_table_start(lines, idx):
            table_lines = [lines[idx], lines[idx + 1]]
            idx += 2
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx])
                idx += 1
            story.append(build_table(table_lines, styles, page_width))
            story.append(Spacer(1, 8))
            continue

        bullet = re.match(r"-\s+(.*)", stripped)
        numbered = re.match(r"(\d+)\.\s+(.*)", stripped)
        if bullet or numbered:
            if bullet:
                text = bullet.group(1)
                bullet_text = "•"
            else:
                text = numbered.group(2)
                bullet_text = f"{numbered.group(1)}."
            story.append(Paragraph(inline_markup(text), styles["list"], bulletText=bullet_text))
            idx += 1
            continue

        paragraph, idx = collect_paragraph(lines, idx)
        if paragraph:
            story.append(Paragraph(inline_markup(paragraph), styles["body"]))
        else:
            idx += 1

    if in_code and code_lines:
        text = "<br/>".join(html.escape(item) for item in code_lines)
        story.append(Paragraph(text, styles["code"]))
    return story


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.75 * cm, f"{doc.page}")
    canvas.restoreState()


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: markdown_to_pdf.py input.md output.pdf", file=sys.stderr)
        return 2

    register_fonts()
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    markdown = input_path.read_text(encoding="utf-8")
    title = next(
        (line[2:].strip() for line in markdown.splitlines() if line.startswith("# ")),
        input_path.stem,
    )

    margin_x = 1.7 * cm
    margin_y = 1.65 * cm
    page_width = A4[0] - 2 * margin_x
    styles = make_styles()
    story = parse_markdown(markdown, styles, page_width)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_y,
        bottomMargin=margin_y,
        title=title,
        author="Stock Movement Prediction",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

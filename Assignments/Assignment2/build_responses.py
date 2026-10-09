"""Build the five-page PDF from the written answers; no models are refitted."""

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate

FOLDER = Path(__file__).resolve().parent


def main():
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.7,
                          leading=12.4, spaceAfter=5)
    title = ParagraphStyle("Title", parent=body, fontName="Helvetica-Bold",
                           fontSize=14, leading=17, spaceAfter=10)
    heading = ParagraphStyle("Heading", parent=body, fontName="Helvetica-Bold",
                             fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=5)
    matrix = ParagraphStyle("Matrix", fontName="Courier", fontSize=8.2,
                            leading=9.7, spaceAfter=6)
    story = []
    paragraph = []
    code = []
    in_code = False

    def inline_text(text):
        text = escape(text)
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        return re.sub(r"`(.*?)`", r'<font name="Courier">\1</font>', text)

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline_text(" ".join(paragraph)), body))
            paragraph.clear()

    for line in (FOLDER / "Assignment2_Responses.md").read_text().splitlines():
        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(Preformatted("\n".join(code), matrix))
                code.clear()
            in_code = not in_code
        elif in_code:
            code.append(line)
        elif not line.strip():
            flush_paragraph()
        elif line == r"\newpage":
            flush_paragraph()
            story.append(PageBreak())
        elif line.startswith("# ") or line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(escape(line.lstrip("# ")), heading if line.startswith("##") else title))
        elif line.startswith("!["):
            flush_paragraph()
            path = re.search(r"\]\((.*?)\)", line).group(1)
            figure = Image(str(FOLDER / path))
            ratio = min(500 / figure.imageWidth, 148 / figure.imageHeight)
            figure.drawWidth = figure.imageWidth * ratio
            figure.drawHeight = figure.imageHeight * ratio
            figure.spaceAfter = 6
            story.append(figure)
        elif line.startswith("- "):
            flush_paragraph()
            story.append(Paragraph(inline_text(line[2:]), body, bulletText="-"))
        else:
            paragraph.append(line)
    flush_paragraph()

    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor("#bbbbbb"))
        canvas.line(44, 37, 568, 37)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(44, 25, "FinTech 545 | Assignment 2")
        canvas.drawRightString(568, 25, f"Page {document.page}")

    document = SimpleDocTemplate(str(FOLDER / "Assignment2_Responses.pdf"),
                                 pagesize=(612, 792), leftMargin=44, rightMargin=44,
                                 topMargin=40, bottomMargin=46,
                                 title="Assignment 2 - Responses", author="Shanpeiyuan")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Created", FOLDER / "Assignment2_Responses.pdf")


if __name__ == "__main__":
    main()

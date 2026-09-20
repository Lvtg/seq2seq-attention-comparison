"""Build the five page report from output/pdf/project_report.md and saved data.

Run with a Python environment containing reportlab and pypdf. All figures are
drawn as PDF vectors directly from the existing experiment artifacts.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable, Paragraph, Table, TableStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
SOURCE = OUT / "project_report.md"
DEST = OUT / "project_report.pdf"
QA = ROOT / "tmp" / "pdfs"
W, H = A4
MARGIN, GAP, BOTTOM = 54, 20, 49
CW = (W - 2 * MARGIN - GAP) / 2

for name, filename in [("ReportSerif", "times.ttf"), ("ReportSerif-Bold", "timesbd.ttf"),
                       ("ReportSerif-Italic", "timesi.ttf"), ("ReportSerif-BoldItalic", "timesbi.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(Path("C:/Windows/Fonts") / filename)))
pdfmetrics.registerFontFamily("ReportSerif", normal="ReportSerif", bold="ReportSerif-Bold",
                            italic="ReportSerif-Italic", boldItalic="ReportSerif-BoldItalic")

STYLES = {
    "body": ParagraphStyle("body", fontName="ReportSerif", fontSize=10.5, leading=12.7,
                           alignment=TA_JUSTIFY, spaceAfter=7, splitLongWords=False),
    "h2": ParagraphStyle("h2", fontName="ReportSerif-Bold", fontSize=12, leading=14.5,
                         spaceBefore=10, spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="ReportSerif-Bold", fontSize=10.8, leading=13,
                         spaceBefore=8, spaceAfter=5),
    "caption": ParagraphStyle("caption", fontName="ReportSerif", fontSize=9.4, leading=11.2,
                              alignment=TA_LEFT, spaceBefore=5, spaceAfter=9),
    "ref": ParagraphStyle("ref", fontName="ReportSerif", fontSize=9.6, leading=11.4,
                          leftIndent=8, firstLineIndent=-8, spaceAfter=7),
    "eq": ParagraphStyle("eq", fontName="ReportSerif", fontSize=10, leading=14.3,
                         alignment=TA_CENTER, spaceBefore=3, spaceAfter=9),
    "cell": ParagraphStyle("cell", fontName="ReportSerif", fontSize=9.3, leading=10.8),
}

MODELS = ["lstm_no_attention", "lstm_attention", "gated_attention", "transformer_small"]
LABELS = ["LSTM", "Attention", "Gate", "Transformer"]
PALETTE = ["#666666", "#205f9e", "#b45925", "#38816a"]


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def markup(text):
    text = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<a href="\2" color="#183c70">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = text.replace("<sub>", '<sub rise="2" size="7">')
    text = text.replace("<super>", '<super rise="3.5" size="7">')
    return text


class Plot(Flowable):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.width = CW
        self.height = {"loss": 176, "length": 179, "gate": 123, "attention": 145}[kind]

    def text(self, x, y, value, size=8, align="left", font="ReportSerif"):
        self.canv.setFillColor(colors.black)
        self.canv.setFont(font, size)
        {"left": self.canv.drawString, "center": self.canv.drawCentredString,
         "right": self.canv.drawRightString}[align](x, y, str(value))

    def axes(self, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, top=20, bottom=28):
        c = self.canv
        left, right = 30, self.width - 7
        low, high = bottom, self.height - top
        fx = lambda value: left + (value - xmin) / (xmax - xmin) * (right - left)
        fy = lambda value: low + (value - ymin) / (ymax - ymin) * (high - low)
        c.setLineWidth(.35)
        for value in yticks:
            c.setStrokeColor(colors.HexColor("#dddddd"))
            c.line(left, fy(value), right, fy(value))
            self.text(left - 5, fy(value) - 2.5, f"{value:g}", 7.6, "right")
        c.setStrokeColor(colors.black)
        c.line(left, low, left, high)
        c.line(left, low, right, low)
        for value, label in xticks:
            c.line(fx(value), low, fx(value), low - 3)
            self.text(fx(value), low - 12, label, 7.6, "center")
        self.text((left + right) / 2, 3, xlabel, 8.3, "center")
        c.saveState()
        c.translate(8, (low + high) / 2)
        c.rotate(90)
        self.text(0, 0, ylabel, 8.3, "center")
        c.restoreState()
        return fx, fy

    def legend(self):
        for i, (name, colour) in enumerate(zip(LABELS, PALETTE)):
            x = 35 + (i % 2) * 94
            y = self.height - 8 - (i // 2) * 11
            self.canv.setStrokeColor(colors.HexColor(colour))
            self.canv.setLineWidth(1.4)
            self.canv.line(x, y + 2, x + 12, y + 2)
            self.text(x + 16, y, name, 8)

    def line(self, xs, ys, fx, fy, colour, marks=False):
        c = self.canv
        c.setStrokeColor(colors.HexColor(colour))
        c.setFillColor(colors.HexColor(colour))
        c.setLineWidth(1.15)
        path = c.beginPath()
        path.moveTo(fx(xs[0]), fy(ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(fx(x), fy(y))
        c.drawPath(path)
        if marks:
            for x, y in zip(xs, ys):
                c.circle(fx(x), fy(y), 2, fill=1, stroke=0)

    def draw(self):
        c = self.canv
        if self.kind == "loss":
            self.legend()
            fx, fy = self.axes(1, 20, 1.7, 3.6, [(i, str(i)) for i in [1, 5, 10, 15, 20]],
                               [2, 2.5, 3, 3.5], "Epoch", "Validation loss", top=29)
            for model, colour in zip(MODELS, PALETTE):
                records = [json.loads(line) for line in (ROOT / "runs" / model / "metrics.jsonl").read_text().splitlines()]
                self.line([r["epoch"] for r in records], [r["valid_loss"] for r in records], fx, fy, colour)
                best = min(records, key=lambda r: r["valid_loss"])
                c.setFillColor(colors.HexColor(colour))
                c.circle(fx(best["epoch"]), fy(best["valid_loss"]), 2.6, fill=1, stroke=0)
        elif self.kind == "length":
            self.legend()
            fx, fy = self.axes(-.08, 2.08, 8, 36,
                               [(0, "1 to 10"), (1, "11 to 20"), (2, "21 to 30")],
                               [10, 15, 20, 25, 30, 35], "Source tokens", "Test BLEU", top=33, bottom=39)
            rows = read_csv("analysis/length_buckets/length_bucket_results.csv")
            for model, colour in zip(MODELS, PALETTE):
                values = [float(next(r["bleu"] for r in rows if r["model_id"] == model
                                    and r["result_set"] == "greedy" and r["bucket"] == bucket))
                          for bucket in ["1-10", "11-20", "21-30"]]
                self.line([0, 1, 2], values, fx, fy, colour, True)
            for x, n in enumerate([283, 661, 54]):
                self.text(fx(x), 14, f"n = {n}", 7.5, "center")
        elif self.kind == "gate":
            fx, fy = self.axes(8, 31, .67, .73, [(i, str(i)) for i in [10, 15, 20, 25, 30]],
                               [.68, .70, .72], "Source tokens", "Mean gate", top=8)
            rows = read_csv("analysis/attention_gate/gate_stats.csv")
            c.setFillColor(colors.HexColor(PALETTE[2]))
            for row in rows:
                c.circle(fx(float(row["source_length_without_eos"])), fy(float(row["gate_mean"])), 2.8,
                         fill=1, stroke=0)
            self.text(self.width - 11, self.height - 12, "n = 12; r = -0.090", 8, "right")
        elif self.kind == "attention":
            record = json.loads((ROOT / "artifacts/gates/gated_attention_test_gate_examples.json").read_text(encoding="utf-8"))[0]
            src, hyp, matrix = record["source_tokens"], record["hypothesis_tokens"], record["attention"]
            left, low = 67, 32
            cellw = (self.width - left - 6) / len(src)
            cellh = (self.height - low - 15) / len(hyp)
            for i, token in enumerate(hyp):
                y = low + (len(hyp) - 1 - i) * cellh
                self.text(left - 4, y + cellh / 2 - 2, token, 6.7, "right")
                for j, value in enumerate(matrix[i]):
                    c.setFillColor(colors.Color(1 - .89 * value, 1 - .64 * value, 1 - .4 * value))
                    c.rect(left + j * cellw, y, cellw + .1, cellh + .1, fill=1, stroke=0)
            for j, token in enumerate(src):
                c.saveState()
                c.translate(left + (j + .5) * cellw, low - 5)
                c.rotate(55)
                self.text(0, 0, token, 6.5, "right")
                c.restoreState()
            self.text(left, self.height - 7, "Attention weight: white 0; blue 1", 7.5)


def make_table(lines):
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in lines if not re.fullmatch(r"[|\s:\-]+", line)]
    count = len(rows[0])
    if count == 2:
        widths = [CW * .55, CW * .45]
    elif count == 3:
        widths = [CW * .46, CW * .28, CW * .26]
    elif count == 4:
        widths = [CW * .37, CW * .21, CW * .21, CW * .21]
    else:
        widths = [CW * .29, CW * .14, CW * .19, CW * .19, CW * .19]
    cells = []
    for i, row in enumerate(rows):
        converted = []
        for j, value in enumerate(row):
            style = ParagraphStyle("table-cell", parent=STYLES["cell"], alignment=TA_LEFT if j == 0 else 2)
            value = markup(value)
            converted.append(Paragraph(f"<b>{value}</b>" if i == 0 else value, style))
        cells.append(converted)
    table = Table(cells, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE", (0, 0), (-1, 0), .6, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), .35, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), .6, colors.black),
    ]))
    table.spaceBefore, table.spaceAfter = 4, 0
    return table


def parse(source):
    title = re.search(r"^# (.+)$", source, re.M)[1]
    author = re.search(r"<!-- author: (.*?) -->", source)[1]
    pages = {}
    for part in re.split(r"<!-- page: (\d+) -->", source)[1:][::2]:
        pages[int(part)] = {}
    chunks = re.split(r"<!-- page: (\d+) -->", source)
    for n in range(1, len(chunks), 2):
        page = int(chunks[n])
        cols = re.split(r"<!-- column: (\d+) -->", chunks[n + 1])
        for j in range(1, len(cols), 2):
            pages[page][int(cols[j])] = cols[j + 1].strip()
    return title, author, pages


def blocks(content):
    references = False
    for block in re.split(r"\n\s*\n", content):
        block = block.strip()
        if not block:
            continue
        if block.startswith("!["):
            kind = re.fullmatch(r"!\[[^]]+\]\(figures/(\w+)\.png\)", block)[1]
            yield Plot(kind), block
        elif block.startswith("|"):
            yield make_table(block.splitlines()), block
        else:
            if block.startswith("### "):
                key, text = "h3", block[4:]
            elif block.startswith("## "):
                key, text = "h2", block[3:]
                if text == "References":
                    references = True
            elif block.startswith('<p class="equation">'):
                key, text = "eq", block[len('<p class="equation">'):-4]
            elif re.match(r"(?:Table|Figure) \d+:", block):
                key, text = "caption", block
            else:
                key, text = ("ref" if references else "body"), block
            yield Paragraph(markup(text.replace("\n", " ")), STYLES[key]), block


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    title, author, pages = parse(source)
    assert sorted(pages) == [1, 2, 3, 4, 5]
    canvas = Canvas(str(DEST), pagesize=A4, pageCompression=1)
    canvas.setTitle(title)
    canvas.setAuthor(author if author != "Course Project Report" else "")
    canvas.setSubject("Multi30k English to German translation course project")
    layout = []
    for page, columns in sorted(pages.items()):
        top = H - 54
        if page == 1:
            title_style = ParagraphStyle("title", fontName="ReportSerif-Bold", fontSize=17, leading=20,
                                         alignment=TA_CENTER)
            title_par = Paragraph(title, title_style)
            _, ht = title_par.wrap(W - 2 * MARGIN, 100)
            title_par.drawOn(canvas, MARGIN, H - 69 - ht)
            canvas.setFont("ReportSerif", 11)
            canvas.drawCentredString(W / 2, H - 128, author)
            top = H - 163
        for col, content in sorted(columns.items()):
            x = MARGIN + (col - 1) * (CW + GAP)
            y = top
            for i, (flowable, raw) in enumerate(blocks(content)):
                before = 0 if i == 0 else flowable.getSpaceBefore()
                y -= before
                _, height = flowable.wrap(CW, H)
                layout.append({"page": page, "column": col, "top": round(y, 2),
                               "bottom": round(y - height, 2), "block": raw[:110]})
                flowable.drawOn(canvas, x, y - height)
                y -= height + flowable.getSpaceAfter()
            print(f"Page {page}, column {col}: bottom={y:.1f}; free={y - BOTTOM:.1f} pt")
        canvas.setFont("ReportSerif", 10)
        canvas.drawCentredString(W / 2, 29, str(page))
        canvas.showPage()
    canvas.save()
    (QA / "report_layout.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")
    reader = PdfReader(DEST)
    assert len(reader.pages) == 5
    extracted = "\n\n".join(p.extract_text() for p in reader.pages)
    assert "\x00" not in extracted, "An unsupported character was found in the PDF"
    (QA / "report_extracted.txt").write_text(extracted, encoding="utf-8")
    overflow = [r for r in layout if r["bottom"] < BOTTOM]
    print(f"PDF: {DEST}; pages: {len(reader.pages)}; words: {len(extracted.split())}")
    if overflow:
        print("OVERFLOW:", json.dumps(overflow, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    figure_dir = OUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm is required to render the Markdown figure assets")
    for kind in ["loss", "length", "gate", "attention"]:
        plot = Plot(kind)
        figure_pdf = QA / f"figure_{kind}.pdf"
        figure_canvas = Canvas(str(figure_pdf), pagesize=(plot.width, plot.height))
        plot.drawOn(figure_canvas, 0, 0)
        figure_canvas.showPage()
        figure_canvas.save()
        subprocess.run([pdftoppm, "-r", "250", "-singlefile", "-png", str(figure_pdf),
                        str(figure_dir / kind)], check=True)


if __name__ == "__main__":
    main()

"""Standalone scientific charts using bundled ReportLab, rendered by PDFium."""
from pathlib import Path
import numpy as np
from reportlab.graphics.shapes import Drawing, String, Rect, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib import colors
from reportlab.graphics import renderPDF
import pypdfium2

PALETTE = ["#1765ad", "#d25d21", "#278761", "#8c4caa", "#333333", "#bd425c"]


def save(drawing, path):
    path = Path(path).with_suffix(".pdf")
    renderPDF.drawToFile(drawing, str(path))
    pdf = pypdfium2.PdfDocument(str(path))
    pdf[0].render(scale=1.7).to_pil().save(path.with_suffix(".png"))
    pdf.close()


def lines(path, series, title, xlabel, ylabel, subtitle=""):
    drawing = Drawing(680, 360)
    drawing.add(String(42, 336, title, fontSize=14, fontName="Helvetica-Bold"))
    drawing.add(String(42, 319, subtitle, fontSize=8, fillColor=colors.HexColor("#555555")))
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 64, 70, 550, 213
    lp.data = [list(zip(np.asarray(x).tolist(), np.asarray(y).tolist())) for name, x, y in series]
    lp.joinedLines = 1
    for i, (name, _, _) in enumerate(series):
        color = colors.HexColor(PALETTE[i % len(PALETTE)])
        lp.lines[i].strokeColor = color
        lp.lines[i].strokeWidth = 1.6
        col, row = i % 3, i // 3
        xx, yy = 64 + col*195, 37 - row*13
        drawing.add(Line(xx, yy, xx+15, yy, strokeColor=color, strokeWidth=2))
        drawing.add(String(xx+20, yy-3, name[:32], fontSize=8))
    lp.xValueAxis.labels.fontSize = 8
    lp.yValueAxis.labels.fontSize = 8
    lp.xValueAxis.labelTextFormat = "%.2g"
    lp.yValueAxis.labelTextFormat = "%.3g"
    drawing.add(lp)
    drawing.add(String(335, 50, xlabel, textAnchor="middle", fontSize=10))
    drawing.add(String(10, 295, ylabel, fontSize=9))
    save(drawing, path)


def heatmaps(path, fields, x, y, titles, units, overall_title, sensors=None):
    n = len(fields)
    drawing = Drawing(365*n, 350)
    drawing.add(String(35, 328, overall_title, fontName="Helvetica-Bold", fontSize=13))
    for k, (field, title, unit) in enumerate(zip(fields, titles, units)):
        xx, yy, width, height = 45+365*k, 65, 280, 210
        # Common limits for quantities with the same title/units must be supplied by caller;
        # each panel here represents a different quantity and has an explicit scale.
        lo, hi = float(np.nanmin(field)), float(np.nanmax(field))
        if hi == lo:
            hi = lo + 1
        nx, ny = field.shape
        sx, sy = max(1, nx//65), max(1, ny//55)
        for i in range(0, nx, sx):
            for j in range(0, ny, sy):
                q = float(np.clip((field[i:i+sx, j:j+sy].mean()-lo)/(hi-lo), 0, 1))
                # A fixed sequential blue-white-orange palette, with displayed min/max.
                a = np.array([.08, .25, .5])
                b = np.array([.98, .95, .85])
                c = np.array([.7, .18, .08])
                rgb = a*(1-2*q)+b*(2*q) if q < .5 else b*(2-2*q)+c*(2*q-1)
                drawing.add(Rect(xx+i/nx*width, yy+j/ny*height,
                                 min(sx,nx-i)/nx*width+.2, min(sy,ny-j)/ny*height+.2,
                                 fillColor=colors.Color(*rgb), strokeColor=None))
        drawing.add(Rect(xx,yy,width,height,fillColor=None,strokeColor=colors.black,strokeWidth=.5))
        drawing.add(String(xx, 292, title, fontName="Helvetica-Bold", fontSize=10))
        drawing.add(String(xx, 278, f"Range: {lo:.3g} to {hi:.3g} {unit}", fontSize=8))
        drawing.add(String(xx+width/2, 40, "x/D", textAnchor="middle", fontSize=10))
        drawing.add(String(xx-28, yy+height/2, "y/D", fontSize=8))
        drawing.add(String(xx, yy-13, f"{x[0]:.2f}", fontSize=8))
        drawing.add(String(xx+width, yy-13, f"{x[-1]:.2f}", textAnchor="end", fontSize=8))
        drawing.add(String(xx-5, yy, f"{y[0]:.2f}", textAnchor="end", fontSize=8))
        drawing.add(String(xx-5, yy+height-3, f"{y[-1]:.2f}", textAnchor="end", fontSize=8))
        if sensors is not None:
            for i,j in sensors:
                drawing.add(Rect(xx+i/(nx-1)*width-1.4, yy+j/(ny-1)*height-1.4, 2.8,2.8,
                                 fillColor=colors.black,strokeColor=colors.white,strokeWidth=.4))
    save(drawing,path)

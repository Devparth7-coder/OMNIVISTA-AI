"""Demo document generator.

Builds a realistic PDF containing paragraphs, headings, a revenue table, a bar
chart, and an architecture diagram (both embedded as raster images with
captions). This makes the MVP acceptance path — upload a mixed PDF, ask a
visual/cross-modal question, get a grounded cited answer — work end to end
without any external service.
"""
from __future__ import annotations

import io
import math
import tempfile

from app.core.config import settings


def generate_demo_pdf() -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

    chart_png = _chart_bytes()
    diagram_png = _diagram_bytes()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.7 * inch, leftMargin=0.7 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                            title="OMNIVISTA Demo Document")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=22, spaceAfter=10)
    h1 = ParagraphStyle("H1c", parent=styles["Heading1"], fontSize=16, spaceBefore=14, spaceAfter=6)
    h2 = ParagraphStyle("H2c", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("BodyC", parent=styles["BodyText"], fontSize=10.5, leading=14)
    caption = ParagraphStyle("Caption", parent=styles["BodyText"], fontSize=9, textColor=colors.grey,
                             spaceBefore=3, spaceAfter=8)

    story = []
    story.append(Paragraph("OMNIVISTA Multimodal Architecture Overview", title_style))
    story.append(Paragraph("A demonstration document containing text, a table, a chart and a diagram.", caption))
    story.append(Spacer(1, 6))

    # --- Section 1 ---------------------------------------------------------
    story.append(Paragraph("1. System Architecture", h1))
    story.append(Paragraph(
        "The platform is built as a set of independent services connected over "
        "HTTPS. A web client renders the UI and talks to an API gateway. The "
        "gateway forwards requests to backend services that implement retrieval "
        "and generation logic. The backend persists document metadata and "
        "conversations in PostgreSQL and stores raw files in object storage.",
        body,
    ))
    story.append(Paragraph(
        "The backend does not read documents directly. Instead, an ingestion "
        "worker parses each upload, extracts text, tables, images and diagrams, "
        "indexes multimodal embeddings into a vector store, and only then marks "
        "the document as ready. Queries are answered from the indexed evidence.",
        body,
    ))

    story.append(Paragraph("1.1 Architecture Diagram", h2))
    story.append(Paragraph(
        "Figure 1 shows the request flow from the client through the API gateway "
        "to the backend and finally the database.",
        body,
    ))
    diagram_path = _write_temp_png(diagram_png)
    diagram_img = RLImage(diagram_path, width=5.4 * inch, height=2.0 * inch)
    story.append(diagram_img)
    story.append(Paragraph(
        "Figure 1: Architecture Diagram showing Client, API Gateway, Backend, Database.",
        caption,
    ))

    story.append(Paragraph("2. Revenue Performance", h1))
    story.append(Paragraph(
        "Revenue grew steadily across the three-year period. The growth is driven "
        "by increased platform usage and higher average contract value.",
        body,
    ))

    # --- Table -------------------------------------------------------------
    table_data = [
        ["Region", "2024", "2025"],
        ["North", "12M", "16M"],
        ["South", "9M", "13M"],
        ["West", "7M", "11M"],
    ]
    tbl = Table(table_data, colWidths=[1.4 * inch, 1.1 * inch, 1.1 * inch])
    tbl.setStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]
    )
    story.append(tbl)
    story.append(Paragraph("Table 1: Revenue by Region (in millions).", caption))

    story.append(Paragraph("3. Year-over-Year Growth", h1))
    story.append(Paragraph(
        "The following chart breaks down revenue by year: 2022 reached 10M, 2023 "
        "reached 14M, and 2024 reached 19M.",
        body,
    ))
    chart_path = _write_temp_png(chart_png)
    chart_img = RLImage(chart_path, width=4.8 * inch, height=2.4 * inch)
    story.append(chart_img)
    story.append(Paragraph(
        "Figure 2: Revenue by Year Bar Chart (2022 10M, 2023 14M, 2024 19M).",
        caption,
    ))

    story.append(Paragraph("4. Conclusion", h1))
    story.append(Paragraph(
        "The architecture separates concerns cleanly, and the revenue trend "
        "is positive. This document demonstrates that combining textual "
        "descriptions, structured tables, and visual charts gives a richer "
        "picture than any single modality alone.",
        body,
    ))

    doc.build(story)
    return buffer.getvalue()


def generate_demo_markdown() -> bytes:
    md = """# OMNIVISTA Demo — Markdown

## Architecture
The system uses an API Gateway between the client and the backend services.

- Client
- API Gateway
- Backend
- Database

## Revenue
Revenue grew each year.
"""
    return md.encode("utf-8")


def _write_temp_png(data: bytes) -> str:
    import tempfile

    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.write(data)
    f.close()
    return f.name


def _chart_bytes() -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 2.6), dpi=120)
    years = ["2022", "2023", "2024"]
    values = [10, 14, 19]
    bars = ax.bar(years, values, color=["#3b82f6", "#6366f1", "#8b5cf6"])
    ax.set_title("Revenue by Year")
    ax.set_ylabel("Revenue (Millions)")
    ax.set_ylim(0, 22)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v}M", ha="center", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _diagram_bytes() -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7, 2.4), dpi=120)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    nodes = [
        ("Client", 1.0, 1.5, "#dbeafe"),
        ("API Gateway", 3.6, 1.5, "#e0e7ff"),
        ("Backend", 6.2, 1.5, "#ede9fe"),
        ("Database", 8.8, 1.5, "#f3e8ff"),
    ]
    for name, x, y, color in nodes:
        box = FancyBboxPatch((x - 0.9, y - 0.5), 1.8, 1.0,
                             boxstyle="round,pad=0.05", fc=color, ec="#4b5563")
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center", fontsize=10, fontweight="bold")
    for i in range(len(nodes) - 1):
        x1 = nodes[i][1] + 0.9
        x2 = nodes[i + 1][1] - 0.9
        arrow = FancyArrowPatch((x1, 1.5), (x2, 1.5), arrowstyle="-|>",
                                mutation_scale=16, color="#374151", lw=1.5)
        ax.add_patch(arrow)
    ax.set_title("System Architecture Flow", fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()

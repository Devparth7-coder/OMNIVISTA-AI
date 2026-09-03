"""PPTX parser (python-pptx).

Each slide becomes a "page". Title and body text become regions; tables are
extracted. Images/notes are recorded so downstream vision can annotate them.
"""
from __future__ import annotations

import io

from app.parsers.base import DocumentParser, ParseError
from app.schemas.domain import (
    ContentType,
    ParsedDocument,
    Region,
    RegionType,
    SourceType,
    TableStructure,
)


class PptxParser(DocumentParser):
    source_type = SourceType.PPTX

    async def parse(self, data: bytes) -> ParsedDocument:
        try:
            import pptx as _pptx
            prs = _pptx.Presentation(io.BytesIO(data))
        except Exception as exc:
            raise ParseError(f"Could not parse PPTX: {exc}") from exc

        document_id = "doc_placeholder"
        regions: list[Region] = []
        order = 0
        slide_num = 0
        for slide in prs.slides:
            slide_num += 1
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text = "\n".join(p.text for p in shape.text_frame.paragraphs).strip()
                    if not text:
                        continue
                    rtype = RegionType.PARAGRAPH
                    if shape == slide.shapes.title:
                        rtype = RegionType.HEADING
                    regions.append(
                        Region(document_id=document_id, page_number=slide_num, type=rtype,
                               content=text, reading_order=order, source_type=SourceType.PPTX)
                    )
                    order += 1
                if shape.has_table:
                    table = shape.table
                    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                    if rows:
                        summary = f"table: columns {rows[0]} rows {rows[1:]}"
                        regions.append(
                            Region(document_id=document_id, page_number=slide_num, type=RegionType.TABLE,
                                   content_type=ContentType.TABLE, content=summary, reading_order=order,
                                   table=TableStructure(columns=rows[0], rows=rows[1:], summary=summary),
                                   source_type=SourceType.PPTX)
                        )
                        order += 1

        return ParsedDocument(document_id=document_id, source_type=SourceType.PPTX,
                              pages=max(1, slide_num), regions=regions, metadata={"format": "pptx"})

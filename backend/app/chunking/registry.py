"""Helpers that assemble :class:`MultimodalChunk` objects from regions."""
from __future__ import annotations

from app.core.config import settings
from app.schemas.domain import ContentType, MultimodalChunk, ParsedDocument, Region


def _section_for(region: Region) -> str:
    return region.hierarchy.get("section", "") or str(region.hierarchy.get("h1", ""))


def build_text_chunk(doc: ParsedDocument, regions: list[Region], extra_visual: str = "",
                     extra_tables: str = "") -> MultimodalChunk:
    text = "\n".join(r.content for r in regions).strip()
    section = _section_for(regions[0]) if regions else ""
    bboxes = [r.bbox for r in regions if r.bbox]
    bbox = _union(bboxes)
    visual = extra_visual.strip()
    table_summary = extra_tables.strip()
    return MultimodalChunk(
        document_id=doc.document_id,
        page_number=regions[0].page_number if regions else 1,
        section=section,
        text=text,
        visual_context=visual,
        region_ids=[r.region_id for r in regions],
        table_ids=_tables_seen(regions),
        image_ids=_images_seen(regions),
        content_type=ContentType.MULTIMODAL if (visual or table_summary) else ContentType.TEXT,
        bbox=bbox,
        source_type=doc.source_type,
    )


def build_table_chunk(doc: ParsedDocument, region: Region) -> MultimodalChunk:
    """Emit a readable table summary — better for retrieval and for table QA."""
    table = region.table
    cols = table.columns if table else []
    rows = table.rows if table else []
    header = ", ".join(str(c) for c in cols if c)
    row_lines = "; ".join(
        f"{r[0]} = " + ", ".join(str(v) for v in r[1:]) for r in rows if r
    )
    text = f"Table. Columns: {header}. Rows: {row_lines}."
    section = _section_for(region)
    return MultimodalChunk(
        document_id=doc.document_id,
        page_number=region.page_number,
        section=section,
        text=text,
        visual_context="",
        region_ids=[region.region_id],
        table_ids=[region.region_id],
        content_type=ContentType.TABLE,
        bbox=region.bbox,
        source_type=doc.source_type,
    )


def build_visual_chunk(doc: ParsedDocument, region: Region, image_meta: dict) -> MultimodalChunk:
    description = image_meta.get("description", "")
    chart_type = image_meta.get("chart_type")
    relationships = image_meta.get("relationships", [])
    caption = image_meta.get("caption", "")
    summary = image_meta.get("summary", "")
    section = _section_for(region)
    visual_parts = [description, caption]
    if chart_type:
        visual_parts.append(f"Visual is a {chart_type.replace('_', ' ')}.")
    if relationships:
        visual_parts.append("Relationships: " + "; ".join(f"{r.get('from','')}→{r.get('to','')}" for r in relationships))
    visual_context = "\n".join(p for p in visual_parts if p)
    text = ""
    # For diagrams, include a short textual representation of the graph.
    if relationships:
        text = "Diagram flow: " + " → ".join(
            [relationships[0].get("from", "")] + [r.get("to", "") for r in relationships]
        )
    if summary:
        text = (text + "\n" if text else "") + f"Chart summary: {summary}"
    if chart_type in ("bar", "line", "pie", "scatter", "chart"):
        content_type = ContentType.CHART
    elif chart_type == "diagram":
        content_type = ContentType.DIAGRAM
    else:
        content_type = ContentType.IMAGE
    return MultimodalChunk(
        document_id=doc.document_id,
        page_number=region.page_number,
        section=section,
        text=text,
        visual_context=visual_context,
        region_ids=[region.region_id],
        image_ids=[region.image_asset_id] if region.image_asset_id else [],
        table_ids=[],
        content_type=content_type,
        bbox=region.bbox,
        source_type=doc.source_type,
    )


def _union(bboxes: list[list[float]]) -> list[float] | None:
    if not bboxes:
        return None
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def _tables_seen(regions: list[Region]) -> list[str]:
    return [r.region_id for r in regions]


def _images_seen(regions: list[Region]) -> list[str]:
    return [r.image_asset_id for r in regions if r.image_asset_id]


def text_budget() -> int:
    return settings.TEXT_CHUNK_CHARS

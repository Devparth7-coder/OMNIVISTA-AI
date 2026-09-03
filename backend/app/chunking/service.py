"""Semantic multimodal chunking.

Rather than naive fixed-size chunks, this walks a page's layout regions in
reading order and groups them into coherent *multimodal* chunks. A single text
chunk absorbs the descriptions of tables and images that appear near it (as its
``visual_context``), so retrieval can return a whole semantic unit of evidence
instead of an arbitrary slice.

It also emits standalone **table** and **visual** chunks (image/chart/diagram) so
modality-specific retrieval and rerouting work cleanly.

All chunks retain page, section, bbox, and region/image/table references needed
for precise clickable citations.
"""
from __future__ import annotations

from app.chunking.registry import build_table_chunk, build_text_chunk, build_visual_chunk, text_budget
from app.core.config import settings
from app.schemas.domain import ContentType, MultimodalChunk, ParsedDocument, Region, RegionType


def chunk_document(doc: ParsedDocument, image_meta: dict[str, dict] | None = None) -> list[MultimodalChunk]:
    """``image_meta`` maps image id -> vision annotation dict."""
    image_meta = image_meta or {}
    chunks: list[MultimodalChunk] = []
    pages = sorted({r.page_number for r in doc.regions})
    for page in pages:
        page_regions = sorted(
            (r for r in doc.regions if r.page_number == page),
            key=lambda r: r.reading_order,
        )
        chunks.extend(_chunk_page(doc, page_regions, image_meta))
    return [c for c in chunks if c.text.strip() or c.visual_context.strip()]


def _chunk_page(doc: ParsedDocument, regions: list[Region], image_meta: dict[str, dict]) -> list[MultimodalChunk]:
    chunks: list[MultimodalChunk] = []
    text_buffer: list[Region] = []
    # Track the most recent table/image annotation so text chunks inherit context.
    last_visual: str = ""
    last_table: str = ""
    budget = text_budget()

    def flush() -> None:
        nonlocal text_buffer, last_visual, last_table
        if text_buffer:
            if text_buffer[0].content_type in (ContentType.TABLE, ContentType.IMAGE, ContentType.CHART, ContentType.DIAGRAM):
                # Standalone handling happens below; skip for text buffer.
                pass
            chunks.append(
                build_text_chunk(doc, text_buffer, extra_visual=last_visual, extra_tables=last_table)
            )
        text_buffer = []
        last_visual = ""
        last_table = ""

    for region in regions:
        if region.content_type in (ContentType.TABLE, ContentType.IMAGE, ContentType.CHART, ContentType.DIAGRAM):
            flush()
            if region.content_type == ContentType.TABLE and region.table:
                chunks.append(build_table_chunk(doc, region))
                last_table = region.table.summary
            elif region.image_asset_id:
                meta = image_meta.get(region.image_asset_id, {})
                chunks.append(build_visual_chunk(doc, region, meta))
                last_visual = meta.get("description", "")
            continue
        # Text-like region
        text_buffer.append(region)
        if _running_chars(text_buffer) >= budget:
            flush()
    flush()
    return chunks


def _running_chars(buffer: list[Region]) -> int:
    return sum(len(r.content) for r in buffer)

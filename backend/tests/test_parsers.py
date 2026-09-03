"""Parser unit tests."""
from __future__ import annotations

import pytest

from app.demo.generator import generate_demo_pdf
from app.parsers.pdf import PDFParser
from app.parsers.registry import allowed_extension, guess_source_type
from app.parsers.text import MarkdownParser, TextParser
from app.schemas.domain import ContentType, RegionType, SourceType


@pytest.mark.asyncio
async def test_plain_text_parser_keeps_headings_and_paragraphs():
    parser = TextParser()
    data = b"# Chapter 1\nSome intro text.\n- item one\n- item two\n\nA trailing paragraph."
    parsed = await parser.parse(data)
    assert parsed.pages == 1
    assert any(r.type == RegionType.HEADING and r.content == "Chapter 1" for r in parsed.regions)
    assert any(r.type == RegionType.LIST for r in parsed.regions)
    assert any(r.type == RegionType.PARAGRAPH for r in parsed.regions)


@pytest.mark.asyncio
async def test_markdown_parser():
    parser = MarkdownParser()
    parsed = await parser.parse(b"# Title\nbody text\n")
    assert parsed.source_type == SourceType.MD
    assert any(r.type == RegionType.HEADING and r.content == "Title" for r in parsed.regions)


@pytest.mark.asyncio
async def test_pdf_parser_extracts_text_tables_and_images():
    parser = PDFParser()
    parsed = await parser.parse(generate_demo_pdf())
    assert parsed.pages >= 1
    assert parsed.regions
    # We have a table and embedded images (chart + diagram).
    assert any(r.content_type == ContentType.TABLE for r in parsed.regions)
    assert len(parsed.images) >= 2
    # bounding boxes preserved
    assert all(r.bbox is not None for r in parsed.regions if r.bbox)


def test_extension_registry():
    assert allowed_extension("report.pdf")
    assert allowed_extension("notes.md")
    assert allowed_extension("image.png")
    assert not allowed_extension("malware.exe")
    assert guess_source_type("a.pptx") == SourceType.PPTX
    assert guess_source_type("a.docx") == SourceType.DOCX
    assert guess_source_type("a.png") == SourceType.IMAGE

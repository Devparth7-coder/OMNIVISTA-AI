"""Parser registry: maps a file extension to a parser, and detects the source
type from an uploaded file. New formats are registered here without touching the
ingestion pipeline.
"""
from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from app.parsers.base import DocumentParser, ParseError
from app.parsers.docx import DocxParser
from app.parsers.pdf import PDFParser
from app.parsers.pptx import PptxParser
from app.parsers.text import MarkdownParser, TextParser
from app.schemas.domain import SourceType

_EXT_PARSERS: dict[str, type[DocumentParser]] = {
    ".pdf": PDFParser,
    ".docx": DocxParser,
    ".pptx": PptxParser,
    ".txt": TextParser,
    ".md": MarkdownParser,
}

# Source types that are images themselves (scanned/single images).
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

_ALLOWED_EXTENSIONS = set(_EXT_PARSERS) | _IMAGE_EXTENSIONS


def is_image_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in _IMAGE_EXTENSIONS


def allowed_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in _ALLOWED_EXTENSIONS


def guess_source_type(filename: str) -> SourceType | None:
    ext = Path(filename).suffix.lower()
    if ext in _IMAGE_EXTENSIONS:
        return SourceType.IMAGE
    if ext == ".pdf":
        return SourceType.PDF
    if ext in (".txt", ".md"):
        return SourceType.TXT
    if ext == ".docx":
        return SourceType.DOCX
    if ext == ".pptx":
        return SourceType.PPTX
    return None


def get_parser(source_type: SourceType) -> DocumentParser:
    mapping: dict[SourceType, type[DocumentParser]] = {
        SourceType.PDF: PDFParser,
        SourceType.DOCX: DocxParser,
        SourceType.PPTX: PptxParser,
        SourceType.TXT: TextParser,
        SourceType.MD: MarkdownParser,
    }
    cls = mapping.get(source_type)
    if cls is None:
        raise ParseError(f"No parser registered for {source_type}")
    return cls()


def get_mime_type(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def sanitize_filename(filename: str) -> str:
    base = Path(filename).name
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base[:120] or "upload"

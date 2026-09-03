"""Core domain models.

These are the structured representations produced by the ingestion pipeline and
consumed by retrieval, reranking, generation, and citation building. They are
deliberately framework-agnostic so the same objects can persist to PostgreSQL
later without change.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Region / layout types
# ---------------------------------------------------------------------------
class RegionType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    DIAGRAM = "diagram"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    EQUATION = "equation"
    CODE = "code"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"
    OTHER = "other"


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    DIAGRAM = "diagram"
    OCR = "ocr"
    MULTIMODAL = "multimodal"


class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"
    MD = "md"
    IMAGE = "image"
    DEMO = "demo"
    OTHER = "other"


BBox = list[float]  # [x1, y1, x2, y2]


class TableCell(BaseModel):
    text: str = ""
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1


class TableStructure(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    summary: str = ""  # textual semantic summary for embedding/retrieval


class Region(BaseModel):
    """A single layout region on a page (paragraph, table, image, ...)."""

    document_id: str
    page_number: int
    region_id: str = Field(default_factory=lambda: new_id("region"))
    type: RegionType = RegionType.PARAGRAPH
    content_type: ContentType = ContentType.TEXT
    bbox: BBox | None = None
    content: str = ""
    reading_order: int = 0
    confidence: float = 1.0
    # structured payloads (set per content type)
    table: TableStructure | None = None
    image_asset_id: str | None = None
    source_type: SourceType = SourceType.OTHER
    is_ocr: bool = False
    hierarchy: dict[str, str] = Field(default_factory=dict)  # section/heading context
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageAsset(BaseModel):
    image_id: str = Field(default_factory=lambda: new_id("img"))
    document_id: str
    page_number: int
    storage_path: str
    bbox: BBox | None = None
    description: str = ""
    objects: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    caption: str = ""
    chart_data: dict[str, Any] | None = None  # parsed chart structure
    mime_type: str = "image/png"
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultimodalChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: new_id("chunk"))
    document_id: str
    page_number: int
    section: str = ""
    text: str = ""
    visual_context: str = ""
    region_ids: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    content_type: ContentType = ContentType.TEXT
    bbox: BBox | None = None
    source_type: SourceType = SourceType.OTHER
    embedding: list[float] | None = None  # set downstream


class ParsedDocument(BaseModel):
    """Result of the parsing/layout stage for one document."""

    document_id: str
    source_type: SourceType
    pages: int = 0
    regions: list[Region] = Field(default_factory=list)
    images: list[ImageAsset] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval / generation
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    content_type: ContentType
    section: str = ""
    text: str = ""
    visual_context: str = ""
    region_ids: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    bbox: BBox | None = None
    score: float = 0.0
    modality: str = "text"


class Citation(BaseModel):
    citation_id: str = Field(default_factory=lambda: new_id("cite"))
    document_id: str
    document_name: str = ""
    page: int
    region_id: str = ""
    content_type: ContentType = ContentType.TEXT
    snippet: str = ""
    bbox: BBox | None = None
    image_ids: list[str] = Field(default_factory=list)


class AnswerEvidence(BaseModel):
    grounded: bool
    confidence: float
    unsupported_claims: list[str] = Field(default_factory=list)
    citations_valid: bool = True
    used_chunks: int = 0


class GenerationResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence: AnswerEvidence
    latency_ms: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    reasoning: list[str] = Field(default_factory=list)


class QueryIntent(BaseModel):
    modality: str = "text"  # text | table | image | diagram | mixed
    entities: list[str] = Field(default_factory=list)
    page: int | None = None
    content_type: ContentType | None = None
    needs_visual: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)
    original: str = ""


# ---------------------------------------------------------------------------
# Document records / status
# ---------------------------------------------------------------------------
class DocumentStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PARSING = "PARSING"
    OCR = "OCR"
    ANALYZING = "ANALYZING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"


class DocumentRecord(BaseModel):
    document_id: str = Field(default_factory=lambda: new_id("doc"))
    name: str
    source_type: SourceType = SourceType.OTHER
    size_bytes: int = 0
    storage_path: str = ""
    status: DocumentStatus = DocumentStatus.UPLOADED
    pages: int = 0
    chunks: int = 0
    image_count: int = 0
    table_count: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    processing_ms: int = 0
    stage_detail: str = ""


class ConversationMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    role: str  # user | assistant
    content: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class Conversation(BaseModel):
    conversation_id: str = Field(default_factory=lambda: new_id("conv"))
    title: str = "New conversation"
    document_ids: list[str] = Field(default_factory=list)
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

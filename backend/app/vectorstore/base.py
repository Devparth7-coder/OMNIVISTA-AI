"""Vector store abstraction.

Every vector carries the metadata required for grounded citations:
document_id, page_number, chunk_id, content_type, section, bbox, source_type.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.domain import ContentType


class SearchHit(BaseModel):
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
    bbox: list[float] | None = None
    score: float = 0.0


class VectorStore(ABC):
    """Interface for upserting/searching/removing vectors."""

    @abstractmethod
    async def upsert(self, chunk_id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        ...

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> int:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...

    @abstractmethod
    async def clear(self) -> None:
        ...

    @abstractmethod
    async def all_hits(self) -> list[SearchHit]:
        """Return every indexed chunk (used for lexical search / knowledge graph)."""
        ...

"""Deterministic, dependency-free embedding provider used in mock mode."""
from __future__ import annotations

from typing import Iterable

from app.core.config import settings
from app.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embed_dim: int | None = None) -> None:
        self.embedding_dim = embed_dim or settings.EMBEDDING_DIM

    async def embed_text(self, texts: Iterable[str]) -> list[list[float]]:
        return await self._async_hashed(texts)

    async def embed_image(self, images: Iterable[bytes]) -> list[list[float]]:
        return [self._embed_from_bytes(b) for b in images]

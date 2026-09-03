"""Reranker abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.domain import RetrievedChunk


class RerankerProvider(ABC):
    @abstractmethod
    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ...

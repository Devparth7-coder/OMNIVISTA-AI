"""Embedding provider abstraction.

The vector store and the rest of the pipeline depend only on this interface, so
the concrete model (OpenAI, Cohere, a local sentence-model, a vision model) can
be swapped without touching retrieval code.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Iterable

_TOKEN_RE = re.compile(r"[\w'-]+|[^\w\s]")


class EmbeddingProvider(ABC):
    embedding_dim: int = 384

    @abstractmethod
    async def embed_text(self, texts: Iterable[str]) -> list[list[float]]:
        """Return a normalized embedding vector for each input text."""

    async def embed_image(self, images: Iterable[bytes]) -> list[list[float]]:
        """Return an embedding for each image (defaults to a text-free hash)."""
        return [self._embed_from_bytes(b) for b in images]

    async def embed_multimodal(self, texts: Iterable[str], images: Iterable[bytes] | None = None) -> list[list[float]]:
        return await self.embed_text(texts)

    async def embed_query(self, query: str) -> list[float]:
        return (await self.embed_text([query]))[0]

    # ---- shared helper for deterministic mock vectors -------------------------
    def _embed_from_bytes(self, blob: bytes) -> list[float]:
        digest = hashlib.sha256(blob).digest()
        vec = [b / 255.0 for b in digest[: self.embedding_dim]]
        return _normalize(vec)

    def _hashed_vector(self, text: str) -> list[float]:
        """Bag-of-features hash embedding.

        Uses character n-grams (+ word features) mapped into a fixed dimension
        so that lexically similar text lands near each other in vector space.
        This is a deliberately-naive stand-in for a real semantic model, but it is
        deterministic and lets the whole pipeline run OOTB with no API keys.
        """
        vec = [0.0] * self.embedding_dim
        lower = text.lower()
        gram_features: list[str] = []
        # word unigrams
        gram_features.extend(_TOKEN_RE.findall(lower))
        # character trigrams
        for i in range(len(lower) - 2):
            gram_features.append(lower[i : i + 3])
        for feat in gram_features:
            idx = int(hashlib.md5(feat.encode()).hexdigest(), 16) % self.embedding_dim
            vec[idx] += 1.0
        return _normalize(vec)

    async def _async_hashed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._hashed_vector(t) for t in texts]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]

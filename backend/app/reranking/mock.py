"""Mock reranker.

Blends the vector score with a lightweight lexical overlap score and a small
document-structure prior (preferring chunks with a section/heading and visual
context for visual queries). Deterministic and dependency-free.
"""
from __future__ import annotations

import re

from app.reranking.base import RerankerProvider
from app.schemas.domain import RetrievedChunk


def _q_tokens(q: str) -> set[str]:
    return {w for w in re.findall(r"[\w'-]+", q.lower()) if len(w) > 1}


class MockRerankerProvider(RerankerProvider):
    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        q_tokens = _q_tokens(query)
        for c in candidates:
            hay = f"{c.text} {c.visual_context} {c.section}"
            c_tokens = _q_tokens(hay)
            overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            lexical = overlap * 0.5
            structure = 0.1 if c.section or c.visual_context else 0.0
            c.score = round(0.5 * c.score + lexical + structure, 5)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

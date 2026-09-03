"""Agentic retrieval loop.

For complex queries the system may retrieve, evaluate whether the evidence is
sufficient, and refine the query / broaden the search before generating. The
loop is hard-capped at ``MAX_RETRIEVAL_ITERATIONS`` to prevent infinite loops.
"""
from __future__ import annotations

from app.core.config import settings
from app.retrieval.hybrid import RetrievalEngine
from app.schemas.domain import DocumentRecord, QueryIntent, RetrievedChunk
from app.services.generation import GenerationService
from app.services.query import analyze_query


class AgenticRag:
    def __init__(self, retrieval: RetrievalEngine, generation: GenerationService) -> None:
        self.retrieval = retrieval
        self.generation = generation

    async def run(self, query: str, documents: dict[str, DocumentRecord],
                  document_ids: list[str] | None = None) -> dict:
        intent = analyze_query(query, document_ids)
        iterations = 0
        all_chunks: list[RetrievedChunk] = []
        refined_query = query
        refreshed = False

        while iterations < settings.MAX_RETRIEVAL_ITERATIONS:
            iterations += 1
            chunks = await self.retrieval.hybrid_search(
                refined_query, intent, document_ids=document_ids or intent.filters.get("document_ids")
            )
            all_chunks = _merge_chunks(all_chunks, chunks)

            # Evaluate the quality of this retrieval pass.
            quality = self._evaluate(chunks, intent)
            if quality["sufficient"]:
                break

            # Refine: relax filters (drop the hard page/type constraint) once,
            # then broaden the query terms.
            if not refreshed and (intent.page is not None or intent.content_type is not None):
                intent.page = None
                intent.content_type = None
                intent.filters.pop("page", None)
                intent.filters.pop("content_type", None)
                refined_query = query
                refreshed = True
                continue
            # Broaden by dropping function words from the query.
            refined_query = _broaden(query, iterations)
            if refined_query.strip() == "":
                break

        # Final generation with merged evidence.
        result = await self.generation.answer(
            refined_query if refined_query else query, all_chunks, documents
        )
        return {
            "answer": result.answer,
            "citations": result.citations,
            "evidence": result.evidence,
            "iterations": iterations,
            "intent": intent,
            "retrieved_chunks": len(all_chunks),
            "total_latency_ms": result.latency_ms,
        }

    def _evaluate(self, chunks: list[RetrievedChunk], intent: QueryIntent) -> dict:
        if not chunks:
            return {"sufficient": False, "reason": "no_chunks"}
        avg = sum(c.score for c in chunks) / len(chunks)
        # A rough sufficiency heuristic: expect a non-trivial top score.
        top = chunks[0].score
        sufficient = top > 0.15 and avg > 0.05
        return {"sufficient": sufficient, "reason": "ok" if sufficient else "weak", "top_score": top, "avg": avg}


def _merge_chunks(a: list[RetrievedChunk], b: list[RetrievedChunk]) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {c.chunk_id: c for c in a}
    for c in b:
        if c.chunk_id in merged:
            merged[c.chunk_id].score = max(merged[c.chunk_id].score, c.score)
        else:
            merged[c.chunk_id] = c
    return sorted(merged.values(), key=lambda c: c.score, reverse=True)


def _broaden(query: str, iteration: int) -> str:
    import re

    stop = {"the", "a", "an", "of", "in", "on", "and", "for", "with", "to", "is", "does", "show", "what", "how", "which", "describe", "explain"}
    words = [w for w in re.findall(r"[\w'\-\d]+", query.lower()) if w not in stop]
    if iteration >= 2 and words:
        # drop the first remaining word to broaden
        words = words[1:]
    return " ".join(words)

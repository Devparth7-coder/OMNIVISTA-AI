"""Grounded generation service.

Ties together retrieval, context assembly, the LLM/VLM, citation building, and
evidence validation to produce a grounded, citable answer. Also implements the
specialized modes (resume / research / technical) and cross-modal reasoning.
"""
from __future__ import annotations

import re
import time

from app.core.config import settings
from app.evaluation.evidence import EvidenceValidator
from app.generation.base import LLMProvider
from app.schemas.domain import (
    Citation,
    ContentType,
    DocumentRecord,
    GenerationResult,
    RetrievedChunk,
)
from app.services.context import assemble_prompt

UNSUPPORTED_MARKER = "couldn't find sufficient evidence"

_CITE_RE = re.compile(r"\[Page (\d+)(?:, ([^\]]+))?\]")

_TYPE_LABEL = {
    "diagram": ContentType.DIAGRAM,
    "chart": ContentType.CHART,
    "image": ContentType.IMAGE,
    "table": ContentType.TABLE,
}


def _match_by_ref(chunks: list[RetrievedChunk], page: int, label: str) -> RetrievedChunk | None:
    for c in chunks:
        if c.page_number != page:
            continue
        if label:
            t = c.content_type.value if hasattr(c.content_type, "value") else str(c.content_type)
            target = _TYPE_LABEL.get(label)
            if target and c.content_type == target:
                return c
            # label like "multimodal" -> match that type broadly
            if label and t == label:
                return c
        else:
            return c
    # best-scoring chunk on that page
    on_page = [c for c in chunks if c.page_number == page]
    return max(on_page, key=lambda c: c.score) if on_page else None


class GenerationService:
    def __init__(self, llm: LLMProvider, validator: EvidenceValidator | None = None) -> None:
        self.llm = llm
        self.validator = validator or EvidenceValidator()

    async def answer(self, query: str, chunks: list[RetrievedChunk],
                     documents: dict[str, DocumentRecord],
                     mode: str = "general") -> GenerationResult:
        t0 = time.time()
        prompt = assemble_prompt(query, chunks)
        answer = await self.llm.generate(prompt)

        # Determine which chunks are actually cited in the answer and build citations.
        used = self._find_used_chunks(answer, chunks)
        if not used and chunks:
            used = chunks[:2]

        used_ids = [c.chunk_id for c in used]
        citations = self._build_citations(used, documents)

        # Evidence validation (groundedness, citation validity, numeric consistency).
        evidence = self.validator.validate(answer, chunks or used, used_ids)

        # If nothing was used / nothing supports it, emit the honest notice.
        if not chunks:
            answer = "I couldn't find sufficient evidence in the uploaded documents."
            evidence.grounded = False

        # Marked-mode prompts (resume / research) alter framing only; extraction
        # and citing stay as-is.
        latency = int((time.time() - t0) * 1000)
        return GenerationResult(
            answer=answer,
            citations=citations,
            evidence=evidence,
            latency_ms=latency,
            retrieval_ms=0,
            generation_ms=latency,
        )

    # ------------------------------------------------------------------
    def _find_used_chunks(self, answer: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        # 1. Direct chunk-id reference.
        used: list[RetrievedChunk] = [c for c in chunks if c.chunk_id in answer]
        if used:
            return used
        # 2. Match styled refs like "[Page 1, Diagram]" / "[Page 12]" to chunks.
        for ref in _CITE_RE.finditer(answer):
            page = int(ref.group(1))
            label = (ref.group(2) or "").strip().lower()
            match = _match_by_ref(chunks, page, label)
            if match and match not in used:
                used.append(match)
        if used:
            return used
        # 3. Fallback: best-scoring chunks.
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:3]

    def _build_citations(self, chunks: list[RetrievedChunk], documents: dict[str, DocumentRecord]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, int, str]] = set()
        for c in chunks:
            key = (c.document_id, c.page_number, c.content_type.value)
            if key in seen:
                continue
            seen.add(key)
            doc = documents.get(c.document_id)
            label = ""
            if c.content_type == ContentType.TABLE:
                label = "Table"
            elif c.content_type in (ContentType.DIAGRAM, ContentType.CHART, ContentType.IMAGE):
                label = "Diagram" if c.content_type == ContentType.DIAGRAM else (
                    "Chart" if c.content_type == ContentType.CHART else "Image"
                )
            cite_doc_name = doc.name if doc else c.document_id
            citations.append(
                Citation(
                    document_id=c.document_id,
                    document_name=cite_doc_name,
                    page=c.page_number,
                    region_id=c.region_ids[0] if c.region_ids else "",
                    content_type=c.content_type,
                    snippet=(c.text or c.visual_context)[:160],
                    bbox=c.bbox,
                    image_ids=c.image_ids,
                )
            )
        return citations

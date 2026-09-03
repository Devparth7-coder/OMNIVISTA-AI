"""Hybrid retrieval: semantic (vector) + lexical (BM25-style), fused, then
reranked. A retrieval router selects which modalities to pull based on the query
intent, so text, table, image, and diagram questions get the right evidence.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from app.core.config import settings
from app.embeddings.base import EmbeddingProvider
from app.reranking.base import RerankerProvider
from app.schemas.domain import ContentType, QueryIntent, RetrievedChunk
from app.vectorstore.base import VectorStore

_TOKEN_RE = re.compile(r"[\w'-]+")


class Bm25:
    """In-memory BM25-style lexical scorer over chunk text."""

    def __init__(self) -> None:
        self._docs: list[dict] = []
        self._doc_freq: dict[str, int] = {}
        self._doc_len: list[int] = []
        self._avg_len: float = 1.0
        self._df: dict[str, int] = {}
        self._k1 = 1.5
        self._b = 0.75

    def build(self, docs: list[str]) -> None:
        self._docs = []
        self._doc_freq = {}
        self._doc_len = []
        token_counts: list[Counter] = []
        for d in docs:
            tokens = _TOKEN_RE.findall(d.lower())
            counts = Counter(tokens)
            token_counts.append(counts)
            self._doc_len.append(len(tokens))
            for tok in tokens:
                self._doc_freq[tok] = self._doc_freq.get(tok, 0) + 1
        self._df = self._doc_freq
        self._avg_len = sum(self._doc_len) / max(len(self._doc_len), 1)
        self._token_counts = token_counts

    def score(self, query: str) -> list[float]:
        if not self._docs:
            return []
        q_tokens = _TOKEN_RE.findall(query.lower())
        n = len(self._docs)
        scores = [0.0] * n
        for q in q_tokens:
            idf = math.log(1 + (n - self._df.get(q, 0) + 0.5) / (self._df.get(q, 0) + 0.5))
            for i, counts in enumerate(self._token_counts):
                tf = counts.get(q, 0)
                if tf == 0:
                    continue
                dl = self._doc_len[i]
                denom = tf + self._k1 * (1 - self._b + self._b * dl / self._avg_len)
                scores[i] += idf * (tf * (self._k1 + 1)) / denom
        return scores


class RetrievalEngine:
    def __init__(self, embeddings: EmbeddingProvider, vector_store: VectorStore,
                 reranker: RerankerProvider) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.reranker = reranker
        self._bm25 = Bm25()

    async def route(self, intent: QueryIntent, document_ids: list[str] | None = None) -> dict[str, list[str]]:
        """Return which modalities to retrieve.

        We deliberately never hard-filter modalities away: a "table" question
        may have its answer in a chart, and a "diagram" question needs nearby
        text. So we always retrieve all modalities and instead *boost* the ones
        the query is most likely about (see ``_boost_modalities``).
        """
        return {"text": True, "table": True, "image": True, "diagram": True}

    @staticmethod
    def _boost_modalities(intent: QueryIntent) -> set[str]:
        if intent.modality == "mixed":
            return set()
        if intent.modality == "diagram":
            return {"diagram", "image"}
        if intent.modality == "chart":
            return {"chart", "table"}
        if intent.modality == "image":
            return {"image", "diagram"}
        if intent.modality == "table":
            return {"table", "chart"}
        return {"text", "multimodal"}

    async def hybrid_search(self, query: str, intent: QueryIntent, document_ids: list[str] | None = None,
                            top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.TOP_K_CANDIDATES
        # Semantic search
        q_vec = await self.embeddings.embed_query(query)
        filters: dict = {}
        if document_ids:
            filters["document_id"] = document_ids  # handled below
        semantic = await self.vector_store.search(q_vec, top_k * 3, filters={})
        semantic = [h for h in semantic if h.score > 0.0]
        # Apply intent page/type filters (vector store handles simple scalar filters).
        semantic = self._apply_filters(semantic, intent, document_ids)

        # Lexical search (hybrid)
        lexical = await self._lexical_search(query, intent, document_ids)

        # Fusion: normalize scores and combine semantic + lexical
        semantic_norm = _normalize_scores(semantic)
        lexical_norm = _normalize_scores(lexical)

        fused: dict[str, RetrievedChunk] = {}
        for hit in semantic_norm:
            chunk = _to_retrieved(hit["hit"])
            chunk.score = 0.6 * hit["norm"]
            fused[chunk.chunk_id] = chunk
        for hit in lexical_norm:
            id_ = hit["id"]
            if id_ in fused:
                fused[id_].score += 0.4 * hit["norm"]
            else:
                chunk = _to_retrieved(hit["hit"])
                chunk.score = 0.4 * hit["norm"]
                fused[chunk.chunk_id] = chunk

        # modality route weighting: boost chunks of the modality the query targets.
        desired = self._boost_modalities(intent)
        if desired:
            for chunk in fused.values():
                if chunk.modality in desired:
                    chunk.score += 0.2

        candidates = list(fused.values())
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:top_k]

        # Rerank
        reranked = await self.reranker.rerank(query, candidates, settings.TOP_K_FINAL)
        return reranked

    def _apply_filters(self, hits, intent: QueryIntent, document_ids: list[str] | None) -> list:
        out = []
        for h in hits:
            if intent.page is not None and h.page_number != intent.page:
                continue
            if document_ids and h.document_id not in document_ids:
                continue
            out.append(h)
        return out

    async def _lexical_search(self, query: str, intent: QueryIntent, document_ids: list[str] | None) -> list:
        all_hits = await self.vector_store.all_hits()
        docs = [h.text + " " + h.visual_context for h in all_hits]
        self._bm25.build(docs)
        scores = self._bm25.score(query)
        scored = []
        for hit, s in zip(all_hits, scores):
            if s <= 0:
                continue
            if intent.page is not None and hit.page_number != intent.page:
                continue
            if document_ids and hit.document_id not in document_ids:
                continue
            scored.append({"id": hit.chunk_id, "hit": hit, "norm": s})
        scored.sort(key=lambda x: x["norm"], reverse=True)
        return scored[: settings.TOP_K_CANDIDATES]


def _normalize_scores(items: list) -> list[dict]:
    if not items:
        return []
    vals = [i.score for i in items]
    mx = max(vals)
    if mx == 0:
        return [{"id": i.chunk_id, "hit": i, "norm": 0.0} for i in items]
    return [{"id": i.chunk_id, "hit": i, "norm": i.score / mx} for i in items]


def _to_retrieved(hit) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        page_number=hit.page_number,
        content_type=hit.content_type,
        section=hit.section,
        text=hit.text,
        visual_context=hit.visual_context,
        region_ids=hit.region_ids,
        image_ids=hit.image_ids,
        table_ids=hit.table_ids,
        bbox=hit.bbox,
        score=hit.score,
        modality=hit.content_type.value,
    )




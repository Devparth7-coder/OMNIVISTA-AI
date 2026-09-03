"""In-memory (and optionally JSON-persisted) vector store used in mock mode."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from app.vectorstore.base import SearchHit, VectorStore


class InMemoryVectorStore(VectorStore):
    def __init__(self, persist_path: str | None = None) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())
            self._vectors = data.get("vectors", {})
            self._meta = data.get("meta", {})
        except Exception:
            pass

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps({"vectors": self._vectors, "meta": self._meta}))

    async def upsert(self, chunk_id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        self._vectors[chunk_id] = embedding
        self._meta[chunk_id] = metadata
        self._save()

    async def search(self, query_embedding: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        scored: list[tuple[float, str]] = []
        for chunk_id, vec in self._vectors.items():
            meta = self._meta.get(chunk_id, {})
            if filters and not self._matches(meta, filters):
                continue
            sim = _cosine_sim(query_embedding, vec)
            scored.append((sim, chunk_id))
        scored.sort(key=lambda t: t[0], reverse=True)
        results: list[SearchHit] = []
        for sim, chunk_id in scored[:top_k]:
            meta = self._meta[chunk_id]
            results.append(
                SearchHit(
                    chunk_id=chunk_id,
                    document_id=str(meta.get("document_id", "")),
                    page_number=int(meta.get("page_number", 0)),
                    content_type=meta.get("content_type", "text"),
                    section=str(meta.get("section", "")),
                    text=str(meta.get("text", "")),
                    visual_context=str(meta.get("visual_context", "")),
                    region_ids=list(meta.get("region_ids", [])),
                    image_ids=list(meta.get("image_ids", [])),
                    table_ids=list(meta.get("table_ids", [])),
                    bbox=meta.get("bbox"),
                    score=round(sim, 5),
                )
            )
        return results

    @staticmethod
    def _matches(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            if key in ("page", "page_number"):
                if int(meta.get("page_number", -1)) != int(expected):
                    return False
            elif key == "document_id":
                if meta.get("document_id") != expected:
                    return False
            elif key == "content_type":
                if str(meta.get("content_type")) != str(expected):
                    return False
            else:
                if meta.get(key) != expected:
                    return False
        return True

    async def delete_document(self, document_id: str) -> int:
        removed = 0
        for chunk_id, meta in list(self._meta.items()):
            if meta.get("document_id") == document_id:
                self._vectors.pop(chunk_id, None)
                self._meta.pop(chunk_id, None)
                removed += 1
        self._save()
        return removed

    async def count(self) -> int:
        return len(self._vectors)

    async def clear(self) -> None:
        self._vectors.clear()
        self._meta.clear()
        self._save()

    async def all_hits(self) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for chunk_id, meta in self._meta.items():
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    document_id=str(meta.get("document_id", "")),
                    page_number=int(meta.get("page_number", 0)),
                    content_type=meta.get("content_type", "text"),
                    section=str(meta.get("section", "")),
                    text=str(meta.get("text", "")),
                    visual_context=str(meta.get("visual_context", "")),
                    region_ids=list(meta.get("region_ids", [])),
                    image_ids=list(meta.get("image_ids", [])),
                    table_ids=list(meta.get("table_ids", [])),
                    bbox=meta.get("bbox"),
                    score=0.0,
                )
            )
        return hits


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

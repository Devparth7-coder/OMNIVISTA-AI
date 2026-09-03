"""Retrieval endpoints: hybrid search and retrieve (for inspection/evaluation)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.deps import get_services
from app.services.query import analyze_query

router = APIRouter(tags=["retrieval"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    document_ids: list[str] | None = None


@router.post("/retrieve")
async def retrieve(req: SearchRequest):
    services = get_services()
    intent = analyze_query(req.query, req.document_ids)
    chunks = await services["retrieval"].hybrid_search(
        req.query, intent, document_ids=req.document_ids, top_k=req.top_k
    )
    return {
        "intent": intent.model_dump(),
        "chunks": [c.model_dump() for c in chunks],
    }


@router.post("/search")
async def search(req: SearchRequest):
    services = get_services()
    intent = analyze_query(req.query, req.document_ids)
    chunks = await services["retrieval"].hybrid_search(
        req.query, intent, document_ids=req.document_ids, top_k=min(req.top_k, 30)
    )
    return {
        "query": req.query,
        "intent": intent.model_dump(),
        "results": [c.model_dump() for c in chunks],
    }

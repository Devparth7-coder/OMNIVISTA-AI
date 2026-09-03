"""Health + status endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import get_services
from app.core.config import settings

router = APIRouter(tags=["system"])


@router.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}


@router.get("/status")
async def status():
    services = get_services()
    return {
        "environment": settings.ENVIRONMENT,
        "mock_mode": settings.is_mock,
        "providers": {
            "llm": settings.LLM_PROVIDER,
            "vision": settings.VISION_PROVIDER,
            "embedding": settings.EMBEDDING_PROVIDER,
            "vector_store": settings.VECTOR_DB_PROVIDER,
            "storage": settings.STORAGE_PROVIDER,
        },
        "total_chunks": await services["vector_store"].count(),
        "documents": len(services["store"].all()),
    }

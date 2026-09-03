"""Demo endpoints: generate a sample multimodal PDF and seed the library."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.api.deps import get_services
from app.schemas.domain import DocumentRecord, new_id

router = APIRouter(prefix="/demo", tags=["demo"])


@router.get("/document")
async def demo_document():
    from app.demo.generator import generate_demo_markdown, generate_demo_pdf

    data = generate_demo_pdf()
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": 'inline; filename="omnivista-demo.pdf"'})


@router.post("/seed", status_code=201)
async def seed_demo(kind: str = Query("pdf")):
    services = get_services()
    from app.demo.generator import generate_demo_markdown, generate_demo_pdf

    if kind == "pdf":
        name, data = "omnivista-demo.pdf", generate_demo_pdf()
    else:
        name, data = "omnivista-demo.md", generate_demo_markdown()

    record = DocumentRecord(
        document_id=new_id("doc"),
        name=name,
        size_bytes=len(data),
        source_type="pdf" if kind == "pdf" else "md",
    )
    services["store"].put(record)
    asyncio.create_task(services["pipeline"].ingest(record.document_id, name, data))
    return record.model_dump(mode="json")

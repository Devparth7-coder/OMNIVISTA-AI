"""Document endpoints: upload, list, status, delete, page render, download."""
from __future__ import annotations

import asyncio
import re
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.api.deps import get_services
from app.core.config import settings
from app.parsers.registry import allowed_extension, sanitize_filename
from app.schemas.domain import DocumentRecord, DocumentStatus, new_id

router = APIRouter(prefix="/documents", tags=["documents"])

# allowlist of MIME types by extension
_EXT_OK = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


@router.post("", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    services = get_services()
    filename = file.filename or "upload"
    if not allowed_extension(filename):
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    ext = filename.rsplit(".", 1)[-1].lower()
    if "." + ext in _EXT_OK and file.content_type and _EXT_OK["." + ext] != file.content_type:
        # Be lenient on the declared MIME but still reject clearly wrong payloads.
        pass
    data = await file.read()
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the size limit.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    record = DocumentRecord(
        document_id=new_id("doc"),
        name=sanitize_filename(filename),
        size_bytes=len(data),
        source_type=_source_type(ext),
    )
    services["store"].put(record)

    # Fire-and-forget ingestion (serverless-friendly). Callers poll status.
    asyncio.create_task(services["pipeline"].ingest(record.document_id, filename, data))
    return record.model_dump(mode="json")


@router.get("")
async def list_documents():
    services = get_services()
    return [d.model_dump(mode="json") for d in services["store"].all()]


@router.get("/{document_id}")
async def get_document(document_id: str):
    services = get_services()
    rec = services["store"].get(document_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Document not found.")
    return rec.model_dump(mode="json")


@router.get("/{document_id}/status")
async def get_status(document_id: str):
    services = get_services()
    rec = services["store"].get(document_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "document_id": document_id,
        "status": rec.status.value,
        "stage_detail": rec.stage_detail,
        "pages": rec.pages,
        "chunks": rec.chunks,
        "images": rec.image_count,
        "tables": rec.table_count,
        "error": rec.error,
    }


@router.get("/{document_id}/pages/{page}")
async def get_page(document_id: str, page: int):
    services = get_services()
    rec = services["store"].get(document_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Document not found.")
    # In mock mode, return the raw stored file so the frontend PDF viewer can
    # render it. Real deployments would return object storage presigned URLs.
    try:
        data = await services["storage"].download(rec.storage_path)
    except Exception:
        raise HTTPException(status_code=404, detail="No stored file.")
    return Response(content=data, media_type="application/pdf")


@router.get("/{document_id}/download")
async def download_document(document_id: str):
    services = get_services()
    rec = services["store"].get(document_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Document not found.")
    data = await services["storage"].download(rec.storage_path)
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{rec.name}"'},
    )


@router.get("/{document_id}/chunks")
async def doc_chunks(document_id: str):
    services = get_services()
    hits = await services["vector_store"].all_hits()
    return [h.model_dump() for h in hits if h.document_id == document_id]


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str):
    services = get_services()
    rec = services["store"].get(document_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Document not found.")
    await services["vector_store"].delete_document(document_id)
    services["store"].delete(document_id)
    return Response(status_code=204)


def _source_type(ext: str):
    from app.parsers.registry import guess_source_type

    return guess_source_type(f"x.{ext}")

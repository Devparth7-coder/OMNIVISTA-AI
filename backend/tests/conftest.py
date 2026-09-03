"""Shared pytest fixtures."""
from __future__ import annotations

import shutil

import pytest
import pytest_asyncio

from app.core.config import settings
from app.demo.generator import generate_demo_pdf


@pytest.fixture(scope="function", autouse=True)
def _clean_persist():
    """Ensure each test starts from a clean persisted state."""
    if settings.STORAGE_DIR:
        shutil.rmtree(settings.STORAGE_DIR, ignore_errors=True)
    from pathlib import Path

    Path(settings.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(settings.STORAGE_DIR, ignore_errors=True)


@pytest_asyncio.fixture
async def container():
    from app.services.container import build_container

    return build_container()


@pytest_asyncio.fixture
async def demo_document(container):
    """Ingest the demo multimodal PDF and return the DocumentRecord."""
    from app.schemas.domain import DocumentRecord, new_id

    data = generate_demo_pdf()
    rec = DocumentRecord(document_id=new_id("doc"), name="demo.pdf", size_bytes=len(data))
    container["store"].put(rec)
    await container["pipeline"].ingest(rec.document_id, "demo.pdf", data)
    return rec

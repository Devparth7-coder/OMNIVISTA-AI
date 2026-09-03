"""Document ingestion pipeline (MVP).

Upload bytes -> parse -> layout regions -> image/diagram understanding (+ OCR for
scanned/image docs) -> multimodal chunking -> embedding -> index.

The pipeline is **asynchronous and resumable**: the API returns a ``document_id``
immediately (status UPLOADED) and this pipeline advances the status through
PARSING / OCR / ANALYZING / CHUNKING / EMBEDDING / INDEXING -> READY, updating
the document store as it goes. Deployments can move the body of
:meth:`IngestPipeline.process` onto a managed queue by swapping the task queue.
"""
from __future__ import annotations

import time
from typing import Callable

from app.chunking.service import chunk_document
from app.core.config import settings
from app.embeddings.base import EmbeddingProvider
from app.ingestion.mock_vision import annotate_images
from app.parsers.base import ParseError
from app.parsers.registry import get_parser, sanitize_filename
from app.schemas.domain import (
    ContentType,
    DocumentRecord,
    DocumentStatus,
    ImageAsset,
    ParsedDocument,
    Region,
    RegionType,
    SourceType,
)
from app.storage.base import StorageProvider
from app.vectorstore.base import VectorStore
from app.vision.base import VisionProvider

StatusCallback = Callable[[DocumentRecord, DocumentStatus, str], None]


class IngestPipeline:
    def __init__(self, store, storage: StorageProvider, vision: VisionProvider,
                 embeddings: EmbeddingProvider, vector_store: VectorStore) -> None:
        self.store = store
        self.storage = storage
        self.vision = vision
        self.embeddings = embeddings
        self.vector_store = vector_store

    async def ingest(self, document_id: str, filename: str, data: bytes, on_status: StatusCallback | None = None) -> DocumentRecord:
        record = self.store.get(document_id)
        if record is None:
            record = DocumentRecord(
                document_id=document_id,
                name=sanitize_filename(filename),
                size_bytes=len(data),
                storage_path="",
                source_type=SourceType.OTHER,
            )
            self.store.put(record)

        t0 = time.time()

        def status(state: DocumentStatus, detail: str = "") -> None:
            r = self.store.get(document_id)
            if r is None:
                return
            r.status = state
            r.stage_detail = detail
            r.updated_at = _now()
            self.store.put(r)
            if on_status:
                on_status(r, state, detail)

        try:
            # store raw bytes
            key = f"{document_id}/{record.name}"
            stored = await self.storage.upload(key, data)
            record = self.store.get(document_id)
            record.storage_path = stored
            self.store.put(record)

            status(DocumentStatus.PARSING, "extracting text, layout and tables")
            parsed = await self._parse(document_id, record.name, data)

            # analyze images with the vision provider
            status(DocumentStatus.ANALYZING, "understanding images, charts and diagrams")
            annotated = await annotate_images(parsed, self.vision)
            image_meta = annotated["image_meta"]
            image_assets = annotated["images"]

            # OCR for image-only / scanned documents
            status(DocumentStatus.OCR, "applying OCR where needed")
            ocr_texts = await self._ocr_scan(parsed, data, record.name)

            status(DocumentStatus.CHUNKING, "building multimodal chunks")
            chunks = chunk_document(parsed, image_meta)

            # attach image assets to parsed doc
            parsed.images = image_assets

            status(DocumentStatus.EMBEDDING, "generating embeddings")
            await self._embed_and_index(parsed, chunks, ocr_texts)

            status(DocumentStatus.INDEXING, "finalizing index")
            record = self.store.get(document_id)
            record.status = DocumentStatus.READY
            record.pages = parsed.pages
            record.chunks = len(chunks)
            record.image_count = len(image_assets)
            record.table_count = sum(1 for r in parsed.regions if r.content_type == ContentType.TABLE)
            record.processing_ms = int((time.time() - t0) * 1000)
            record.updated_at = _now()
            record.stage_detail = ""
            record.error = None
            self.store.put(record)
            return record
        except Exception as exc:  # noqa: BLE001
            record = self.store.get(document_id)
            if record:
                record.status = DocumentStatus.FAILED
                record.error = str(exc)
                record.updated_at = _now()
                self.store.put(record)
            raise

    # ------------------------------------------------------------------
    async def _parse(self, document_id: str, filename: str, data: bytes) -> ParsedDocument:
        from app.parsers.registry import guess_source_type, is_image_extension

        source_type = guess_source_type(filename)
        if source_type == SourceType.IMAGE:
            parsed = ParsedDocument(document_id=document_id, source_type=SourceType.IMAGE, pages=1)
            parsed.regions = [
                Region(
                    document_id=document_id, page_number=1, type=RegionType.IMAGE,
                    content_type=ContentType.IMAGE, content="", reading_order=0,
                    image_asset_id=f"img_{document_id}_0_0", bbox=[0, 0, 1, 1],
                )
            ]
            return parsed
        parser = get_parser(source_type)
        parsed = await parser.parse(data)
        parsed.document_id = document_id
        # Assign document_id to each region/image
        for r in parsed.regions:
            r.document_id = document_id
        for img in parsed.images:
            img.document_id = document_id
        return parsed

    async def _ocr_scan(self, parsed: ParsedDocument, data: bytes, filename: str) -> dict[int, str]:
        """OCR image-only or scanned documents (mock returns nothing)."""
        if parsed.source_type != SourceType.IMAGE:
            return {}
        from app.ocr.mock import MockOCRProvider

        ocr = MockOCRProvider()
        result = await ocr.extract_text(data, page=1)
        lines = [l.text for l in result.lines if l.text]
        if lines:
            return {1: "\n".join(lines)}
        return {}

    async def _embed_and_index(self, parsed: ParsedDocument, chunks, ocr_texts: dict[int, str]) -> None:
        # Embed text chunks
        texts = [c.text if c.text else c.visual_context for c in chunks]
        vectors = await self.embeddings.embed_text(texts)
        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec
            await self.vector_store.upsert(
                chunk.chunk_id,
                vec,
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "content_type": chunk.content_type.value,
                    "section": chunk.section,
                    "text": chunk.text,
                    "visual_context": chunk.visual_context,
                    "region_ids": chunk.region_ids,
                    "image_ids": chunk.image_ids,
                    "table_ids": chunk.table_ids,
                    "bbox": chunk.bbox,
                    "source_type": chunk.source_type.value,
                },
            )


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)

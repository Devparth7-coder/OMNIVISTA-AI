"""Mock OCR provider. In mock mode image documents have no extractable text,
so it returns no lines (indicating the OCR stage ran but found nothing to read).
"""
from __future__ import annotations

from app.ocr.base import OCRProvider, OCRResult


class MockOCRProvider(OCRProvider):
    async def extract_text(self, image_bytes: bytes, page: int = 0) -> OCRResult:
        return OCRResult(page=page, lines=[], provider="mock")

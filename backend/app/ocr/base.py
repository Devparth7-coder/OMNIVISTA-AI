"""OCR provider abstraction (for scanned PDFs and image-based documents)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class OCRLine(BaseModel):
    text: str
    confidence: float = 0.95
    bbox: list[float] | None = None


class OCRResult(BaseModel):
    page: int
    lines: list[OCRLine] = Field(default_factory=list)
    provider: str = "mock"
    metadata: dict[str, Any] = Field(default_factory=dict)


class OCRProvider(ABC):
    @abstractmethod
    async def extract_text(self, image_bytes: bytes, page: int = 0) -> OCRResult:
        ...

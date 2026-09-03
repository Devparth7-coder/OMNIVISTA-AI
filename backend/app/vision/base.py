"""Vision-Language provider abstraction (image/chart/diagram understanding)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class VisionAnnotation(BaseModel):
    description: str = ""
    objects: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    chart_type: str | None = None  # bar, line, pie, scatter, diagram, ...
    chart_data: dict[str, Any] = Field(default_factory=dict)
    relationships: list[dict[str, str]] = Field(default_factory=list)
    summary: str = ""
    confidence: float = 0.9


class VisionProvider(ABC):
    @abstractmethod
    async def describe(self, image_bytes: bytes, hint: str = "") -> VisionAnnotation:
        """Describe an image/chart/diagram, optionally using surrounding hint."""

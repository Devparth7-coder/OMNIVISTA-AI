"""LLM provider abstraction: plain generation, structured generation, vision."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None) -> str:
        ...

    @abstractmethod
    async def structured_generate(self, prompt: str, schema: dict[str, Any], system: str | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    async def vision_generate(self, prompt: str, images: list[bytes], system: str | None = None) -> str:
        ...

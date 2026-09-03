"""Document parser abstraction.

A parser turns raw file bytes into a :class:`ParsedDocument` containing layout
regions (title/heading/paragraph/list/table/image/caption...) with bounding
boxes and reading order where available. The registry lets new formats be added
without changing the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from app.schemas.domain import ParsedDocument, SourceType

ParserFactory = Callable[[], "DocumentParser"]


class ParseError(Exception):
    pass


class DocumentParser(ABC):
    source_type: SourceType = SourceType.OTHER

    @abstractmethod
    async def parse(self, data: bytes) -> ParsedDocument:
        ...

    def extract_images(self) -> bool:
        return False

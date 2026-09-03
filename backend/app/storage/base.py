"""Object storage abstraction. Raw files never live in the database."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import BinaryIO


class StorageProvider(ABC):
    @abstractmethod
    async def upload(self, key: str, data: bytes, content_type: str | None = None) -> str:
        """Persist bytes under ``key`` and return an addressable path/URI."""

    @abstractmethod
    async def download(self, key: str) -> bytes:
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        ...

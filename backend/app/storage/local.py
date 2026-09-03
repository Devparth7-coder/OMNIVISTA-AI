"""Local-disk object storage for development/mock mode."""
from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.storage.base import StorageProvider


class LocalStorageProvider(StorageProvider):
    def __init__(self, root: str | None = None) -> None:
        self._root = Path(root or settings.STORAGE_DIR).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Strip unsafe path separators to prevent traversal.
        safe = key.replace("..", "").lstrip("/")
        p = (self._root / safe).resolve()
        if self._root.resolve() not in p.parents:
            raise ValueError("invalid storage key")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    async def upload(self, key: str, data: bytes, content_type: str | None = None) -> str:
        p = self._path(key)
        p.write_bytes(data)
        return str(p.relative_to(self._root))

    async def download(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    async def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

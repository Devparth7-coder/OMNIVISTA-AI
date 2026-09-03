"""Task queue abstraction for asynchronous (serverless-friendly) ingestion."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TaskQueue(ABC):
    @abstractmethod
    async def enqueue(self, task_name: str, payload: dict[str, Any]) -> str:
        ...

    @abstractmethod
    async def status(self, task_id: str) -> dict[str, Any]:
        ...

    @abstractmethod
    async def wait(self, task_id: str) -> dict[str, Any]:
        ...

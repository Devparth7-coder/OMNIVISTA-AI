"""Dependency-free task queue.

``InlineTaskQueue`` executes the task immediately when enqueued and tracks
state in memory. For real deployments, swap in a provider backed by a managed
queue (Redis/Celery, Cloud Tasks, SQS...) via the same ``TaskQueue`` interface,
and switch ``TASK_QUEUE_PROVIDER`` accordingly.
"""
from __future__ import annotations

from typing import Any

from app.tasks.base import TaskQueue


class InlineTaskQueue(TaskQueue):
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def enqueue(self, task_name: str, payload: dict[str, Any]) -> str:
        task_id = f"task_{len(self._store) + 1}"
        self._store[task_id] = {"task": task_name, "status": "pending", "payload": payload}
        # Inline mode: run synchronously and record completion.
        try:
            await self._run(task_name, payload)
            self._store[task_id]["status"] = "completed"
        except Exception as exc:  # noqa: BLE001
            self._store[task_id]["status"] = "failed"
            self._store[task_id]["error"] = str(exc)
        return task_id

    async def status(self, task_id: str) -> dict[str, Any]:
        return self._store.get(task_id, {})

    async def wait(self, task_id: str) -> dict[str, Any]:
        return self._store.get(task_id, {})

    async def _run(self, task_name: str, payload: dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError("Subclasses must implement _run")

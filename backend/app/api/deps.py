"""FastAPI dependency helpers for the container."""
from __future__ import annotations

from typing import AsyncIterator

from app.services.container import get_container


def get_services() -> dict:
    return get_container()

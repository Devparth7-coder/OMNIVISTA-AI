"""In-memory document registry (with optional JSON persistence).

Production would back this with PostgreSQL (SQLAlchemy + Alembic). Keeping it
behind a small store interface lets the API run in mock mode with zero setup.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.schemas.domain import Conversation, DocumentRecord


class DocumentStore:
    def __init__(self, persist_path: str | None = None, conversation_store: "ConversationStore | None" = None) -> None:
        self._docs: dict[str, DocumentRecord] = {}
        self._convs = conversation_store or ConversationStore()
        self._persist_path = Path(persist_path) if persist_path else None
        self._indexed_names: dict[str, str] = {}
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())
            for d in data.get("documents", []):
                rec = DocumentRecord.model_validate(d)
                self._docs[rec.document_id] = rec
        except Exception:
            pass

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
            json.dumps({"documents": [d.model_dump(mode="json") for d in self._docs.values()]}, default=str)
        )

    def put(self, record: DocumentRecord) -> None:
        self._docs[record.document_id] = record
        self._save()

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._docs.get(document_id)

    def all(self) -> list[DocumentRecord]:
        return sorted(self._docs.values(), key=lambda d: d.created_at, reverse=True)

    def delete(self, document_id: str) -> bool:
        if document_id in self._docs:
            del self._docs[document_id]
            self._save()
            return True
        return False

    @property
    def conversations(self) -> "ConversationStore":
        return self._convs


class ConversationStore:
    def __init__(self) -> None:
        self._convs: dict[str, "Conversation"] = {}

    def put(self, conv: "Conversation") -> None:
        self._convs[conv.conversation_id] = conv

    def get(self, conversation_id: str) -> "Conversation | None":
        return self._convs.get(conversation_id)

    def all(self) -> list["Conversation"]:
        return sorted(self._convs.values(), key=lambda c: c.updated_at, reverse=True)

    def delete(self, conversation_id: str) -> bool:
        return bool(self._convs.pop(conversation_id, None))

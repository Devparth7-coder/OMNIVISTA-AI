"""Chat orchestration.

Handles conversation state (history), document-scoped and multi-document
chat, runs the agentic retrieval + grounded generation, records messages,
and exposes a streamable answer iterator for the UI.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.domain import Conversation, ConversationMessage, DocumentRecord
from app.services.agent import AgenticRag
from app.services.generation import GenerationService
from app.services.store import DocumentStore


class ChatService:
    def __init__(self, store: DocumentStore, agentic: AgenticRag, generation: GenerationService) -> None:
        self.store = store
        self.agentic = agentic
        self.generation = generation

    def load_documents(self, document_ids: list[str] | None) -> dict[str, DocumentRecord]:
        docs: dict[str, DocumentRecord] = {}
        if not document_ids:
            for rec in self.store.all():
                if rec.status == "READY":
                    docs[rec.document_id] = rec
            return docs
        for did in document_ids:
            rec = self.store.get(did)
            if rec and rec.status == "READY":
                docs[did] = rec
        return docs

    async def send(self, message: str, conversation_id: str | None = None,
                   document_ids: list[str] | None = None) -> dict:
        conv = self._get_or_create(conversation_id)
        if document_ids:
            conv.document_ids = document_ids
        ready_docs = self.load_documents(conv.document_ids or document_ids)
        ready_ids = list(ready_docs.keys())

        conv.messages.append(ConversationMessage(role="user", content=message))
        if len(conv.messages) == 1:
            conv.title = message[:60]

        result = await self.agentic.run(message, ready_docs, document_ids=ready_ids or None)

        assistant_msg = ConversationMessage(
            role="assistant",
            content=result["answer"],
            citations=result["citations"],
        )
        conv.messages.append(assistant_msg)
        conv.updated_at = _now()
        self.store.conversations.put(conv)

        return {
            "conversation_id": conv.conversation_id,
            "answer": result["answer"],
            "citations": [c.model_dump() for c in result["citations"]],
            "evidence": result["evidence"].model_dump(),
            "iterations": result["iterations"],
            "intent": result["intent"].model_dump(),
            "retrieved_chunks": result["retrieved_chunks"],
            "message_id": assistant_msg.message_id,
        }

    async def stream(self, message: str, conversation_id: str | None = None,
                     document_ids: list[str] | None = None):
        """Yield serialized chunks for SSE streaming (answer split into parts)."""
        result = await self.send(message, conversation_id, document_ids)
        answer = result["answer"]
        for i in range(0, len(answer), 24):
            yield answer[i : i + 24]
        yield "__DONE__" + result["conversation_id"]

    def _get_or_create(self, conversation_id: str | None) -> Conversation:
        if conversation_id:
            conv = self.store.conversations.get(conversation_id)
            if conv:
                return conv
        conv = Conversation()
        self.store.conversations.put(conv)
        return conv


def _now() -> datetime:
    return datetime.now(timezone.utc)

"""Chat endpoints: single-shot answer, streaming answer, conversations."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_services

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    document_ids: list[str] | None = None
    stream: bool = False


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    document_ids: list[str] | None = None


@router.post("")
async def chat(req: ChatRequest):
    services = get_services()
    return await services["chat"].send(req.message, req.conversation_id, req.document_ids)


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    services = get_services()

    async def gen():
        async for token in services["chat"].stream(req.message, req.conversation_id, req.document_ids):
            data = json.dumps({"token": token})
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/conversations")
async def list_conversations():
    services = get_services()
    return [
        {"conversation_id": c.conversation_id, "title": c.title,
         "updated_at": c.updated_at.isoformat(), "message_count": len(c.messages)}
        for c in services["store"].conversations.all()
    ]


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    services = get_services()
    conv = services["store"].conversations.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv.model_dump(mode="json")

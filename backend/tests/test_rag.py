"""End-to-end RAG integration tests (the MVP acceptance path)."""
from __future__ import annotations

import pytest

from app.services.query import analyze_query


@pytest.mark.asyncio
async def test_ingest_indexes_multimodal_chunks(container, demo_document):
    hits = await container["vector_store"].all_hits()
    assert len(hits) > 0
    types = {h.content_type.value for h in hits}
    assert "table" in types
    assert "diagram" in types
    assert "chart" in types


@pytest.mark.asyncio
async def test_diagram_question_is_grounded(container, demo_document):
    res = await container["agentic"].run(
        "What does the architecture diagram show?", {demo_document.document_id: demo_document}
    )
    assert res["intent"].modality == "diagram"
    assert "couldn't find sufficient" not in res["answer"]
    assert res["evidence"].grounded
    assert any(c.page == 1 for c in res["citations"])


@pytest.mark.asyncio
async def test_connector_question(container, demo_document):
    res = await container["agentic"].run(
        "What component connects the client to the backend?", {demo_document.document_id: demo_document}
    )
    assert "api gateway" in res["answer"].lower()
    assert res["evidence"].grounded


@pytest.mark.asyncio
async def test_cross_modal_question(container, demo_document):
    res = await container["agentic"].run(
        "Does the architecture described in the text match the architecture shown in the diagram?",
        {demo_document.document_id: demo_document},
    )
    assert "consistent" in res["answer"].lower() or "diagram" in res["answer"].lower()


@pytest.mark.asyncio
async def test_table_lookup(container, demo_document):
    res = await container["agentic"].run(
        "What is the revenue in 2025 for the North region?",
        {demo_document.document_id: demo_document},
    )
    assert "16M" in res["answer"]
    assert res["evidence"].grounded


@pytest.mark.asyncio
async def test_chart_question(container, demo_document):
    res = await container["agentic"].run(
        "Which year had the highest revenue?", {demo_document.document_id: demo_document}
    )
    assert "2024" in res["answer"]


def test_query_intent_routing():
    assert analyze_query("What does the architecture diagram show?").modality == "diagram"
    assert analyze_query("What is the revenue in 2025?").modality in ("table", "chart")
    assert analyze_query("Describe the platform in the text.").modality == "text"
    mixed = analyze_query("Does the text match the diagram?")
    assert mixed.modality == "mixed"

"""Analytics endpoints (overview + knowledge graph + inspect)."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import get_services
from app.schemas.domain import ContentType

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("")
async def analytics():
    services = get_services()
    store = services["store"]
    docs = store.all()
    hits = await services["vector_store"].all_hits()

    by_type: dict[str, int] = {}
    pages = 0
    images = 0
    tables = 0
    for d in docs:
        pages += d.pages
        images += d.image_count
        tables += d.table_count
    for h in hits:
        key = h.content_type.value if isinstance(h.content_type, ContentType) else str(h.content_type)
        by_type[key] = by_type.get(key, 0) + 1

    return {
        "documents": len(docs),
        "pages_processed": pages,
        "images_analyzed": images,
        "tables_extracted": tables,
        "chunks": len(hits),
        "chunks_by_modality": by_type,
        "queries": 0,
        "avg_retrieval_latency_ms": 0,
        "avg_answer_latency_ms": 0,
        "groundedness": 0.0,
        "citation_coverage": 0.0,
        "documents_list": [d.model_dump(mode="json") for d in docs],
    }


@router.get("/knowledge-graph")
async def knowledge_graph():
    services = get_services()
    hits = await services["vector_store"].all_hits()
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for h in hits:
        if h.content_type in ("diagram", "image", "chart") or "diagram" in (h.text or "").lower():
            for line in (h.text or "").split("\n"):
                if "Diagram flow:" in line:
                    flow = line.split("Diagram flow:")[1].split("→")
                    names = [f.strip() for f in flow if f.strip()]
                    for n in names:
                        key = n.lower()
                        nodes.setdefault(key, {"id": key, "label": n, "type": "entity"})
                    for i in range(len(names) - 1):
                        edges.append({"source": names[i].lower(), "target": names[i + 1].lower()})
    return {"nodes": list(nodes.values()), "edges": edges}

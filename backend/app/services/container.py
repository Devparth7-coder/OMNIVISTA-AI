"""Dependency container.

Wires the pluggable providers (LLM, vision, embeddings, vector store, storage,
task queue, reranker) into the concrete services used by the API. In mock mode
(MOCK_MODE=true) every provider is the deterministic mock implementation, so the
entire platform runs with zero external dependencies or API keys.
"""
from __future__ import annotations

from app.core.config import settings
from app.evaluation.evidence import EvidenceValidator
from app.generation.base import LLMProvider
from app.generation.mock import MockLLMProvider
from app.ingestion.pipeline import IngestPipeline
from app.reranking.base import RerankerProvider
from app.reranking.mock import MockRerankerProvider
from app.retrieval.hybrid import RetrievalEngine
from app.services.agent import AgenticRag
from app.services.chat import ChatService
from app.services.generation import GenerationService
from app.services.store import DocumentStore
from app.storage.base import StorageProvider
from app.storage.local import LocalStorageProvider
from app.vectorstore.base import VectorStore
from app.vectorstore.in_memory import InMemoryVectorStore
from app.embeddings.base import EmbeddingProvider
from app.embeddings.mock import MockEmbeddingProvider
from app.vision.base import VisionProvider
from app.vision.mock import MockVisionProvider

_PERSIST = settings.STORAGE_DIR + "/index.json"


def build_container():
    storage: StorageProvider = LocalStorageProvider()
    embeddings: EmbeddingProvider = MockEmbeddingProvider()
    vector_store: VectorStore = InMemoryVectorStore(persist_path=_PERSIST)
    vision: VisionProvider = MockVisionProvider()
    llm: LLMProvider = MockLLMProvider()
    reranker: RerankerProvider = MockRerankerProvider()
    store = DocumentStore(persist_path=settings.STORAGE_DIR + "/documents.json")

    retrieval = RetrievalEngine(embeddings, vector_store, reranker)
    generation = GenerationService(llm, EvidenceValidator())
    agentic = AgenticRag(retrieval, generation)
    pipeline = IngestPipeline(store, storage, vision, embeddings, vector_store)
    chat = ChatService(store, agentic, generation)

    return {
        "storage": storage,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "vision": vision,
        "llm": llm,
        "reranker": reranker,
        "store": store,
        "retrieval": retrieval,
        "generation": generation,
        "agentic": agentic,
        "pipeline": pipeline,
        "chat": chat,
    }


_container: dict | None = None


def get_container() -> dict:
    global _container
    if _container is None:
        _container = build_container()
    return _container

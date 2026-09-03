"""Application configuration.

All values are loaded from environment variables (see `.env.example`).
The MVP runs in ``ENVIRONMENT=development`` with every AI/data provider in
**mock mode**, so no external API keys or database are required to run the full
pipeline.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Backend root: .../backend  (config.py -> core -> app -> backend)
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = BACKEND_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Runtime -------------------------------------------------------------
    ENVIRONMENT: str = "development"
    APP_NAME: str = "OMNIVISTA AI"
    APP_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    MOCK_MODE: bool = True  # force mock providers even if keys exist

    # ---- CORS ------------------------------------------------------------------
    CORS_ORIGINS: str = "http://localhost:3000"  # comma-separated

    # ---- Provider selection -----------------------------------------------------
    LLM_PROVIDER: str = "mock"
    VISION_PROVIDER: str = "mock"
    EMBEDDING_PROVIDER: str = "mock"
    VECTOR_DB_PROVIDER: str = "memory"
    STORAGE_PROVIDER: str = "local"
    TASK_QUEUE_PROVIDER: str = "inline"
    RERANKER_PROVIDER: str = "mock"
    OCR_PROVIDER: str = "mock"

    # ---- External (unused in mock mode, documented for real deployments) --------
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    EMBEDDING_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = ""
    DATABASE_URL: str = ""
    BLOB_READ_WRITE_TOKEN: str = ""

    # ---- Processing knobs ---------------------------------------------------------
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024  # 25 MB
    MAX_PAGES: int = 200
    TEXT_CHUNK_CHARS: int = 1400
    TEXT_CHUNK_OVERLAP: int = 160
    TOP_K_CANDIDATES: int = 30
    TOP_K_FINAL: int = 8
    MAX_RETRIEVAL_ITERATIONS: int = 3
    EMBEDDING_DIM: int = 384

    # ---- Storage directories --------------------------------------------------------
    UPLOAD_DIR: str = str(DEFAULT_DATA_DIR / "uploads")
    STORAGE_DIR: str = str(DEFAULT_DATA_DIR / "storage")

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def is_mock(self) -> bool:
        return self.MOCK_MODE or self.LLM_PROVIDER in ("mock", "memory")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

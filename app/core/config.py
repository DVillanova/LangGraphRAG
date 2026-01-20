"""Application configuration using pydantic-settings."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "langgraph_rag_docs"

    # Embedding settings (strong multilingual model suitable for 8 GB VRAM)
    embedding_model: str = "intfloat/multilingual-e5-large"

    # Document storage
    documents_dir: Path = Path("./data/documents")

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # RAG settings tuned for large documents
    chunk_size: int = 900
    chunk_overlap: int = 200
    retriever_k: int = 6
    max_document_length: int = 1_000_000

    # Batch processing settings (for large documents)
    embedding_batch_size: int = 100  # Embeddings to generate at once
    upsert_batch_size: int = 50  # Points to upsert to Qdrant at once

    @property
    def qdrant_url(self) -> str:
        """Get the full Qdrant URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

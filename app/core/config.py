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

    # Language settings
    language: str = "es"  # Spanish
    use_spanish_stopwords: bool = False  # Optional BM25 optimization

    # Hybrid search weights (tuned for Spanish)
    bm25_weight: float = 0.35
    semantic_weight: float = 0.65

    # Re-ranking settings (Spanish-optimized model)
    use_reranking: bool = True
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_top_k: int = 20  # Candidates to re-rank
    final_top_k: int = 6  # Final results

    # Query expansion
    use_query_expansion: bool = True
    num_query_variants: int = 3
    query_expansion_language: str = "es"

    # Semantic chunking
    use_semantic_chunking: bool = True
    semantic_chunk_size: int = 1000
    semantic_breakpoint_threshold: float = 0.5

    @property
    def qdrant_url(self) -> str:
        """Get the full Qdrant URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

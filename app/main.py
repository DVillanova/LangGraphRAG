"""FastAPI application entry point."""

import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.schemas import HealthResponse
from app.api.routes import chat, documents
from app.services.vector_store import get_vector_store_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: Initialize services
    settings = get_settings()

    # Ensure vector store collection exists
    try:
        vector_store = get_vector_store_service()
        vector_store.ensure_collection()
        print(f"✓ Connected to Qdrant at {settings.qdrant_url}")
    except Exception as e:
        print(f"⚠ Warning: Could not connect to Qdrant: {e}")

    yield

    # Shutdown: Cleanup if needed
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="LangGraph RAG Agent",
    description="A locally-operating Retrieval Augmented Generation agent using LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for future web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check the health of all services."""
    settings = get_settings()
    status = "healthy"
    ollama_status = "unknown"
    qdrant_status = "unknown"
    collection_info = None

    # Check Ollama
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            if response.status_code == 200:
                ollama_status = "connected"
            else:
                ollama_status = "error"
                status = "degraded"
    except Exception:
        ollama_status = "disconnected"
        status = "degraded"

    # Check Qdrant
    try:
        vector_store = get_vector_store_service()
        collection_info = vector_store.get_collection_info()
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "disconnected"
        status = "degraded"

    return HealthResponse(
        status=status,
        ollama=ollama_status,
        qdrant=qdrant_status,
        collection_info=collection_info,
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LangGraph RAG Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "/api/chat",
            "stream": "/api/chat/stream",
            "search": "/api/chat/search",
            "documents": "/api/documents",
        },
    }


def run_server():
    """Run the FastAPI server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run_server()

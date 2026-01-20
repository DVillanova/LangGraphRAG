"""Pydantic schemas for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


# Chat schemas
class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    message: str = Field(..., description="User message to send to the RAG agent")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    response: str = Field(..., description="Agent response")
    sources: Optional[List[str]] = Field(
        default=None, description="Source documents used for the response"
    )


class StreamChunk(BaseModel):
    """Schema for streaming response chunks."""

    node: str = Field(..., description="Name of the node that produced this update")
    content: str = Field(..., description="Content of the update")
    done: bool = Field(default=False, description="Whether this is the final chunk")


# Document schemas
class DocumentUploadResponse(BaseModel):
    """Response schema for document upload."""

    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    chunks_indexed: int = Field(..., description="Number of chunks indexed")
    message: str = Field(..., description="Status message")


class DocumentInfo(BaseModel):
    """Schema for document information."""

    doc_id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="Source filename")
    chunk_count: int = Field(..., description="Number of chunks from this document")


class DocumentListResponse(BaseModel):
    """Response schema for listing documents."""

    documents: List[DocumentInfo] = Field(..., description="List of indexed documents")
    total: int = Field(..., description="Total number of documents")


class DocumentDeleteResponse(BaseModel):
    """Response schema for document deletion."""

    doc_id: str = Field(..., description="Deleted document identifier")
    chunks_deleted: int = Field(..., description="Number of chunks deleted")
    file_deleted: bool = Field(..., description="Whether the file was deleted")
    message: str = Field(..., description="Status message")


# Health schemas
class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    ollama: str = Field(..., description="Ollama connection status")
    qdrant: str = Field(..., description="Qdrant connection status")
    collection_info: Optional[dict] = Field(
        default=None, description="Qdrant collection information"
    )


# Search schemas
class SearchRequest(BaseModel):
    """Request schema for search endpoint."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=4, ge=1, le=20, description="Maximum results to return")


class SearchResult(BaseModel):
    """Schema for a single search result."""

    content: str = Field(..., description="Chunk content")
    source: str = Field(..., description="Source document")
    score: float = Field(..., description="Relevance score")
    doc_id: str = Field(..., description="Document ID")
    page_number: Optional[int] = Field(default=None, description="Page number in source document")


class SearchResponse(BaseModel):
    """Response schema for search endpoint."""

    query: str = Field(..., description="Original query")
    results: List[SearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Number of results")

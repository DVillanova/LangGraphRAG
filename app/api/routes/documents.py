"""Document management API routes."""

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.api.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentInfo,
    DocumentDeleteResponse,
)
from app.services.document_service import get_document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document.

    Supported file types: .pdf, .txt, .csv, .html, .htm

    The document will be:
    1. Saved to the documents storage directory
    2. Split into chunks
    3. Embedded and indexed in the vector store
    """
    doc_service = get_document_service()

    # Validate file extension
    filename = file.filename or "unknown"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    supported = doc_service.get_supported_extensions()
    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported types: {', '.join(supported)}",
        )

    try:
        # Read file content
        content = await file.read()

        # Process and index document
        doc_id, chunks_indexed = doc_service.add_document_from_bytes(
            content=content,
            filename=filename,
        )

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=filename,
            chunks_indexed=chunks_indexed,
            message=f"Successfully indexed {chunks_indexed} chunks from {filename}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents.

    Returns metadata about each document including:
    - Document ID
    - Source filename
    - Number of chunks
    """
    try:
        doc_service = get_document_service()
        documents = doc_service.list_documents()

        doc_infos = [
            DocumentInfo(
                doc_id=doc["doc_id"],
                source=doc["source"],
                chunk_count=doc["chunk_count"],
            )
            for doc in documents
        ]

        return DocumentListResponse(
            documents=doc_infos,
            total=len(doc_infos),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str):
    """Delete a document from the index.

    This will:
    1. Remove all chunks from the vector store
    2. Delete the stored file if it exists
    """
    try:
        doc_service = get_document_service()
        chunks_deleted, file_deleted = doc_service.delete_document(doc_id)

        if chunks_deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {doc_id}",
            )

        return DocumentDeleteResponse(
            doc_id=doc_id,
            chunks_deleted=chunks_deleted,
            file_deleted=file_deleted,
            message=f"Successfully deleted document with {chunks_deleted} chunks",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-types")
async def get_supported_types():
    """Get list of supported document types."""
    doc_service = get_document_service()
    return {
        "supported_extensions": doc_service.get_supported_extensions(),
    }

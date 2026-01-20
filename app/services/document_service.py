"""Document processing service with multiple loader support."""

import uuid
import shutil
import re
from pathlib import Path
from typing import List, Optional, Tuple
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.vector_store import get_vector_store_service, VectorStoreService


# Supported file extensions and their loaders
SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}


def sanitize_text(text: str) -> str:
    """Sanitize text by removing invalid characters and surrogates.
    
    This fixes encoding issues from PDF extraction that can cause
    'surrogates not allowed' errors.
    
    Args:
        text: Text to sanitize.
        
    Returns:
        Sanitized text safe for UTF-8 encoding.
    """
    if not text:
        return text
    
    # Remove surrogate characters (invalid UTF-8)
    # Surrogates are in range U+D800 to U+DFFF
    text = text.encode('utf-8', errors='surrogateescape').decode('utf-8', errors='replace')
    
    # Remove any remaining problematic characters
    text = re.sub(r'[\ud800-\udfff]', '', text)
    
    # Replace common problematic characters with their ASCII equivalents
    replacements = {
        '\ufffd': '',  # Replacement character
        '\x00': '',    # Null bytes
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


class DocumentService:
    """Service for processing and managing documents."""

    def __init__(
        self,
        vector_store: VectorStoreService | None = None,
    ):
        """Initialize the document service.

        Args:
            vector_store: Vector store service instance.
        """
        self.settings = get_settings()
        self._vector_store = vector_store
        self._text_splitter: RecursiveCharacterTextSplitter | None = None

        # Ensure documents directory exists
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    @property
    def vector_store(self) -> VectorStoreService:
        """Get the vector store service."""
        if self._vector_store is None:
            self._vector_store = get_vector_store_service()
        return self._vector_store

    @property
    def documents_dir(self) -> Path:
        """Get the documents directory path."""
        return Path(self.settings.documents_dir)

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get the text splitter instance."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        return self._text_splitter

    def get_loader_for_file(self, file_path: Path):
        """Get the appropriate loader for a file.

        Args:
            file_path: Path to the file.

        Returns:
            Loader class for the file type.

        Raises:
            ValueError: If file type is not supported.
        """
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            raise ValueError(
                f"Unsupported file type: {suffix}. Supported types: {supported}"
            )
        return SUPPORTED_EXTENSIONS[suffix]

    def load_document(self, file_path: Path) -> List[Document]:
        """Load a document from file.

        Args:
            file_path: Path to the document file.

        Returns:
            List of Document objects.
        """
        loader_class = self.get_loader_for_file(file_path)
        loader = loader_class(str(file_path))
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List of document chunks.
        """
        return self.text_splitter.split_documents(documents)

    def add_document(
        self,
        file_path: Path,
        copy_to_storage: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[str, int]:
        """Add a document to the vector store.

        Args:
            file_path: Path to the document file.
            copy_to_storage: Whether to copy the file to storage directory.
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            Tuple of (doc_id, number of chunks indexed).
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Copy file to storage if requested
        if copy_to_storage:
            dest_path = self.documents_dir / f"{doc_id}_{file_path.name}"
            shutil.copy2(file_path, dest_path)
            source_name = file_path.name
        else:
            source_name = str(file_path)

        # Load and process document
        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)

        # Extract and sanitize text content from chunks
        texts = [sanitize_text(chunk.page_content) for chunk in chunks]

        # Prepare metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            metadata["original_file"] = file_path.name
            metadatas.append(metadata)

        # Add to vector store
        self.vector_store.add_documents(
            texts=texts,
            doc_id=doc_id,
            source=source_name,
            metadatas=metadatas,
            progress_callback=progress_callback,
        )

        return doc_id, len(chunks)

    def add_document_from_bytes(
        self,
        content: bytes,
        filename: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[str, int]:
        """Add a document from bytes content.

        Args:
            content: File content as bytes.
            filename: Original filename.
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            Tuple of (doc_id, number of chunks indexed).
        """
        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Save to storage
        dest_path = self.documents_dir / f"{doc_id}_{filename}"
        dest_path.write_bytes(content)

        # Load and process
        documents = self.load_document(dest_path)
        chunks = self.split_documents(documents)

        # Extract and sanitize text content
        texts = [sanitize_text(chunk.page_content) for chunk in chunks]

        # Prepare metadata
        metadatas = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata["original_file"] = filename
            metadatas.append(metadata)

        # Add to vector store
        self.vector_store.add_documents(
            texts=texts,
            doc_id=doc_id,
            source=filename,
            metadatas=metadatas,
            progress_callback=progress_callback,
        )

        return doc_id, len(chunks)

    def delete_document(self, doc_id: str) -> Tuple[int, bool]:
        """Delete a document and its file.

        Args:
            doc_id: Document ID to delete.

        Returns:
            Tuple of (chunks deleted, file deleted).
        """
        # Delete from vector store
        chunks_deleted = self.vector_store.delete_document(doc_id)

        # Try to delete the file
        file_deleted = False
        for file_path in self.documents_dir.glob(f"{doc_id}_*"):
            file_path.unlink()
            file_deleted = True

        return chunks_deleted, file_deleted

    def list_documents(self) -> List[dict]:
        """List all indexed documents.

        Returns:
            List of document metadata.
        """
        return self.vector_store.list_documents()

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Search for relevant document chunks.

        Args:
            query: Search query.
            limit: Maximum results to return.

        Returns:
            List of matching chunks with metadata.
        """
        if limit is None:
            limit = self.settings.retriever_k

        return self.vector_store.search(query=query, limit=limit)

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions.
        """
        return list(SUPPORTED_EXTENSIONS.keys())


@lru_cache
def get_document_service() -> DocumentService:
    """Get cached document service instance."""
    return DocumentService()

"""Vector store service using Qdrant."""

import re
import uuid
from typing import List, Optional
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.services.embeddings import get_embedding_service, EmbeddingService


def sanitize_text(text: str) -> str:
    """Sanitize text by removing invalid characters and surrogates.
    
    Args:
        text: Text to sanitize.
        
    Returns:
        Sanitized text safe for UTF-8 encoding.
    """
    if not text:
        return text
    
    # Remove surrogate characters (invalid UTF-8)
    text = text.encode('utf-8', errors='surrogateescape').decode('utf-8', errors='replace')
    text = re.sub(r'[\ud800-\udfff]', '', text)
    text = text.replace('\ufffd', '').replace('\x00', '')
    
    return text


class VectorStoreService:
    """Service for managing document vectors in Qdrant."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        """Initialize the vector store service.

        Args:
            client: Qdrant client instance. Creates one if not provided.
            embedding_service: Embedding service instance.
        """
        self.settings = get_settings()
        self._client = client
        self._embedding_service = embedding_service
        self._collection_initialized = False

    @property
    def client(self) -> QdrantClient:
        """Lazy load the Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
            )
        return self._client

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get the embedding service."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self.settings.qdrant_collection_name

    def ensure_collection(self) -> None:
        """Ensure the collection exists, create if it doesn't."""
        if self._collection_initialized:
            return

        try:
            self.client.get_collection(self.collection_name)
        except (UnexpectedResponse, Exception):
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_service.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )

        self._collection_initialized = True

    def add_documents(
        self,
        texts: List[str],
        doc_id: str,
        source: str,
        metadatas: Optional[List[dict]] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[str]:
        """Add document chunks to the vector store.

        Handles large documents by batching embeddings and upserts.

        Args:
            texts: List of text chunks to add.
            doc_id: Parent document ID (for grouping chunks).
            source: Source filename or identifier.
            metadatas: Optional list of metadata dicts for each chunk.
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            List of generated point IDs.
        """
        self.ensure_collection()

        total_chunks = len(texts)
        embed_batch_size = self.settings.embedding_batch_size
        upsert_batch_size = self.settings.upsert_batch_size

        point_ids = []

        # Process in batches to handle large documents
        for batch_start in range(0, total_chunks, embed_batch_size):
            batch_end = min(batch_start + embed_batch_size, total_chunks)
            batch_texts = texts[batch_start:batch_end]

            # Generate embeddings for this batch
            batch_embeddings = self.embedding_service.embed_texts(batch_texts)

            # Prepare points for this batch
            points = []
            for i, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                global_index = batch_start + i
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)

                payload = {
                    "content": text,
                    "source": source,
                    "doc_id": doc_id,
                    "chunk_index": global_index,
                }

                # Add any additional metadata
                if metadatas and global_index < len(metadatas):
                    payload.update(metadatas[global_index])

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            # Upsert points in smaller batches to avoid payload limits
            for upsert_start in range(0, len(points), upsert_batch_size):
                upsert_end = min(upsert_start + upsert_batch_size, len(points))
                batch_points = points[upsert_start:upsert_end]

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points,
                )

            # Report progress
            if progress_callback:
                progress_callback(batch_end, total_chunks)

        return point_ids

    def search(
        self,
        query: str,
        limit: int = 4,
        score_threshold: float = 0.0,
    ) -> List[dict]:
        """Search for similar documents.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.

        Returns:
            List of matching documents with content and metadata.
        """
        self.ensure_collection()

        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Search using query_points (newer API in qdrant-client 1.10+)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            score_threshold=score_threshold if score_threshold > 0 else None,
            with_payload=True,
        )

        # Format results and sanitize content
        documents = []
        for point in results.points:
            documents.append({
                "id": point.id,
                "content": sanitize_text(point.payload.get("content", "")),
                "source": point.payload.get("source", ""),
                "doc_id": point.payload.get("doc_id", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "score": point.score,
            })

        return documents

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document.

        Args:
            doc_id: The parent document ID.

        Returns:
            Number of points deleted.
        """
        self.ensure_collection()

        # First, find all points with this doc_id
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )

        point_ids = [point.id for point in scroll_result[0]]

        if point_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )

        return len(point_ids)

    def list_documents(self) -> List[dict]:
        """List all unique documents in the store.

        Returns:
            List of unique documents with their metadata.
        """
        self.ensure_collection()

        # Scroll through all points and collect unique doc_ids
        documents = {}
        offset = None

        while True:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            points, next_offset = scroll_result

            if not points:
                break

            for point in points:
                doc_id = point.payload.get("doc_id")
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "doc_id": doc_id,
                        "source": point.payload.get("source", ""),
                        "chunk_count": 0,
                    }
                if doc_id:
                    documents[doc_id]["chunk_count"] += 1

            if next_offset is None:
                break
            offset = next_offset

        return list(documents.values())

    def get_collection_info(self) -> dict:
        """Get information about the collection.

        Returns:
            Collection statistics and info.
        """
        self.ensure_collection()

        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }


@lru_cache
def get_vector_store_service() -> VectorStoreService:
    """Get cached vector store service instance."""
    return VectorStoreService()

"""Cross-encoder re-ranking service for improved retrieval quality."""

from typing import List, Dict, Tuple
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import get_settings


class RerankerService:
    """Service for re-ranking documents using cross-encoder models."""
    
    def __init__(self, model_name: str | None = None):
        """Initialize the reranker service.
        
        Args:
            model_name: Name of the cross-encoder model to use.
                       Defaults to settings value.
        """
        settings = get_settings()
        self.model_name = model_name or settings.rerank_model
        self._model: CrossEncoder | None = None
    
    @property
    def model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int | None = None,
    ) -> List[Dict]:
        """Re-rank documents using cross-encoder.
        
        Args:
            query: The search query
            documents: List of document dictionaries with 'content' field
            top_k: Number of top results to return (None returns all)
            
        Returns:
            Re-ranked list of documents with updated scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            content = doc.get("content", "")
            pairs.append([query, content])
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Combine documents with new scores
        reranked_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            # Store original score for reference
            if "score" in doc_copy:
                doc_copy["original_score"] = doc_copy["score"]
            # Update with cross-encoder score
            doc_copy["score"] = float(score)
            doc_copy["rerank_score"] = float(score)
            reranked_docs.append(doc_copy)
        
        # Sort by new scores
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]
        
        return reranked_docs
    
    def score_pairs(
        self,
        query_document_pairs: List[Tuple[str, str]],
    ) -> List[float]:
        """Score query-document pairs.
        
        Args:
            query_document_pairs: List of (query, document) tuples
            
        Returns:
            List of relevance scores
        """
        if not query_document_pairs:
            return []
        
        scores = self.model.predict(query_document_pairs)
        return [float(s) for s in scores]


@lru_cache
def get_reranker_service() -> RerankerService:
    """Get cached reranker service instance.
    
    Returns:
        RerankerService instance
    """
    return RerankerService()

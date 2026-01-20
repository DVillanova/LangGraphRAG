"""Hybrid search combining BM25 and semantic search for Spanish documents."""

import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.core.config import get_settings


# Spanish stopwords (common words to optionally filter)
SPANISH_STOPWORDS = {
    "el", "la", "de", "que", "y", "a", "en", "un", "ser", "se", "no", "haber",
    "por", "con", "su", "para", "como", "estar", "tener", "le", "lo", "todo",
    "pero", "más", "hacer", "o", "poder", "decir", "este", "ir", "otro", "ese",
    "si", "me", "ya", "ver", "porque", "dar", "cuando", "él", "muy", "sin",
    "vez", "mucho", "saber", "qué", "sobre", "mi", "alguno", "mismo", "yo",
    "también", "hasta", "año", "dos", "querer", "entre", "así", "primero",
    "desde", "grande", "eso", "ni", "nos", "llegar", "pasar", "tiempo", "ella",
    "sí", "día", "uno", "bien", "poco", "deber", "entonces", "poner", "cosa",
    "tanto", "hombre", "parecer", "nuestro", "tan", "donde", "ahora", "parte",
    "después", "vida", "quedar", "siempre", "creer", "hablar", "llevar", "dejar",
    "nada", "cada", "seguir", "menos", "nuevo", "encontrar", "algo", "solo",
    "decir", "aunque", "bueno", "tal", "quien", "cual", "cuando", "donde"
}


class SpanishTokenizer:
    """Tokenizer optimized for Spanish text."""
    
    def __init__(self, use_stopwords: bool = False, lowercase: bool = True):
        """Initialize Spanish tokenizer.
        
        Args:
            use_stopwords: Whether to remove Spanish stopwords
            lowercase: Whether to lowercase text
        """
        self.use_stopwords = use_stopwords
        self.lowercase = lowercase
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Spanish text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Lowercase if requested
        if self.lowercase:
            text = text.lower()
        
        # Split on whitespace and punctuation, but preserve Spanish characters
        # This regex keeps letters (including Spanish accented chars) and numbers
        tokens = re.findall(r'[a-záéíóúñü]+', text, re.UNICODE | re.IGNORECASE)
        
        # Remove stopwords if requested
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in SPANISH_STOPWORDS]
        
        return tokens


class BM25Index:
    """BM25 index for keyword-based retrieval."""
    
    def __init__(
        self,
        tokenizer: Optional[SpanishTokenizer] = None,
        storage_path: Optional[Path] = None,
    ):
        """Initialize BM25 index.
        
        Args:
            tokenizer: Tokenizer for text processing
            storage_path: Path to store/load index
        """
        self.tokenizer = tokenizer or SpanishTokenizer(use_stopwords=False)
        self.storage_path = storage_path
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.metadatas: List[Dict] = []
        
        # Try to load existing index
        if storage_path and storage_path.exists():
            self.load()
    
    def add_documents(
        self,
        texts: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """Add documents to the BM25 index.
        
        Args:
            texts: Document texts
            doc_ids: Document IDs
            metadatas: Optional metadata for each document
        """
        self.documents.extend(texts)
        self.doc_ids.extend(doc_ids)
        
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(texts))
        
        # Rebuild BM25 index
        self._build_index()
    
    def _build_index(self) -> None:
        """Build or rebuild the BM25 index."""
        if not self.documents:
            self.bm25 = None
            return
        
        # Tokenize all documents
        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in self.documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with scores
        """
        if not self.bm25 or not self.documents:
            return []
        
        # Tokenize query
        tokenized_query = self.tokenizer.tokenize(query)
        
        if not tokenized_query:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    "content": self.documents[idx],
                    "doc_id": self.doc_ids[idx],
                    "score": float(scores[idx]),
                    "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                })
        
        return results
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all entries with a specific doc_id.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Number of entries deleted
        """
        indices_to_keep = [
            i for i, did in enumerate(self.doc_ids)
            if did != doc_id
        ]
        
        deleted_count = len(self.doc_ids) - len(indices_to_keep)
        
        if deleted_count > 0:
            self.documents = [self.documents[i] for i in indices_to_keep]
            self.doc_ids = [self.doc_ids[i] for i in indices_to_keep]
            self.metadatas = [self.metadatas[i] for i in indices_to_keep]
            self._build_index()
        
        return deleted_count
    
    def save(self) -> None:
        """Save the BM25 index to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "metadatas": self.metadatas,
        }
        
        with open(self.storage_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self) -> None:
        """Load the BM25 index from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data.get("documents", [])
            self.doc_ids = data.get("doc_ids", [])
            self.metadatas = data.get("metadatas", [])
            
            # Rebuild BM25 index
            self._build_index()
        except Exception as e:
            print(f"Warning: Failed to load BM25 index: {e}")
            self.documents = []
            self.doc_ids = []
            self.metadatas = []
            self.bm25 = None
    
    def clear(self) -> None:
        """Clear the entire index."""
        self.documents = []
        self.doc_ids = []
        self.metadatas = []
        self.bm25 = None
        
        if self.storage_path and self.storage_path.exists():
            self.storage_path.unlink()


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic search using RRF."""
    
    def __init__(
        self,
        bm25_weight: float = 0.35,
        semantic_weight: float = 0.65,
    ):
        """Initialize hybrid retriever.
        
        Args:
            bm25_weight: Weight for BM25 scores
            semantic_weight: Weight for semantic scores
        """
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        
        # Normalize weights
        total_weight = bm25_weight + semantic_weight
        self.bm25_weight /= total_weight
        self.semantic_weight /= total_weight
    
    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict],
        semantic_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """Combine results using Reciprocal Rank Fusion.
        
        Args:
            bm25_results: Results from BM25 search
            semantic_results: Results from semantic search
            k: RRF parameter (default 60)
            
        Returns:
            Merged and ranked results
        """
        # Create a dictionary to store combined scores
        doc_scores: Dict[str, Dict] = {}
        
        # Add BM25 scores (using RRF)
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.get("id", result.get("doc_id", ""))
            chunk_key = f"{doc_id}_{result.get('chunk_index', 0)}"
            
            if chunk_key not in doc_scores:
                doc_scores[chunk_key] = {
                    "result": result,
                    "bm25_score": 0,
                    "semantic_score": 0,
                }
            
            # RRF score: 1 / (k + rank)
            doc_scores[chunk_key]["bm25_score"] = 1.0 / (k + rank)
        
        # Add semantic scores (using RRF)
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result.get("id", result.get("doc_id", ""))
            chunk_key = f"{doc_id}_{result.get('chunk_index', 0)}"
            
            if chunk_key not in doc_scores:
                doc_scores[chunk_key] = {
                    "result": result,
                    "bm25_score": 0,
                    "semantic_score": 0,
                }
            
            # RRF score
            doc_scores[chunk_key]["semantic_score"] = 1.0 / (k + rank)
        
        # Calculate combined scores
        for chunk_key in doc_scores:
            bm25_score = doc_scores[chunk_key]["bm25_score"]
            semantic_score = doc_scores[chunk_key]["semantic_score"]
            
            # Weighted combination
            combined_score = (
                self.bm25_weight * bm25_score +
                self.semantic_weight * semantic_score
            )
            
            doc_scores[chunk_key]["combined_score"] = combined_score
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True
        )
        
        # Format results
        results = []
        for chunk_key, data in sorted_docs:
            result = data["result"].copy()
            result["score"] = data["combined_score"]
            result["bm25_contribution"] = data["bm25_score"]
            result["semantic_contribution"] = data["semantic_score"]
            results.append(result)
        
        return results


@lru_cache
def get_bm25_index() -> BM25Index:
    """Get cached BM25 index instance.
    
    Returns:
        BM25Index instance
    """
    settings = get_settings()
    storage_path = Path(settings.documents_dir) / "bm25_index.pkl"
    
    tokenizer = SpanishTokenizer(
        use_stopwords=settings.use_spanish_stopwords,
        lowercase=True
    )
    
    return BM25Index(tokenizer=tokenizer, storage_path=storage_path)


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    """Get cached hybrid retriever instance.
    
    Returns:
        HybridRetriever instance
    """
    settings = get_settings()
    return HybridRetriever(
        bm25_weight=settings.bm25_weight,
        semantic_weight=settings.semantic_weight,
    )

"""Query expansion service for generating multiple query variants."""

from typing import List
from functools import lru_cache

from langchain_ollama import ChatOllama

from app.core.config import get_settings


# Spanish query expansion prompt
QUERY_EXPANSION_PROMPT_ES = """Genera {num_variants} reformulaciones alternativas de esta pregunta para recuperar información relevante:
Original: {query}

Enfócate en usar sinónimos, diferentes palabras clave y perspectivas distintas, manteniendo la misma intención de búsqueda.
Proporciona solo las {num_variants} preguntas reformuladas, una por línea, sin numeración ni explicaciones."""


class QueryExpander:
    """Service for expanding queries into multiple variants."""
    
    def __init__(self, llm: ChatOllama | None = None):
        """Initialize query expander.
        
        Args:
            llm: LLM instance for query generation
        """
        self.settings = get_settings()
        self._llm = llm
    
    @property
    def llm(self) -> ChatOllama:
        """Lazy load the LLM."""
        if self._llm is None:
            self._llm = ChatOllama(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                temperature=0.3,  # Slightly creative for variations
            )
        return self._llm
    
    def expand_query(
        self,
        query: str,
        num_variants: int | None = None,
        include_original: bool = True,
    ) -> List[str]:
        """Expand a query into multiple variants.
        
        Args:
            query: Original query
            num_variants: Number of variants to generate (None uses settings)
            include_original: Whether to include original query in results
            
        Returns:
            List of query variants (including original if requested)
        """
        if num_variants is None:
            num_variants = self.settings.num_query_variants
        
        # Generate query variants
        prompt = QUERY_EXPANSION_PROMPT_ES.format(
            query=query,
            num_variants=num_variants,
        )
        
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            
            # Parse response - expect one query per line
            variants = []
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Split by newlines and clean
            lines = content.strip().split('\n')
            for line in lines:
                # Remove numbering, bullets, and extra whitespace
                cleaned = line.strip()
                # Remove common numbering patterns
                cleaned = cleaned.lstrip('0123456789.-) ')
                
                if cleaned and len(cleaned) > 5:  # Basic quality filter
                    variants.append(cleaned)
            
            # Limit to requested number
            variants = variants[:num_variants]
            
        except Exception as e:
            print(f"Warning: Query expansion failed: {e}")
            variants = []
        
        # Always include original query
        if include_original:
            return [query] + variants
        else:
            return variants if variants else [query]
    
    def expand_with_fallback(
        self,
        query: str,
        num_variants: int | None = None,
    ) -> List[str]:
        """Expand query with simple fallback if LLM fails.
        
        Args:
            query: Original query
            num_variants: Number of variants to generate
            
        Returns:
            List of query variants
        """
        variants = self.expand_query(query, num_variants, include_original=True)
        
        # If expansion failed, use simple heuristics
        if len(variants) <= 1:
            # Add simple variants
            simple_variants = [query]
            
            # Variant 1: More general (remove specific words)
            words = query.split()
            if len(words) > 3:
                general = ' '.join(words[:len(words)//2])
                simple_variants.append(general)
            
            # Variant 2: More specific (add context)
            if "qué" in query.lower() or "cómo" in query.lower():
                simple_variants.append(query + " explicación detallada")
            
            return simple_variants[:num_variants + 1] if num_variants else simple_variants
        
        return variants


@lru_cache
def get_query_expander() -> QueryExpander:
    """Get cached query expander instance.
    
    Returns:
        QueryExpander instance
    """
    return QueryExpander()

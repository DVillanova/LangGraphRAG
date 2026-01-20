"""Retriever tools for the RAG agent."""

from langchain.tools import tool

from app.core.config import get_settings
from app.services.document_service import get_document_service
from app.rag.query_expander import get_query_expander


@tool
def retrieve_documents(query: str) -> str:
    """Search and return relevant information from the indexed documents.

    Use this tool when you need to find information from the document knowledge base.
    The tool performs advanced hybrid search (BM25 + semantic) with optional query expansion
    and re-ranking for optimal retrieval quality.

    Args:
        query: The search query to find relevant documents.

    Returns:
        Concatenated content from the most relevant document chunks with citations.
    """
    settings = get_settings()
    doc_service = get_document_service()
    
    # Step 1: Query expansion (if enabled)
    queries_to_search = [query]
    if settings.use_query_expansion:
        try:
            expander = get_query_expander()
            queries_to_search = expander.expand_with_fallback(query)
        except Exception as e:
            print(f"Warning: Query expansion failed, using original query: {e}")
            queries_to_search = [query]
    
    # Step 2: Retrieve documents for each query variant
    all_results = {}  # Use dict to deduplicate by chunk ID
    
    for search_query in queries_to_search:
        try:
            # Hybrid search with re-ranking is handled in doc_service.search
            results = doc_service.search(
                query=search_query,
                use_reranking=settings.use_reranking,
            )
            
            # Deduplicate results by chunk ID, keeping highest score
            for result in results:
                chunk_id = result.get("id", "")
                if chunk_id:
                    if chunk_id not in all_results or result.get("score", 0) > all_results[chunk_id].get("score", 0):
                        all_results[chunk_id] = result
        
        except Exception as e:
            print(f"Warning: Search failed for query '{search_query}': {e}")
            continue
    
    # Convert back to list and sort by score
    results = sorted(
        all_results.values(),
        key=lambda x: x.get("score", 0),
        reverse=True
    )
    
    # Limit to final_top_k
    results = results[:settings.final_top_k]

    if not results:
        return "No relevant documents found for the query."

    # Step 3: Format results with source attribution including page numbers
    formatted_results = []
    for result in results:
        source = result.get("source", "Unknown")
        content = result.get("content", "")
        score = result.get("score", 0)
        
        # Build citation header with page number if available
        citation_parts = [f"Fuente: {source}"]
        
        if "page_number" in result:
            page_num = result["page_number"]
            citation_parts.append(f"Página: {page_num}")
        
        citation_parts.append(f"relevancia: {score:.2f}")
        citation = ", ".join(citation_parts)
        
        formatted_results.append(
            f"[{citation}]\n{content}"
        )

    return "\n\n---\n\n".join(formatted_results)


# Export the tool for use in the agent
retriever_tool = retrieve_documents

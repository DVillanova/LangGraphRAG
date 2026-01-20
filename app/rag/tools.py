"""Retriever tools for the RAG agent."""

from langchain.tools import tool

from app.services.document_service import get_document_service


@tool
def retrieve_documents(query: str) -> str:
    """Search and return relevant information from the indexed documents.

    Use this tool when you need to find information from the document knowledge base.
    The tool performs semantic search across all indexed documents.

    Args:
        query: The search query to find relevant documents.

    Returns:
        Concatenated content from the most relevant document chunks.
    """
    doc_service = get_document_service()
    results = doc_service.search(query=query)

    if not results:
        return "No relevant documents found for the query."

    # Format results with source attribution
    formatted_results = []
    for result in results:
        source = result.get("source", "Unknown")
        content = result.get("content", "")
        score = result.get("score", 0)
        formatted_results.append(
            f"[Source: {source} (relevance: {score:.2f})]\n{content}"
        )

    return "\n\n---\n\n".join(formatted_results)


# Export the tool for use in the agent
retriever_tool = retrieve_documents

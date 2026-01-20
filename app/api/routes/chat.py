"""Chat API routes."""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse, SearchResult
from app.rag.agent import chat, get_final_response
from app.services.document_service import get_document_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Send a message to the RAG agent and get a response.

    The agent will decide whether to retrieve documents or respond directly
    based on the user's question.
    """
    try:
        result = chat(request.message, stream=False)
        response_text = get_final_response(result)

        # Extract sources from tool messages if available
        sources = []
        for msg in result.get("messages", []):
            if hasattr(msg, "type") and msg.type == "tool":
                # Parse source information from tool response
                content = msg.content if hasattr(msg, "content") else ""
                if "[Source:" in content:
                    # Extract source names
                    import re
                    source_matches = re.findall(r"\[Source: ([^\]]+)\]", content)
                    sources.extend([s.split(" (relevance")[0] for s in source_matches])

        return ChatResponse(
            response=response_text,
            sources=sources if sources else None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Send a message to the RAG agent and stream the response.

    Returns a server-sent events stream with updates from each node.
    """

    async def generate():
        try:
            for chunk in chat(request.message, stream=True):
                for node, update in chunk.items():
                    # Get the last message content
                    content = ""
                    if "messages" in update and update["messages"]:
                        last_msg = update["messages"][-1]
                        if hasattr(last_msg, "content"):
                            content = last_msg.content
                        elif isinstance(last_msg, dict):
                            content = last_msg.get("content", "")

                    data = {
                        "node": node,
                        "content": content,
                        "done": False,
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            # Send final done signal
            yield f"data: {json.dumps({'node': 'end', 'content': '', 'done': True})}\n\n"

        except Exception as e:
            error_data = {"node": "error", "content": str(e), "done": True}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """Search the document index without RAG processing.

    This endpoint performs a direct semantic search without
    involving the LLM for answer generation.
    """
    try:
        doc_service = get_document_service()
        results = doc_service.search(query=request.query, limit=request.limit)

        search_results = [
            SearchResult(
                content=r["content"],
                source=r["source"],
                score=r["score"],
                doc_id=r["doc_id"],
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            results=search_results,
            total=len(search_results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

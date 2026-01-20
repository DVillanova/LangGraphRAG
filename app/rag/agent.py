"""LangGraph RAG agent assembly."""

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.rag.nodes import (
    generate_query_or_respond,
    grade_documents,
    rewrite_question,
    generate_answer,
    ask_clarification,
)
from app.rag.tools import retriever_tool


# Maximum number of question rewrites before asking for clarification
MAX_REWRITE_ATTEMPTS = 2


class RAGState(MessagesState):
    """Extended state for RAG agent with rewrite tracking."""
    
    rewrite_count: int = 0


def grade_documents_with_limit(
    state: RAGState,
) -> Literal["generate_answer", "rewrite_question", "ask_clarification"]:
    """Grade documents and check rewrite limit.
    
    Routes to:
    - generate_answer: if documents are relevant
    - rewrite_question: if documents not relevant and under limit
    - ask_clarification: if rewrite limit reached
    """
    # Check if we've exceeded rewrite attempts
    rewrite_count = state.get("rewrite_count", 0)
    
    if rewrite_count >= MAX_REWRITE_ATTEMPTS:
        return "ask_clarification"
    
    # Otherwise, use normal grading
    return grade_documents(state)


def increment_rewrite_count(state: RAGState) -> dict:
    """Increment the rewrite counter and rewrite the question."""
    # Get current count
    current_count = state.get("rewrite_count", 0)
    
    # Rewrite the question using the existing function
    result = rewrite_question(state)
    
    # Add the incremented count
    result["rewrite_count"] = current_count + 1
    
    return result


def create_rag_agent():
    """Create and compile the RAG agent graph.

    The graph implements the following flow:
    1. generate_query_or_respond: LLM decides to retrieve or respond directly
    2. If tool call -> retrieve: Execute retriever tool
    3. grade_documents: Check if retrieved docs are relevant
    4. If relevant -> generate_answer: Generate final response
    5. If not relevant and under limit -> rewrite_question: Reformulate and retry
    6. If rewrite limit reached -> ask_clarification: Ask user to clarify

    Returns:
        Compiled LangGraph agent.
    """
    # Create the workflow graph with extended state
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", increment_rewrite_count)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("ask_clarification", ask_clarification)

    # Set entry point
    workflow.add_edge(START, "generate_query_or_respond")

    # Add conditional edge from generate_query_or_respond
    # - If LLM calls the retriever tool -> go to retrieve
    # - If LLM responds directly -> end
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # After retrieval, grade the documents with limit check
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents_with_limit,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
            "ask_clarification": "ask_clarification",
        },
    )

    # After generating answer, end
    workflow.add_edge("generate_answer", END)

    # After rewriting question, try again
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # After asking for clarification, end
    workflow.add_edge("ask_clarification", END)

    # Compile the graph
    return workflow.compile()


# Create a singleton instance
_agent = None


def get_rag_agent():
    """Get the RAG agent singleton instance.

    Returns:
        Compiled RAG agent.
    """
    global _agent
    if _agent is None:
        _agent = create_rag_agent()
    return _agent


def chat(message: str, history: list = None, stream: bool = False):
    """Send a message to the RAG agent with conversation history.

    Args:
        message: User message to process.
        history: Optional list of previous messages for context.
                 Each message should be a dict with 'role' and 'content'.
        stream: Whether to stream the response.

    Returns:
        Agent response (generator if streaming, dict otherwise).
    """
    agent = get_rag_agent()
    
    # Build messages list with history
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})
    
    input_state = {"messages": messages, "rewrite_count": 0}

    if stream:
        return agent.stream(input_state)
    else:
        result = agent.invoke(input_state)
        return result


def get_final_response(result: dict) -> str:
    """Extract the final response from agent result.

    Args:
        result: Agent execution result.

    Returns:
        The final response text.
    """
    if "messages" in result and result["messages"]:
        last_message = result["messages"][-1]
        if hasattr(last_message, "content"):
            return last_message.content
        elif isinstance(last_message, dict):
            return last_message.get("content", "")
    return ""

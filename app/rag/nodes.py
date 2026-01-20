"""Graph nodes for the RAG agent."""

from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.tools import retriever_tool


# Prompts (Spanish)
GRADE_PROMPT = """Eres un evaluador que determina la relevancia de un documento recuperado con respecto 
a la pregunta del usuario.

Aquí está el documento recuperado:

{context}

Aquí está la pregunta del usuario: {question}

Si el documento contiene palabras clave o significado semántico relacionado con la pregunta del usuario, 
califícalo como relevante.
Da una puntuación binaria 'yes' o 'no' para indicar si el documento es relevante para la pregunta."""

REWRITE_PROMPT = """Eres un reformulador de preguntas que convierte una pregunta inicial en una versión 
mejorada optimizada para la recuperación de documentos. Analiza la entrada e intenta razonar sobre la 
intención semántica subyacente.

Aquí está la pregunta inicial:

{question}

Formula una pregunta mejorada:"""

GENERATE_PROMPT = """Eres un asistente para tareas de respuesta a preguntas. Utiliza los siguientes 
fragmentos de contexto recuperado para responder la pregunta. Si no sabes la respuesta, simplemente 
di que no lo sabes. Mantén la respuesta concisa, pero no olvides ningún detalle relevante.

IMPORTANTE: Siempre cita la fuente y el número de página cuando respondas. Usa el formato:
[Fuente: nombre_archivo.pdf, Página: X]

Si múltiples fuentes contienen información relevante, cítalas todas.

Pregunta: {question}

Contexto: {context}"""

CLARIFICATION_MESSAGE = """[NO_CONTEXT_FOUND] No pude encontrar información relevante en los documentos indexados 
para responder tu pregunta. Esto puede deberse a que:

1. La información no está en los documentos indexados
2. La pregunta necesita ser más específica
3. Se utilizan términos diferentes en los documentos

Por favor, intenta reformular tu pregunta con más detalle o utilizando términos alternativos. 
También puedes verificar que el documento relevante esté indexado usando el comando `rag list`."""

# System message for conversation context
SYSTEM_MESSAGE = """Eres un asistente experto que responde preguntas basándose ÚNICAMENTE en documentos indexados.
Tienes acceso a una herramienta de búsqueda que te permite recuperar información relevante de los documentos.

Instrucciones CRÍTICAS:
- DEBES usar la herramienta de búsqueda para TODAS las preguntas sobre contenido de documentos
- NO respondas directamente preguntas sobre reglas, mecánicas, o información específica del juego
- Usa la herramienta de búsqueda incluso para preguntas de seguimiento o clarificaciones
- Considera el contexto de la conversación anterior al formular tus búsquedas
- Si el usuario hace referencia a algo mencionado anteriormente, incluye ese contexto en tu búsqueda
- Responde siempre en español"""


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def get_llm() -> ChatOllama:
    """Get the configured LLM instance."""
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0,
    )


def should_use_retrieval(query: str, conversation_history: list) -> bool:
    """Determine if a query should trigger document retrieval.

    Args:
        query: The current user query
        conversation_history: Previous messages in the conversation

    Returns:
        True if retrieval should be used, False for direct response
    """
    # Keywords that strongly indicate document retrieval is needed
    strong_retrieval_keywords = [
        'reglas', 'regla', 'mecánicas', 'mecánica', 'manual', 'consulta', 'consultar',
        'página', 'p.', 'capítulo', 'sección', 'apartado', 'forja', 'creación',
        'objeto', 'objetos', 'personaje', 'personajes', 'habilidades', 'habilidad',
        'magia', 'mágico', 'don', 'divino', 'sobrenatural', 'combate', 'lucha',
        'daño', 'ataque', 'defensa', 'arma', 'armas', 'equipo', 'equipamiento'
    ]

    # Question words that often need document lookup
    question_words = ['cómo', 'qué', 'cuál', 'cuáles', 'dónde', 'cuándo', 'por qué']

    query_lower = query.lower()

    # Check for strong retrieval keywords
    has_strong_keywords = any(keyword in query_lower for keyword in strong_retrieval_keywords)

    # Check for question words combined with game terms
    has_question = any(word in query_lower for word in question_words)

    # Check if this is a follow-up conversation (likely needs context)
    is_follow_up = len(conversation_history) > 2  # More than just the current query

    # Force retrieval for:
    # 1. Queries with strong retrieval keywords
    # 2. Questions in follow-up conversations
    # 3. Explicit requests to consult/ check the manual
    return has_strong_keywords or (has_question and is_follow_up) or 'consulta' in query_lower or 'manual' in query_lower


def generate_query_or_respond(state: MessagesState) -> dict:
    """Generate a query using the retriever tool or respond directly.

    This node analyzes the query and decides whether to retrieve documents
    or respond directly based on keywords and conversation context.

    Args:
        state: Current graph state with messages.

    Returns:
        Updated state with the LLM response.
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    # Get the current user message
    current_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            current_query = msg.content
            break

    # Check if retrieval is needed
    needs_retrieval = should_use_retrieval(current_query, state["messages"])

    if needs_retrieval:
        # Force retrieval by creating a tool call in the correct format
        tool_call = {
            "name": "retrieve_documents",
            "args": {"query": current_query},
            "id": f"retrieval_{len(state['messages'])}"
        }

        response = AIMessage(
            content="Voy a buscar la información relevante en los documentos.",
            tool_calls=[tool_call]
        )
        return {"messages": [response]}

    # For general conversation, use the LLM normally
    llm = get_llm()
    llm_with_tools = llm.bind_tools([retriever_tool])

    messages = [SystemMessage(content=SYSTEM_MESSAGE)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Grade the relevance of retrieved documents.

    This conditional edge determines whether retrieved documents
    are relevant to the question and routes accordingly.

    Args:
        state: Current graph state with messages.

    Returns:
        Name of the next node: 'generate_answer' or 'rewrite_question'.
    """
    # Get the current question (last human message to focus on clarifications)
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (
            hasattr(msg, "type") and msg.type == "human"
        ):
            question = msg.content
            break

    # Get the retrieved context (last tool message)
    context = state["messages"][-1].content

    # Grade using structured output
    llm = get_llm()
    grader = llm.with_structured_output(GradeDocuments)

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader.invoke([{"role": "user", "content": prompt}])

    if response.binary_score.lower() == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


def rewrite_question(state: MessagesState) -> dict:
    """Rewrite the question for better retrieval.

    This node reformulates the question when retrieved documents
    are not relevant.

    Args:
        state: Current graph state with messages.

    Returns:
        Updated state with the rewritten question.
    """
    # Get the current question (last human message to focus on clarifications)
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (
            hasattr(msg, "type") and msg.type == "human"
        ):
            question = msg.content
            break

    llm = get_llm()
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])

    return {"messages": [HumanMessage(content=response.content)]}


def generate_answer(state: MessagesState) -> dict:
    """Generate the final answer using retrieved context.

    This node produces the final response based on the current
    question and the relevant retrieved documents.

    Args:
        state: Current graph state with messages.

    Returns:
        Updated state with the generated answer.
    """
    # Get the current question (last human message to focus on clarifications)
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (
            hasattr(msg, "type") and msg.type == "human"
        ):
            question = msg.content
            break

    # Get the retrieved context (last tool message)
    context = state["messages"][-1].content

    llm = get_llm()
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])

    return {"messages": [response]}


def ask_clarification(state: MessagesState) -> dict:
    """Ask the user to clarify their question.

    This node is called when the agent fails to find relevant documents
    after multiple rewrite attempts.

    Args:
        state: Current graph state with messages.

    Returns:
        Updated state with clarification request message.
    """
    from langchain_core.messages import AIMessage
    
    return {"messages": [AIMessage(content=CLARIFICATION_MESSAGE)]}

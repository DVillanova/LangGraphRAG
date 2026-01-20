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
di que no lo sabes. Mantén la respuesta concisa, pero no olvides ningún detalle relevante. Cita la referencia
del documento en la respuesta.

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
SYSTEM_MESSAGE = """Eres un asistente experto que responde preguntas basándose en documentos indexados.
Tienes acceso a una herramienta de búsqueda que te permite recuperar información relevante de los documentos.

Instrucciones importantes:
- Usa la herramienta de búsqueda para encontrar información antes de responder preguntas sobre documentos
- Considera el contexto de la conversación anterior al formular tus búsquedas
- Si el usuario hace referencia a algo mencionado anteriormente (como "la tabla que mencionaste"), 
  incluye ese contexto en tu búsqueda
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


def generate_query_or_respond(state: MessagesState) -> dict:
    """Generate a query using the retriever tool or respond directly.

    This node calls the LLM to decide whether to retrieve documents
    or respond directly to the user based on the current conversation.

    Args:
        state: Current graph state with messages.

    Returns:
        Updated state with the LLM response.
    """
    from langchain_core.messages import SystemMessage
    
    llm = get_llm()
    llm_with_tools = llm.bind_tools([retriever_tool])
    
    # Prepend system message to provide context instructions
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
    # Get the original question (first human message)
    question = ""
    for msg in state["messages"]:
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
    # Get the original question
    question = ""
    for msg in state["messages"]:
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

    This node produces the final response based on the original
    question and the relevant retrieved documents.

    Args:
        state: Current graph state with messages.

    Returns:
        Updated state with the generated answer.
    """
    # Get the original question
    question = ""
    for msg in state["messages"]:
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

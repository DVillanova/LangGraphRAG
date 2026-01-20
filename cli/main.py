"""CLI interface for the LangGraph RAG Agent."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.services.document_service import get_document_service
from app.rag.agent import chat, get_final_response
from cli.translations import t, set_language, get_language

# Create CLI app
cli = typer.Typer(
    name="rag",
    help="Agente RAG con LangGraph - Sistema de Generación Aumentada por Recuperación local",
    add_completion=False,
)

console = Console()


def language_callback(lang: str):
    """Callback to set language before command execution."""
    if lang:
        set_language(lang)


@cli.callback()
def main_callback(
    lang: str = typer.Option(
        "es",
        "--lang", "-l",
        help="Idioma / Language: 'es' (español) or 'en' (english)",
        callback=language_callback,
        is_eager=True,
    ),
):
    """LangGraph RAG Agent - Sistema de Generación Aumentada por Recuperación."""
    pass


@cli.command()
def serve(
    host: str = typer.Option(None, "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(None, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """Iniciar el servidor FastAPI. / Start the FastAPI server."""
    import uvicorn

    settings = get_settings()
    host = host or settings.api_host
    port = port or settings.api_port

    console.print(
        Panel(
            f"[bold green]{t('serve_starting')}[/bold green]\n\n"
            f"Host: {host}\n"
            f"Port: {port}\n"
            f"Docs: http://{host}:{port}/docs",
            title=t("serve_title"),
        )
    )

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command(name="chat")
def chat_command():
    """Iniciar chat interactivo. / Start an interactive chat session."""
    console.print(
        Panel(
            t("chat_welcome"),
            title=t("chat_title"),
        )
    )

    # Maintain conversation history for context
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = console.input(f"\n[bold green]{t('chat_prompt')}:[/bold green] ").strip()

            if not user_input:
                continue

            # Handle special commands (support both Spanish and English)
            if user_input.lower() in ("exit", "quit", "q", "salir"):
                console.print(f"[yellow]{t('goodbye')}[/yellow]")
                break

            if user_input.lower() in ("help", "ayuda"):
                console.print(t("chat_help_text"))
                continue

            if user_input.lower() in ("clear", "limpiar"):
                console.clear()
                continue
            
            # New command to reset conversation history
            if user_input.lower() in ("reset", "nuevo", "new"):
                conversation_history.clear()
                console.print(f"[cyan]{t('chat_reset')}[/cyan]")
                continue

            # Process with RAG agent, passing conversation history
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task(t("chat_thinking"), total=None)
                result = chat(user_input, history=conversation_history, stream=False)
                response = get_final_response(result)

            # Check if this was a failed retrieval (clarification message)
            # Don't add failed exchanges to history to avoid confusing the model
            is_clarification = "[NO_CONTEXT_FOUND]" in response
            
            if is_clarification:
                # Remove the marker for display
                response = response.replace("[NO_CONTEXT_FOUND] ", "")
            else:
                # Only add successful exchanges to conversation history
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})
                
                # Limit history to last 10 exchanges (20 messages) to avoid context overflow
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]

            # Display response
            console.print(f"\n[bold blue]{t('chat_assistant')}:[/bold blue]")
            console.print(Markdown(response))

        except KeyboardInterrupt:
            console.print(f"\n[yellow]{t('interrupted')}[/yellow]")
        except Exception as e:
            console.print(f"\n[red]{t('error')}: {e}[/red]")


@cli.command(name="add")
def add_document(
    file_path: Path = typer.Argument(..., help="Ruta al documento / Path to document"),
):
    """Añadir documento al índice. / Add a document to the RAG index."""
    if not file_path.exists():
        console.print(f"[red]{t('error')}: {t('add_file_not_found', path=file_path)}[/red]")
        raise typer.Exit(1)

    doc_service = get_document_service()

    # Check if file type is supported
    supported = doc_service.get_supported_extensions()
    suffix = file_path.suffix.lower()
    if suffix not in supported:
        console.print(
            f"[red]{t('error')}: {t('add_unsupported', suffix=suffix, supported=', '.join(supported))}[/red]"
        )
        raise typer.Exit(1)

    # First, show a spinner while loading and splitting the document
    console.print(f"[cyan]{t('add_loading', filename=file_path.name)}[/cyan]")

    try:
        # Load and split first to get chunk count
        documents = doc_service.load_document(file_path)
        chunks = doc_service.split_documents(documents)
        total_chunks = len(chunks)

        console.print(f"[cyan]{t('add_splitting', count=total_chunks)}[/cyan]")

        # Now index with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(t("add_indexing"), total=total_chunks)

            def update_progress(current, total):
                progress.update(task, completed=current)

            doc_id, indexed_chunks = doc_service.add_document(
                file_path,
                progress_callback=update_progress,
            )

        console.print(
            Panel(
                t("add_success", filename=file_path.name, doc_id=doc_id, chunks=indexed_chunks),
                title=t("add_success_title"),
            )
        )
    except Exception as e:
        console.print(f"[red]{t('add_error')}: {e}[/red]")
        raise typer.Exit(1)


@cli.command(name="list")
def list_documents():
    """Listar documentos indexados. / List all indexed documents."""
    doc_service = get_document_service()

    try:
        documents = doc_service.list_documents()

        if not documents:
            console.print(f"[yellow]{t('list_empty')}[/yellow]")
            console.print(t("list_empty_hint"))
            return

        table = Table(title=t("list_title"))
        table.add_column(t("list_col_id"), style="cyan", no_wrap=True)
        table.add_column(t("list_col_source"), style="green")
        table.add_column(t("list_col_chunks"), justify="right", style="magenta")

        for doc in documents:
            table.add_row(
                doc["doc_id"][:8] + "...",
                doc["source"],
                str(doc["chunk_count"]),
            )

        console.print(table)
        console.print(f"\n{t('list_total', count=len(documents))}")

    except Exception as e:
        console.print(f"[red]{t('list_error')}: {e}[/red]")
        raise typer.Exit(1)


@cli.command(name="remove")
def remove_document(
    doc_id: str = typer.Argument(..., help="ID del documento / Document ID (can be partial)"),
    force: bool = typer.Option(False, "--force", "-f", help="Omitir confirmación / Skip confirmation"),
):
    """Eliminar documento del índice. / Remove a document from the RAG index."""
    doc_service = get_document_service()

    # Find matching documents
    documents = doc_service.list_documents()
    matches = [d for d in documents if d["doc_id"].startswith(doc_id)]

    if not matches:
        console.print(f"[red]{t('remove_not_found', doc_id=doc_id)}[/red]")
        raise typer.Exit(1)

    if len(matches) > 1:
        console.print(f"[yellow]{t('remove_multiple', doc_id=doc_id)}[/yellow]")
        for doc in matches:
            console.print(f"  - {doc['doc_id'][:12]}... ({doc['source']})")
        console.print(f"\n{t('remove_specify')}")
        raise typer.Exit(1)

    target_doc = matches[0]

    # Confirm deletion
    if not force:
        confirm = typer.confirm(
            t("remove_confirm", source=target_doc['source'], chunks=target_doc['chunk_count'])
        )
        if not confirm:
            console.print(f"[yellow]{t('remove_cancelled')}[/yellow]")
            raise typer.Exit(0)

    try:
        chunks_deleted, file_deleted = doc_service.delete_document(target_doc["doc_id"])
        file_deleted_str = t("remove_yes") if file_deleted else t("remove_no")

        console.print(
            Panel(
                t("remove_success", source=target_doc['source'], chunks=chunks_deleted, file_deleted=file_deleted_str),
                title=t("remove_title"),
            )
        )
    except Exception as e:
        console.print(f"[red]{t('remove_error')}: {e}[/red]")
        raise typer.Exit(1)


@cli.command(name="search")
def search_documents(
    query: str = typer.Argument(..., help="Consulta de búsqueda / Search query"),
    limit: int = typer.Option(4, "--limit", "-n", help="Máximo de resultados / Maximum results"),
):
    """Buscar en documentos (sin RAG). / Search indexed documents (without RAG)."""
    doc_service = get_document_service()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(t("search_searching"), total=None)
        results = doc_service.search(query=query, limit=limit)

    if not results:
        console.print(f"[yellow]{t('search_no_results')}[/yellow]")
        return

    console.print(f"\n[bold]{t('search_results_for')}:[/bold] {query}\n")

    for i, result in enumerate(results, 1):
        console.print(
            Panel(
                f"{result['content'][:500]}{'...' if len(result['content']) > 500 else ''}",
                title=f"[{i}] {result['source']} (score: {result['score']:.3f})",
                border_style="blue",
            )
        )


@cli.command(name="status")
def show_status():
    """Mostrar estado de servicios. / Show the status of all services."""
    import httpx

    settings = get_settings()

    console.print(f"\n[bold]{t('status_title')}[/bold]\n")

    # Check Ollama
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            console.print(f"[green]✓ Ollama[/green]: {t('status_connected')}")
            if models:
                console.print(f"  {t('status_models')}: {', '.join(models[:5])}")
        else:
            console.print(f"[yellow]⚠ Ollama[/yellow]: {t('status_error')} (status {response.status_code})")
    except Exception as e:
        console.print(f"[red]✗ Ollama[/red]: {t('status_disconnected')} ({e})")

    # Check Qdrant
    try:
        from app.services.vector_store import get_vector_store_service

        vector_store = get_vector_store_service()
        info = vector_store.get_collection_info()
        console.print(f"[green]✓ Qdrant[/green]: {t('status_connected')}")
        console.print(f"  {t('status_collection')}: {info['name']}")
        console.print(f"  {t('status_vectors')}: {info['vectors_count']}")
    except Exception as e:
        console.print(f"[red]✗ Qdrant[/red]: {t('status_disconnected')} ({e})")

    # Show configuration
    console.print(f"\n[bold]{t('status_config_title')}[/bold]")
    console.print(f"  {t('status_ollama_url')}: {settings.ollama_base_url}")
    console.print(f"  {t('status_ollama_model')}: {settings.ollama_model}")
    console.print(f"  {t('status_qdrant')}: {settings.qdrant_host}:{settings.qdrant_port}")
    console.print(f"  {t('status_embedding_model')}: {settings.embedding_model}")
    console.print(f"  {t('status_documents_dir')}: {settings.documents_dir}")


@cli.command(name="init")
def initialize():
    """Inicializar sistema RAG. / Initialize the RAG system."""
    settings = get_settings()

    console.print(
        Panel(
            f"[bold cyan]{t('init_initializing')}[/bold cyan]",
            title=t("init_title"),
        )
    )

    # Create documents directory
    docs_dir = Path(settings.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] {t('init_docs_dir')}: {docs_dir}")

    # Initialize embedding model (this downloads it if needed)
    console.print(f"[yellow]...[/yellow] {t('init_loading_embed')}")
    try:
        from app.services.embeddings import get_embedding_service

        embed_service = get_embedding_service()
        dim = embed_service.embedding_dimension
        console.print(f"[green]✓[/green] {t('init_embed_loaded', dim=dim)}")
    except Exception as e:
        console.print(f"[red]✗[/red] {t('init_embed_failed')}: {e}")
        raise typer.Exit(1)

    # Initialize vector store collection
    console.print(f"[yellow]...[/yellow] {t('init_connecting_qdrant')}")
    try:
        from app.services.vector_store import get_vector_store_service

        vector_store = get_vector_store_service()
        vector_store.ensure_collection()
        console.print(f"[green]✓[/green] {t('init_qdrant_ready', name=settings.qdrant_collection_name)}")
    except Exception as e:
        console.print(f"[red]✗[/red] {t('init_qdrant_failed')}: {e}")
        console.print(f"\n[yellow]{t('init_qdrant_hint')}[/yellow]")
        console.print("  docker-compose up -d")
        raise typer.Exit(1)

    # Check Ollama
    console.print(f"[yellow]...[/yellow] {t('init_checking_ollama')}")
    try:
        import httpx

        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            console.print(f"[green]✓[/green] {t('init_ollama_connected')}")

            # Check if model is available
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            if settings.ollama_model in models or any(
                settings.ollama_model in m for m in models
            ):
                console.print(f"[green]✓[/green] {t('init_model_available', model=settings.ollama_model)}")
            else:
                console.print(f"[yellow]⚠[/yellow] {t('init_model_not_found', model=settings.ollama_model)}")
                console.print(f"  {t('init_model_hint', model=settings.ollama_model)}")
    except Exception as e:
        console.print(f"[red]✗[/red] {t('init_ollama_failed')}: {e}")
        console.print(f"\n[yellow]{t('init_ollama_hint')}[/yellow]")
        console.print("  ollama serve")

    console.print(
        Panel(
            t("init_complete"),
            title=t("init_complete_title"),
        )
    )


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()

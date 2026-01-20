"""CLI translations for Spanish and English."""

TRANSLATIONS = {
    "es": {
        # General
        "app_help": "Agente RAG con LangGraph - Sistema de Generación Aumentada por Recuperación local",
        "goodbye": "¡Hasta luego!",
        "interrupted": "Interrumpido. Escribe 'salir' para terminar.",
        "error": "Error",
        
        # Serve command
        "serve_help": "Iniciar el servidor FastAPI.",
        "serve_host_help": "Host para vincular",
        "serve_port_help": "Puerto para vincular",
        "serve_reload_help": "Habilitar recarga automática",
        "serve_title": "Agente RAG con LangGraph",
        "serve_starting": "Iniciando Servidor del Agente RAG",
        
        # Chat command
        "chat_help": "Iniciar una sesión de chat interactiva con el agente RAG.",
        "chat_title": "Chat Interactivo",
        "chat_welcome": (
            "[bold cyan]Chat del Agente RAG[/bold cyan]\n\n"
            "Haz preguntas sobre tus documentos indexados.\n"
            "Escribe 'salir' o 'exit' para terminar la sesión.\n"
            "Escribe 'ayuda' para más comandos."
        ),
        "chat_prompt": "Tú",
        "chat_assistant": "Asistente",
        "chat_thinking": "Pensando...",
        "chat_help_text": (
            "\n[bold]Comandos:[/bold]\n"
            "  salir, exit, q - Salir del chat\n"
            "  ayuda, help - Mostrar este mensaje de ayuda\n"
            "  limpiar, clear - Limpiar la pantalla\n"
            "  nuevo, reset - Iniciar nueva conversación (borrar historial)\n"
        ),
        "chat_reset": "Historial de conversación borrado. Nueva conversación iniciada.",
        
        # Add command
        "add_help": "Añadir un documento al índice RAG.",
        "add_file_help": "Ruta al documento a añadir",
        "add_loading": "Cargando {filename}...",
        "add_splitting": "Documento dividido en {count} fragmentos. Indexando...",
        "add_indexing": "Indexando fragmentos",
        "add_success_title": "Documento Añadido",
        "add_success": (
            "[green]✓ Documento indexado exitosamente[/green]\n\n"
            "Archivo: {filename}\n"
            "ID del documento: {doc_id}\n"
            "Fragmentos indexados: {chunks}"
        ),
        "add_error": "Error al indexar documento",
        "add_file_not_found": "Archivo no encontrado: {path}",
        "add_unsupported": "Tipo de archivo no soportado: {suffix}\nTipos soportados: {supported}",
        
        # List command
        "list_help": "Listar todos los documentos indexados.",
        "list_title": "Documentos Indexados",
        "list_empty": "No hay documentos indexados todavía.",
        "list_empty_hint": "Usa 'rag add <archivo>' para añadir documentos.",
        "list_col_id": "ID del Documento",
        "list_col_source": "Fuente",
        "list_col_chunks": "Fragmentos",
        "list_total": "Total de documentos: {count}",
        "list_error": "Error al listar documentos",
        
        # Remove command
        "remove_help": "Eliminar un documento del índice RAG.",
        "remove_id_help": "ID del documento a eliminar (puede ser parcial)",
        "remove_force_help": "Omitir confirmación",
        "remove_not_found": "No se encontró documento con ID: {doc_id}",
        "remove_multiple": "Múltiples documentos coinciden con '{doc_id}':",
        "remove_specify": "Por favor, proporciona un ID más específico.",
        "remove_confirm": "¿Eliminar documento '{source}' ({chunks} fragmentos)?",
        "remove_cancelled": "Cancelado.",
        "remove_title": "Documento Eliminado",
        "remove_success": (
            "[green]✓ Documento eliminado[/green]\n\n"
            "Documento: {source}\n"
            "Fragmentos eliminados: {chunks}\n"
            "Archivo eliminado: {file_deleted}"
        ),
        "remove_error": "Error al eliminar documento",
        "remove_yes": "Sí",
        "remove_no": "No",
        
        # Search command
        "search_help": "Buscar en documentos indexados (sin procesamiento RAG).",
        "search_query_help": "Consulta de búsqueda",
        "search_limit_help": "Máximo de resultados",
        "search_searching": "Buscando...",
        "search_no_results": "No se encontraron resultados.",
        "search_results_for": "Resultados de búsqueda para",
        
        # Status command
        "status_help": "Mostrar el estado de todos los servicios.",
        "status_title": "Estado de Servicios",
        "status_connected": "Conectado",
        "status_disconnected": "Desconectado",
        "status_error": "Error",
        "status_models": "Modelos",
        "status_collection": "Colección",
        "status_vectors": "Vectores",
        "status_config_title": "Configuración",
        "status_ollama_url": "URL de Ollama",
        "status_ollama_model": "Modelo de Ollama",
        "status_qdrant": "Qdrant",
        "status_embedding_model": "Modelo de embeddings",
        "status_documents_dir": "Directorio de documentos",
        
        # Init command
        "init_help": "Inicializar el sistema RAG (crear colección, verificar servicios).",
        "init_title": "Configuración",
        "init_initializing": "Inicializando Sistema RAG",
        "init_docs_dir": "Directorio de documentos",
        "init_loading_embed": "Cargando modelo de embeddings...",
        "init_embed_loaded": "Modelo de embeddings cargado (dimensión: {dim})",
        "init_embed_failed": "Error al cargar modelo de embeddings",
        "init_connecting_qdrant": "Conectando a Qdrant...",
        "init_qdrant_ready": "Colección de Qdrant lista: {name}",
        "init_qdrant_failed": "Error al conectar con Qdrant",
        "init_qdrant_hint": "Asegúrate de que Qdrant esté ejecutándose:",
        "init_checking_ollama": "Verificando Ollama...",
        "init_ollama_connected": "Ollama conectado",
        "init_model_available": "Modelo disponible: {model}",
        "init_model_not_found": "Modelo no encontrado: {model}",
        "init_model_hint": "Ejecuta: ollama pull {model}",
        "init_ollama_failed": "Ollama no conectado",
        "init_ollama_hint": "Asegúrate de que Ollama esté ejecutándose:",
        "init_complete_title": "Listo",
        "init_complete": (
            "[bold green]¡Inicialización completada![/bold green]\n\n"
            "Próximos pasos:\n"
            "1. Añadir documentos: rag add <archivo>\n"
            "2. Iniciar chat: rag chat\n"
            "3. O iniciar la API: rag serve"
        ),
    },
    "en": {
        # General
        "app_help": "LangGraph RAG Agent - A locally-operating Retrieval Augmented Generation system",
        "goodbye": "Goodbye!",
        "interrupted": "Interrupted. Type 'exit' to quit.",
        "error": "Error",
        
        # Serve command
        "serve_help": "Start the FastAPI server.",
        "serve_host_help": "Host to bind to",
        "serve_port_help": "Port to bind to",
        "serve_reload_help": "Enable auto-reload",
        "serve_title": "LangGraph RAG Agent",
        "serve_starting": "Starting RAG Agent Server",
        
        # Chat command
        "chat_help": "Start an interactive chat session with the RAG agent.",
        "chat_title": "Interactive Chat",
        "chat_welcome": (
            "[bold cyan]RAG Agent Chat[/bold cyan]\n\n"
            "Ask questions about your indexed documents.\n"
            "Type 'exit' or 'quit' to end the session.\n"
            "Type 'help' for more commands."
        ),
        "chat_prompt": "You",
        "chat_assistant": "Assistant",
        "chat_thinking": "Thinking...",
        "chat_help_text": (
            "\n[bold]Commands:[/bold]\n"
            "  exit, quit, q - Exit the chat\n"
            "  help - Show this help message\n"
            "  clear - Clear the screen\n"
            "  new, reset - Start new conversation (clear history)\n"
        ),
        "chat_reset": "Conversation history cleared. New conversation started.",
        
        # Add command
        "add_help": "Add a document to the RAG index.",
        "add_file_help": "Path to the document to add",
        "add_loading": "Loading {filename}...",
        "add_splitting": "Document split into {count} chunks. Indexing...",
        "add_indexing": "Indexing chunks",
        "add_success_title": "Document Added",
        "add_success": (
            "[green]✓ Document indexed successfully[/green]\n\n"
            "File: {filename}\n"
            "Document ID: {doc_id}\n"
            "Chunks indexed: {chunks}"
        ),
        "add_error": "Error indexing document",
        "add_file_not_found": "File not found: {path}",
        "add_unsupported": "Unsupported file type: {suffix}\nSupported types: {supported}",
        
        # List command
        "list_help": "List all indexed documents.",
        "list_title": "Indexed Documents",
        "list_empty": "No documents indexed yet.",
        "list_empty_hint": "Use 'rag add <file>' to add documents.",
        "list_col_id": "Document ID",
        "list_col_source": "Source",
        "list_col_chunks": "Chunks",
        "list_total": "Total documents: {count}",
        "list_error": "Error listing documents",
        
        # Remove command
        "remove_help": "Remove a document from the RAG index.",
        "remove_id_help": "Document ID to remove (can be partial)",
        "remove_force_help": "Skip confirmation",
        "remove_not_found": "No document found matching: {doc_id}",
        "remove_multiple": "Multiple documents match '{doc_id}':",
        "remove_specify": "Please provide a more specific ID.",
        "remove_confirm": "Delete document '{source}' ({chunks} chunks)?",
        "remove_cancelled": "Cancelled.",
        "remove_title": "Document Removed",
        "remove_success": (
            "[green]✓ Document removed[/green]\n\n"
            "Document: {source}\n"
            "Chunks deleted: {chunks}\n"
            "File deleted: {file_deleted}"
        ),
        "remove_error": "Error removing document",
        "remove_yes": "Yes",
        "remove_no": "No",
        
        # Search command
        "search_help": "Search indexed documents (without RAG processing).",
        "search_query_help": "Search query",
        "search_limit_help": "Maximum results",
        "search_searching": "Searching...",
        "search_no_results": "No results found.",
        "search_results_for": "Search Results for",
        
        # Status command
        "status_help": "Show the status of all services.",
        "status_title": "Service Status",
        "status_connected": "Connected",
        "status_disconnected": "Disconnected",
        "status_error": "Error",
        "status_models": "Models",
        "status_collection": "Collection",
        "status_vectors": "Vectors",
        "status_config_title": "Configuration",
        "status_ollama_url": "Ollama URL",
        "status_ollama_model": "Ollama Model",
        "status_qdrant": "Qdrant",
        "status_embedding_model": "Embedding Model",
        "status_documents_dir": "Documents Dir",
        
        # Init command
        "init_help": "Initialize the RAG system (create collection, check services).",
        "init_title": "Setup",
        "init_initializing": "Initializing RAG System",
        "init_docs_dir": "Documents directory",
        "init_loading_embed": "Loading embedding model...",
        "init_embed_loaded": "Embedding model loaded (dimension: {dim})",
        "init_embed_failed": "Failed to load embedding model",
        "init_connecting_qdrant": "Connecting to Qdrant...",
        "init_qdrant_ready": "Qdrant collection ready: {name}",
        "init_qdrant_failed": "Failed to connect to Qdrant",
        "init_qdrant_hint": "Make sure Qdrant is running:",
        "init_checking_ollama": "Checking Ollama...",
        "init_ollama_connected": "Ollama connected",
        "init_model_available": "Model available: {model}",
        "init_model_not_found": "Model not found: {model}",
        "init_model_hint": "Run: ollama pull {model}",
        "init_ollama_failed": "Ollama not connected",
        "init_ollama_hint": "Make sure Ollama is running:",
        "init_complete_title": "Ready",
        "init_complete": (
            "[bold green]Initialization complete![/bold green]\n\n"
            "Next steps:\n"
            "1. Add documents: rag add <file>\n"
            "2. Start chatting: rag chat\n"
            "3. Or start the API: rag serve"
        ),
    },
}


# Default language
_current_lang = "es"


def set_language(lang: str) -> None:
    """Set the current language."""
    global _current_lang
    if lang in TRANSLATIONS:
        _current_lang = lang


def get_language() -> str:
    """Get the current language."""
    return _current_lang


def t(key: str, **kwargs) -> str:
    """Get a translated string.
    
    Args:
        key: Translation key
        **kwargs: Format arguments
        
    Returns:
        Translated and formatted string
    """
    text = TRANSLATIONS.get(_current_lang, TRANSLATIONS["es"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text

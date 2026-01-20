# LangGraph RAG Agent

A production-grade, locally-operating Retrieval Augmented Generation (RAG) agent built with LangGraph. This system allows you to index documents and query them using a local LLM, all running on your machine.

## Features

- **Fully Local Operation**: Uses Ollama for LLM inference and local embeddings
- **Multiple Document Types**: Supports PDF, TXT, CSV, and HTML files
- **Production-Ready API**: FastAPI endpoints for easy integration
- **Interactive CLI**: Rich command-line interface for document management and chat
- **Intelligent Retrieval**: LangGraph-powered agent that decides when to retrieve context
- **Document Management**: Add, list, and remove documents from the vector store

## Architecture

```
                    User Interface Layer
              CLI Interface / Future Web UI
                          |
                          v
                    FastAPI Server
    /chat    /chat/stream    /documents    /health
                          |
                          v
                RAG Agent (LangGraph)
  generate_query -> retrieve -> grade -> generate_answer
                      ^                        |
              rewrite_question <---------------+
                          |
        +-----------------+------------------+
        |                 |                  |
        v                 v                  v
     Ollama           Qdrant           Sentence
   (LLM: 3B)       (Vector DB)       Transformers
```

## Prerequisites

1. **Python 3.10+**

2. **Ollama** - Install from [ollama.ai](https://ollama.ai)
   ```bash
   # After installation, pull the model
   ollama pull llama3.2:3b
   ```

3. **Docker** - For running Qdrant vector database

## Installation

1. **Clone the repository**
   ```bash
   cd LangGraphRAG
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   # Or: pip install -r requirements.txt
   ```

4. **Configure environment** (optional)
   ```bash
   cp env.example .env
   # Edit .env to customize settings
   ```

5. **Start Qdrant**
   ```bash
   docker-compose up -d
   ```

6. **Initialize the system**
   ```bash
   rag init
   ```

## Usage

### CLI Commands

```bash
# Initialize the system (first time setup)
rag init

# Check service status
rag status

# Add a document to the index
rag add path/to/document.pdf
rag add research_paper.txt

# List indexed documents
rag list

# Remove a document
rag remove <doc_id>

# Search documents directly
rag search "your query here"

# Start interactive chat
rag chat

# Start the API server
rag serve
```

### Interactive Chat

Start an interactive session with the RAG agent:

```bash
rag chat
```

Then ask questions about your indexed documents. The agent will automatically retrieve relevant context when needed.

### API Endpoints

Start the server with `rag serve`, then access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send a message and get a response |
| `/api/chat/stream` | POST | Stream the response (SSE) |
| `/api/chat/search` | POST | Direct semantic search |
| `/api/documents` | GET | List all indexed documents |
| `/api/documents` | POST | Upload and index a document |
| `/api/documents/{id}` | DELETE | Remove a document |
| `/health` | GET | Check service health |
| `/docs` | GET | OpenAPI documentation |

#### Example API Usage

```python
import httpx

# Chat with the RAG agent
response = httpx.post(
    "http://localhost:8000/api/chat",
    json={"message": "What does the document say about X?"}
)
print(response.json())

# Upload a document
with open("document.pdf", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/documents",
        files={"file": f}
    )
print(response.json())
```

## Configuration

Configuration can be set via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM model to use |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION_NAME` | `langgraph_rag_docs` | Collection name |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Sentence transformer model |
| `DOCUMENTS_DIR` | `./data/documents` | Document storage path |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `CHUNK_SIZE` | `900` | Chunk size for splitting large documents |
| `CHUNK_OVERLAP` | `200` | Overlap between splits to preserve context |
| `RETRIEVER_K` | `6` | Number of chunks retrieved before grading |
| `MAX_DOCUMENT_LENGTH` | `1000000` | Soft limit on characters per document to protect RAM |

### Large document tuning

The defaults above are sized for large PDFs with the new `intfloat/multilingual-e5-large` embeddings.
If you still push against memory limits, reduce `CHUNK_SIZE` or `RETRIEVER_K`, or increase `CHUNK_OVERLAP` to keep
context connected without creating too many vectors.

## Supported Document Types

- PDF (`.pdf`)
- Plain text (`.txt`)
- CSV (`.csv`)
- HTML (`.html`, `.htm`)

## Project Structure

```
LangGraphRAG/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py         # Chat endpoints
│   │   │   └── documents.py    # Document CRUD
│   │   └── schemas.py          # Pydantic models
│   ├── core/
│   │   └── config.py           # Settings
│   ├── rag/
│   │   ├── agent.py            # LangGraph agent
│   │   ├── nodes.py            # Graph nodes
│   │   └── tools.py            # Retriever tool
│   └── services/
│       ├── document_service.py # Document processing
│       ├── vector_store.py     # Qdrant operations
│       └── embeddings.py       # Embedding service
├── cli/
│   └── main.py                 # CLI interface
├── data/documents/             # Document storage
├── docker-compose.yml          # Qdrant container
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Troubleshooting

### Ollama not connecting
```bash
# Make sure Ollama is running
ollama serve

# Check if the model is available
ollama list

# Pull the model if needed
ollama pull llama3.2:3b
```

### Qdrant not connecting
```bash
# Start Qdrant with Docker
docker-compose up -d

# Check if it's running
docker ps

# View logs
docker-compose logs qdrant
```

### Memory issues with embeddings
The default embedding model (`all-MiniLM-L6-v2`) is lightweight. If you experience memory issues:
1. Reduce `chunk_size` in settings
2. Process fewer documents at once

## License

MIT License

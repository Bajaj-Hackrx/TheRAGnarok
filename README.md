# Agentic RAG System

A production-ready, end-to-end agentic RAG system for contextual question-answering over insurance, legal, and compliance documents. Built with FastAPI, ChromaDB, sentence-transformers, and OpenRouter LLM API, orchestrated using LangGraph.

## Features

- **Multi-format document processing**: PDF, DOCX, EML, TXT
- **Semantic vector search**: ChromaDB with cosine similarity
- **Agent orchestration**: LangGraph + LangChain workflow management
- **Structured JSON responses**: Consistent, typed API responses
- **Production-ready**: Docker support, health checks, logging
- **Flexible LLM backend**: OpenRouter API (Meta Llama, GPT-4o, Gemini, etc.)

## Architecture

```
agentic-rag-system/
├── backend/
│   ├── main.py                # FastAPI application entry point
│   ├── agents/                # Agent orchestration (LangGraph)
│   │   ├── entity_extractor.py   # Entity extraction agent
│   │   ├── vector_searcher.py    # Vector search agent
│   │   ├── final_reasoner.py     # Decision reasoning agent
│   │   └── coordinator.py        # LangGraph workflow coordinator
│   ├── services/              # Core services
│   │   ├── chromadb_service.py   # Vector database operations
│   │   ├── llm_service.py        # OpenRouter LLM integration
│   │   └── ingestion_service.py  # Document processing
│   ├── models/                # Pydantic schemas and config
│   │   ├── schemas.py            # API request/response models
│   │   └── config.py             # Configuration management
│   └── requirements.txt       # Python dependencies
├── data/
│   ├── documents/             # Document uploads
│   └── chroma_db/             # Persistent vector storage
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Multi-service deployment
└── .env.example              # Environment configuration template
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker (optional, recommended)
- OpenRouter API key ([get one here](https://openrouter.ai/))

### 2. Environment Setup

```bash
# Clone or create the project directory
cd agentic-rag-system

# Copy environment template
cp .env.example .env

# Edit .env with your OpenRouter API key
# OPENROUTER_API_KEY=your_api_key_here
```

### 3. Installation Options

#### Option A: Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run with Docker directly
docker build -t agentic-rag .
docker run -p 8000:8000 --env-file .env agentic-rag
```

#### Option B: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Run the application
python main.py
```

### 4. Verify Installation

Visit `http://localhost:8000/docs` for interactive API documentation or check system health:

```bash
curl http://localhost:8000/health
```

## API Usage

### Document Upload

Upload and index documents for Q&A:

```bash
# Upload a PDF document
curl -X POST "http://localhost:8000/upload/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"

# Response:
{
  "message": "Document 'your_document.pdf' successfully processed and indexed",
  "document_id": "your_document_a1b2c3d4",
  "chunks_processed": 25
}
```

### Query Documents

Ask natural language questions:

```bash
# Query the document collection
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the coverage limit for dental procedures?",
    "max_results": 5
  }'

# Response:
{
  "decision": "approved",
  "confidence": 0.85,
  "reasoning": "Based on the policy documents, dental procedures are covered up to $2,000 annually...",
  "conditions": "Subject to deductible and in-network provider requirements",
  "citations": ["policy_doc_chunk_12", "benefits_summary_chunk_5"],
  "entities_found": {
    "procedure": "dental procedures",
    "coverage_amount": "$2,000",
    "additional_entities": {}
  },
  "search_results": [...],
  "processing_time": 2.34
}
```

## Advanced Configuration

### Model Selection

Configure different LLM models via environment variables:

```bash
# Use different OpenRouter models
DEFAULT_MODEL=anthropic/claude-3.5-sonnet
DEFAULT_MODEL=openai/gpt-4o
DEFAULT_MODEL=google/gemini-pro-1.5
DEFAULT_MODEL=deepseek/deepseek-chat-v3-0324:free
```

### Document Processing

Adjust chunking and processing parameters:

```bash
CHUNK_SIZE=500          # Larger chunks for more context
CHUNK_OVERLAP=100       # More overlap for better continuity
MAX_FILE_SIZE_MB=100    # Allow larger files
```

### Performance Tuning

For production deployment:

```bash
# Use multiple Gunicorn workers
gunicorn backend.main:app -w 8 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or configure via Docker Compose
docker-compose up --scale agentic-rag=4
```

## Agent Workflow

The system uses a multi-agent architecture orchestrated by LangGraph:

1. **Entity Extractor**: Parses user query to identify key entities (age, procedures, conditions, etc.)
2. **Vector Searcher**: Performs hybrid semantic + keyword search over document collection
3. **Final Reasoner**: Generates structured decisions with confidence scores and citations

### Workflow State

Each query maintains state through the workflow:

```python
{
  "query": "user question",
  "entities": {"age": "65", "procedure": "cardiac surgery"},
  "search_results": [...],
  "final_decision": {
    "decision": "approved",
    "confidence": 0.85,
    "reasoning": "...",
    "citations": [...]
  }
}
```

## API Endpoints

### Core Endpoints

- `POST /upload/` - Upload and index documents
- `POST /query/` - Query document collection
- `GET /health` - System health check
- `GET /status/` - Detailed system status
- `GET /docs` - Interactive API documentation

### Response Format

All responses follow consistent JSON schemas:

```python
# Query Response
{
  "decision": "approved|rejected|requires_review|insufficient_info",
  "confidence": 0.85,  # 0.0 to 1.0
  "reasoning": "Detailed explanation...",
  "conditions": "Optional conditions...",
  "citations": ["chunk_id_1", "chunk_id_2"],
  "entities_found": {...},
  "search_results": [...],
  "processing_time": 2.34
}
```

## Troubleshooting

### Common Issues

#### 1. ChromaDB SQLite Version Error

```bash
# For some deployment environments, add to requirements.txt:
pysqlite3-binary==0.5.2.post2

# Or set environment variable:
SYSTEM_VERSION_COMPAT=0 pip install chromadb
```

#### 2. OpenRouter API Errors

```bash
# Verify API key is set
echo $OPENROUTER_API_KEY

# Check model availability
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models
```

#### 3. Memory Issues

```bash
# Reduce batch size for large documents
CHUNK_SIZE=200
MAX_FILE_SIZE_MB=25

# Or increase Docker memory limit
docker run -m 4g agentic-rag
```

### Debugging

Enable debug mode for detailed logging:

```bash
DEBUG=true python backend/main.py
```

Check logs:

```bash
# Docker logs
docker-compose logs -f

# Local logs
tail -f agentic_rag.log
```

## Production Deployment

### Security Considerations

1. **API Keys**: Use secure secret management
2. **CORS**: Configure `allow_origins` for production domains
3. **Rate Limiting**: Add rate limiting middleware
4. **Authentication**: Add API authentication if needed

### Monitoring

Health checks are available at `/health` endpoint:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status/
```

### Scaling

The system supports horizontal scaling:

```bash
# Scale with Docker Compose
docker-compose up --scale agentic-rag=4

# Or use Kubernetes/cloud deployment
kubectl apply -f k8s-deployment.yaml
```

## Development

### Adding New Document Types

1. Extend `DocumentType` enum in `models/schemas.py`
2. Add processing logic in `services/ingestion_service.py`
3. Update file validation in upload endpoint

### Customizing Agents

1. Create new agent in `agents/` directory
2. Add to workflow in `agents/coordinator.py`
3. Update state schema if needed

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/

# Test with sample documents
python scripts/test_upload.py
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review logs for error details
3. Verify all environment variables are set correctly
4. Ensure OpenRouter API key has sufficient credits

---

**Built with ❤️ using FastAPI, ChromaDB, LangGraph, and OpenRouter**

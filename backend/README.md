# Document Intelligence RAG System - Backend

This is the backend component of the Document Intelligence RAG system. It provides a complete Retrieval-Augmented Generation (RAG) pipeline for document processing, retrieval, and question answering.

## Architecture

The system consists of the following components:

1. **Document Processor**: Handles document ingestion, text extraction, metadata extraction, and chunking.
2. **Hybrid Retriever**: Combines vector search, BM25, and TF-IDF for robust retrieval.
3. **Reranker**: Uses Cohere and TART models to rerank search results for improved relevance.
4. **LLM Service**: Integrates with OpenRouter to provide LLM-powered responses with citations.
5. **API Layer**: FastAPI endpoints for document upload, chat, and system management.

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Redis
- PostgreSQL
- API keys for:
  - OpenRouter (for LLM access)
  - Cohere (for reranking)
  - Pinecone (for vector storage)

### Environment Setup

1. **Clone the repository**

2. **Set up environment variables**

   The `.env` file has been created with placeholder values. You need to replace these with your actual API keys:

   ```
   OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
   COHERE_API_KEY=YOUR_COHERE_API_KEY
   PINECONE_API_KEY=YOUR_PINECONE_API_KEY
   PINECONE_ENVIRONMENT=YOUR_PINECONE_ENVIRONMENT
   PINECONE_INDEX=docqa-index
   ```

3. **Run the setup script**

   ```bash
   cd backend
   python scripts/setup_environment.py
   ```

   This will:
   - Check for required dependencies
   - Set up a Python virtual environment
   - Install required packages
   - Download the spaCy language model (en_core_web_md)
   - Start Docker containers for Redis and PostgreSQL
   - Initialize the database and Pinecone index

   > **Note:** If you encounter issues with the spaCy model, you can manually install it:
   > ```bash
   > python -m spacy download en_core_web_md
   > ```
   > Or use the dedicated script:
   > ```bash
   > python scripts/setup_spacy.py
   > ```

4. **Quick Test**

   After setting up your environment variables, you can run a quick test to verify everything works:

   ```bash
   python scripts/quick_test.py
   ```

   This will test the complete RAG pipeline with a simple example document.

### Running the Application

1. **Start the application**

   ```bash
   python scripts/dev.py
   ```

   This will start the FastAPI application on http://localhost:8000

2. **API Documentation**

   Once the application is running, you can access the API documentation at:
   - http://localhost:8000/docs (Swagger UI)
   - http://localhost:8000/redoc (ReDoc)

## API Endpoints

### Document Management

- `POST /api/v1/documents/upload`: Upload and process a document
- `GET /api/v1/documents/status/{document_id}`: Check document processing status
- `DELETE /api/v1/documents/{document_id}`: Delete a document

### Chat

- `POST /api/v1/chat/stream`: Stream chat responses with RAG
- `POST /api/v1/chat`: Non-streaming chat endpoint

### Health Checks

- `GET /health`: System health check
- `GET /api/v1/chat/health`: Chat service health check
- `GET /api/v1/documents/health`: Document service health check

## Development

### Running Tests

```bash
pytest backend/tests
```

### Code Quality

```bash
black backend
isort backend
pylint backend
mypy backend
```

## Monitoring

The system includes monitoring endpoints that expose Prometheus metrics:

- `GET /metrics`: System metrics

You can visualize these metrics using Grafana by running:

```bash
python scripts/setup_monitoring.py
```

This will start Prometheus and Grafana containers and configure them to scrape metrics from the application.

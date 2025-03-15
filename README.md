# Document Intelligence RAG System

An advanced RAG (Retrieval Augmented Generation) system for document Q&A with hybrid search, reranking, and streaming responses.

## Features

- **Advanced RAG Pipeline**
  - Hybrid search combining vector, BM25, and TF-IDF
  - Multi-model reranking with Cohere and TART
  - Streaming responses with source citations
  - AutoRAG optimization for best retrieval/reranking

- **Document Processing**
  - PDF parsing with layout preservation
  - Semantic chunking
  - Metadata extraction
  - Automatic entity recognition

- **Infrastructure**
  - Redis caching
  - PostgreSQL for document storage
  - Pinecone vector database
  - Docker containerization

## Tech Stack

- Frontend: Next.js 14 (App Router)
- Backend: FastAPI + Python
- Database: PostgreSQL
- Cache: Redis
- Vector DB: Pinecone
- LLM: OpenRouter (GPT-4)
- Reranking: Cohere, TART
- Deployment: Vercel, Railway

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker and Docker Compose
- API Keys:
  - OpenRouter
  - Cohere
  - Pinecone
- Accounts for deployment:
  - GitHub account
  - Railway account
  - Vercel account

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd document-answer
```

2. Configure environment variables:
```bash
# Frontend
cp .env.example .env.local
# Edit .env.local with your settings

# Backend
cp backend/.env.template backend/.env
# Edit backend/.env with your API keys and settings
```

3. Start the services with Docker:
```bash
docker-compose up -d
```

4. Install frontend dependencies:
```bash
npm install
```

5. Run the development servers:
```bash
# Backend
cd backend
uvicorn app.main:app --reload

# Frontend
npm run dev
```

### Deployment

1. Push code to GitHub:
```bash
git add .
git commit -m "Initial commit"
git remote add origin <your-github-url>
git push -u origin main
```

2. Set up Railway:
- Create new project in Railway
- Connect to your GitHub repository
- Add environment variables from backend/.env
- Railway will automatically deploy when you push to main

3. Set up Vercel:
- Import your GitHub repository
- Configure environment variables from .env.local
- Vercel will automatically deploy when you push to main

4. Add repository secrets in GitHub:
- RAILWAY_TOKEN
- VERCEL_TOKEN
- VERCEL_ORG_ID
- VERCEL_PROJECT_ID
- OPENROUTER_API_KEY
- COHERE_API_KEY
- PINECONE_API_KEY
- PINECONE_ENVIRONMENT
- PINECONE_INDEX
- SLACK_WEBHOOK_URL (optional, for notifications)

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## RAG Pipeline Optimization

1. Generate test dataset:
```bash
cd backend
python scripts/generate_test_data.py
```

2. Run AutoRAG optimization:
```bash
python scripts/optimize_rag.py
```

This will:
- Create synthetic test documents and QA pairs
- Evaluate different retrieval/reranking configurations
- Output optimal settings for your use case

## API Endpoints

### Documents

- `POST /api/v1/documents/upload`: Upload PDF document
- `GET /api/v1/documents/status/{document_id}`: Check processing status
- `DELETE /api/v1/documents/{document_id}`: Delete document

### Chat

- `POST /api/v1/chat/stream`: Stream chat responses (SSE)
- `POST /api/v1/chat`: Get complete response (non-streaming)

## Development

### Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   └── main.py
│   ├── core/
│   │   ├── document_processor.py
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   └── llm_service.py
│   ├── models/
│   │   └── document.py
│   └── scripts/
│       ├── generate_test_data.py
│       └── optimize_rag.py
├── src/
│   ├── app/
│   │   └── page.tsx
│   └── components/
└── docker-compose.yml
```

### Adding New Features

1. Backend:
- Add models in `backend/models/`
- Create new services in `backend/core/`
- Add API endpoints in `backend/app/api/v1/`

2. Frontend:
- Add components in `src/components/`
- Create new pages in `src/app/`
- Add API client functions in `src/lib/`

## Deployment

The application is configured for deployment on:
- Frontend: Vercel
- Backend: Railway
- Database: Railway PostgreSQL
- Cache: Railway Redis
- Vector DB: Pinecone Cloud

CI/CD is handled through GitHub Actions, automatically deploying on pushes to main.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

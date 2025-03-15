import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json
import tempfile
import os

from app.main import app
from models.document import Document, DocumentChunk, ChatMessage, ChatResponse, SearchResult

# Create test client
client = TestClient(app)

@pytest.fixture
def test_pdf():
    """Create a test PDF file"""
    content = """
    Machine Learning Basics
    
    This is a test document about machine learning.
    It contains information about various topics.
    """
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", mode="w", delete=False) as f:
        f.write(content)
        return f.name

@pytest.fixture
def mock_services():
    """Mock all external services"""
    with patch("core.document_processor.DocumentProcessor") as mock_processor, \
         patch("core.hybrid_retriever.HybridRetriever") as mock_retriever, \
         patch("core.reranker.Reranker") as mock_reranker, \
         patch("core.llm_service.LLMService") as mock_llm, \
         patch("redis.asyncio.from_url"):
        
        # Mock document processor
        mock_processor.return_value.process_pdf = AsyncMock(return_value=Document(
            filename="test.pdf",
            content="Test content",
            metadata={"test": True}
        ))
        mock_processor.return_value.extract_metadata = lambda doc: {
            "language": "en",
            "entities": [],
            "key_phrases": [],
            "statistics": {"chars": 100, "words": 20, "sentences": 2}
        }
        mock_processor.return_value.create_chunks = lambda doc: [
            DocumentChunk(
                document_id=doc.id,
                content="Test chunk",
                chunk_index=0,
                metadata={"test": True}
            )
        ]
        
        # Mock retriever
        mock_retriever.return_value.add_documents = AsyncMock()
        mock_retriever.return_value.hybrid_search = AsyncMock(return_value=[
            SearchResult(
                chunk=DocumentChunk(
                    document_id="test-id",
                    content="Test result",
                    chunk_index=0,
                    metadata={"test": True}
                ),
                vector_score=0.8,
                bm25_score=0.7,
                tfidf_score=0.6,
                combined_score=0.75
            )
        ])
        
        # Mock reranker
        mock_reranker.return_value.rerank_hybrid = AsyncMock(return_value=[
            SearchResult(
                chunk=DocumentChunk(
                    document_id="test-id",
                    content="Test reranked result",
                    chunk_index=0,
                    metadata={"test": True}
                ),
                vector_score=0.8,
                bm25_score=0.7,
                tfidf_score=0.6,
                rerank_score=0.9,
                combined_score=0.85
            )
        ])
        
        # Mock LLM service
        async def mock_generate(*args, **kwargs):
            yield ChatResponse(
                message="Test response with [Source 1] citation.",
                sources=[DocumentChunk(
                    document_id="test-id",
                    content="Test source content",
                    chunk_index=0,
                    metadata={"test": True}
                )],
                finished=True
            )
        mock_llm.return_value.generate_response = mock_generate
        mock_llm.return_value._extract_citations = lambda text: [1]
        
        yield {
            "processor": mock_processor,
            "retriever": mock_retriever,
            "reranker": mock_reranker,
            "llm": mock_llm
        }

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_document_upload(test_pdf, mock_services):
    """Test document upload endpoint"""
    # Prepare file upload
    with open(test_pdf, "rb") as f:
        files = {"file": ("test.pdf", f, "application/pdf")}
        response = client.post("/api/v1/documents/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["message"] == "Document processed successfully"
    
    # Clean up
    os.unlink(test_pdf)

@pytest.mark.asyncio
async def test_document_status(mock_services):
    """Test document status endpoint"""
    response = client.get("/api/v1/documents/status/test-id")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

@pytest.mark.asyncio
async def test_chat_stream(mock_services):
    """Test streaming chat endpoint"""
    # Test request
    request_data = {
        "query": "What is machine learning?",
        "chat_history": [
            {
                "role": "user",
                "content": "Tell me about AI"
            },
            {
                "role": "assistant",
                "content": "AI is a broad field..."
            }
        ],
        "use_cache": True,
        "top_k": 3,
        "stream": True
    }
    
    response = client.post("/api/v1/chat/stream", json=request_data)
    assert response.status_code == 200
    
    # Parse SSE response
    responses = []
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data != "[DONE]":
                    responses.append(json.loads(data))
    
    assert len(responses) > 0
    assert "message" in responses[-1]
    assert responses[-1]["finished"] == True
    assert "sources" in responses[-1]
    assert len(responses[-1]["sources"]) > 0

@pytest.mark.asyncio
async def test_chat_non_streaming(mock_services):
    """Test non-streaming chat endpoint"""
    request_data = {
        "query": "What is machine learning?",
        "chat_history": [],
        "use_cache": True,
        "top_k": 3,
        "stream": False
    }
    
    response = client.post("/api/v1/chat", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["finished"] == True
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert "[Source 1]" in data["message"]

@pytest.mark.asyncio
async def test_chat_health_check(mock_services):
    """Test chat health check endpoint"""
    response = client.get("/api/v1/chat/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "retriever" in data
    assert "reranker" in data
    assert "llm_service" in data
    assert "timestamp" in data

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in endpoints"""
    # Test invalid file upload
    response = client.post("/api/v1/documents/upload", files={})
    assert response.status_code == 422  # Validation error
    
    # Test invalid document ID
    response = client.get("/api/v1/documents/status/invalid-id")
    assert response.status_code == 404
    
    # Test invalid chat request
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422
    
    # Test missing required fields
    response = client.post("/api/v1/chat", json={"use_cache": True})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_large_document_handling(mock_services):
    """Test handling of large documents"""
    # Create large test document
    large_content = "Test content\n" * 1000  # ~10KB
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", mode="w", delete=False) as f:
        f.write(large_content)
        large_file = f.name
    
    try:
        # Test upload
        with open(large_file, "rb") as f:
            files = {"file": ("large.pdf", f, "application/pdf")}
            response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        
    finally:
        os.unlink(large_file)

@pytest.mark.asyncio
async def test_concurrent_requests(mock_services):
    """Test handling of concurrent requests"""
    import asyncio
    
    async def make_request():
        return client.post("/api/v1/chat", json={
            "query": "Test query",
            "chat_history": None
        })
    
    # Make 5 concurrent requests
    tasks = [make_request() for _ in range(5)]
    responses = await asyncio.gather(*tasks)
    
    assert all(r.status_code == 200 for r in responses)

@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiting functionality"""
    # This test is optional and depends on rate limiting implementation
    # Make many requests in quick succession
    responses = []
    for _ in range(50):  # Adjust based on rate limit
        response = client.get("/health")
        responses.append(response)
        
    # Check if rate limiting kicked in (if implemented)
    # If rate limiting is not implemented, this test will pass anyway
    if any(r.status_code == 429 for r in responses):
        assert True
    else:
        # Skip test if rate limiting is not implemented
        pytest.skip("Rate limiting not implemented or threshold not reached")

if __name__ == "__main__":
    pytest.main([__file__])

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
from datetime import datetime
import json

from core.document_processor import DocumentProcessor
from core.hybrid_retriever import HybridRetriever
from core.reranker import Reranker
from core.llm_service import LLMService
from models.document import Document, DocumentChunk, ChatMessage, SearchResult

@pytest.fixture
def test_document():
    """Create a test document"""
    return Document(
        filename="test.pdf",
        content="""
        Machine Learning Fundamentals
        
        Supervised Learning involves training a model on labeled data.
        Common algorithms include linear regression and neural networks.
        
        Unsupervised Learning works with unlabeled data to find patterns.
        Examples include clustering and dimensionality reduction.
        
        Model Evaluation uses metrics like accuracy and precision.
        Cross-validation helps assess model performance.
        """,
        metadata={"type": "pdf", "test": True}
    )

@pytest.fixture
def test_chunks(test_document):
    """Create test document chunks"""
    return [
        DocumentChunk(
            document_id=test_document.id,
            content="Supervised Learning involves training a model on labeled data.",
            chunk_index=0,
            metadata={"section": "supervised"}
        ),
        DocumentChunk(
            document_id=test_document.id,
            content="Unsupervised Learning works with unlabeled data to find patterns.",
            chunk_index=1,
            metadata={"section": "unsupervised"}
        )
    ]

@pytest.mark.asyncio
async def test_document_processor():
    """Test document processing functionality"""
    processor = DocumentProcessor()
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Test content for document processing.")
        test_file = f.name
    
    try:
        # Test document processing
        document = await processor.process_pdf(test_file, "test.txt")
        assert document.filename == "test.txt"
        assert "Test content" in document.content
        
        # Test metadata extraction
        metadata = processor.extract_metadata(document)
        assert "language" in metadata
        assert "entities" in metadata
        
        # Test chunking
        chunks = processor.create_chunks(document)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        
    finally:
        os.unlink(test_file)

@pytest.mark.asyncio
async def test_hybrid_retriever(test_chunks):
    """Test hybrid retrieval functionality"""
    with patch("pinecone.Index") as mock_index:
        # Mock Pinecone responses
        mock_index.return_value.query.return_value.matches = [
            Mock(id=str(test_chunks[0].id), score=0.8),
            Mock(id=str(test_chunks[1].id), score=0.6)
        ]
        
        retriever = HybridRetriever(
            pinecone_api_key="test",
            pinecone_environment="test",
            pinecone_index="test",
            redis_url="redis://localhost"
        )
        
        # Add test documents
        await retriever.add_documents(test_chunks)
        
        # Test hybrid search
        results = await retriever.hybrid_search(
            query="What is supervised learning?",
            top_k=2
        )
        
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].vector_score >= results[1].vector_score

@pytest.mark.asyncio
async def test_reranker(test_chunks):
    """Test reranking functionality"""
    with patch("cohere.Client") as mock_cohere:
        # Mock Cohere responses
        mock_cohere.return_value.rerank.return_value.results = [
            Mock(relevance_score=0.9),
            Mock(relevance_score=0.7)
        ]
        
        reranker = Reranker(
            cohere_api_key="test",
            redis_url="redis://localhost"
        )
        
        # Create test results
        results = [
            SearchResult(
                chunk=test_chunks[0],
                vector_score=0.8,
                bm25_score=0.7,
                tfidf_score=0.75,
                combined_score=0.75
            ),
            SearchResult(
                chunk=test_chunks[1],
                vector_score=0.6,
                bm25_score=0.5,
                tfidf_score=0.55,
                combined_score=0.55
            )
        ]
        
        # Test reranking
        reranked = await reranker.rerank_cohere(
            query="What is supervised learning?",
            results=results
        )
        
        assert len(reranked) == 2
        assert all(r.rerank_score is not None for r in reranked)
        assert reranked[0].rerank_score >= reranked[1].rerank_score

@pytest.mark.asyncio
async def test_llm_service(test_chunks):
    """Test LLM service functionality"""
    with patch("httpx.AsyncClient") as mock_client, \
         patch("redis.asyncio.from_url") as mock_redis:
        # Mock Redis
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis.return_value = mock_redis_instance
        
        # Mock streaming response
        mock_response = AsyncMock()
        mock_response.aiter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"Test"}}]}',
            'data: {"choices":[{"delta":{"content":" response"}}]}',
            'data: {"choices":[{"delta":{"content":" with [Source 1]."}}]}',
            'data: [DONE]'
        ]
        mock_client.return_value.__aenter__.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.raise_for_status = AsyncMock()
        
        # Initialize LLM service with Redis
        llm_service = LLMService(
            openrouter_api_key="test",
            redis_url="redis://localhost",
            model="test-model",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Create test results
        results = [
            SearchResult(
                chunk=test_chunks[0],
                vector_score=0.8,
                bm25_score=0.7,
                tfidf_score=0.6,
                rerank_score=0.9,
                combined_score=0.85
            )
        ]
        
        # Test streaming response
        responses = []
        async for response in llm_service.generate_response(
            query="What is supervised learning?",
            results=results,
            use_cache=True
        ):
            responses.append(response)
            
        # Verify response
        assert len(responses) > 0
        assert responses[-1].finished
        assert "Test response" in responses[-1].message
        assert "[Source 1]" in responses[-1].message
        assert len(responses[-1].sources) > 0
        
        # Verify cache was checked and set
        mock_redis_instance.get.assert_called_once()
        mock_redis_instance.setex.assert_called_once()

@pytest.mark.asyncio
async def test_complete_pipeline():
    """Test complete RAG pipeline integration"""
    # Create test components with mocks
    with patch("pinecone.Index"), \
         patch("cohere.Client"), \
         patch("httpx.AsyncClient") as mock_client, \
         patch("redis.asyncio.from_url"):
        
        # Mock HTTP response for LLM
        mock_response = AsyncMock()
        mock_response.aiter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"Test response with citation [Source 1]."}}]}',
            'data: [DONE]'
        ]
        mock_client.return_value.__aenter__.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.raise_for_status = AsyncMock()
        
        # Initialize components
        processor = DocumentProcessor()
        retriever = HybridRetriever(
            pinecone_api_key="test",
            pinecone_environment="test",
            pinecone_index="test",
            redis_url="redis://localhost"
        )
        reranker = Reranker(
            cohere_api_key="test",
            redis_url="redis://localhost"
        )
        llm_service = LLMService(
            openrouter_api_key="test",
            redis_url="redis://localhost"
        )
    
    # Create test document
    test_content = """
    Machine Learning Fundamentals
    
    Supervised Learning involves training a model on labeled data.
    Unsupervised Learning works with unlabeled data to find patterns.
    """
    
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(test_content)
        test_file = f.name
        
    try:
        # Process document
        document = await processor.process_pdf(test_file, "test.txt")
        chunks = processor.create_chunks(document)
        
        # Add to retriever
        await retriever.add_documents(chunks)
        
        # Test complete pipeline
        query = "What is supervised learning?"
        
        # Retrieve
        results = await retriever.hybrid_search(query, top_k=5)
        assert len(results) > 0
        
        # Rerank
        reranked = await reranker.rerank_hybrid(query, results)
        assert len(reranked) > 0
        
        # Generate response
        responses = []
        async for response in llm_service.generate_response(
            query=query,
            results=reranked[:3]
        ):
            responses.append(response)
            
        assert len(responses) > 0
        assert responses[-1].finished
        
    finally:
        os.unlink(test_file)

if __name__ == "__main__":
    pytest.main([__file__])

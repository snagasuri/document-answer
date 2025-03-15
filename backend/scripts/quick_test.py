#!/usr/bin/env python
"""
Quick test script for the RAG pipeline.
Run this after setting up your environment variables to verify everything works.
"""

import asyncio
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import tempfile
import json

from core.document_processor import DocumentProcessor
from core.hybrid_retriever import HybridRetriever
from core.reranker import Reranker
from core.llm_service import LLMService
from models.document import Document, ChatMessage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rag_pipeline():
    """Test the complete RAG pipeline with a simple example"""
    logger.info("Starting RAG pipeline test...")
    
    # Check required environment variables
    required_vars = [
        "OPENROUTER_API_KEY",
        "COHERE_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please update your .env file with the required API keys")
        return
    
    # Initialize components
    logger.info("Initializing RAG components...")
    processor = DocumentProcessor()
    retriever = HybridRetriever(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
        pinecone_index=os.getenv("PINECONE_INDEX"),
        redis_url=os.getenv("REDIS_URL")
    )
    reranker = Reranker(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        redis_url=os.getenv("REDIS_URL")
    )
    llm_service = LLMService(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        redis_url=os.getenv("REDIS_URL"),
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000"))
    )
    
    # Create test document
    test_content = """
    Document Intelligence RAG System
    
    This system uses Retrieval-Augmented Generation (RAG) to provide accurate answers
    based on document knowledge. It combines vector search, BM25, and TF-IDF for
    hybrid retrieval, then uses reranking to improve result quality.
    
    The system includes document processing, chunking, embedding, and retrieval components.
    It uses OpenRouter for LLM access, Pinecone for vector storage, and Cohere for reranking.
    
    The RAG pipeline improves answer quality by providing relevant context to the LLM,
    reducing hallucinations and improving factual accuracy.
    """
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(test_content.encode())
        test_file = f.name
    
    try:
        # Process document
        logger.info("Processing test document...")
        document = await processor.process_pdf(
            file_path=test_file,
            filename="test_document.txt"
        )
        
        # Extract metadata
        metadata = processor.extract_metadata(document)
        document.metadata.update(metadata)
        logger.info(f"Extracted metadata: {json.dumps(metadata, indent=2)}")
        
        # Create chunks
        chunks = processor.create_chunks(document)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Add to retriever
        logger.info("Adding chunks to retriever...")
        await retriever.add_documents(chunks)
        
        # Test query
        query = "What components does the RAG system include?"
        logger.info(f"Testing query: '{query}'")
        
        # Retrieve results
        logger.info("Retrieving results...")
        results = await retriever.hybrid_search(query, top_k=5)
        logger.info(f"Retrieved {len(results)} results")
        
        # Rerank results
        logger.info("Reranking results...")
        reranked_results = await reranker.rerank_hybrid(query, results)
        
        # Generate response
        logger.info("Generating response...")
        response_text = ""
        async for response in llm_service.generate_response(
            query=query,
            results=reranked_results[:3],
            chat_history=[]
        ):
            if response.finished:
                response_text = response.message
                logger.info("\nFinal Response:")
                logger.info(response.message)
                logger.info("\nSources:")
                for source in response.sources:
                    logger.info(f"- {source.content[:100]}...")
        
        logger.info("\nRAG pipeline test completed successfully!")
        return response_text
        
    except Exception as e:
        logger.error(f"Error testing RAG pipeline: {str(e)}")
        raise
    finally:
        # Clean up
        os.unlink(test_file)

if __name__ == "__main__":
    asyncio.run(test_rag_pipeline())

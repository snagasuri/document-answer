#!/usr/bin/env python
"""
Test script for the HybridRetriever to verify the fixes for the namespace issue.
This script tests the HybridRetriever directly with a specific session ID.
"""

import os
import sys
import asyncio
import logging
import uuid
from dotenv import load_dotenv
import PyPDF2

# Add the parent directory to the path so we can import from the core module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hybrid_retriever import HybridRetriever
from models.document import DocumentChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def process_document(file_path: str, session_id: str):
    """Process a document and add it to Pinecone using HybridRetriever"""
    logger.info(f"Processing document: {file_path} for session: {session_id}")
    
    # Extract text from PDF
    logger.info(f"Extracting text from PDF")
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    # Create simple chunks (for testing purposes)
    logger.info(f"Creating chunks from document text")
    chunk_size = 1000
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i+chunk_size]
        chunks.append(chunk_text)
    
    # Create DocumentChunk objects
    logger.info(f"Creating {len(chunks)} DocumentChunk objects")
    doc_id = uuid.uuid4()
    doc_chunks = []
    for i, chunk_text in enumerate(chunks):
        chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            content=chunk_text,
            metadata={"source": os.path.basename(file_path), "chunk_index": i},
            chunk_index=i
        )
        doc_chunks.append(chunk)
    
    # Initialize HybridRetriever with the specific session ID
    logger.info(f"Initializing HybridRetriever with session ID: {session_id}")
    retriever = HybridRetriever(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
        pinecone_index=os.getenv("PINECONE_INDEX"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        session_id=session_id
    )
    
    # Add chunks to Pinecone
    logger.info(f"Adding chunks to Pinecone with namespace: session_{session_id}")
    await retriever.add_documents(doc_chunks)
    
    # Test query
    logger.info(f"Testing query with session ID: {session_id}")
    test_query = "test query"
    results = await retriever.hybrid_search(test_query, top_k=5)
    
    if results:
        logger.info(f"Query successful! Found {len(results)} results.")
        for i, result in enumerate(results[:3]):  # Show first 3 results
            logger.info(f"Result {i+1}: ID={result.chunk.id}, Score={result.combined_score:.4f}")
            content_preview = result.chunk.content[:100]
            logger.info(f"Content preview: {content_preview}...")
    else:
        logger.warning(f"Query returned no results for session ID: {session_id}")
    
    return len(results) > 0

async def main():
    """Main function to run the test"""
    # Check for environment variables
    required_env_vars = [
        "PINECONE_API_KEY", 
        "PINECONE_ENVIRONMENT", 
        "PINECONE_INDEX"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return
    
    # Generate a test session ID
    test_session_id = str(uuid.uuid4())
    logger.info(f"Generated test session ID: {test_session_id}")
    
    # Process document
    document_path = "/Users/sriram/Documents/document-answer/oleve.pdf"
    if not os.path.exists(document_path):
        logger.error(f"Document not found: {document_path}")
        return
    
    success = await process_document(document_path, test_session_id)
    
    if success:
        logger.info("Test completed successfully! The HybridRetriever is working correctly.")
        logger.info("This confirms that the namespace issue has been fixed.")
    else:
        logger.error("Test failed! The HybridRetriever is not working correctly.")
        logger.error("The namespace issue may still exist.")

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python
"""
Test script for the Cohere API integration.
This is a simplified version that tests the reranking service.
"""

import os
import logging
from dotenv import load_dotenv
import httpx
import json
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCohereClient:
    """A simplified version of the Cohere client for testing"""
    
    def __init__(self, api_key: str):
        """Initialize the Cohere client"""
        self.api_key = api_key
        self.base_url = "https://api.cohere.ai/v1"
        
    async def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_n: int = 3,
        model: str = "rerank-english-v2.0"
    ) -> Dict[str, Any]:
        """Rerank documents"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "model": model
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/rerank",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            raise

async def main():
    """Run the test"""
    # Load environment variables
    load_dotenv()
    
    # Check if Cohere API key is set
    api_key = os.getenv("COHERE_API_KEY")
    
    if not api_key:
        logger.error("COHERE_API_KEY environment variable is not set")
        return
        
    # Create Cohere client
    cohere_client = SimpleCohereClient(api_key=api_key)
    
    # Test query and documents
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital and largest city of Germany.",
        "London is the capital and largest city of England and the United Kingdom.",
        "France is a country in Western Europe.",
        "The Eiffel Tower is located in Paris, France."
    ]
    
    # Rerank documents
    logger.info(f"Reranking documents for query: {query}")
    try:
        results = await cohere_client.rerank(query, documents)
        logger.info(f"Reranking results: {json.dumps(results, indent=2)}")
        
        # Print reranked documents
        logger.info("Reranked documents:")
        for i, result in enumerate(results.get("results", [])):
            index = result.get("index")
            relevance_score = result.get("relevance_score")
            document = documents[index]
            logger.info(f"{i+1}. Score: {relevance_score:.4f} - {document}")
        
        logger.info("Cohere reranking test completed successfully!")
    except Exception as e:
        logger.error(f"Cohere reranking test failed: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

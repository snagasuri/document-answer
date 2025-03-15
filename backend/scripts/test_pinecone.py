#!/usr/bin/env python
"""
Test script for the Pinecone API integration.
This is a simplified version that tests the vector database connection.
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

class SimplePineconeClient:
    """A simplified version of the Pinecone client for testing"""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        """Initialize the Pinecone client"""
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.base_url = f"https://{index_name}.svc.{environment}.pinecone.io"
        
    async def describe_index(self) -> Dict[str, Any]:
        """Describe the index"""
        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/describe_index_stats",
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error describing index: {str(e)}")
            raise
            
    async def query(self, vector: List[float], top_k: int = 5) -> Dict[str, Any]:
        """Query the index"""
        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/query",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error querying index: {str(e)}")
            raise

async def main():
    """Run the test"""
    # Load environment variables
    load_dotenv()
    
    # Check if Pinecone API key is set
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX")
    
    if not api_key or not environment or not index_name:
        logger.error("Pinecone environment variables are not set")
        return
        
    # Create Pinecone client
    pinecone_client = SimplePineconeClient(
        api_key=api_key,
        environment=environment,
        index_name=index_name
    )
    
    # Describe index
    logger.info(f"Describing index: {index_name}")
    try:
        index_stats = await pinecone_client.describe_index()
        logger.info(f"Index stats: {json.dumps(index_stats, indent=2)}")
        
        # Get vector count
        vector_count = index_stats.get("total_vector_count", 0)
        logger.info(f"Vector count: {vector_count}")
        
        # Get dimension
        dimension = index_stats.get("dimension", 0)
        logger.info(f"Dimension: {dimension}")
        
        logger.info("Pinecone connection test completed successfully!")
    except Exception as e:
        logger.error(f"Pinecone connection test failed: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

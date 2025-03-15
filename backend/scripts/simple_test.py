#!/usr/bin/env python
"""
Simple test script for the RAG pipeline.
This is a minimal version that only tests the environment setup.
"""

import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    """Test the environment setup"""
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = [
        "OPENROUTER_API_KEY",
        "COHERE_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX"
    ]
    
    # Check which variables are set
    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✓ {var} is set")
        else:
            logger.warning(f"✗ {var} is not set")
    
    # Print Pinecone environment details
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index = os.getenv("PINECONE_INDEX")
    
    logger.info(f"Pinecone Environment: {pinecone_env}")
    logger.info(f"Pinecone Index: {pinecone_index}")
    
    logger.info("Environment test completed")

if __name__ == "__main__":
    logger.info("Starting simple environment test...")
    test_environment()
    logger.info("Test completed successfully!")

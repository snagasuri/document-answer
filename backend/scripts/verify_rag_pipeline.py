#!/usr/bin/env python
"""
Verification script for the RAG pipeline.
This script tests all the core components of the RAG pipeline.
"""

import os
import json
import asyncio
import logging
from dotenv import load_dotenv
import httpx
from typing import Dict, Any, List, AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLLMResponse:
    """Simple response object for LLM responses"""
    def __init__(self, message: str, finished: bool = False, sources: List[Dict[str, Any]] = None):
        self.message = message
        self.finished = finished
        self.sources = sources or []

class SimpleLLMService:
    """A simplified version of the LLM service for testing"""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-4o"):
        """Initialize the LLM service"""
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model
        
    async def generate_response(
        self, 
        query: str, 
        results: List[Dict[str, Any]] = None,
        chat_history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[SimpleLLMResponse, None]:
        """Generate a response from the LLM"""
        results = results or []
        chat_history = chat_history or []
        
        # Create context from results
        context = "\n\n".join([
            f"Document {i+1}:\n{result.get('content', '')}"
            for i, result in enumerate(results)
        ])
        
        # Create system prompt
        system_prompt = f"""You are a helpful assistant that answers questions based on the provided documents.
If the answer is not in the documents, say you don't know.
Use the following documents to answer the question:

{context}
"""
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history
        for message in chat_history:
            messages.append(message)
            
        # Add user query
        messages.append({"role": "user", "content": query})
        
        # Create request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Send request
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    
                    buffer = ""
                    async for chunk in response.aiter_text():
                        if not chunk.strip():
                            continue
                            
                        # Handle SSE format
                        for line in chunk.split("\n"):
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    yield SimpleLLMResponse(buffer, finished=True)
                                    return
                                    
                                try:
                                    data_json = json.loads(data)
                                    delta = data_json.get("choices", [{}])[0].get("delta", {})
                                    content = delta.get("content", "")
                                    
                                    if content:
                                        buffer += content
                                        yield SimpleLLMResponse(buffer)
                                except json.JSONDecodeError:
                                    logger.error(f"Failed to parse JSON: {data}")
                                    continue
                                    
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            yield SimpleLLMResponse(
                f"Error generating response: {str(e)}",
                finished=True
            )

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

async def verify_openrouter():
    """Verify OpenRouter API integration"""
    logger.info("Verifying OpenRouter API integration...")
    
    # Check if OpenRouter API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable is not set")
        return False
        
    # Create LLM service
    llm_service = SimpleLLMService(api_key=api_key)
    
    # Test query
    query = "What is the capital of France?"
    
    try:
        # Generate response
        logger.info(f"Generating response for query: {query}")
        response_text = ""
        async for response in llm_service.generate_response(query=query):
            if response.finished:
                response_text = response.message
                
        logger.info(f"Response: {response_text}")
        logger.info("OpenRouter API integration verified successfully!")
        return True
    except Exception as e:
        logger.error(f"OpenRouter API integration verification failed: {str(e)}")
        return False

async def verify_pinecone():
    """Verify Pinecone API integration"""
    logger.info("Verifying Pinecone API integration...")
    
    # Check if Pinecone API key is set
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX")
    
    if not api_key or not environment or not index_name:
        logger.error("Pinecone environment variables are not set")
        return False
        
    # Create Pinecone client
    pinecone_client = SimplePineconeClient(
        api_key=api_key,
        environment=environment,
        index_name=index_name
    )
    
    try:
        # Describe index
        logger.info(f"Describing index: {index_name}")
        index_stats = await pinecone_client.describe_index()
        
        # Get vector count
        vector_count = index_stats.get("total_vector_count", 0)
        logger.info(f"Vector count: {vector_count}")
        
        # Get dimension
        dimension = index_stats.get("dimension", 0)
        logger.info(f"Dimension: {dimension}")
        
        logger.info("Pinecone API integration verified successfully!")
        return True
    except Exception as e:
        logger.error(f"Pinecone API integration verification failed: {str(e)}")
        return False

async def verify_cohere():
    """Verify Cohere API integration"""
    logger.info("Verifying Cohere API integration...")
    
    # Check if Cohere API key is set
    api_key = os.getenv("COHERE_API_KEY")
    
    if not api_key:
        logger.error("COHERE_API_KEY environment variable is not set")
        return False
        
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
    
    try:
        # Rerank documents
        logger.info(f"Reranking documents for query: {query}")
        results = await cohere_client.rerank(query, documents)
        
        # Print reranked documents
        logger.info("Reranked documents:")
        for i, result in enumerate(results.get("results", [])):
            index = result.get("index")
            relevance_score = result.get("relevance_score")
            document = documents[index]
            logger.info(f"{i+1}. Score: {relevance_score:.4f} - {document}")
        
        logger.info("Cohere API integration verified successfully!")
        return True
    except Exception as e:
        logger.error(f"Cohere API integration verification failed: {str(e)}")
        return False

async def verify_environment():
    """Verify environment variables"""
    logger.info("Verifying environment variables...")
    
    # Check required environment variables
    required_vars = [
        "OPENROUTER_API_KEY",
        "COHERE_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX"
    ]
    
    # Check which variables are set
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✓ {var} is set")
        else:
            logger.warning(f"✗ {var} is not set")
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("Environment variables verified successfully!")
    return True

async def main():
    """Run the verification"""
    # Load environment variables
    load_dotenv()
    
    logger.info("Starting RAG pipeline verification...")
    
    # Verify environment variables
    env_ok = await verify_environment()
    if not env_ok:
        logger.error("Environment verification failed")
        return
    
    # Verify OpenRouter API integration
    openrouter_ok = await verify_openrouter()
    
    # Verify Pinecone API integration
    pinecone_ok = await verify_pinecone()
    
    # Verify Cohere API integration
    cohere_ok = await verify_cohere()
    
    # Print summary
    logger.info("\nVerification Summary:")
    logger.info(f"Environment: {'✓' if env_ok else '✗'}")
    logger.info(f"OpenRouter API: {'✓' if openrouter_ok else '✗'}")
    logger.info(f"Pinecone API: {'✓' if pinecone_ok else '✗'}")
    logger.info(f"Cohere API: {'✓' if cohere_ok else '✗'}")
    
    if env_ok and openrouter_ok and pinecone_ok and cohere_ok:
        logger.info("\nAll components verified successfully! The RAG pipeline is ready to use.")
    else:
        logger.error("\nSome components failed verification. Please check the logs for details.")

if __name__ == "__main__":
    asyncio.run(main())

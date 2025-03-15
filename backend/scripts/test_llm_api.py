#!/usr/bin/env python
"""
Test script for the OpenRouter API integration.
This is a simplified version that tests the LLM service without requiring all dependencies.
"""

import os
import json
import asyncio
import logging
from dotenv import load_dotenv
import httpx
from typing import Dict, Any, List, Optional, AsyncGenerator

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

async def main():
    """Run the test"""
    # Load environment variables
    load_dotenv()
    
    # Check if OpenRouter API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable is not set")
        return
        
    # Create LLM service
    llm_service = SimpleLLMService(api_key=api_key)
    
    # Test query
    query = "What is the capital of France?"
    
    # Generate response
    logger.info(f"Generating response for query: {query}")
    async for response in llm_service.generate_response(query=query):
        if response.finished:
            logger.info(f"\nFinal response: {response.message}")
        else:
            print(".", end="", flush=True)
            
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())

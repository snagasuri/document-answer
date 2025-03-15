"""
Token counter service for tracking token usage in LLM requests.
"""

import tiktoken
from typing import List, Dict, Optional, Union
import logging

from core.config import MODEL_CONTEXT_SIZES

logger = logging.getLogger(__name__)

class TokenCounterService:
    """Service for counting tokens in messages"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """Initialize token counter with model"""
        self.model_name = model_name
        # Map model names to encoding
        model_map = {
            "gpt-4o": "cl100k_base",
            "gpt-4": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "claude-3-opus": "cl100k_base",  # Approximation
            "claude-3-sonnet": "cl100k_base",  # Approximation
            "claude-3-haiku": "cl100k_base",  # Approximation
        }
        
        # Get encoding name based on model
        encoding_name = model_map.get(self.model_name.split("/")[-1], "cl100k_base")
        
        # Get encoding
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            logger.info(f"Initialized token counter with encoding: {encoding_name}")
        except KeyError:
            # Fallback to cl100k_base if encoding not found
            logger.warning(f"Encoding {encoding_name} not found, falling back to cl100k_base")
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
    def count_tokens(self, text: Union[str, List[int]]) -> int:
        """Count tokens in text"""
        if not text:
            return 0
            
        try:
            # If text is already tokenized
            if isinstance(text, list):
                return len(text)
                
            # Encode the text and count tokens
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback to character-based approximation
            return len(text) // 4  # Rough approximation: ~4 chars per token
        
    def count_message_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in a list of messages"""
        if not messages:
            return 0
            
        try:
            # Count tokens in each message
            token_count = 0
            for message in messages:
                # Count tokens in content
                content = message.get("content", "")
                token_count += self.count_tokens(content)
                
                # Add tokens for message format (role, etc.)
                # This is an approximation based on OpenAI's tokenization
                token_count += 4  # ~4 tokens for message format
                
                # Add tokens for role
                role = message.get("role", "")
                token_count += self.count_tokens(role)
            
            # Add tokens for overall formatting
            token_count += 2  # ~2 tokens for overall formatting
            
            return token_count
        except Exception as e:
            logger.error(f"Error counting message tokens: {e}")
            # Fallback to a simple approximation
            return sum(len(m.get("content", "")) // 4 for m in messages)
        
    def get_model_context_size(self) -> int:
        """Get the context size for the current model"""
        # Extract model name without provider prefix
        model_name = self.model_name.split("/")[-1]
        
        # Return context size or default
        return MODEL_CONTEXT_SIZES.get(model_name, 8192)
        
    def get_token_usage_info(
        self, 
        prompt_tokens: int, 
        completion_tokens: int
    ) -> Dict[str, Union[int, float, Dict]]:
        """Get token usage information including pricing estimates"""
        # Define pricing per 1K tokens (approximate)
        pricing = {
            "gpt-4o": {"prompt": 0.01, "completion": 0.03},
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
            "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
            "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
            "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
        }
        
        # Extract model name without provider prefix
        model_name = self.model_name.split("/")[-1]
        
        # Get pricing for the model
        model_pricing = pricing.get(model_name, {"prompt": 0.01, "completion": 0.03})
        
        # Calculate cost
        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        total_cost = prompt_cost + completion_cost
        
        # Return token usage info
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": {
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "total_cost": total_cost,
                "currency": "USD"
            }
        }

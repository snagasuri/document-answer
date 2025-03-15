"""
Enhanced LLM service with token counting and context window management.
"""

import json
import logging
import asyncio
import uuid
from typing import List, Dict, Optional, AsyncGenerator, Any, Tuple
from datetime import datetime

from models.document import SearchResult, ChatMessage, ChatResponse, DocumentChunk
from core.llm_service import LLMService
from core.token_counter import TokenCounterService

logger = logging.getLogger(__name__)

class EnhancedLLMService(LLMService):
    """Enhanced LLM service with token counting and context management"""
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced LLM service"""
        super().__init__(*args, **kwargs)
        # Extract model name without provider prefix
        model_name = self.model.split("/")[-1] if "/" in self.model else self.model
        self.token_counter = TokenCounterService(model_name)
        self.max_context_tokens = self.token_counter.get_model_context_size()
        logger.info(f"Initialized EnhancedLLMService with model {self.model}, max context: {self.max_context_tokens} tokens")
        
    def _manage_context_window(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None
    ) -> List[Dict]:
        """Manage context window by truncating history if needed"""
        max_tokens = max_tokens or self.max_context_tokens
        
        # Always keep system message
        system_message = next((m for m in messages if m["role"] == "system"), None)
        other_messages = [m for m in messages if m["role"] != "system"]
        
        # Count tokens in all messages
        total_tokens = self.token_counter.count_message_tokens(messages)
        
        # If we're within limits, return all messages
        if total_tokens <= max_tokens * 0.8:  # Keep 20% buffer
            return messages
            
        logger.info(f"Context window management: total tokens {total_tokens} exceeds 80% of max {max_tokens}, truncating history")
            
        # We need to truncate history
        # Strategy: Keep the most recent messages, and remove older ones
        # Always keep the system message and the last user message
        
        # Sort by recency (assuming messages are in chronological order)
        # Keep the last N messages that fit within our token budget
        truncated_messages = [system_message] if system_message else []
        remaining_budget = max_tokens * 0.8
        
        if system_message:
            system_tokens = self.token_counter.count_tokens(system_message["content"]) + 4
            remaining_budget -= system_tokens
        
        # Always include the most recent user message
        last_user_message = next((m for m in reversed(other_messages) if m["role"] == "user"), None)
        if last_user_message:
            last_user_tokens = self.token_counter.count_tokens(last_user_message["content"]) + 4
            remaining_budget -= last_user_tokens
            
        # Add as many recent messages as will fit in the budget
        recent_messages = []
        for message in reversed(other_messages):
            if message == last_user_message:
                continue  # Skip, we'll add it later
                
            message_tokens = self.token_counter.count_tokens(message["content"]) + 4
            if remaining_budget - message_tokens >= 0:
                recent_messages.insert(0, message)
                remaining_budget -= message_tokens
            else:
                break
                
        # Add the last user message at the end
        if last_user_message:
            recent_messages.append(last_user_message)
            
        # Combine all messages
        if system_message:
            final_messages = [system_message] + recent_messages
        else:
            final_messages = recent_messages
            
        # Log truncation info
        original_count = len(messages)
        truncated_count = len(final_messages)
        logger.info(f"Context window management: truncated from {original_count} to {truncated_count} messages")
        
        return final_messages
            
    async def generate_response_with_token_count(
        self,
        query: str,
        results: List[SearchResult],
        chat_history: Optional[List] = None,
        use_cache: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response with token counting"""
        try:
            # Format messages for the API
            logger.info("Formatting messages for API")
            messages = self._format_messages(query, results, chat_history)
            
            # Apply context window management
            logger.info("Applying context window management")
            managed_messages = self._manage_context_window(messages)
            
            # Count input tokens
            logger.info("Counting input tokens")
            input_tokens = self.token_counter.count_message_tokens(managed_messages)
            logger.info(f"Input tokens: {input_tokens}")
            
            # Check cache if enabled
            if use_cache and self.redis:
                cache_key = self._generate_cache_key(query, results)
                cached_response = await self._check_cache(cache_key)
                
                if cached_response:
                    try:
                        # Parse cached response
                        cached_data = json.loads(cached_response)
                        message = cached_data.get("message", "")
                        source_ids = cached_data.get("source_ids", [])
                        
                        # Get source chunks, including placeholders for missing sources
                        sources = []
                        for i in source_ids:
                            if 0 < i <= len(results):
                                # Source exists in results
                                sources.append(results[i-1].chunk)
                            else:
                                # Source doesn't exist, create a placeholder
                                logger.warning(f"Citation [Source {i}] refers to a non-existent source (only {len(results)} sources available)")
                                placeholder = DocumentChunk(
                                    id=f"placeholder-source-{i}",
                                    content=f"Source {i} content not available (citation refers to a source beyond the {len(results)} sources provided to the LLM)",
                                    metadata={"placeholder": True, "missing_source_index": i}
                                )
                                sources.append(placeholder)
                        
                        # Count completion tokens
                        completion_tokens = self.token_counter.count_tokens(message)
                        total_tokens = input_tokens + completion_tokens
                        
                        # Create token usage info
                        token_usage = {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                        
                        # Add pricing info if available
                        try:
                            token_usage_with_pricing = self.token_counter.get_token_usage_info(
                                prompt_tokens=input_tokens,
                                completion_tokens=completion_tokens
                            )
                            token_usage = token_usage_with_pricing
                        except Exception as e:
                            logger.warning(f"Error getting token pricing info: {e}")
                        
                        # Return cached response with token info
                        yield {
                            "message": message,
                            "sources": sources,
                            "finished": True,
                            "token_usage": token_usage,
                            "context_window": {
                                "used_tokens": total_tokens,
                                "max_tokens": self.max_context_tokens,
                                "remaining_tokens": self.max_context_tokens - total_tokens
                            }
                        }
                        return
                    except Exception as e:
                        logger.warning(f"Error processing cached response: {e}")
                        # Continue with normal processing if cache processing fails
            
            # Stream response
            buffer = ""
            sources = []
            
            try:
                async for response in self._stream_response(managed_messages, results):
                    if response.finished:
                        # Final response
                        buffer = response.message
                        sources = response.sources or []
                        
                        # Count tokens in the full response
                        completion_tokens = self.token_counter.count_tokens(buffer)
                        total_tokens = input_tokens + completion_tokens
                        
                        # Create token usage info
                        token_usage = {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                        
                        # Add pricing info if available
                        try:
                            token_usage_with_pricing = self.token_counter.get_token_usage_info(
                                prompt_tokens=input_tokens,
                                completion_tokens=completion_tokens
                            )
                            token_usage = token_usage_with_pricing
                        except Exception as e:
                            logger.warning(f"Error getting token pricing info: {e}")
                        
                        # Cache the response if enabled
                        if use_cache and self.redis:
                            try:
                                # Extract citations
                                cited_sources = self._extract_citations(buffer)
                                
                                cache_data = {
                                    "message": buffer,
                                    "source_ids": cited_sources,
                                    "token_usage": token_usage,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                                await self._save_to_cache(
                                    self._generate_cache_key(query, results),
                                    json.dumps(cache_data)
                                )
                            except Exception as e:
                                logger.warning(f"Error caching response: {e}")
                        
                        # Return final response with token info
                        # Convert response to dict safely
                        response_dict = {}
                        try:
                            # Log the response object for debugging
                            logger.info(f"Response object type: {type(response)}")
                            logger.info(f"Response object attributes: {dir(response)}")
                            logger.info(f"Response message: {response.message if hasattr(response, 'message') else 'No message attribute'}")
                            
                            # Try model_dump() first (Pydantic v2)
                            if hasattr(response, 'model_dump'):
                                response_dict = response.model_dump()
                                logger.info("Used model_dump() for response conversion")
                            # Fall back to dict() (Pydantic v1)
                            elif hasattr(response, 'dict'):
                                response_dict = response.dict()
                                logger.info("Used dict() for response conversion")
                            # Manual conversion if needed
                            else:
                                response_dict = {
                                    "message": response.message if hasattr(response, 'message') else "",
                                    "sources": response.sources if hasattr(response, 'sources') else [],
                                    "finished": response.finished if hasattr(response, 'finished') else True
                                }
                                logger.info("Used manual conversion for response")
                        except Exception as e:
                            logger.warning(f"Error converting response to dict: {e}")
                            logger.info(f"Falling back to buffer-based response with buffer: {buffer[:100]}...")
                            response_dict = {
                                "message": buffer if buffer is not None else "",
                                "sources": sources if sources is not None else [],
                                "finished": True
                            }
                            
                        # Add token usage and context window info
                        response_dict.update({
                            "token_usage": token_usage,
                            "context_window": {
                                "used_tokens": total_tokens,
                                "max_tokens": self.max_context_tokens,
                                "remaining_tokens": self.max_context_tokens - total_tokens
                            }
                        })
                        
                        yield response_dict
                    else:
                        # Streaming response
                        buffer = response.message
                        
                        # Count tokens in the current response
                        current_completion_tokens = self.token_counter.count_tokens(buffer)
                        current_total_tokens = input_tokens + current_completion_tokens
                        
                        # Convert response to dict safely
                        response_dict = {}
                        try:
                            # Try model_dump() first (Pydantic v2)
                            if hasattr(response, 'model_dump'):
                                response_dict = response.model_dump()
                                logger.info(f"Used model_dump() for streaming response conversion")
                            # Fall back to dict() (Pydantic v1)
                            elif hasattr(response, 'dict'):
                                response_dict = response.dict()
                                logger.info(f"Used dict() for streaming response conversion")
                            # Manual conversion if needed
                            else:
                                response_dict = {
                                    "message": response.message if hasattr(response, 'message') else "",
                                    "sources": response.sources if hasattr(response, 'sources') else [],
                                    "finished": response.finished if hasattr(response, 'finished') else False
                                }
                                logger.info(f"Used manual conversion for streaming response")
                                
                            # Log sources information for debugging
                            sources = response_dict.get("sources", [])
                            logger.info(f"Streaming response has {len(sources)} sources")
                            if sources:
                                # Convert UUID objects to strings for JSON serialization
                                for source in sources:
                                    if hasattr(source, 'id') and isinstance(source.id, uuid.UUID):
                                        source.id = str(source.id)
                                    if hasattr(source, 'document_id') and isinstance(source.document_id, uuid.UUID):
                                        source.document_id = str(source.document_id)
                                    if hasattr(source, 'metadata') and source.metadata:
                                        for key, value in source.metadata.items():
                                            if isinstance(value, uuid.UUID):
                                                source.metadata[key] = str(value)
                                
                                logger.info(f"First source preview: {str(sources[0])[:100]}...")
                        except Exception as e:
                            logger.warning(f"Error converting response to dict: {e}")
                            response_dict = {
                                "message": buffer if buffer is not None else "",
                                "sources": getattr(response, 'sources', []) or [],
                                "finished": False
                            }
                            logger.info(f"Used fallback conversion for streaming response")
                        
                        # Add token usage and context window info
                        response_dict.update({
                            "token_usage": {
                                "prompt_tokens": input_tokens,
                                "completion_tokens": current_completion_tokens,
                                "total_tokens": current_total_tokens
                            },
                            "context_window": {
                                "used_tokens": current_total_tokens,
                                "max_tokens": self.max_context_tokens,
                                "remaining_tokens": self.max_context_tokens - current_total_tokens
                            }
                        })
                        
                        yield response_dict
            except Exception as e:
                logger.error(f"Error streaming response: {e}")
                # Continue to error handling
                    
        except Exception as e:
            logger.error(f"Error generating response with token count: {e}")
            # Return error response with empty token usage
            yield {
                "message": f"Error: Failed to generate response - {str(e)}",
                "sources": [],
                "finished": True,
                "token_usage": {
                    "prompt_tokens": input_tokens if 'input_tokens' in locals() else 0,
                    "completion_tokens": 0,
                    "total_tokens": input_tokens if 'input_tokens' in locals() else 0
                },
                "context_window": {
                    "used_tokens": input_tokens if 'input_tokens' in locals() else 0,
                    "max_tokens": self.max_context_tokens,
                    "remaining_tokens": self.max_context_tokens - (input_tokens if 'input_tokens' in locals() else 0)
                }
            }

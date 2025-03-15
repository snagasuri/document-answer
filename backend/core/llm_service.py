import asyncio
import logging
import json
import re
import hashlib
import uuid
from typing import List, Dict, Optional, AsyncGenerator, Tuple
import httpx
import redis.asyncio as redis
from datetime import datetime

from models.document import SearchResult, ChatMessage, ChatResponse, DocumentChunk
from core.config import LLM_CONFIG, settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(
        self,
        openrouter_api_key: str,
        redis_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        cache_ttl: int = 3600,
    ):
        """Initialize LLM service with OpenRouter"""
        self.api_key = openrouter_api_key
        self.model = model or LLM_CONFIG.get("model", "openai/gpt-4o")
        self.temperature = temperature or LLM_CONFIG.get("temperature", 0.7)
        self.max_tokens = max_tokens or LLM_CONFIG.get("max_tokens", 1000)
        self.top_p = top_p or LLM_CONFIG.get("top_p", 0.9)
        self.api_base = "https://openrouter.ai/api/v1"
        
        # Initialize Redis for caching if URL provided
        self.redis = None
        self.cache_ttl = cache_ttl
        if redis_url:
            self.redis = redis.from_url(redis_url)
            
        # Citation tracking
        self.citation_pattern = re.compile(r'\[Source (\d+)\]')
        
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the conversation"""
        return """You are a helpful AI assistant with access to document knowledge. 
        
        IMPORTANT: Thoroughly read ALL provided context before answering. Each source is clearly marked with:
        ===== SOURCE X BEGINS =====
        [content]
        ===== SOURCE X ENDS =====
        
        CRITICAL CITATION RULES:
        1. You MUST use the exact source numbers from the section markers
        2. NEVER cite a source number that doesn't exist in the context
        3. Only use source numbers that appear in the markers (e.g. if there are 2 sources, only use [Source 1] and [Source 2])
        4. Every factual statement must include a citation
        
        Example citation: "According to the document, machine learning models can overfit when they learn noise in the training data [Source 2]."
        
        For multiple sources: "This approach improved efficiency [Source 1] while maintaining accuracy [Source 2]"
        
        Never state that information is not available if it appears in any context source.
        If the context does contain relevant information, summarize it comprehensively with appropriate citations.
        
        If you're unsure or the context truly doesn't contain the answer, say so clearly.
        Keep responses clear and well-structured."""
        
    def _build_context_prompt(self, results: List[SearchResult]) -> str:
        """Build context prompt from search results"""
        try:
            # First, group chunks by document ID
            doc_groups = {}
            for idx, r in enumerate(results):
                doc_id = str(r.chunk.document_id)  # Convert UUID to string for dict key
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = {
                        'index': len(doc_groups) + 1,  # Source number (1-based)
                        'chunks': []
                    }
                doc_groups[doc_id]['chunks'].append(r)
            
            # Log document grouping for debugging
            logger.info(f"Grouped {len(results)} chunks into {len(doc_groups)} documents")
            for doc_id, group in doc_groups.items():
                logger.info(f"Document {doc_id}: Source {group['index']} with {len(group['chunks'])} chunks")
            
            context_parts = []
            for i, result in enumerate(results, 1):
                logger.info(f"Processing result {i}")
                
                # Format scores for display
                try:
                    logger.info(f"Formatting scores for result {i}")
                    score_info = ""
                    
                    if result.vector_score is not None:
                        score_info += f"Vector: {result.vector_score:.2f}, "
                    else:
                        score_info += "Vector: N/A, "
                        
                    if result.bm25_score is not None:
                        score_info += f"BM25: {result.bm25_score:.2f}, "
                    else:
                        score_info += "BM25: N/A, "
                        
                    if result.tfidf_score is not None:
                        score_info += f"TF-IDF: {result.tfidf_score:.2f}"
                    else:
                        score_info += "TF-IDF: N/A"
                        
                    if result.rerank_score is not None:
                        score_info += f", Rerank: {result.rerank_score:.2f}"
                except Exception as e:
                    logger.error(f"Error formatting scores: {e}")
                    score_info = "Scores: N/A"
                
                # Add document metadata if available
                try:
                    logger.info(f"Adding metadata for result {i}")
                    metadata = ""
                    if result.chunk.metadata:
                        if "document_id" in result.chunk.metadata:
                            metadata += f"Document ID: {result.chunk.metadata.get('document_id')}, "
                        if "chunk_index" in result.chunk.metadata:
                            metadata += f"Chunk: {result.chunk.metadata.get('chunk_index')}, "
                        if "start_char" in result.chunk.metadata and "end_char" in result.chunk.metadata:
                            metadata += f"Position: {result.chunk.metadata.get('start_char')}-{result.chunk.metadata.get('end_char')}"
                except Exception as e:
                    logger.error(f"Error adding metadata: {e}")
                    metadata = ""

                # Format the context with clear section markers to help LLM parse content
                try:
                    logger.info(f"Formatting context for result {i}")
                    
                    # Get source index from document groups
                    doc_id = str(result.chunk.document_id)
                    source_index = doc_groups[doc_id]['index']
                    
                    # Set citation_index based on source document
                    result.chunk.metadata['citation_index'] = source_index
                    result.chunk.citation_index = source_index
                    
                    # Only show source marker for first chunk of each document
                    if result == doc_groups[doc_id]['chunks'][0]:
                        context = f"\n===== SOURCE {source_index} BEGINS =====\n"
                        context += f"RELEVANCE: {score_info}\n"
                        if metadata:
                            context += f"METADATA: {metadata}\n"
                        context += f"CONTENT:\n"
                    
                    # Add chunk content
                    context += f"{result.chunk.content}\n"
                    
                    # Close source marker for last chunk of document
                    if result == doc_groups[doc_id]['chunks'][-1]:
                        context += f"===== SOURCE {source_index} ENDS =====\n"
                    
                    context_parts.append(context)
                except Exception as e:
                    logger.error(f"Error formatting context: {e}")
                    context_parts.append(f"\n[Source {i}] (Error formatting context)\n")
                
            logger.info(f"Joining {len(context_parts)} context parts")
            return "\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error building context prompt: {e}")
            return "Error building context prompt"
        
    def _format_messages(
        self,
        query: str,
        results: List[SearchResult],
        chat_history: Optional[List] = None
    ) -> List[Dict]:
        """Format messages for the LLM API"""
        try:
            logger.info("Building system prompt")
            system_prompt = self._build_system_prompt()
            
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            # Add chat history if provided
            logger.info(f"Adding chat history: {chat_history is not None}")
            if chat_history:
                for msg in chat_history[-5:]:  # Include last 5 messages for context
                    # Handle both ChatMessage objects and dictionaries
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        # ChatMessage object
                        messages.append({"role": msg.role, "content": msg.content})
                    elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        # Dictionary with role and content keys
                        messages.append({"role": msg['role'], "content": msg['content']})
                    else:
                        logger.warning(f"Skipping invalid chat message format: {msg}")
            
            # Add context and query
            logger.info(f"Building context prompt with {len(results)} results")
            context = self._build_context_prompt(results)
            
            # Add instructions for citation
            citation_instructions = """
            IMPORTANT: When citing sources, you must use the exact source number from the section markers.
            For example, if you use information from the section marked "SOURCE 2", cite it as [Source 2].
            Do not make up source numbers or use numbers that don't match the source markers.
            Every piece of information from the context must have a citation.
            If you're unsure about something, indicate that clearly rather than guessing.
            """
            
            logger.info(f"Adding user message with query: {query[:50]}...")
            messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\n{citation_instructions}\n\nQuestion: {query}"
            })
            
            logger.info(f"Formatted {len(messages)} messages")
            return messages
        except Exception as e:
            logger.error(f"Error formatting messages: {e}")
            raise
        
    async def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if response is cached"""
        if not self.redis:
            return None
            
        try:
            cached = await self.redis.get(cache_key)
            if cached:
                logger.info(f"Cache hit for key: {cache_key}")
                return cached
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            
        return None
        
    async def _save_to_cache(self, cache_key: str, response: str) -> None:
        """Save response to cache"""
        if not self.redis:
            return
            
        try:
            await self.redis.setex(
                cache_key,
                self.cache_ttl,
                response
            )
            logger.info(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _generate_cache_key(self, query: str, results: List[SearchResult]) -> str:
        """Generate cache key from query and results"""
        # Create a deterministic representation of the query and results
        result_ids = [str(r.chunk.id) for r in results]
        cache_input = f"{query}:{','.join(result_ids)}"
        return f"llm:response:{hashlib.md5(cache_input.encode()).hexdigest()}"
    
    def _extract_citations(self, text: str) -> List[int]:
        """Extract source citations from text"""
        citations = []
        for match in self.citation_pattern.finditer(text):
            try:
                source_num = int(match.group(1))
                citations.append(source_num)
            except ValueError:
                continue
        return sorted(list(set(citations)))
    
    def _process_buffer(self, buffer: str, sentence_end_chars: str = ".!?") -> Tuple[str, str]:
        """Process buffer to extract complete sentences"""
        # Find the last sentence end character
        last_end_pos = -1
        for char in sentence_end_chars:
            pos = buffer.rfind(char)
            if pos > last_end_pos:
                last_end_pos = pos
                
        if last_end_pos == -1:
            # No complete sentence found
            return "", buffer
            
        # Include the sentence end character and any closing quotes, brackets, etc.
        end_pos = last_end_pos + 1
        while end_pos < len(buffer) and buffer[end_pos] in '")]}>\'\"':
            end_pos += 1
            
        # Split the buffer
        complete = buffer[:end_pos]
        remainder = buffer[end_pos:].lstrip()
        
        return complete, remainder
                    
    async def _stream_response(
        self,
        messages: List[Dict],
        results: List[SearchResult]
    ) -> AsyncGenerator[ChatResponse, None]:
        """Stream response from OpenRouter API using efficient streaming approach"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # Use streaming approach with client.stream
                async with client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://localhost:3000",  # Required by OpenRouter
                        "Content-Type": "application/json",
                        "X-Title": "Document Intelligence RAG"  # Identify our application
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "stream": True
                    }
                ) as response:
                    response.raise_for_status()
                    
                    buffer = ""
                    full_response = ""
                    
                    # Use aiter_lines for line-by-line processing
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                            
                        data = line[6:].strip()  # Remove "data: " prefix
                        
                        if data == "[DONE]":
                            # Send any remaining content
                            if buffer:
                                full_response += buffer
                                # Extract citations from the full response
                                cited_sources = self._extract_citations(full_response)
                                
                                # Group results by document ID first
                                doc_groups = {}
                                for idx, r in enumerate(results):
                                    doc_id = str(r.chunk.document_id)
                                    if doc_id not in doc_groups:
                                        doc_groups[doc_id] = {
                                            'index': len(doc_groups) + 1,
                                            'chunks': []
                                        }
                                    doc_groups[doc_id]['chunks'].append(r)

                                # Get the cited source chunks using document groups
                                sources = []
                                for i in cited_sources:
                                    # Find document group with this index
                                    doc_group = next(
                                        (group for group in doc_groups.values() if group['index'] == i),
                                        None
                                    )
                                    if doc_group:
                                        # Add first chunk from document as source
                                        sources.append(doc_group['chunks'][0].chunk)
                                    else:
                                        # Log warning for invalid citation
                                        logger.warning(f"LLM cited non-existent source: [Source {i}] (only {len(doc_groups)} sources available)")
                                
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
                                
                                yield ChatResponse(
                                    message=full_response,
                                    sources=sources,
                                    finished=True
                                )
                            break
                        
                        # Skip OpenRouter processing comments
                        if data.startswith(":"):
                            continue
                            
                        try:
                            parsed = json.loads(data)
                            
                            if not parsed.get("choices"):
                                continue
                                
                            delta = parsed["choices"][0].get("delta", {})
                            
                            # Check for end of response
                            if delta.get("finish_reason"):
                                if buffer:
                                    full_response += buffer
                                    # Extract citations
                                    cited_sources = self._extract_citations(full_response)
                                    
                                    # Group results by document ID first
                                    doc_groups = {}
                                    for idx, r in enumerate(results):
                                        doc_id = str(r.chunk.document_id)
                                        if doc_id not in doc_groups:
                                            doc_groups[doc_id] = {
                                                'index': len(doc_groups) + 1,
                                                'chunks': []
                                            }
                                        doc_groups[doc_id]['chunks'].append(r)

                                    # Get the cited source chunks using document groups
                                    sources = []
                                    for i in cited_sources:
                                        # Find document group with this index
                                        doc_group = next(
                                            (group for group in doc_groups.values() if group['index'] == i),
                                            None
                                        )
                                        if doc_group:
                                            # Add first chunk from document as source
                                            sources.append(doc_group['chunks'][0].chunk)
                                        else:
                                            # Log warning for invalid citation
                                            logger.warning(f"LLM cited non-existent source: [Source {i}] (only {len(doc_groups)} sources available)")
                                    
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
                                    
                                    yield ChatResponse(
                                        message=full_response,
                                        sources=sources,
                                        finished=True
                                    )
                                break
                            
                            # Get content from delta
                            content = delta.get("content", "")
                            if not content:
                                continue
                                
                            # Process content
                            buffer += content
                            full_response += content
                            
                            # Extract citations from the current buffer to provide sources during streaming
                            current_citations = self._extract_citations(buffer)
                            
                            # Group results by document ID first
                            doc_groups = {}
                            for idx, r in enumerate(results):
                                doc_id = str(r.chunk.document_id)
                                if doc_id not in doc_groups:
                                    doc_groups[doc_id] = {
                                        'index': len(doc_groups) + 1,
                                        'chunks': []
                                    }
                                doc_groups[doc_id]['chunks'].append(r)

                            # Get the cited source chunks using document groups
                            current_sources = []
                            for i in current_citations:
                                # Find document group with this index
                                doc_group = next(
                                    (group for group in doc_groups.values() if group['index'] == i),
                                    None
                                )
                                if doc_group:
                                    # Add first chunk from document as source
                                    current_sources.append(doc_group['chunks'][0].chunk)
                                else:
                                    # Log warning for invalid citation
                                    logger.warning(f"LLM cited non-existent source: [Source {i}] (only {len(doc_groups)} sources available)")
                            
                            # Convert UUID objects to strings for JSON serialization
                            for source in current_sources:
                                if hasattr(source, 'id') and isinstance(source.id, uuid.UUID):
                                    source.id = str(source.id)
                                if hasattr(source, 'document_id') and isinstance(source.document_id, uuid.UUID):
                                    source.document_id = str(source.document_id)
                                if hasattr(source, 'metadata') and source.metadata:
                                    for key, value in source.metadata.items():
                                        if isinstance(value, uuid.UUID):
                                            source.metadata[key] = str(value)
                            
                            # Check if we have complete sentences to yield
                            complete, remaining = self._process_buffer(buffer)
                            if complete:
                                buffer = remaining
                                yield ChatResponse(
                                    message=complete,
                                    sources=current_sources,  # Include sources during streaming
                                    finished=False
                                )
                            
                        except json.JSONDecodeError:
                            # Ignore invalid JSON (could be comments or other non-JSON data)
                            continue
                                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error streaming response: {e.response.status_code} - {e.response.text}")
                error_msg = f"Error: API request failed with status {e.response.status_code}"
                yield ChatResponse(
                    message=error_msg,
                    sources=[],  # Initialize with empty list
                    finished=True
                )
            except httpx.RequestError as e:
                logger.error(f"Request error streaming response: {e}")
                error_msg = "Error: Failed to connect to LLM API"
                yield ChatResponse(
                    message=error_msg,
                    sources=[],  # Initialize with empty list
                    finished=True
                )
            except Exception as e:
                logger.error(f"Error streaming response: {e}")
                error_msg = f"Error: {str(e)}"
                if buffer:
                    error_msg = f"{buffer}\n\nError: Stream interrupted - {str(e)}"
                
                yield ChatResponse(
                    message=error_msg,
                    sources=[],  # Initialize with empty list
                    finished=True
                )
                    
    async def generate_response(
        self,
        query: str,
        results: List[SearchResult],
        chat_history: Optional[List[ChatMessage]] = None,
        use_cache: bool = True
    ) -> AsyncGenerator[ChatResponse, None]:
        """Generate streaming response with source citations"""
        try:
            # Check cache if enabled
            if use_cache and self.redis:
                cache_key = self._generate_cache_key(query, results)
                cached_response = await self._check_cache(cache_key)
                
                if cached_response:
                    # Parse cached response
                    cached_data = json.loads(cached_response)
                    message = cached_data.get("message", "")
                    source_ids = cached_data.get("source_ids", [])
                    
                    # Group results by document ID first
                    doc_groups = {}
                    for idx, r in enumerate(results):
                        doc_id = str(r.chunk.document_id)
                        if doc_id not in doc_groups:
                            doc_groups[doc_id] = {
                                'index': len(doc_groups) + 1,
                                'chunks': []
                            }
                        doc_groups[doc_id]['chunks'].append(r)

                    # Get source chunks using document groups
                    sources = []
                    for i in source_ids:
                        # Find document group with this index
                        doc_group = next(
                            (group for group in doc_groups.values() if group['index'] == i),
                            None
                        )
                        if doc_group:
                            # Add first chunk from document as source
                            sources.append(doc_group['chunks'][0].chunk)
                        else:
                            # Log warning for invalid citation
                            logger.warning(f"LLM cited non-existent source: [Source {i}] (only {len(doc_groups)} sources available)")
                    
                    # Return cached response
                    yield ChatResponse(
                        message=message,
                        sources=sources,
                        finished=True
                    )
                    return
            
            # Format messages for the API
            messages = self._format_messages(query, results, chat_history)
            
            # Track the full response for caching
            full_response = ""
            cited_sources = []
            
            # Stream response
            async for response in self._stream_response(messages, results):
                if response.finished:
                    full_response = response.message
                    cited_sources = self._extract_citations(full_response)
                    
                    # Cache the response if enabled
                    if use_cache and self.redis:
                        cache_data = {
                            "message": full_response,
                            "source_ids": cited_sources,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        await self._save_to_cache(
                            self._generate_cache_key(query, results),
                            json.dumps(cache_data)
                        )
                
                yield response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield ChatResponse(
                message=f"Error: Failed to generate response - {str(e)}",
                sources=[],  # Initialize with empty list
                finished=True
            )

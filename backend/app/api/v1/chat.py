"""
Chat API endpoints for RAG application.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
import json
import logging
import time
import re
import math
from datetime import datetime
import structlog
from pydantic import BaseModel, Field
from bson import ObjectId

from models.document import ChatMessage, ChatResponse
from core.hybrid_retriever import HybridRetriever
from core.reranker import Reranker
from core.enhanced_llm_service import EnhancedLLMService
from core.mongodb_service import MongoDBService
from core.config import settings

# Configure structured logging
logger = structlog.get_logger()
router = APIRouter(prefix="/chat", tags=["chat"])

# Helper function to sanitize response data
def _sanitize_response_data(response: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize response data to ensure all values are valid for JSON serialization"""
    sanitized = response.copy()
    
    # Ensure response has valid content
    if 'content' not in sanitized and 'message' in sanitized:
        sanitized['content'] = sanitized['message']
    
    # Ensure token usage values are valid numbers
    if 'token_usage' in sanitized:
        token_usage = sanitized['token_usage']
        if 'prompt_tokens' in token_usage and (token_usage['prompt_tokens'] is None or isinstance(token_usage['prompt_tokens'], float) and math.isnan(token_usage['prompt_tokens'])):
            token_usage['prompt_tokens'] = 0
        if 'completion_tokens' in token_usage and (token_usage['completion_tokens'] is None or isinstance(token_usage['completion_tokens'], float) and math.isnan(token_usage['completion_tokens'])):
            token_usage['completion_tokens'] = 0
        if 'total_tokens' in token_usage and (token_usage['total_tokens'] is None or isinstance(token_usage['total_tokens'], float) and math.isnan(token_usage['total_tokens'])):
            token_usage['total_tokens'] = 0
    
    # Ensure context window values are valid numbers
    if 'context_window' in sanitized:
        context_window = sanitized['context_window']
        if 'used_tokens' in context_window and (context_window['used_tokens'] is None or isinstance(context_window['used_tokens'], float) and math.isnan(context_window['used_tokens'])):
            context_window['used_tokens'] = 0
        if 'max_tokens' in context_window and (context_window['max_tokens'] is None or isinstance(context_window['max_tokens'], float) and math.isnan(context_window['max_tokens'])):
            context_window['max_tokens'] = 100000
        if 'remaining_tokens' in context_window and (context_window['remaining_tokens'] is None or isinstance(context_window['remaining_tokens'], float) and math.isnan(context_window['remaining_tokens'])):
            context_window['remaining_tokens'] = 100000
    
    # Ensure tokenUsage is also present (frontend might expect this format)
    if 'token_usage' in sanitized and 'tokenUsage' not in sanitized:
        sanitized['tokenUsage'] = {
            'prompt_tokens': sanitized['token_usage'].get('prompt_tokens', 0),
            'completion_tokens': sanitized['token_usage'].get('completion_tokens', 0),
            'total_tokens': sanitized['token_usage'].get('total_tokens', 0)
        }
    
    # Ensure contextWindow is also present (frontend might expect this format)
    if 'context_window' in sanitized and 'contextWindow' not in sanitized:
        sanitized['contextWindow'] = {
            'used_tokens': sanitized['context_window'].get('used_tokens', 0),
            'max_tokens': sanitized['context_window'].get('max_tokens', 100000),
            'remaining_tokens': sanitized['context_window'].get('remaining_tokens', 0)
        }
    
    return sanitized

# Request models
class ChatRequest(BaseModel):
    """Chat request model"""
    query: str
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list)
    use_cache: bool = True
    top_k: int = Field(default=3, ge=1, le=10)
    stream: bool = True

class CreateSessionRequest(BaseModel):
    """Request to create a new chat session"""
    title: Optional[str] = None
    clerk_user_id: Optional[str] = "anonymous"  # Default user ID for non-authenticated requests

class ChatSessionRequest(BaseModel):
    """Chat request with session ID"""
    query: str
    use_cache: bool = True
    top_k: int = Field(default=3, ge=1, le=10)
    stream: bool = True

class UpdateSessionRequest(BaseModel):
    """Request to update a chat session"""
    title: Optional[str] = None
    is_active: Optional[bool] = None

# Dependencies
async def get_mongodb_service():
    """Get MongoDB service instance"""
    return MongoDBService()

async def get_hybrid_retriever(session_id: Optional[str] = None):
    """Get hybrid retriever instance with session-specific namespace"""
    logger.info(f"Creating HybridRetriever for session: {session_id}")
    retriever = HybridRetriever(
        pinecone_api_key=settings.PINECONE_API_KEY,
        pinecone_environment=settings.PINECONE_ENVIRONMENT,
        pinecone_index=settings.PINECONE_INDEX,
        redis_url=settings.REDIS_URL,
        session_id=session_id,
        vector_weight=float(settings.VECTOR_WEIGHT),
        bm25_weight=float(settings.BM25_WEIGHT),
        tfidf_weight=float(settings.TFIDF_WEIGHT)
    )
    # Initialize the retriever to ensure document loading is complete
    await retriever.initialize()
    logger.info(f"Initialized HybridRetriever with namespace: {retriever.namespace}")
    return retriever

async def get_reranker():
    """Get reranker instance with config"""
    return Reranker(
        cohere_api_key=settings.COHERE_API_KEY,
        redis_url=settings.REDIS_URL,
        cache_ttl=settings.CACHE_TTL
    )

async def get_enhanced_llm_service():
    """Get enhanced LLM service instance with config"""
    return EnhancedLLMService(
        openrouter_api_key=settings.OPENROUTER_API_KEY,
        redis_url=settings.REDIS_URL,
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        cache_ttl=settings.CACHE_TTL
    )

# Metrics tracking
async def log_rag_metrics(
    query: str,
    results: List[Any],
    response: Dict[str, Any],
    duration: float
):
    """Log RAG metrics for monitoring"""
    try:
        # Extract citations
        citation_pattern = re.compile(r'\[Source (\d+)\]')
        citations = []
        for match in citation_pattern.finditer(response.get("message", "")):
            try:
                source_num = int(match.group(1))
                citations.append(source_num)
            except ValueError:
                continue
        
        # Unique citations
        unique_citations = sorted(list(set(citations)))
        
        # Log metrics
        logger.info(
            "rag_metrics",
            query_length=len(query),
            result_count=len(results),
            response_length=len(response.get("message", "")),
            citation_count=len(citations),
            unique_citations=len(unique_citations),
            cited_sources=unique_citations,
            duration_seconds=duration,
            token_usage=response.get("token_usage", {})
        )
    except Exception as e:
        logger.error("error_logging_metrics", error=str(e))

from fastapi import Header

# Session endpoints
@router.post("/sessions")
async def create_chat_session(
    request: CreateSessionRequest,
    mongodb_service: MongoDBService = Depends(get_mongodb_service),
    authorization: str = Header(None)
):
    """Create a new chat session"""
    try:
        # Extract user ID from auth token if available
        clerk_user_id = "anonymous"
        if authorization:
            try:
                scheme, token = authorization.split()
                if scheme.lower() == "bearer":
                    # Decode JWT token to get user ID
                    import jwt
                    payload = jwt.decode(
                        token,
                        options={"verify_signature": False}
                    )
                    if "sub" in payload:
                        clerk_user_id = payload["sub"]
                        logger.info(f"Creating session for authenticated user: {clerk_user_id}")
            except Exception as e:
                logger.error(f"Error extracting user ID from token: {str(e)}")
                # Continue with anonymous user if token parsing fails
        
        # Use the extracted user ID or fallback to the one in the request
        session = await mongodb_service.create_chat_session(
            clerk_user_id=clerk_user_id,
            title=request.title
        )
        
        # Convert any ObjectId in documents to strings
        if "documents" in session and session["documents"]:
            session["documents"] = [str(doc_id) if isinstance(doc_id, ObjectId) else doc_id for doc_id in session["documents"]]
            
        return session
    except Exception as e:
        logger.error("Failed to create chat session", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@router.get("/sessions")
async def list_chat_sessions(
    mongodb_service: MongoDBService = Depends(get_mongodb_service),
    authorization: str = Header(None)
):
    """List all chat sessions for a user"""
    try:
        # Extract user ID from auth token if available
        clerk_user_id = "anonymous"
        if authorization:
            try:
                scheme, token = authorization.split()
                if scheme.lower() == "bearer":
                    # Decode JWT token to get user ID
                    import jwt
                    payload = jwt.decode(
                        token,
                        options={"verify_signature": False}
                    )
                    if "sub" in payload:
                        clerk_user_id = payload["sub"]
                        logger.info(f"Listing sessions for authenticated user: {clerk_user_id}")
            except Exception as e:
                logger.error(f"Error extracting user ID from token: {str(e)}")
                # Continue with anonymous user if token parsing fails
        
        sessions = await mongodb_service.list_chat_sessions(clerk_user_id)
        
        # Convert any ObjectId in documents to strings
        for session in sessions:
            if "documents" in session and session["documents"]:
                session["documents"] = [str(doc_id) if isinstance(doc_id, ObjectId) else doc_id for doc_id in session["documents"]]
        
        return sessions
    except Exception as e:
        logger.error("Failed to list chat sessions", error=str(e))
        # Return empty list as fallback
        return []

@router.get("/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Get a chat session by ID"""
    try:
        session = await mongodb_service.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Convert any ObjectId in documents to strings
        if "documents" in session and session["documents"]:
            session["documents"] = [str(doc_id) if isinstance(doc_id, ObjectId) else doc_id for doc_id in session["documents"]]
        
        return session
    except Exception as e:
        logger.error(f"Failed to get chat session {session_id}", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@router.patch("/sessions/{session_id}")
async def update_chat_session(
    session_id: str,
    request: UpdateSessionRequest,
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Update a chat session"""
    try:
        # Verify session exists
        session = await mongodb_service.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Prepare updates
        updates = {}
        if request.title is not None:
            updates["title"] = request.title
        if request.is_active is not None:
            updates["isActive"] = request.is_active
            
        if not updates:
            # Convert any ObjectId in documents to strings before returning
            if "documents" in session and session["documents"]:
                session["documents"] = [str(doc_id) if isinstance(doc_id, ObjectId) else doc_id for doc_id in session["documents"]]
            return session
            
        # Update session
        updated_session = await mongodb_service.update_chat_session(session_id, updates)
        if not updated_session:
            raise HTTPException(status_code=500, detail="Failed to update session")
        
        # Convert any ObjectId in documents to strings
        if "documents" in updated_session and updated_session["documents"]:
            updated_session["documents"] = [str(doc_id) if isinstance(doc_id, ObjectId) else doc_id for doc_id in updated_session["documents"]]
            
        return updated_session
    except Exception as e:
        logger.error(f"Failed to update chat session {session_id}", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Delete a chat session"""
    try:
        # Verify session exists
        session = await mongodb_service.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete session
        success = await mongodb_service.delete_chat_session(session_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete session")
            
        return {"success": True, "message": "Session deleted"}
    except Exception as e:
        logger.error(f"Failed to delete chat session {session_id}", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@router.get("/sessions/{session_id}/messages")
async def get_chat_messages(
    session_id: str,
    limit: int = 50,
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Get messages for a chat session"""
    try:
        # Verify session exists
        session = await mongodb_service.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = await mongodb_service.get_chat_messages(session_id, limit)
        return messages
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get chat messages for session {session_id}", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error retrieving messages: {str(e)}")

@router.get("/sessions/{session_id}/token-usage")
async def get_session_token_usage(
    session_id: str,
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Get token usage for a chat session"""
    try:
        # Verify session exists
        session = await mongodb_service.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        usage = await mongodb_service.get_session_token_usage(session_id)
        return usage
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get token usage for session {session_id}", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error retrieving token usage: {str(e)}")

# Chat endpoints with session management
async def stream_session_response(
    session_id: str,
    query: str,
    clerk_user_id: str,
    retriever: HybridRetriever,
    reranker: Reranker,
    llm_service: EnhancedLLMService,
    mongodb_service: MongoDBService,
    background_tasks: BackgroundTasks,
    use_cache: bool = True,
    top_k: int = 3
):
    """Stream chat response with session management"""
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    log = logger.bind(
        request_id=request_id, 
        query=query[:50], 
        session_id=session_id,
        user_id=clerk_user_id
    )
    
    try:
        log.info("session_request_started")
        
        # Get chat history
        db_messages = await mongodb_service.get_chat_messages(session_id)
        
        # Convert to format expected by LLM service
        chat_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in db_messages
        ]
        
        # Add user message to session
        user_message = await mongodb_service.add_chat_message(
            session_id=session_id,
            role="user",
            content=query
        )
        
        # Get initial search results
        log.info("retrieval_started", top_k=top_k*3)
        search_results = await retriever.hybrid_search(
            query=query,
            top_k=top_k*3,
            use_cache=use_cache
        )
        log.info("retrieval_completed", result_count=len(search_results))
        
        # Rerank results
        log.info("reranking_started")
        reranked_results = await reranker.rerank_hybrid(
            query=query,
            results=search_results,
            use_cache=use_cache
        )
        log.info("reranking_completed")
        
        # Keep top results after reranking
        top_results = reranked_results[:top_k]
        
        # Stream LLM response with token counting
        log.info("llm_generation_started")
        final_response = None
        assistant_message_id = None
        
        async for response in llm_service.generate_response_with_token_count(
            query=query,
            results=top_results,
            chat_history=chat_history,
            use_cache=use_cache
        ):
            # Track final response for metrics
            if response.get("finished", False):
                final_response = response
                duration = time.time() - start_time
                
                # Add assistant message to session
                assistant_message = await mongodb_service.add_chat_message(
                    session_id=session_id,
                    role="assistant",
                    content=response.get("message", "")
                )
                assistant_message_id = assistant_message["_id"]
                
                # Update token usage
                token_usage = response.get("token_usage", {})
                if token_usage:
                    # Update user message token count
                    await mongodb_service.update_token_usage(
                        message_id=user_message["_id"],
                        usage={"prompt_tokens": token_usage.get("prompt_tokens", 0)}
                    )
                    
                    # Update assistant message token count
                    await mongodb_service.update_token_usage(
                        message_id=assistant_message_id,
                        usage={"completion_tokens": token_usage.get("completion_tokens", 0)}
                    )
                
                log.info(
                    "llm_generation_completed", 
                    duration=duration,
                    response_length=len(response.get("message", "")),
                    source_count=len(response.get("sources", [])),
                    token_usage=token_usage
                )
                
                # Log metrics in background
                background_tasks.add_task(
                    log_rag_metrics,
                    query,
                    top_results,
                    response,
                    duration
                )
                
                # Convert any ObjectId in sources to strings
                if "sources" in response:
                    for source in response["sources"]:
                        if "id" in source and isinstance(source["id"], ObjectId):
                            source["id"] = str(source["id"])
            
            # Ensure response has valid content
            if 'content' not in response and 'message' in response:
                response['content'] = response['message']
            
            # Sanitize response data to ensure all values are valid
            sanitized_response = _sanitize_response_data(response)
                    
            # Convert sanitized response to SSE format
            yield f"data: {json.dumps(sanitized_response)}\n\n"
            
        # End stream
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        duration = time.time() - start_time
        log.error(
            "session_request_failed",
            error=str(e),
            duration=duration,
            error_type=type(e).__name__
        )
        
        error_response = {
            "message": f"Error: {str(e)}",
            "finished": True
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"

@router.post("/sessions/{session_id}/stream")
async def chat_session_stream(
    session_id: str,
    request: ChatSessionRequest,
    background_tasks: BackgroundTasks,
    mongodb_service: MongoDBService = Depends(get_mongodb_service),
    reranker: Reranker = Depends(get_reranker),
    llm_service: EnhancedLLMService = Depends(get_enhanced_llm_service)
):
    """Stream chat responses with session management"""
    try:
        # Verify session exists
        session = await mongodb_service.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Convert any ObjectId in documents to strings
        if "documents" in session and session["documents"]:
            session["documents"] = [str(doc_id) if isinstance(doc_id, ObjectId) else doc_id for doc_id in session["documents"]]
        
        # Get the clerk_user_id from the session
        clerk_user_id = session.get("clerkUserId", "anonymous")
        
        # Get session-specific retriever
        retriever = await get_hybrid_retriever(session_id)
        
        # Create streaming response
        return StreamingResponse(
            stream_session_response(
                session_id=session_id,
                query=request.query,
                clerk_user_id=clerk_user_id,
                retriever=retriever,
                reranker=reranker,
                llm_service=llm_service,
                mongodb_service=mongodb_service,
                background_tasks=background_tasks,
                use_cache=request.use_cache,
                top_k=request.top_k
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Failed to stream chat response for session {session_id}", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error streaming response: {str(e)}")

# Legacy chat endpoints (without session management)
async def stream_chat_response(
    query: str,
    chat_history: Optional[List[ChatMessage]],
    retriever: HybridRetriever,
    reranker: Reranker,
    llm_service: EnhancedLLMService,
    background_tasks: BackgroundTasks,
    use_cache: bool = True,
    top_k: int = 3
):
    """Stream chat response with RAG pipeline"""
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    log = logger.bind(request_id=request_id, query=query[:50])
    
    try:
        log.info("rag_request_started")
        
        # Get initial search results
        log.info("retrieval_started", top_k=top_k*3)  # Get more for reranking
        search_results = await retriever.hybrid_search(
            query=query,
            top_k=top_k*3,  # Get more results for reranking
            use_cache=use_cache
        )
        log.info("retrieval_completed", result_count=len(search_results))
        
        # Rerank results
        log.info("reranking_started")
        reranked_results = await reranker.rerank_hybrid(
            query=query,
            results=search_results,
            use_cache=use_cache
        )
        log.info("reranking_completed")
        
        # Keep top results after reranking
        top_results = reranked_results[:top_k]
        
        # Convert chat history to format expected by LLM service
        formatted_history = None
        if chat_history:
            formatted_history = [
                {"role": msg.role, "content": msg.content}
                for msg in chat_history
            ]
        
        # Stream LLM response
        log.info("llm_generation_started")
        final_response = None
        
        async for response in llm_service.generate_response_with_token_count(
            query=query,
            results=top_results,
            chat_history=formatted_history,
            use_cache=use_cache
        ):
            # Track final response for metrics
            if response.get("finished", True):
                final_response = response
                duration = time.time() - start_time
                log.info(
                    "llm_generation_completed", 
                    duration=duration,
                    response_length=len(response.get("message", "")),
                    source_count=len(response.get("sources", [])),
                    token_usage=response.get("token_usage", {})
                )
                
                # Log metrics in background
                background_tasks.add_task(
                    log_rag_metrics,
                    query,
                    top_results,
                    response,
                    duration
                )
                
                # Convert any ObjectId in sources to strings
                if "sources" in response:
                    for source in response["sources"]:
                        if "id" in source and isinstance(source["id"], ObjectId):
                            source["id"] = str(source["id"])
            
            # Ensure response has valid content
            if 'content' not in response and 'message' in response:
                response['content'] = response['message']
            
            # Sanitize response data to ensure all values are valid
            sanitized_response = _sanitize_response_data(response)
            
            # Convert sanitized response to SSE format
            yield f"data: {json.dumps(sanitized_response)}\n\n"
            
        # End stream
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        duration = time.time() - start_time
        log.error(
            "rag_request_failed",
            error=str(e),
            duration=duration,
            error_type=type(e).__name__
        )
        
        error_response = {
            "message": f"Error: {str(e)}",
            "finished": True
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: Reranker = Depends(get_reranker),
    llm_service: EnhancedLLMService = Depends(get_enhanced_llm_service)
):
    """Stream chat responses with RAG"""
    return StreamingResponse(
        stream_chat_response(
            query=request.query,
            chat_history=request.chat_history,
            retriever=retriever,
            reranker=reranker,
            llm_service=llm_service,
            background_tasks=background_tasks,
            use_cache=request.use_cache,
            top_k=request.top_k
        ),
        media_type="text/event-stream"
    )

@router.post("/")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: Reranker = Depends(get_reranker),
    llm_service: EnhancedLLMService = Depends(get_enhanced_llm_service)
):
    """Non-streaming chat endpoint"""
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    log = logger.bind(request_id=request_id, query=request.query[:50])
    
    try:
        log.info("rag_request_started", streaming=False)
        
        # Get initial search results
        log.info("retrieval_started", top_k=request.top_k*3)
        search_results = await retriever.hybrid_search(
            query=request.query,
            top_k=request.top_k*3,
            use_cache=request.use_cache
        )
        log.info("retrieval_completed", result_count=len(search_results))
        
        # Rerank results
        log.info("reranking_started")
        reranked_results = await reranker.rerank_hybrid(
            query=request.query,
            results=search_results,
            use_cache=request.use_cache
        )
        log.info("reranking_completed")
        
        # Keep top results
        top_results = reranked_results[:request.top_k]
        
        # Convert chat history to format expected by LLM service
        formatted_history = None
        if request.chat_history:
            formatted_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.chat_history
            ]
        
        # Get full response
        log.info("llm_generation_started")
        full_response = None
        async for response in llm_service.generate_response_with_token_count(
            query=request.query,
            results=top_results,
            chat_history=formatted_history,
            use_cache=request.use_cache
        ):
            if response.get("finished", False):
                full_response = response
                
        if full_response:
            duration = time.time() - start_time
            log.info(
                "llm_generation_completed", 
                duration=duration,
                response_length=len(full_response.get("message", "")),
                source_count=len(full_response.get("sources", [])),
                token_usage=full_response.get("token_usage", {})
            )
            
            # Log metrics in background
            background_tasks.add_task(
                log_rag_metrics,
                request.query,
                top_results,
                full_response,
                duration
            )
            
            # Convert any ObjectId in sources to strings
            if "sources" in full_response:
                for source in full_response["sources"]:
                    if "id" in source and isinstance(source["id"], ObjectId):
                        source["id"] = str(source["id"])
                
        return full_response
        
    except Exception as e:
        duration = time.time() - start_time
        log.error(
            "rag_request_failed",
            error=str(e),
            duration=duration,
            error_type=type(e).__name__
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

# Health check endpoint
@router.get("/health")
async def health_check(
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: Reranker = Depends(get_reranker),
    llm_service: EnhancedLLMService = Depends(get_enhanced_llm_service),
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Check RAG pipeline health"""
    try:
        # Simple test query
        test_query = "What is machine learning?"
        
        # Test retrieval
        results = await retriever.hybrid_search(
            query=test_query,
            top_k=3,
            use_cache=False
        )
        
        # Test reranking
        reranked = await reranker.rerank_hybrid(
            query=test_query,
            results=results,
            use_cache=False
        )
        
        # Don't actually call the LLM API to save costs
        # Just check that the service is initialized
        assert llm_service.api_key is not None
        
        # Test MongoDB connection
        # Just check that we can connect to the database
        client = mongodb_service.client
        await client.admin.command('ping')
        
        return {
            "status": "healthy",
            "retriever": "ok",
            "reranker": "ok",
            "llm_service": "ok",
            "mongodb": "ok",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail=f"RAG pipeline unhealthy: {str(e)}"
        )

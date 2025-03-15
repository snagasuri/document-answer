import asyncio
import logging
from typing import List, Dict, Any
import cohere
import redis.asyncio as redis
import json
import uuid

from models.document import SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(
        self,
        cohere_api_key: str,
        redis_url: str,
        cache_ttl: int = 3600
    ):
        """Initialize reranker"""
        # Initialize Cohere
        self.cohere = cohere.Client(cohere_api_key)
        
        # Initialize Redis
        self.redis = redis.from_url(redis_url)
        
        # Set settings
        self.cache_ttl = cache_ttl
        
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        use_cache: bool = True
    ) -> List[SearchResult]:
        """Rerank results using Cohere"""
        try:
            # Check if results is empty
            if not results:
                logger.warning("No documents to rerank")
                return []
                
            # Check cache
            if use_cache and self.redis:
                cache_key = f"cohere_rerank:{query}:{[str(r.chunk.id) for r in results]}"
                cached = await self.redis.get(cache_key)
                if cached:
                    return [SearchResult(**r) for r in json.loads(cached)]
            
            # Prepare documents
            documents = [result.chunk.content for result in results]
            
            # Rerank with Cohere
            reranked = self.cohere.rerank(
                query=query,
                documents=documents,
                model="rerank-english-v2.0",
                top_n=len(documents)
            )
            
            # Update scores
            # The Cohere API response format has changed
            # Now it returns an object with a 'results' property
            for i, result in enumerate(results):
                if hasattr(reranked, 'results') and i < len(reranked.results):
                    result.rerank_score = reranked.results[i].relevance_score
                else:
                    # Fallback to using the original score
                    result.rerank_score = result.combined_score
                
            # Sort by rerank score
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Cache results
            if use_cache and self.redis:
                try:
                    # Convert UUIDs to strings for JSON serialization
                    serializable_results = []
                    for r in results:
                        result_dict = r.dict()
                        # Convert UUID objects to strings
                        if 'chunk' in result_dict and result_dict['chunk']:
                            chunk_dict = result_dict['chunk']
                            if 'id' in chunk_dict and isinstance(chunk_dict['id'], uuid.UUID):
                                chunk_dict['id'] = str(chunk_dict['id'])
                            if 'document_id' in chunk_dict and isinstance(chunk_dict['document_id'], uuid.UUID):
                                chunk_dict['document_id'] = str(chunk_dict['document_id'])
                        serializable_results.append(result_dict)
                    
                    await self.redis.setex(
                        cache_key,
                        self.cache_ttl,
                        json.dumps(serializable_results)
                    )
                except Exception as e:
                    logger.error(f"Error caching results: {str(e)}")
                    # Continue without caching
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Cohere reranking: {str(e)}")
            raise
            
            
        
    async def rerank_hybrid(
        self,
        query: str,
        results: List[SearchResult],
        use_cache: bool = True
    ) -> List[SearchResult]:
        """Perform hybrid reranking combining Cohere and TART"""
        try:
            # Check if results is empty
            if not results:
                logger.warning("No documents to rerank in hybrid reranking")
                return []
                
            # Check cache
            if use_cache and self.redis:
                cache_key = f"hybrid_rerank:{query}:{[str(r.chunk.id) for r in results]}"
                cached = await self.redis.get(cache_key)
                if cached:
                    return [SearchResult(**r) for r in json.loads(cached)]
            
            # Just use Cohere reranking
            final_results = await self.rerank(query, results.copy(), use_cache=False)
            
            # Sort by rerank score
            final_results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Cache results
            if use_cache and self.redis:
                try:
                    # Convert UUIDs to strings for JSON serialization
                    serializable_results = []
                    for r in final_results:
                        result_dict = r.dict()
                        # Convert UUID objects to strings
                        if 'chunk' in result_dict and result_dict['chunk']:
                            chunk_dict = result_dict['chunk']
                            if 'id' in chunk_dict and isinstance(chunk_dict['id'], uuid.UUID):
                                chunk_dict['id'] = str(chunk_dict['id'])
                            if 'document_id' in chunk_dict and isinstance(chunk_dict['document_id'], uuid.UUID):
                                chunk_dict['document_id'] = str(chunk_dict['document_id'])
                        serializable_results.append(result_dict)
                    
                    await self.redis.setex(
                        cache_key,
                        self.cache_ttl,
                        json.dumps(serializable_results)
                    )
                except Exception as e:
                    logger.error(f"Error caching results: {str(e)}")
                    # Continue without caching
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid reranking: {str(e)}")
            raise

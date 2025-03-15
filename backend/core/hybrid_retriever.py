import asyncio
import logging
import json
import random
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import redis.asyncio as redis
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid

from models.document import DocumentChunk, SearchResult
from core.mongodb_service import MongoDBService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeEmbeddingService:
    """Service for generating embeddings using Pinecone's llama-text-embed-v2"""
    
    def __init__(self, api_key: str):
        """Initialize the Pinecone embedding service"""
        self.api_key = api_key
        self.model = "llama-text-embed-v2"
        self.dimension = 1024  # Default dimension for llama-text-embed-v2
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Pinecone Inference API"""
        if not texts:
            return []
            
        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=self.api_key)
            
            # Process in batches to avoid rate limits
            batch_size = 10
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Generate embeddings using Pinecone Inference
                batch_embeddings = pc.inference.embed(
                    model=self.model,
                    inputs=batch,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                
                # Extract embedding values
                batch_values = [item["values"] for item in batch_embeddings]
                embeddings.extend(batch_values)
                
                # Add a small delay to avoid rate limits
                await asyncio.sleep(0.5)
                
            return embeddings
                
        except Exception as e:
            logger.error(f"Error generating embeddings with Pinecone: {str(e)}")
            logger.warning("Falling back to simple embedding method.")
            return self._fallback_embeddings(texts)
    
    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate simple fallback embeddings when API is unavailable"""
        logger.info("Using fallback embedding method for document processing")
        # Simple embedding: create a vector matching the dimension of llama-text-embed-v2
        dimension = self.dimension  # 1024 for llama-text-embed-v2
        embeddings = []
        
        for text in texts:
            # Create a deterministic but simple embedding based on the text
            # This is NOT a good embedding, just a fallback to allow the system to continue
            import hashlib
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Use the hash to seed a random number generator
            import random
            random.seed(hash_bytes)
            
            # Generate a random vector with consistent dimensions
            vector = [random.uniform(-1, 1) for _ in range(dimension)]
            
            # Normalize the vector
            magnitude = sum(x**2 for x in vector) ** 0.5
            if magnitude > 0:
                vector = [x/magnitude for x in vector]
                
            embeddings.append(vector)
            
        return embeddings

class HybridRetriever:
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_environment: str,
        pinecone_index: str,
        redis_url: str,
        session_id: str = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.15,
        tfidf_weight: float = 0.15
    ):
        """Initialize hybrid retriever"""
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index)
        self.pinecone_api_key = pinecone_api_key
        
        # Set session ID for namespace
        self.session_id = session_id
        # Use session-specific namespace for proper multi-tenancy
        self.namespace = f"session_{session_id}" if session_id else "default"
        logger.info(f"Initialized HybridRetriever with namespace: {self.namespace} (implementing multi-tenancy)")
        
        # Initialize Redis
        self.redis = redis.from_url(redis_url)
        
        # Initialize embedding service using Pinecone's llama-text-embed-v2
        self.embedding_service = PineconeEmbeddingService(api_key=pinecone_api_key)
        logger.info(f"Using embedding model: {self.embedding_service.model}")
        
        # No longer using SentenceTransformer - fully migrated to Pinecone's llama-text-embed-v2
        
        # Initialize BM25 and TF-IDF
        self.bm25 = None
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = None
        self.documents = []
        
        # Set weights
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight
        
        # Cache settings
        self.cache_ttl = 3600  # 1 hour
        
        # Session-specific cache keys
        self.cache_prefix = f"session_{session_id}_" if session_id else ""
        
        # Flag to track initialization status
        self.initialized = False
        
        # We don't automatically load documents here anymore
        # Instead, we'll do it explicitly in the initialize method
    
    async def initialize(self):
        """Ensure document loading is complete before searches are performed"""
        if self.initialized:
            return self
            
        if self.session_id:
            logger.info(f"Initializing HybridRetriever for session: {self.session_id}")
            await self._load_documents_from_mongodb(self.session_id)
        
        self.initialized = True
        return self
        
    async def _load_documents_from_mongodb(self, session_id: str):
        """Load document chunks from MongoDB for the given session"""
        try:
            # Initialize MongoDB service
            mongodb_service = MongoDBService()
            
            # Get all documents for the session
            try:
                documents = await mongodb_service.get_session_documents(session_id)
            except Exception as e:
                logger.error(f"Error getting session documents: {str(e)}")
                return
            
            if not documents:
                logger.warning(f"No documents found for session {session_id}")
                return
                
            # Load chunks for each document
            for doc in documents:
                doc_id = doc["_id"]
                try:
                    chunks = await mongodb_service.get_document_chunks(doc_id)
                except Exception as e:
                    logger.error(f"Error getting document chunks: {str(e)}")
                    continue
                
                # Convert MongoDB chunks to DocumentChunk objects
                document_chunks = []
                for chunk in chunks:
                    try:
                        # Generate deterministic UUIDs from MongoDB ObjectId strings
                        # Use try-except to handle potential UUID conversion errors
                        try:
                            chunk_id = uuid.uuid5(uuid.NAMESPACE_OID, str(chunk["_id"]))
                        except Exception as uuid_err:
                            logger.warning(f"Error creating UUID from chunk ID: {str(uuid_err)}")
                            # Fallback to a new UUID
                            chunk_id = uuid.uuid4()
                            
                        try:
                            document_id = uuid.uuid5(uuid.NAMESPACE_OID, str(chunk["documentId"]))
                        except Exception as uuid_err:
                            logger.warning(f"Error creating UUID from document ID: {str(uuid_err)}")
                            # Fallback to a new UUID
                            document_id = uuid.uuid4()
                        
                        document_chunk = DocumentChunk(
                            id=chunk_id,
                            document_id=document_id,
                            content=chunk["content"],
                            metadata=chunk["metadata"],
                            chunk_index=chunk["chunkIndex"]
                        )
                        document_chunks.append(document_chunk)
                    except Exception as e:
                        logger.error(f"Error creating DocumentChunk from MongoDB data: {str(e)}")
                        continue
                
                # Add chunks to the retriever
                if document_chunks:
                    # Add to local documents list
                    self.documents.extend(document_chunks)
                    
                    # Update BM25
                    texts = [chunk.content for chunk in document_chunks]
                    tokenized_texts = [text.split() for text in texts]
                    if self.bm25 is None:
                        self.bm25 = BM25Okapi(tokenized_texts)
                    else:
                        # Fix for "truth value of an array" error - ensure we're not using NumPy arrays in boolean context
                        self.bm25 = BM25Okapi(
                            [doc.content.split() for doc in self.documents]
                        )
                    
                    # Update TF-IDF
                    if self.tfidf_matrix is None:
                        self.tfidf_matrix = self.tfidf.fit_transform(texts)
                    else:
                        self.tfidf_matrix = self.tfidf.fit_transform(
                            [doc.content for doc in self.documents]
                        )
                    
                    logger.info(f"Loaded {len(document_chunks)} chunks for document {doc_id} in session {session_id}")
            
            logger.info(f"Loaded a total of {len(self.documents)} document chunks for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error loading documents from MongoDB: {str(e)}")
            # Log the full exception traceback for better debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
    async def verify_vectors_searchable(self, vector_id: str, max_attempts: int = 10, initial_delay: float = 1.0, max_delay: float = 10.0):
        """Verify that vectors are searchable after upserting"""
        # First try fetch operation to confirm vector exists
        fetch_verified = False
        current_delay = initial_delay
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Check if the vector exists in the index using fetch
                fetch_response = self.index.fetch(
                    ids=[vector_id],
                    namespace=self.namespace
                )
                
                if vector_id in fetch_response.vectors:
                    logger.info(f"Vector {vector_id} verified as existing via fetch after {attempt} attempts")
                    fetch_verified = True
                    break
                
                logger.warning(f"Vector {vector_id} not found via fetch in attempt {attempt}/{max_attempts}")
                
                # Exponential backoff with jitter
                current_delay = min(current_delay * 1.5, max_delay)
                jitter = current_delay * 0.2 * random.random()
                await asyncio.sleep(current_delay + jitter)
                
            except Exception as e:
                logger.error(f"Error verifying vector existence: {str(e)}")
                # Exponential backoff with jitter
                current_delay = min(current_delay * 1.5, max_delay)
                jitter = current_delay * 0.2 * random.random()
                await asyncio.sleep(current_delay + jitter)
        
        if not fetch_verified:
            logger.error(f"Could not verify vector {vector_id} exists after {max_attempts} attempts")
            return False
            
        # Now verify the vector is actually queryable
        logger.info("Vector exists, now verifying it is queryable...")
        current_delay = initial_delay
        
        # Generate a test query embedding
        test_query = "test query for verification"
        logger.info(f"Generating embedding for test query: '{test_query}'")
        
        try:
            query_embeddings = await self.embedding_service.generate_embeddings([test_query])
            query_embedding = query_embeddings[0]
            
            # Try multiple times to query with increasing delays
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.info(f"Attempt {attempt}/{max_attempts} to query vectors in namespace: {self.namespace}")
                    
                    # Query with the actual embedding
                    query_response = self.index.query(
                        vector=query_embedding,
                        top_k=10,  # Use a reasonable top_k
                        namespace=self.namespace,
                        include_metadata=True
                    )
                    
                    # Check if we got any matches
                    if hasattr(query_response, 'matches') and len(query_response.matches) > 0:
                        match_count = len(query_response.matches)
                        match_ids = [match.id for match in query_response.matches]
                        logger.info(f"Query returned {match_count} matches. Vector verification succeeded.")
                        logger.info(f"First few match IDs: {match_ids[:5]}")
                        return True
                    
                    # If no matches with the embedding, try a dummy vector as fallback
                    logger.warning(f"No matches found with test query embedding in attempt {attempt}")
                    
                    # Try with a dummy vector
                    dummy_vector = [0.1] * 1024  # Create a dummy vector with 1024 dimensions
                    
                    dummy_response = self.index.query(
                        vector=dummy_vector,
                        top_k=100,  # Use a larger top_k for dummy vector
                        namespace=self.namespace,
                        include_metadata=True
                    )
                    
                    if hasattr(dummy_response, 'matches') and len(dummy_response.matches) > 0:
                        match_count = len(dummy_response.matches)
                        match_ids = [match.id for match in dummy_response.matches]
                        logger.info(f"Dummy query returned {match_count} matches. Vector verification succeeded.")
                        logger.info(f"First few match IDs: {match_ids[:5]}")
                        return True
                    
                    logger.warning(f"No matches found with dummy vector in attempt {attempt}")
                    
                    # Exponential backoff with jitter
                    current_delay = min(current_delay * 1.5, max_delay)
                    jitter = current_delay * 0.2 * random.random()
                    logger.info(f"Waiting {current_delay + jitter:.2f}s before next attempt...")
                    await asyncio.sleep(current_delay + jitter)
                    
                except Exception as e:
                    logger.error(f"Error during query verification: {str(e)}")
                    # Exponential backoff with jitter
                    current_delay = min(current_delay * 1.5, max_delay)
                    jitter = current_delay * 0.2 * random.random()
                    await asyncio.sleep(current_delay + jitter)
            
            logger.error(f"Could not verify vectors are queryable after {max_attempts} attempts")
            return False
            
        except Exception as e:
            logger.error(f"Error generating test query embedding: {str(e)}")
            return False
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """
        Add documents to retrieval indices
        
        Returns:
            bool: True if documents were successfully added and verified as searchable,
                  False otherwise
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Prepare documents for BM25 and TF-IDF
            texts = [chunk.content for chunk in chunks]
            self.documents.extend(chunks)
            
            # Update BM25
            tokenized_texts = [text.split() for text in texts]
            if self.bm25 is None:
                self.bm25 = BM25Okapi(tokenized_texts)
            else:
                self.bm25 = BM25Okapi(
                    [doc.content.split() for doc in self.documents]
                )
            
            # Update TF-IDF
            # Fix for "truth value of an array" error - use proper check for sparse matrix
            if self.tfidf_matrix is None:
                self.tfidf_matrix = self.tfidf.fit_transform(texts)
            else:
                self.tfidf_matrix = self.tfidf.fit_transform(
                    [doc.content for doc in self.documents]
                )
            
            # Generate embeddings using Pinecone's llama-text-embed-v2
            logger.info(f"Generating embeddings for {len(texts)} chunks using {self.embedding_service.model}")
            embed_start_time = asyncio.get_event_loop().time()
            embeddings = await self.embedding_service.generate_embeddings(texts)
            embed_time = asyncio.get_event_loop().time() - embed_start_time
            logger.info(f"Embedding generation completed in {embed_time:.2f}s")
            
            # Add to Pinecone
            vectors = []
            for i, chunk in enumerate(chunks):
                vector = {
                    "id": str(chunk.id),
                    "values": embeddings[i],
                    "metadata": {
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content[:1000]  # First 1000 chars for preview
                    }
                }
                vectors.append(vector)
            
            # Upsert in batches of 100
            batch_size = 100
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone namespace: {self.namespace}")
            upsert_start_time = asyncio.get_event_loop().time()
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                batch_start_time = asyncio.get_event_loop().time()
                response = self.index.upsert(vectors=batch, namespace=self.namespace)
                batch_time = asyncio.get_event_loop().time() - batch_start_time
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1} in {batch_time:.2f}s with response: {response}")
            
            upsert_total_time = asyncio.get_event_loop().time() - upsert_start_time
            logger.info(f"Total upsert time: {upsert_total_time:.2f}s")
            
            # Add a much longer delay after upserting to ensure Pinecone index is updated
            logger.info("Adding a much longer delay after upserting to ensure index is updated...")
            await asyncio.sleep(15.0)  # Increased from 10.0 to 15.0 seconds
            
            # Verify vectors are searchable
            if vectors:
                logger.info("Verifying vectors are searchable...")
                sample_id = vectors[0]["id"]
                
                # This is the critical change - we now wait for vectors to be truly searchable
                # and return the verification result
                is_searchable = await self.verify_vectors_searchable(sample_id)
                
                if is_searchable:
                    logger.info("Vectors verified as searchable!")
                else:
                    logger.warning("Vectors were not verified as searchable after multiple attempts")
                    
                total_time = asyncio.get_event_loop().time() - start_time
                logger.info(f"Added {len(chunks)} documents to indices in namespace: {self.namespace} in {total_time:.2f}s")
                
                return is_searchable
            
            # If no vectors were added, consider it a failure
            logger.warning("No vectors were added to Pinecone")
            return False
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
            
    def _adapt_dimensions(self, embedding, target_dim=1024):
        """Adapt embedding dimensions to match Pinecone index"""
        current_dim = len(embedding)
        
        if current_dim == target_dim:
            return embedding
            
        if current_dim > target_dim:
            # Truncate
            return embedding[:target_dim]
        else:
            # Pad with zeros
            padding = np.zeros(target_dim - current_dim)
            return np.concatenate([embedding, padding])
    
    async def vector_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Perform vector search"""
        try:
            # Generate query embedding using Pinecone's llama-text-embed-v2
            logger.info(f"Generating embedding for query: '{query}' using {self.embedding_service.model}")
            start_time = asyncio.get_event_loop().time()
            query_embeddings = await self.embedding_service.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            embedding_time = asyncio.get_event_loop().time() - start_time
            
            # Log embedding dimensions and generation time
            logger.info(f"Query embedding dimensions: {len(query_embedding)}, generation time: {embedding_time:.2f}s")
            
            # Log first few values of the embedding for debugging
            embedding_preview = [f"{val:.4f}" for val in query_embedding[:5]]
            logger.info(f"Embedding preview (first 5 values): {embedding_preview}")
            
            # Log namespace being queried
            logger.info(f"Querying Pinecone namespace: {self.namespace} with top_k={top_k}")
            
            # Search Pinecone with timing
            start_time = asyncio.get_event_loop().time()
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            search_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Pinecone search completed in {search_time:.2f}s")
            
            # Log raw Pinecone response in a more readable format
            match_count = len(results.matches) if hasattr(results, 'matches') else 0
            logger.info(f"Pinecone returned {match_count} matches")
            
            if match_count > 0:
                # Log the first few matches for debugging
                for i, match in enumerate(results.matches[:3]):  # Log first 3 matches
                    logger.info(f"Match {i+1}: ID={match.id}, Score={match.score:.4f}")
                    if hasattr(match, 'metadata') and match.metadata:
                        content_preview = match.metadata.get('content', '')[:100]
                        logger.info(f"Content preview: {content_preview}...")
            
            # Check if we have matches
            if not results.matches:
                logger.warning(f"No matches found in Pinecone namespace: {self.namespace}")
                
                # Try a query with a dummy vector as a test
                logger.info("Attempting a test query with a dummy vector to check if index has any vectors...")
                dummy_vector = [0.1] * 1024  # Create a dummy vector with 1024 dimensions
                
                test_results = self.index.query(
                    vector=dummy_vector,
                    top_k=100,  # Use a large top_k to increase chances of finding vectors
                    namespace=self.namespace,
                    include_metadata=True
                )
                
                test_match_count = len(test_results.matches) if hasattr(test_results, 'matches') else 0
                if test_match_count > 0:
                    logger.info(f"Test query found {test_match_count} matches, suggesting the index has vectors but they don't match the query")
                    match_ids = [match.id for match in test_results.matches[:5]]
                    logger.info(f"First few match IDs from test query: {match_ids}")
                else:
                    logger.warning(f"Test query also found no matches. The namespace {self.namespace} may be empty or not yet indexed")
                
                return []
            
            # Convert to SearchResults
            search_results = []
            for match in results.matches:
                # Find corresponding chunk
                chunk = next(
                    (c for c in self.documents if str(c.id) == match.id),
                    None
                )
                
                # If chunk not found in local documents but we have metadata, create a temporary chunk
                if not chunk and match.metadata and 'content' in match.metadata:
                    try:
                        # Generate deterministic UUIDs from strings with error handling
                        try:
                            chunk_id = uuid.uuid5(uuid.NAMESPACE_OID, match.id)
                        except Exception as uuid_err:
                            logger.warning(f"Error creating UUID from match ID: {str(uuid_err)}")
                            # Fallback to a new UUID
                            chunk_id = uuid.uuid4()
                            
                        doc_id_str = match.metadata.get('document_id', str(uuid.uuid4()))
                        try:
                            document_id = uuid.uuid5(uuid.NAMESPACE_OID, doc_id_str)
                        except Exception as uuid_err:
                            logger.warning(f"Error creating UUID from document ID: {str(uuid_err)}")
                            # Fallback to a new UUID
                            document_id = uuid.uuid4()
                        
                        # Create a temporary DocumentChunk from metadata
                        chunk = DocumentChunk(
                            id=chunk_id,
                            document_id=document_id,
                            content=match.metadata.get('content', ''),
                            metadata={
                                'source': match.metadata.get('document_id', 'unknown'),
                                'chunk_index': match.metadata.get('chunk_index', 0)
                            },
                            chunk_index=match.metadata.get('chunk_index', 0)
                        )
                        # Add to local documents for future use
                        self.documents.append(chunk)
                        logger.info(f"Created temporary chunk from metadata for ID: {match.id}")
                    except Exception as e:
                        logger.error(f"Error creating temporary chunk: {str(e)}")
                        continue
                
                if chunk:
                    result = SearchResult(
                        chunk=chunk,
                        vector_score=float(match.score),
                        bm25_score=None,
                        tfidf_score=None,
                        combined_score=float(match.score)
                    )
                    search_results.append(result)
            
            # If no chunks were found in our local documents, log a warning
            if not search_results:
                logger.warning("Matches found in Pinecone but no corresponding chunks in local documents")
                
            return search_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            raise
            
    async def bm25_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Perform BM25 search"""
        try:
            # Check if BM25 is initialized
            if self.bm25 is None or not self.documents:
                logger.warning("BM25 not initialized or no documents available")
                return []
                
            # Tokenize query
            tokenized_query = query.split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top k indices
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            # Convert to SearchResults
            search_results = []
            for idx in top_indices:
                result = SearchResult(
                    chunk=self.documents[idx],
                    vector_score=None,
                    bm25_score=float(scores[idx]),
                    tfidf_score=None,
                    combined_score=float(scores[idx])
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            raise
            
    async def tfidf_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Perform TF-IDF search"""
        try:
            # Check if TF-IDF matrix is initialized
            if self.tfidf_matrix is None or not self.documents:
                logger.warning("TF-IDF matrix not initialized or no documents available")
                return []
                
            # Transform query
            query_vector = self.tfidf.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(
                query_vector,
                self.tfidf_matrix
            ).flatten()
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Convert to SearchResults
            search_results = []
            for idx in top_indices:
                result = SearchResult(
                    chunk=self.documents[idx],
                    vector_score=None,
                    bm25_score=None,
                    tfidf_score=float(similarities[idx]),
                    combined_score=float(similarities[idx])
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {str(e)}")
            raise
            
    def normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores to [0, 1] range"""
        if not results:
            return results
            
        # Get score lists
        vector_scores = [r.vector_score for r in results if r.vector_score is not None]
        bm25_scores = [r.bm25_score for r in results if r.bm25_score is not None]
        tfidf_scores = [r.tfidf_score for r in results if r.tfidf_score is not None]
        
        # Normalize each score type
        for result in results:
            if result.vector_score is not None and vector_scores:
                min_score = min(vector_scores)
                max_score = max(vector_scores)
                if max_score == min_score:
                    result.vector_score = 1.0  # All scores are equal, so normalize to 1.0
                else:
                    result.vector_score = (result.vector_score - min_score) / (max_score - min_score)
                    
            if result.bm25_score is not None and bm25_scores:
                min_score = min(bm25_scores)
                max_score = max(bm25_scores)
                if max_score == min_score:
                    result.bm25_score = 1.0  # All scores are equal, so normalize to 1.0
                else:
                    result.bm25_score = (result.bm25_score - min_score) / (max_score - min_score)
                    
            if result.tfidf_score is not None and tfidf_scores:
                min_score = min(tfidf_scores)
                max_score = max(tfidf_scores)
                if max_score == min_score:
                    result.tfidf_score = 1.0  # All scores are equal, so normalize to 1.0
                else:
                    result.tfidf_score = (result.tfidf_score - min_score) / (max_score - min_score)
        
        return results
        
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector, BM25, and TF-IDF"""
        try:
            # Ensure initialization is complete
            if not self.initialized:
                logger.info("HybridRetriever not initialized, initializing now...")
                await self.initialize()
                # Add a much longer delay after initialization to ensure Pinecone index is ready
                logger.info("Adding a longer delay after initialization to ensure index is ready...")
                await asyncio.sleep(7.0)  # Increased from 3.0 to 7.0 seconds
                
            # Check cache
            if use_cache and self.redis:
                cache_key = f"{self.cache_prefix}hybrid_search:{query}:{top_k}"
                cached = await self.redis.get(cache_key)
                if cached:
                    try:
                        return [SearchResult(**r) for r in json.loads(cached)]
                    except Exception as e:
                        logger.error(f"Error deserializing cached results: {str(e)}")
                        # Continue with fresh search
            
            logger.info(f"Performing hybrid search with query: '{query}' in namespace: {self.namespace}")
            
            # Perform searches
            vector_results = await self.vector_search(query, top_k)
            bm25_results = await self.bm25_search(query, top_k)
            tfidf_results = await self.tfidf_search(query, top_k)
            
            logger.info(f"Search results - Vector: {len(vector_results)}, BM25: {len(bm25_results)}, TF-IDF: {len(tfidf_results)}")
            
            # Normalize scores
            vector_results = self.normalize_scores(vector_results)
            bm25_results = self.normalize_scores(bm25_results)
            tfidf_results = self.normalize_scores(tfidf_results)
            
            # Combine results
            chunk_scores = {}
            for result in vector_results + bm25_results + tfidf_results:
                chunk_id = str(result.chunk.id)
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        "chunk": result.chunk,
                        "vector_score": 0.0,
                        "bm25_score": 0.0,
                        "tfidf_score": 0.0
                    }
                if result.vector_score is not None:
                    chunk_scores[chunk_id]["vector_score"] = result.vector_score
                if result.bm25_score is not None:
                    chunk_scores[chunk_id]["bm25_score"] = result.bm25_score
                if result.tfidf_score is not None:
                    chunk_scores[chunk_id]["tfidf_score"] = result.tfidf_score
            
            # Calculate combined scores
            final_results = []
            for chunk_id, scores in chunk_scores.items():
                combined_score = (
                    self.vector_weight * scores["vector_score"] +
                    self.bm25_weight * scores["bm25_score"] +
                    self.tfidf_weight * scores["tfidf_score"]
                )
                result = SearchResult(
                    chunk=scores["chunk"],
                    vector_score=scores["vector_score"],
                    bm25_score=scores["bm25_score"],
                    tfidf_score=scores["tfidf_score"],
                    combined_score=combined_score
                )
                final_results.append(result)
            
            # Sort by combined score
            final_results.sort(key=lambda x: x.combined_score, reverse=True)
            final_results = final_results[:top_k]
            
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
            logger.error(f"Error in hybrid search: {str(e)}")
            raise

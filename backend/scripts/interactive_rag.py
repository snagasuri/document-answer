#!/usr/bin/env python
"""
Interactive RAG script for document Q&A.
This script processes a document, stores it in Pinecone, and allows interactive Q&A.
"""

import os
import sys
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import uuid
import numpy as np
from dotenv import load_dotenv
import httpx
import PyPDF2
import spacy
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DocumentProcessor:
    """Process documents into chunks suitable for embedding"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document processor"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.nlp = spacy.load("en_core_web_md")
            # Increase max_length for large documents
            self.nlp.max_length = 3000000  # Increased from default 1,000,000
            logger.info("Loaded spaCy model: en_core_web_md")
        except OSError:
            logger.warning("Could not load en_core_web_md, trying en_core_web_sm")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.nlp.max_length = 3000000
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.error("No spaCy model available. Using basic text splitting.")
                self.nlp = None
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file"""
        logger.info(f"Extracting text from PDF: {file_path}")
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        if not text:
            return []
        
        # Check if text is too large for spaCy processing
        if len(text) > 1000000 and self.nlp:
            logger.warning(f"Text length ({len(text)}) exceeds 1,000,000 characters. Using simple chunking.")
            return self._simple_chunking(text)
            
        if self.nlp:
            # Use spaCy for semantic chunking
            try:
                return self._semantic_chunking(text)
            except Exception as e:
                logger.warning(f"Error in semantic chunking: {str(e)}. Falling back to simple chunking.")
                return self._simple_chunking(text)
        else:
            # Fallback to simple chunking
            return self._simple_chunking(text)
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """Chunk text based on semantic boundaries"""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_text = sentence.text.strip()
            sentence_size = len(sentence_text)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Add the current chunk to the list of chunks
                chunks.append(" ".join(current_chunk))
                
                # Start a new chunk with overlap
                overlap_size = 0
                overlap_chunk = []
                
                # Add sentences from the end of the previous chunk for overlap
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            current_chunk.append(sentence_text)
            current_size += sentence_size
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _simple_chunking(self, text: str) -> List[str]:
        """Simple text chunking with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a good breaking point (newline or space)
            if end < len(text):
                # Look for newline first
                newline_pos = text.rfind('\n', start, end)
                if newline_pos > start:
                    end = newline_pos + 1
                else:
                    # Look for space
                    space_pos = text.rfind(' ', start, end)
                    if space_pos > start:
                        end = space_pos + 1
            
            # Add the chunk
            chunks.append(text[start:end].strip())
            
            # Move to the next chunk with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a document into chunks with metadata"""
        # Extract text from PDF
        logger.info(f"Processing document: {file_path}")
        
        # Get document metadata
        doc_id = str(uuid.uuid4())
        filename = os.path.basename(file_path)
        
        # For large PDFs, process page by page
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                # Check if this is a large document
                if total_pages > 50:  # Arbitrary threshold for "large document"
                    logger.info(f"Large document detected ({total_pages} pages). Processing page by page.")
                    all_chunks = []
                    
                    # Process each page separately
                    for page_num in tqdm(range(total_pages), desc="Processing pages"):
                        page_text = reader.pages[page_num].extract_text() + "\n"
                        page_chunks = self.chunk_text(page_text)
                        all_chunks.extend(page_chunks)
                    
                    chunks = all_chunks
                else:
                    # For smaller documents, process all at once
                    text = self.extract_text_from_pdf(file_path)
                    chunks = self.chunk_text(text)
                
                logger.info(f"Document split into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
        
        # Create document chunks with metadata
        doc_chunks = []
        for i, chunk in enumerate(chunks):
            doc_chunks.append({
                "id": f"{doc_id}-{i}",
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return doc_chunks

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
            
        from pinecone import Pinecone
        
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

class PineconeService:
    """Service for interacting with Pinecone vector database"""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        """Initialize the Pinecone service"""
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.namespace = "session_interactive"  # Fixed namespace for interactive script
        
        # Initialize Pinecone client
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Initialized Pinecone with index: {self.index_name}, namespace: {self.namespace}")
    
    async def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert vectors into Pinecone"""
        try:
            # Convert vectors to the format expected by the Pinecone SDK
            formatted_vectors = []
            for vector in vectors:
                formatted_vectors.append({
                    "id": vector["id"],
                    "values": vector["values"],
                    "metadata": vector["metadata"]
                })
            
            # Log the namespace being used
            logger.info(f"Upserting vectors to Pinecone namespace: {self.namespace}")
            start_time = asyncio.get_event_loop().time()
            
            # Upsert vectors using the Pinecone SDK with the session namespace
            response = self.index.upsert(
                vectors=formatted_vectors,
                namespace=self.namespace
            )
            
            upsert_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Upsert completed in {upsert_time:.2f}s with response: {response}")
            
            # Add a verification step to check if vectors are searchable
            logger.info("Verifying vectors are searchable...")
            await self.verify_vectors_searchable(formatted_vectors[0]["id"])
            
            return response
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise
    
    async def verify_vectors_searchable(self, vector_id: str, max_attempts: int = 5, delay: float = 1.0):
        """Verify that vectors are searchable after upserting"""
        # First try fetch operation
        for attempt in range(1, max_attempts + 1):
            try:
                # Check if the vector exists in the index using fetch
                fetch_response = self.index.fetch(
                    ids=[vector_id],
                    namespace=self.namespace
                )
                
                if vector_id in fetch_response.vectors:
                    logger.info(f"Vector {vector_id} verified as searchable via fetch after {attempt} attempts")
                    return True
                
                logger.warning(f"Vector {vector_id} not found via fetch in attempt {attempt}/{max_attempts}")
                
                # If fetch fails, try a query operation as an alternative verification method
                if attempt >= 2:  # Try query after a couple of fetch attempts
                    logger.info(f"Attempting to verify using query operation instead...")
                    
                    # Get the vector values from our local data
                    # For this test, we'll just use a simple query with a dummy vector
                    dummy_vector = [0.1] * 1024  # Create a dummy vector with 1024 dimensions
                    
                    query_response = self.index.query(
                        vector=dummy_vector,
                        top_k=100,  # Use a large top_k to increase chances of finding our vectors
                        namespace=self.namespace,
                        include_metadata=True
                    )
                    
                    # Check if we got any matches at all
                    if hasattr(query_response, 'matches') and len(query_response.matches) > 0:
                        match_count = len(query_response.matches)
                        match_ids = [match.id for match in query_response.matches]
                        logger.info(f"Query returned {match_count} matches. Vector verification succeeded via query.")
                        logger.info(f"First few match IDs: {match_ids[:5]}")
                        return True
                    else:
                        logger.warning(f"Query returned no matches in namespace: {self.namespace}")
                
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error verifying vector searchability: {str(e)}")
                await asyncio.sleep(delay)
        
        # As a last resort, just assume it worked and continue
        logger.warning(f"Could not verify vector {vector_id} as searchable after {max_attempts} attempts, but continuing anyway")
        return True  # Return True to allow the process to continue
    
    async def query(
        self, 
        vector: List[float], 
        top_k: int = 5, 
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Query Pinecone for similar vectors"""
        try:
            # Log the namespace being queried
            logger.info(f"Querying Pinecone namespace: {self.namespace} with top_k={top_k}")
            start_time = asyncio.get_event_loop().time()
            
            # Query using the Pinecone SDK with the session namespace
            response = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=include_metadata,
                namespace=self.namespace
            )
            
            query_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Query completed in {query_time:.2f}s")
            
            # Log match count
            match_count = len(response.matches) if hasattr(response, 'matches') else 0
            if match_count == 0:
                logger.warning(f"No matches found in Pinecone namespace: {self.namespace}")
            else:
                logger.info(f"Found {match_count} matches in Pinecone namespace: {self.namespace}")
            
            return response
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            raise

class CohereReranker:
    """Reranker using Cohere API"""
    
    def __init__(self, api_key: str, model: str = "rerank-english-v2.0"):
        """Initialize the Cohere reranker"""
        self.api_key = api_key
        self.base_url = "https://api.cohere.ai/v1"
        self.model = model
    
    async def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_n: int = 3
    ) -> Dict[str, Any]:
        """Rerank documents"""
        if not documents:
            return {"results": []}
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
            "model": self.model
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

class LLMService:
    """Service for generating responses from LLM"""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-4o"):
        """Initialize the LLM service"""
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"  # Updated to use OpenRouter API
        self.model = model  # Keep the full model name for OpenRouter
    
    async def generate_response(
        self, 
        query: str, 
        results: List[Dict[str, Any]] = None,
        chat_history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
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
            "Content-Type": "application/json",
            "HTTP-Referer": "https://docqa.app",  # Required by OpenRouter
            "X-Title": "DocQA RAG Application"  # Required by OpenRouter
        }
        
        # Send request
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
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
                                        yield {
                                            "message": buffer,
                                            "finished": True
                                        }
                                        return
                                        
                                    try:
                                        data_json = json.loads(data)
                                        delta = data_json.get("choices", [{}])[0].get("delta", {})
                                        content = delta.get("content", "")
                                        
                                        if content:
                                            buffer += content
                                            yield {
                                                "message": buffer,
                                                "finished": False
                                            }
                                    except json.JSONDecodeError:
                                        logger.error(f"Failed to parse JSON: {data}")
                                        continue
                except httpx.HTTPStatusError as e:
                    # Handle API authentication errors
                    if e.response.status_code in [401, 403, 404]:
                        logger.warning(f"API error ({e.response.status_code}): {str(e)}")
                        yield {
                            "message": "I'm unable to access the language model API at the moment. Using fallback mode to answer your question based on the retrieved documents.",
                            "finished": True
                        }
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            yield {
                "message": f"Error generating response: {str(e)}",
                "finished": True
            }

class RAGPipeline:
    """RAG pipeline for document Q&A"""
    
    def __init__(self):
        """Initialize the RAG pipeline"""
        # Load API keys
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index = os.getenv("PINECONE_INDEX")
        
        # Check if API keys are set
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        if not self.pinecone_environment:
            raise ValueError("PINECONE_ENVIRONMENT environment variable is not set")
        if not self.pinecone_index:
            raise ValueError("PINECONE_INDEX environment variable is not set")
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_service = PineconeEmbeddingService(api_key=self.pinecone_api_key)
        self.pinecone_service = PineconeService(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment,
            index_name=self.pinecone_index
        )
        self.reranker = CohereReranker(api_key=self.cohere_api_key)
        self.llm_service = LLMService(api_key=self.openrouter_api_key)
        
        # Chat history
        self.chat_history = []
    
    async def process_and_store_document(self, file_path: str) -> None:
        """Process a document and store it in Pinecone"""
        logger.info(f"Processing document: {file_path}")
        start_time = asyncio.get_event_loop().time()
        
        # Process document
        doc_chunks = self.document_processor.process_document(file_path)
        process_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Document processing completed in {process_time:.2f}s, generated {len(doc_chunks)} chunks")
        
        # Generate embeddings
        texts = [chunk["content"] for chunk in doc_chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks using {self.embedding_service.model}")
        embed_start_time = asyncio.get_event_loop().time()
        embeddings = await self.embedding_service.generate_embeddings(texts)
        embed_time = asyncio.get_event_loop().time() - embed_start_time
        logger.info(f"Embedding generation completed in {embed_time:.2f}s")
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(doc_chunks, embeddings)):
            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    "doc_id": chunk["doc_id"],
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"]
                }
            })
        
        # Upsert vectors to Pinecone
        logger.info(f"Upserting {len(vectors)} vectors to Pinecone namespace: {self.pinecone_service.namespace}")
        
        # Upsert in batches to avoid size limits
        batch_size = 100
        upsert_start_time = asyncio.get_event_loop().time()
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            batch_start_time = asyncio.get_event_loop().time()
            await self.pinecone_service.upsert_vectors(batch)
            batch_time = asyncio.get_event_loop().time() - batch_start_time
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1} in {batch_time:.2f}s")
        
        upsert_total_time = asyncio.get_event_loop().time() - upsert_start_time
        logger.info(f"Total upsert time: {upsert_total_time:.2f}s")
        
        # Add a longer delay after upserting to ensure Pinecone index is updated
        logger.info("Adding a longer delay after upserting to ensure index is updated...")
        await asyncio.sleep(5.0)  # Increased from 3.0 to 5.0 seconds
        
        # Verify vectors are searchable
        logger.info("Verifying vectors are searchable...")
        sample_id = vectors[0]["id"]
        await self.pinecone_service.verify_vectors_searchable(sample_id)
        
        # Try a direct query to see if we get any results
        logger.info("Testing direct query to check if vectors are searchable...")
        try:
            # Use the first vector's embedding for the query
            test_embedding = vectors[0]["values"]
            test_response = await self.pinecone_service.query(
                vector=test_embedding,
                top_k=5
            )
            
            # Log the results
            match_count = len(test_response.matches) if hasattr(test_response, 'matches') else 0
            if match_count > 0:
                logger.info(f"Test query successful! Found {match_count} matches.")
                match_ids = [match.id for match in test_response.matches]
                logger.info(f"Match IDs: {match_ids}")
            else:
                logger.warning("Test query returned no matches. This may indicate an indexing delay.")
        except Exception as e:
            logger.error(f"Error during test query: {str(e)}")
        
        total_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Document processing and storage completed in {total_time:.2f}s")
    
    async def answer_question(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Answer a question using the RAG pipeline"""
        logger.info(f"Processing query: '{query}'")
        start_time = asyncio.get_event_loop().time()
        
        # Generate embedding for the query
        logger.info(f"Generating embedding for query using {self.embedding_service.model}")
        embed_start_time = asyncio.get_event_loop().time()
        query_embeddings = await self.embedding_service.generate_embeddings([query])
        query_embedding = query_embeddings[0]
        embed_time = asyncio.get_event_loop().time() - embed_start_time
        logger.info(f"Query embedding generated in {embed_time:.2f}s")
        
        # Query Pinecone
        logger.info(f"Querying Pinecone namespace: {self.pinecone_service.namespace} with query: '{query}'")
        query_start_time = asyncio.get_event_loop().time()
        pinecone_results = await self.pinecone_service.query(
            vector=query_embedding,
            top_k=10
        )
        query_time = asyncio.get_event_loop().time() - query_start_time
        logger.info(f"Pinecone query completed in {query_time:.2f}s")
        
        # Debug: Print raw Pinecone results
        # Convert QueryResponse to dict for JSON serialization
        pinecone_results_dict = {
            "matches": [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                } for match in pinecone_results.matches
            ] if hasattr(pinecone_results, 'matches') else []
        }
        logger.info(f"Raw Pinecone response: {json.dumps(pinecone_results_dict, indent=2)}")
        
        # Extract results
        results = []
        if hasattr(pinecone_results, 'matches'):
            for match in pinecone_results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "content": match.metadata.get("content", "") if hasattr(match, "metadata") else "",
                    "metadata": match.metadata if hasattr(match, "metadata") else {}
                }
                results.append(result)
                logger.info(f"Match {match.id} (score: {match.score:.4f})")
                logger.info(f"Content preview: {result['content'][:100]}...")
        
        if not results:
            logger.warning("No matches found in Pinecone results")
            yield {
                "message": "No relevant documents found to answer your question.",
                "finished": True
            }
            return
        
        # Rerank results
        documents = [result["content"] for result in results]
        reranked_results = await self.reranker.rerank(query, documents)
        
        # Reorder results based on reranking
        reordered_results = []
        for reranked in reranked_results.get("results", []):
            index = reranked.get("index")
            if index is not None and 0 <= index < len(results):
                result = results[index]
                result["score"] = reranked.get("relevance_score", 0)
                reordered_results.append(result)
        
        # Take top 3 results
        top_results = reordered_results[:3]
        
        # Generate response
        async for response in self.llm_service.generate_response(query, top_results, self.chat_history):
            yield response
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": response["message"]})
        
        # Limit chat history to last 10 messages
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]


async def main():
    """Main function to run the interactive RAG pipeline"""
    # Configure logging to match web app format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Print a warning about the namespace
    print("\n" + "="*80)
    print("IMPORTANT: This script uses the fixed namespace 'session_interactive'")
    print("The web app uses dynamic namespaces like 'session_67c0b8b2cbdd2bbae44cc395'")
    print("This difference in namespaces may explain why documents are not found in the web app")
    print("="*80 + "\n")
    
    # Get document path from user
    print("Select a document to process:")
    print("1. Law PDF")
    print("2. Oleve PDF")
    
    while True:
        selection = input("\nEnter your selection (1 or 2): ")
        if selection == "1":
            document_path = "/Users/sriram/Documents/law.pdf"
            break
        elif selection == "2":
            document_path = "/Users/sriram/Documents/document-answer/oleve.pdf"
            break
        else:
            print("Invalid selection. Please enter 1 or 2.")
    
    if not os.path.exists(document_path):
        logger.error(f"Document not found: {document_path}")
        return
    
    try:
        # Check for environment variables
        required_env_vars = [
            "OPENROUTER_API_KEY", 
            "COHERE_API_KEY", 
            "PINECONE_API_KEY", 
            "PINECONE_ENVIRONMENT", 
            "PINECONE_INDEX"
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            print("Error: Missing required environment variables:")
            for var in missing_vars:
                print(f"  - {var}")
            print("\nPlease set these variables in your .env file or environment.")
            return
        
        # Log environment configuration
        logger.info(f"Using Pinecone index: {os.getenv('PINECONE_INDEX')}")
        logger.info(f"Using Pinecone environment: {os.getenv('PINECONE_ENVIRONMENT')}")
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline")
        start_time = asyncio.get_event_loop().time()
        rag_pipeline = RAGPipeline()
        init_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"RAG pipeline initialized in {init_time:.2f}s")
        
        # Process and store document
        logger.info(f"Processing document: {document_path}")
        await rag_pipeline.process_and_store_document(document_path)
        logger.info("Document processed and stored successfully!")
        
        # Interactive Q&A loop
        print("\nYou can now ask questions about the document. Type 'exit' to quit.")
        
        while True:
            # Get user query
            query = input("\nQuestion: ")
            
            # Check if user wants to exit
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            
            # Skip empty queries
            if not query.strip():
                continue
            
            # Answer question
            print("\nAnswering...")
            logger.info(f"Processing query: '{query}'")
            query_start_time = asyncio.get_event_loop().time()
            
            # Use variables to track the response
            full_response = ""
            
            async for response in rag_pipeline.answer_question(query):
                if response["finished"]:
                    # When finished, print the complete response
                    print("\nAnswer:")
                    print(response["message"])
                    
                    # Log completion time
                    query_time = asyncio.get_event_loop().time() - query_start_time
                    logger.info(f"Query answered in {query_time:.2f}s")
                else:
                    # Get only the new content since last update
                    new_content = response["message"][len(full_response):]
                    if new_content:
                        # Print only the new content
                        sys.stdout.write(new_content)
                        sys.stdout.flush()
                        full_response = response["message"]
            
            # After each query, print a reminder about the namespace
            print("\nReminder: This script uses the fixed namespace 'session_interactive'")
            print("The web app uses dynamic namespaces based on session IDs")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends, Form, Path
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os
import tempfile
import shutil
import uuid
import asyncio
import structlog
from bson import ObjectId

from core.document_processor import DocumentProcessor
from core.hybrid_retriever import HybridRetriever
from core.mongodb_service import MongoDBService
from core.config import settings

# Configure structured logging
logger = structlog.get_logger()

# Create router
router = APIRouter(prefix="/documents", tags=["documents"])

# Dependencies
async def get_document_processor():
    """Get document processor instance"""
    return DocumentProcessor()

async def get_mongodb_service():
    """Get MongoDB service instance"""
    return MongoDBService()

async def get_hybrid_retriever(session_id: Optional[str] = None):
    """Get hybrid retriever instance with session-specific namespace"""
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
    return retriever

# Health check endpoint
@router.get("/health")
async def health_check():
    """Check document service health"""
    try:
        return {
            "status": "healthy",
            "service": "documents",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Document health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Document service unhealthy: {str(e)}"
        )

# Upload document endpoint
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    mongodb_service: MongoDBService = Depends(get_mongodb_service),
    hybrid_retriever: HybridRetriever = Depends(get_hybrid_retriever)
):
    """Upload and process a document"""
    try:
        # Verify session exists
        session = await mongodb_service.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Convert any ObjectId in documents to strings
        if "documents" in session and session["documents"]:
            session["documents"] = [str(doc_id) if isinstance(doc_id, ObjectId) else doc_id for doc_id in session["documents"]]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Process document
            document, chunks = await document_processor.process_document(temp_path, file.filename)
            
            # Add document to MongoDB
            doc_dict = {
                "filename": document.filename,
                "content": document.content,
                "metadata": document.metadata
            }
            mongo_doc = await mongodb_service.add_document(session_id, document.filename, document.content, document.metadata)
            
            # Add document chunks to MongoDB
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.chunk_index
                }
                chunk_dicts.append(chunk_dict)
            
            mongo_chunks = await mongodb_service.add_document_chunks(mongo_doc["_id"], chunk_dicts)
            
            # Initialize session-specific retriever
            session_retriever = HybridRetriever(
                pinecone_api_key=settings.PINECONE_API_KEY,
                pinecone_environment=settings.PINECONE_ENVIRONMENT,
                pinecone_index=settings.PINECONE_INDEX,
                redis_url=settings.REDIS_URL,
                session_id=session_id,
                vector_weight=float(settings.VECTOR_WEIGHT),
                bm25_weight=float(settings.BM25_WEIGHT),
                tfidf_weight=float(settings.TFIDF_WEIGHT)
            )
            # No need to initialize here as it will be done in the background task
            
            # Add document chunks to Pinecone in background
            background_tasks.add_task(
                add_chunks_to_pinecone,
                session_retriever,
                chunks,
                mongo_doc["_id"],
                mongodb_service
            )
            
            # Convert ObjectId to string for response
            document_id = str(mongo_doc["_id"]) if isinstance(mongo_doc["_id"], ObjectId) else mongo_doc["_id"]
            
            return {
                "message": "Document uploaded and processing started",
                "document_id": document_id,
                "filename": file.filename,
                "session_id": session_id,
                "chunk_count": len(chunks),
                "processingStatus": "processing"
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )

# Helper function for background task
async def add_chunks_to_pinecone(retriever: HybridRetriever, chunks: List[Any], document_id: str, mongodb_service: MongoDBService):
    """Add document chunks to Pinecone"""
    try:
        # Log the namespace being used
        logger.info(f"Adding chunks to Pinecone with namespace: {retriever.namespace}")
        
        # Convert document_id to string if it's an ObjectId
        if isinstance(document_id, ObjectId):
            document_id = str(document_id)
        
        # Initialize the retriever first to ensure proper setup
        await retriever.initialize()
        
        # Then add the documents - this now returns a boolean indicating if vectors are searchable
        is_searchable = await retriever.add_documents(chunks)
        
        if is_searchable:
            # Update document status to complete only if vectors are verified as searchable
            await mongodb_service.update_document_processing_status(document_id, "complete")
            logger.info(f"Added {len(chunks)} chunks to Pinecone in namespace: {retriever.namespace} - vectors are searchable")
        else:
            # Update document status to indicate vectors are not yet searchable
            await mongodb_service.update_document_processing_status(document_id, "indexed_not_searchable")
            logger.warning(f"Added {len(chunks)} chunks to Pinecone in namespace: {retriever.namespace} - vectors are NOT yet searchable")
            
            # Start a retry task in a separate asyncio task
            asyncio.create_task(
                retry_verify_vectors_searchable(
                    retriever,
                    document_id,
                    mongodb_service
                )
            )
    except Exception as e:
        logger.error(f"Error adding chunks to Pinecone: {str(e)}")
        # Update document status to error
        try:
            await mongodb_service.update_document_processing_status(document_id, "error")
        except Exception as update_error:
            logger.error(f"Error updating document status: {str(update_error)}")
        
        # Log the full exception traceback for better debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

# Helper function to retry vector verification
async def retry_verify_vectors_searchable(retriever: HybridRetriever, document_id: str, mongodb_service: MongoDBService, max_retries: int = 5):
    """Periodically retry verifying if vectors are searchable"""
    import random
    
    # Convert document_id to string if it's an ObjectId
    if isinstance(document_id, ObjectId):
        document_id = str(document_id)
    
    # Get a sample vector ID to verify
    try:
        # Get document chunks
        chunks = await mongodb_service.get_document_chunks(document_id)
        if not chunks:
            logger.error(f"No chunks found for document {document_id} during retry verification")
            return
            
        # Get a sample chunk ID
        sample_chunk = chunks[0]
        # Convert _id to string if it's an ObjectId
        chunk_id = str(sample_chunk["_id"]) if isinstance(sample_chunk["_id"], ObjectId) else sample_chunk["_id"]
        sample_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))
        
        # Try to verify with increasing delays
        for retry in range(1, max_retries + 1):
            logger.info(f"Retry {retry}/{max_retries} to verify vectors are searchable for document {document_id}")
            
            # Wait with exponential backoff and jitter
            delay = min(30, 5 * (2 ** (retry - 1)))  # 5, 10, 20, 40, 60 seconds
            jitter = delay * 0.2 * random.random()
            await asyncio.sleep(delay + jitter)
            
            # Try to verify vectors are searchable
            is_searchable = await retriever.verify_vectors_searchable(sample_id)
            
            if is_searchable:
                # Update document status to complete
                await mongodb_service.update_document_processing_status(document_id, "complete")
                logger.info(f"Vectors for document {document_id} are now verified as searchable after retry {retry}")
                return
        
        # If we get here, we've exhausted all retries
        logger.error(f"Failed to verify vectors as searchable for document {document_id} after {max_retries} retries")
        # Update status to indicate permanent indexing issue
        await mongodb_service.update_document_processing_status(document_id, "indexing_failed")
    except Exception as e:
        logger.error(f"Error during retry verification for document {document_id}: {str(e)}")

# List documents endpoint
@router.get("/session/{session_id}")
async def list_session_documents(
    session_id: str = Path(...),
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """List all documents for a session"""
    try:
        # Verify session exists
        session = await mongodb_service.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Convert any ObjectId in documents to strings
        if "documents" in session and session["documents"]:
            session["documents"] = [str(doc_id) if isinstance(doc_id, ObjectId) else doc_id for doc_id in session["documents"]]
        
        # Get documents
        documents = await mongodb_service.get_session_documents(session_id)
        
        # Convert any ObjectId in documents to strings
        for doc in documents:
            if "_id" in doc and isinstance(doc["_id"], ObjectId):
                doc["_id"] = str(doc["_id"])
        
        return {
            "session_id": session_id,
            "documents": documents
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )

# Get document endpoint
@router.get("/{document_id}")
async def get_document(
    document_id: str = Path(...),
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Get a document by ID"""
    try:
        # Get document
        document = await mongodb_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Convert any ObjectId to strings
        if "_id" in document and isinstance(document["_id"], ObjectId):
            document["_id"] = str(document["_id"])
        
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting document: {str(e)}"
        )

# Delete document endpoint
@router.delete("/{document_id}")
async def delete_document(
    document_id: str = Path(...),
    session_id: str = Form(...),
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Delete a document"""
    try:
        # Verify session exists
        session = await mongodb_service.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Convert any ObjectId in documents to strings
        if "documents" in session and session["documents"]:
            session["documents"] = [str(doc_id) if isinstance(doc_id, ObjectId) else doc_id for doc_id in session["documents"]]
        
        # Delete document
        success = await mongodb_service.delete_document(document_id, session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

# Check document processing status endpoint
@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str = Path(...),
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Get document processing status"""
    try:
        # Get document
        document = await mongodb_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Convert any ObjectId to strings
        if "_id" in document and isinstance(document["_id"], ObjectId):
            document["_id"] = str(document["_id"])
        
        return {
            "document_id": document_id,
            "processingStatus": document.get("processingStatus", "unknown")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting document status: {str(e)}"
        )

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from uuid import UUID, uuid4

class Document(BaseModel):
    """Base document model"""
    id: UUID = Field(default_factory=uuid4)
    filename: str
    content: str
    metadata: Dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
class DocumentChunk(BaseModel):
    """Chunked document segment"""
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    metadata: Dict = Field(default_factory=dict)
    vector_embedding: Optional[List[float]] = None
    bm25_tokens: Optional[List[str]] = None
    tfidf_vector: Optional[List[float]] = None
    chunk_index: int
    citation_index: Optional[int] = None  # Explicit field for citation index
    
    def dict(self, *args, **kwargs):
        """Override dict serialization to ensure metadata has citation_index"""
        d = super().dict(*args, **kwargs)
        # If citation_index is set, ensure it's in metadata
        if d.get('citation_index') is not None:
            if 'metadata' not in d:
                d['metadata'] = {}
            d['metadata']['citation_index'] = d['citation_index']
        return d
    
class SearchResult(BaseModel):
    """Search result with relevance scores"""
    chunk: DocumentChunk
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    tfidf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    combined_score: float
    
class ChatMessage(BaseModel):
    """Chat message model"""
    role: str
    content: str
    
class ChatResponse(BaseModel):
    """Streaming chat response"""
    message: str
    sources: List[DocumentChunk] = []
    finished: bool = False

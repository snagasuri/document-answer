import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import spacy
from PyPDF2 import PdfReader
import uuid
from datetime import datetime

from models.document import Document, DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor"""
        try:
            # Load spaCy model for text processing
            # Using medium model for better accuracy in entity recognition and chunking
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            # Fallback to small model if medium model is not installed
            logger.warning("en_core_web_md not found. Falling back to en_core_web_sm. "
                          "For better results, run: python -m spacy download en_core_web_md")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("No spaCy model found. Please run: python -m spacy download en_core_web_md")
                raise
        
        # Configure chunking parameters
        self.chunk_size = 500
        self.chunk_overlap = 100
        
    async def process_pdf(self, file_path: str, filename: str) -> Document:
        """Process PDF document"""
        try:
            # Read PDF
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                text = ""
                metadata = {}
                
                # Extract metadata
                if pdf.metadata:
                    metadata = {
                        "title": pdf.metadata.get("/Title", ""),
                        "author": pdf.metadata.get("/Author", ""),
                        "subject": pdf.metadata.get("/Subject", ""),
                        "keywords": pdf.metadata.get("/Keywords", ""),
                        "created": pdf.metadata.get("/CreationDate", ""),
                        "modified": pdf.metadata.get("/ModDate", ""),
                        "pages": len(pdf.pages)
                    }
                
                # Extract text with layout preservation
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
                    
                # Create document
                document = Document(
                    id=uuid.uuid4(),
                    filename=filename,
                    content=text,
                    metadata={
                        **metadata,
                        "source": "pdf",
                        "processed_at": datetime.utcnow().isoformat()
                    }
                )
                
                return document
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
            
    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract additional metadata from document content"""
        try:
            # Process text with spaCy
            doc = self.nlp(document.content[:10000])  # Process first 10K chars for efficiency
            
            # Extract entities
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]
            
            # Extract key phrases (noun chunks)
            key_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Detect language
            language = doc.lang_
            
            # Basic text statistics
            stats = {
                "chars": len(document.content),
                "words": len(document.content.split()),
                "sentences": len(list(doc.sents))
            }
            
            metadata = {
                "entities": entities[:100],  # Limit to top 100 entities
                "key_phrases": key_phrases[:100],  # Limit to top 100 phrases
                "language": language,
                "statistics": stats
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            raise
            
    def create_chunks(self, document: Document) -> List[DocumentChunk]:
        """Create overlapping chunks from document content"""
        try:
            chunks = []
            content = document.content
            
            # Process with spaCy for sentence boundaries
            doc = self.nlp(content)
            sentences = list(doc.sents)
            
            current_chunk = ""
            current_sentences = []
            chunk_index = 0
            
            for sentence in sentences:
                # Add sentence to current chunk
                if len(current_chunk) + len(sentence.text) <= self.chunk_size:
                    current_chunk += sentence.text + " "
                    current_sentences.append(sentence)
                else:
                    # Create chunk
                    if current_chunk:
                        chunk = DocumentChunk(
                            id=uuid.uuid4(),
                            document_id=document.id,
                            content=current_chunk.strip(),
                            chunk_index=chunk_index,
                            metadata={
                                "start_char": current_sentences[0].start_char,
                                "end_char": current_sentences[-1].end_char,
                                "num_sentences": len(current_sentences)
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        
                        # Handle overlap
                        overlap_text = ""
                        overlap_sentences = []
                        for sent in reversed(current_sentences):
                            if len(overlap_text) + len(sent.text) <= self.chunk_overlap:
                                overlap_text = sent.text + " " + overlap_text
                                overlap_sentences.insert(0, sent)
                            else:
                                break
                                
                        current_chunk = overlap_text
                        current_sentences = overlap_sentences
                    
                    # Add current sentence
                    current_chunk += sentence.text + " "
                    current_sentences.append(sentence)
            
            # Add final chunk
            if current_chunk:
                chunk = DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=document.id,
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    metadata={
                        "start_char": current_sentences[0].start_char,
                        "end_char": current_sentences[-1].end_char,
                        "num_sentences": len(current_sentences)
                    }
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise
            
    async def process_document(self, file_path: str, filename: str) -> tuple[Document, List[DocumentChunk]]:
        """Process document end-to-end"""
        try:
            # Process document
            document = await self.process_pdf(file_path, filename)
            
            # Extract metadata
            metadata = self.extract_metadata(document)
            document.metadata.update(metadata)
            
            # Create chunks
            chunks = self.create_chunks(document)
            
            return document, chunks
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}")
            raise

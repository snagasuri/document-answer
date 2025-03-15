import asyncio
import logging
import typer
from pathlib import Path
import json
from datetime import datetime
import uuid
from typing import List, Dict
import random

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.core.config import settings
from backend.scripts.init_db import Document, DocumentChunk, ChatSession, ChatMessage
from backend.scripts.generate_test_data import TestDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer()

class DatabaseSeeder:
    def __init__(self):
        """Initialize seeder"""
        self.engine = create_async_engine(settings.POSTGRES_URL)
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
    async def seed_test_documents(self, num_docs: int = 5):
        """Seed test documents and chunks"""
        try:
            # Generate test data
            test_gen = TestDataGenerator(settings.OPENROUTER_API_KEY)
            topics = [
                {
                    "name": "Machine Learning",
                    "sections": [
                        "Supervised Learning",
                        "Unsupervised Learning",
                        "Model Evaluation"
                    ]
                },
                {
                    "name": "Deep Learning",
                    "sections": [
                        "Neural Networks",
                        "CNN",
                        "RNN",
                        "Transformers"
                    ]
                },
                {
                    "name": "Natural Language Processing",
                    "sections": [
                        "Text Processing",
                        "Word Embeddings",
                        "Language Models"
                    ]
                }
            ]
            
            docs_df, qa_df = await test_gen.create_test_dataset(
                topics=topics[:num_docs],
                output_dir="test_data"
            )
            
            # Create documents and chunks
            async with self.async_session() as session:
                for _, row in docs_df.iterrows():
                    # Create document
                    doc = Document(
                        filename=f"{row['topic']}.pdf",
                        content=row['content'],
                        doc_metadata={
                            "topic": row["topic"],
                            "source": "test_data",
                            "generated": True
                        }
                    )
                    session.add(doc)
                    await session.flush()
                    
                    # Create chunks
                    chunk_size = 500
                    content = row['content']
                    chunks = [
                        content[i:i + chunk_size]
                        for i in range(0, len(content), chunk_size)
                    ]
                    
                    for i, chunk_content in enumerate(chunks):
                        chunk = DocumentChunk(
                            document_id=doc.id,
                            content=chunk_content,
                            chunk_index=i,
                            chunk_metadata={
                                "topic": row["topic"],
                                "chunk_size": len(chunk_content)
                            }
                        )
                        session.add(chunk)
                        
                await session.commit()
                
            logger.info(f"Seeded {len(docs_df)} documents with chunks")
            
        except Exception as e:
            logger.error(f"Error seeding test documents: {str(e)}")
            raise
            
    async def seed_test_chat_sessions(self, num_sessions: int = 3):
        """Seed test chat sessions and messages"""
        try:
            async with self.async_session() as session:
                for _ in range(num_sessions):
                    # Create session
                    chat_session = ChatSession(
                        session_metadata={
                            "source": "test_data",
                            "generated": True
                        }
                    )
                    session.add(chat_session)
                    await session.flush()
                    
                    # Create messages
                    num_messages = random.randint(3, 8)
                    for i in range(num_messages):
                        role = "user" if i % 2 == 0 else "assistant"
                        content = (
                            "What is machine learning?"
                            if role == "user"
                            else "Machine learning is a subset of artificial intelligence..."
                        )
                        
                        message = ChatMessage(
                            session_id=chat_session.id,
                            role=role,
                            content=content,
                            message_metadata={
                                "sequence": i,
                                "generated": True
                            }
                        )
                        session.add(message)
                        
                await session.commit()
                
            logger.info(f"Seeded {num_sessions} chat sessions with messages")
            
        except Exception as e:
            logger.error(f"Error seeding test chat sessions: {str(e)}")
            raise
            
    async def clean_test_data(self):
        """Clean up test data"""
        try:
            async with self.async_session() as session:
                # Delete test documents
                await session.execute(
                    text("""
                        DELETE FROM documents
                        WHERE doc_metadata->>'generated' = 'true'
                    """)
                )
                
                # Delete test chat sessions
                await session.execute(
                    text("""
                        DELETE FROM chat_sessions
                        WHERE session_metadata->>'generated' = 'true'
                    """)
                )
                
                await session.commit()
                
            logger.info("Cleaned up test data")
            
        except Exception as e:
            logger.error(f"Error cleaning test data: {str(e)}")
            raise
            
    async def export_data(self, output_dir: str = "exported_data"):
        """Export database data to JSON files"""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            async with self.async_session() as session:
                # Export documents
                result = await session.execute(
                    text("SELECT * FROM documents")
                )
                documents = result.fetchall()
                
                with open(f"{output_dir}/documents.json", "w") as f:
                    json.dump(
                        [dict(doc._mapping) for doc in documents],
                        f,
                        indent=2,
                        default=str
                    )
                    
                # Export chunks
                result = await session.execute(
                    text("SELECT * FROM document_chunks")
                )
                chunks = result.fetchall()
                
                with open(f"{output_dir}/chunks.json", "w") as f:
                    json.dump(
                        [dict(chunk._mapping) for chunk in chunks],
                        f,
                        indent=2,
                        default=str
                    )
                    
                # Export chat data
                result = await session.execute(
                    text("SELECT * FROM chat_sessions")
                )
                sessions = result.fetchall()
                
                with open(f"{output_dir}/chat_sessions.json", "w") as f:
                    json.dump(
                        [dict(session._mapping) for session in sessions],
                        f,
                        indent=2,
                        default=str
                    )
                    
            logger.info(f"Data exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise

@app.command()
def seed_all():
    """Seed all test data"""
    async def _seed_all():
        seeder = DatabaseSeeder()
        await seeder.seed_test_documents()
        await seeder.seed_test_chat_sessions()
        
    asyncio.run(_seed_all())

@app.command()
def clean_all():
    """Clean up all test data"""
    async def _clean_all():
        seeder = DatabaseSeeder()
        await seeder.clean_test_data()
        
    asyncio.run(_clean_all())

@app.command()
def export_all(output_dir: str = "exported_data"):
    """Export all data"""
    async def _export_all():
        seeder = DatabaseSeeder()
        await seeder.export_data(output_dir)
        
    asyncio.run(_export_all())

if __name__ == "__main__":
    app()

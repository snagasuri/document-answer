import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Integer, JSON, text
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create base model
Base = declarative_base()

class Document(Base):
    """Document model for storing uploaded files"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    doc_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
class DocumentChunk(Base):
    """Chunk model for storing document segments"""
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_metadata = Column(JSON, nullable=False, default=dict)
    chunk_index = Column(Integer, nullable=False)
    vector_id = Column(String)  # Pinecone vector ID
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
class ChatSession(Base):
    """Chat session model"""
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    session_metadata = Column(JSON, nullable=False, default=dict)
    
class ChatMessage(Base):
    """Chat message model"""
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    message_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

async def init_db():
    """Initialize database tables"""
    try:
        # Create async engine
        engine = create_async_engine(
            settings.POSTGRES_URL,
            echo=True
        )
        
        # Don't create tables directly, rely on Alembic migrations
        logger.info("Database engine initialized. Tables should be created using Alembic migrations.")
        
        # Create session factory
        async_session = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        await engine.dispose()

async def create_tables_directly():
    """Create tables directly using SQLAlchemy (fallback method)"""
    try:
        # Create async engine
        engine = create_async_engine(
            settings.POSTGRES_URL,
            echo=True
        )
        
        # Create all tables directly
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Tables created directly using SQLAlchemy")
        
    except Exception as e:
        logger.error(f"Direct table creation failed: {str(e)}")
        raise
    finally:
        await engine.dispose()

async def verify_tables_exist():
    """Verify that required tables exist in the database"""
    try:
        engine = create_async_engine(settings.POSTGRES_URL)
        tables_to_check = ["documents", "document_chunks", "chat_sessions", "chat_messages"]
        missing_tables = []
        
        async with engine.connect() as conn:
            for table in tables_to_check:
                result = await conn.execute(text(f"SELECT to_regclass('public.{table}')"))
                exists = result.scalar()
                if not exists:
                    missing_tables.append(table)
        
        if missing_tables:
            logger.warning(f"Missing tables detected: {', '.join(missing_tables)}")
            return False
        else:
            logger.info("All required tables exist")
            return True
            
    except Exception as e:
        logger.error(f"Table verification failed: {str(e)}")
        return False
    finally:
        await engine.dispose()

async def create_indexes():
    """Create database indexes"""
    try:
        engine = create_async_engine(settings.POSTGRES_URL)
        
        async with engine.begin() as conn:
            # Create indexes for faster queries
            from sqlalchemy import text
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON document_chunks(vector_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON chat_messages(session_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_created_at ON chat_messages(created_at DESC)"))
            
        logger.info("Successfully created database indexes")
        
    except Exception as e:
        logger.error(f"Index creation failed: {str(e)}")
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_db())
    asyncio.run(create_indexes())

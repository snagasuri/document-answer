from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import structlog

from core.config import settings, REDIS_CONFIG, PINECONE_CONFIG, MONGODB_CONFIG
from app.api.v1 import documents, chat

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for connections
redis_client = None
db_session = None
mongodb_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    # Startup
    logger.info("Starting services")
    
    try:
        # Initialize Redis
        global redis_client
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        logger.info("Redis connected")
        
        # Initialize PostgreSQL
        engine = create_async_engine(settings.POSTGRES_URL)
        global db_session
        db_session = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        # Verify connection
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connected")
        
        # Initialize Pinecone (if needed)
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            logger.info("Pinecone initialized")
        except Exception as e:
            logger.warning(f"Pinecone initialization failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down services")
    try:
        if redis_client:
            await redis_client.close()
    except Exception as e:
        logger.error(f"Redis shutdown failed: {str(e)}")

# Create FastAPI app
app = FastAPI(
    title="Document Intelligence RAG API",
    description="Advanced RAG system with hybrid search and reranking",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for logging
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    logger.info(f"Request started: {method} {path}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"Request completed: {method} {path} - {response.status_code} ({duration:.3f}s)")
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Request failed: {method} {path} - {str(e)} ({duration:.3f}s)")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

# Include API routers
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check service health"""
    try:
        # Check Redis
        redis_status = "not_checked"
        if redis_client:
            try:
                await redis_client.ping()
                redis_status = "connected"
            except Exception as e:
                logger.error(f"Redis health check failed: {str(e)}")
                redis_status = "error"
        
        # Check PostgreSQL
        postgres_status = "not_checked"
        try:
            if db_session:
                async with db_session() as session:
                    await session.execute(text("SELECT 1"))
                    postgres_status = "connected"
            else:
                postgres_status = "not_initialized"
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {str(e)}")
            postgres_status = "error"
        
        # Consider the service healthy if at least one service is connected
        is_healthy = redis_status == "connected" or postgres_status == "connected"
        
        return JSONResponse(
            status_code=200 if is_healthy else 503,
            content={
                "status": "healthy" if is_healthy else "unhealthy",
                "services": {
                    "redis": redis_status,
                    "postgresql": postgres_status
                },
                "version": "1.0.0",
                "environment": settings.ENVIRONMENT
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Document Intelligence RAG API"}

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )

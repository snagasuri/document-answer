from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import json

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    OPENROUTER_API_KEY: str
    COHERE_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX: str
    
    # Database
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    
    @property
    def POSTGRES_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # MongoDB
    MONGODB_URI: str = "mongodb+srv://ramnag2003:QnsLyJvMohgPexIF@cluster0.wsbsm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    MONGODB_DB: str = "rag"
    
    # Redis
    REDIS_URL: str
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    REDIS_DB: str = "0"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = '["http://localhost:3000"]'
    DEBUG: str = "false"
    ENVIRONMENT: str = "development"
    
    @property
    def CORS_ORIGINS_LIST(self) -> List[str]:
        return json.loads(self.CORS_ORIGINS)
    
    # Authentication
    CLERK_SECRET_KEY: str = "sk_test_RnS7uV21jNdtqTX3KytGwbpSt2vIG6CeqhWGGFcXAJ"
    CLERK_PUBLISHABLE_KEY: str = "pk_test_c3BlY2lhbC1zaGVwaGVyZC03LmNsZXJrLmFjY291bnRzLmRldiQ"
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    VECTOR_TOP_K: int = 10
    FINAL_TOP_K: int = 3
    CACHE_TTL: int = 3600
    CACHE_MAX_SIZE: str = "1000"
    MODEL_NAME: str = "openai/gpt-4o"
    OPENROUTER_MODEL: str = "openai/gpt-4o"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.2
    TFIDF_WEIGHT: float = 0.1
    RERANK_THRESHOLD: str = "0.5"
    
    # Deployment Settings
    VERCEL_TOKEN: str = "YOUR_VERCEL_TOKEN"
    VERCEL_ORG_ID: str = "YOUR_VERCEL_ORG_ID"
    VERCEL_PROJECT_ID: str = "YOUR_VERCEL_PROJECT_ID"
    RAILWAY_TOKEN: str = "YOUR_RAILWAY_TOKEN"
    
    # Monitoring Settings
    PROMETHEUS_PORT: str = "8001"
    GRAFANA_PORT: str = "3000"
    ALERTMANAGER_PORT: str = "9093"
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "rag.log"
    
    # Security Settings
    JWT_SECRET_KEY: str = "REPLACE_WITH_SECURE_RANDOM_STRING"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: str = "30"
    
    # Test Settings
    TEST_POSTGRES_URL: str = "postgresql://postgres:postgres@localhost:5432/docqa_test"
    TEST_REDIS_URL: str = "redis://localhost:6379/1"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in the .env file

# Create settings instance
settings = Settings()

# Database configuration
DATABASE_CONFIG = {
    "connections": {
        "default": settings.POSTGRES_URL
    },
    "apps": {
        "models": {
            "models": ["models"],
            "default_connection": "default",
        }
    }
}

# MongoDB configuration
MONGODB_CONFIG = {
    "uri": settings.MONGODB_URI,
    "db": settings.MONGODB_DB,
    "collections": {
        "users": "users",
        "chat_sessions": "chat_sessions",
        "chat_messages": "chat_messages",
        "token_usage": "token_usage"
    }
}

# Redis configuration
REDIS_CONFIG = {
    "url": settings.REDIS_URL,
    "encoding": "utf-8",
    "decode_responses": True
}

# Pinecone configuration
PINECONE_CONFIG = {
    "api_key": settings.PINECONE_API_KEY,
    "environment": settings.PINECONE_ENVIRONMENT,
    "index_name": settings.PINECONE_INDEX
}

# LLM configuration
LLM_CONFIG = {
    "model": settings.MODEL_NAME,
    "temperature": settings.TEMPERATURE,
    "max_tokens": settings.MAX_TOKENS
}

# RAG configuration
RAG_CONFIG = {
    "chunk_size": settings.CHUNK_SIZE,
    "chunk_overlap": settings.CHUNK_OVERLAP,
    "vector_top_k": settings.VECTOR_TOP_K,
    "final_top_k": settings.FINAL_TOP_K,
    "cache_ttl": settings.CACHE_TTL
}

# Authentication configuration
AUTH_CONFIG = {
    "clerk_secret_key": settings.CLERK_SECRET_KEY,
    "clerk_publishable_key": settings.CLERK_PUBLISHABLE_KEY
}

# Model context sizes
MODEL_CONTEXT_SIZES = {
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
}

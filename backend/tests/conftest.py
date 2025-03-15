import pytest
import os
import asyncio
from typing import AsyncGenerator, Generator
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import pinecone
from dotenv import load_dotenv

# Load test environment variables
load_dotenv()

# Test database URL
TEST_POSTGRES_URL = "postgresql://postgres:postgres@localhost:5432/docqa_test"
TEST_REDIS_URL = "redis://localhost:6379/1"  # Use DB 1 for testing

# Configure test settings
pytest_plugins = [
    "pytest_asyncio",
]

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test case"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Create a Redis client for testing"""
    client = redis.from_url(TEST_REDIS_URL)
    try:
        await client.ping()
        yield client
    finally:
        await client.flushdb()  # Clear test database
        await client.close()

@pytest.fixture(scope="session")
async def db_engine():
    """Create a test database engine"""
    engine = create_async_engine(
        TEST_POSTGRES_URL,
        echo=True
    )
    yield engine
    await engine.dispose()

@pytest.fixture(scope="session")
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session"""
    async_session = sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    async with async_session() as session:
        yield session

@pytest.fixture(scope="session")
def pinecone_client():
    """Initialize Pinecone client for testing"""
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY", "test-key"),
        environment=os.getenv("PINECONE_ENVIRONMENT", "test-env")
    )
    return pinecone

@pytest.fixture(autouse=True)
async def cleanup_after_test(redis_client, db_session):
    """Clean up after each test"""
    yield
    # Clear Redis test database
    await redis_client.flushdb()
    
    # Rollback any pending database transactions
    await db_session.rollback()

@pytest.fixture
def test_api_keys():
    """Provide test API keys"""
    return {
        "openrouter": os.getenv("OPENROUTER_API_KEY", "test-key"),
        "cohere": os.getenv("COHERE_API_KEY", "test-key"),
        "pinecone": os.getenv("PINECONE_API_KEY", "test-key")
    }

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    env_vars = {
        "OPENROUTER_API_KEY": "test-key",
        "COHERE_API_KEY": "test-key",
        "PINECONE_API_KEY": "test-key",
        "PINECONE_ENVIRONMENT": "test-env",
        "PINECONE_INDEX": "test-index",
        "POSTGRES_URL": TEST_POSTGRES_URL,
        "REDIS_URL": TEST_REDIS_URL
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars

@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "vector_top_k": 5,
        "final_top_k": 3,
        "cache_ttl": 300,
        "model_name": "openai/gpt-4",
        "temperature": 0.7,
        "max_tokens": 500
    }

@pytest.fixture
def test_documents():
    """Provide test document content"""
    return [
        {
            "filename": "ml_basics.pdf",
            "content": """
            Machine Learning Basics
            
            Supervised learning is a type of machine learning where models learn from labeled data.
            The model learns to map inputs to outputs based on example pairs.
            
            Common algorithms include:
            - Linear Regression
            - Decision Trees
            - Neural Networks
            """,
            "metadata": {"type": "textbook", "subject": "machine learning"}
        },
        {
            "filename": "deep_learning.pdf",
            "content": """
            Deep Learning Fundamentals
            
            Neural networks are composed of layers of artificial neurons.
            Deep learning models can automatically learn hierarchical features.
            
            Key concepts:
            - Backpropagation
            - Activation Functions
            - Loss Functions
            """,
            "metadata": {"type": "textbook", "subject": "deep learning"}
        }
    ]

@pytest.fixture
def test_queries():
    """Provide test queries and expected responses"""
    return [
        {
            "query": "What is supervised learning?",
            "expected_sources": ["ml_basics.pdf"],
            "expected_keywords": ["labeled data", "inputs", "outputs"]
        },
        {
            "query": "Explain neural networks",
            "expected_sources": ["deep_learning.pdf"],
            "expected_keywords": ["layers", "neurons", "backpropagation"]
        }
    ]

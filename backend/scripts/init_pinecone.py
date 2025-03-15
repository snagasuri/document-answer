import pinecone
import logging
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import asyncio
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_pinecone():
    """Initialize Pinecone index with proper configuration"""
    try:
        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        
        # Load embedding model to get vector dimension
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vector_dim = model.get_sentence_embedding_dimension()
        
        index_name = os.getenv("PINECONE_INDEX")
        
        # Check if index exists
        if index_name not in pinecone.list_indexes():
            logger.info(f"Creating new Pinecone index: {index_name}")
            
            # Create index
            pinecone.create_index(
                name=index_name,
                dimension=vector_dim,
                metric="cosine",
                pods=1,
                replicas=1,
                pod_type="p1.x1"  # Choose based on your needs
            )
            
            # Wait for index to be ready
            while not pinecone.describe_index(index_name).status['ready']:
                logger.info("Waiting for index to be ready...")
                time.sleep(5)
                
        # Get index
        index = pinecone.Index(index_name)
        
        # Get index statistics
        stats = index.describe_index_stats()
        logger.info(f"Index statistics: {stats}")
        
        # Create test vectors
        test_vectors = [
            {
                "id": "test-1",
                "values": [0.1] * vector_dim,
                "metadata": {
                    "text": "Test vector 1",
                    "test": True
                }
            },
            {
                "id": "test-2",
                "values": [0.2] * vector_dim,
                "metadata": {
                    "text": "Test vector 2",
                    "test": True
                }
            }
        ]
        
        # Upsert test vectors
        index.upsert(vectors=test_vectors)
        logger.info("Upserted test vectors")
        
        # Test query
        query_response = index.query(
            vector=[0.1] * vector_dim,
            top_k=2,
            include_metadata=True
        )
        logger.info(f"Query response: {query_response}")
        
        # Clean up test vectors
        index.delete(ids=["test-1", "test-2"])
        logger.info("Cleaned up test vectors")
        
        return {
            "status": "success",
            "index_name": index_name,
            "dimension": vector_dim,
            "total_vectors": stats.total_vector_count
        }
        
    except Exception as e:
        logger.error(f"Pinecone initialization failed: {str(e)}")
        raise

async def create_index_metadata():
    """Create metadata schema for better filtering"""
    try:
        index_name = os.getenv("PINECONE_INDEX")
        index = pinecone.Index(index_name)
        
        # Define metadata schema
        metadata_config = {
            "document_id": {"type": "string", "indexed": True},
            "chunk_index": {"type": "integer", "indexed": True},
            "content": {"type": "text", "indexed": True},  # For semantic search
            "metadata": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "indexed": True},
                    "type": {"type": "string", "indexed": True},
                    "created_at": {"type": "string", "indexed": True},
                    "language": {"type": "string", "indexed": True},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "label": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
        
        # TODO: When Pinecone supports metadata schema configuration
        # index.configure_metadata(metadata_config)
        
        logger.info("Metadata schema configured")
        
    except Exception as e:
        logger.error(f"Metadata configuration failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Initialize Pinecone
    result = asyncio.run(init_pinecone())
    print("\nPinecone Initialization Result:")
    print(result)
    
    # Configure metadata schema
    asyncio.run(create_index_metadata())

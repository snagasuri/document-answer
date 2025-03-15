import asyncio
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import tempfile
import shutil

from core.document_processor import DocumentProcessor
from core.hybrid_retriever import HybridRetriever
from core.reranker import Reranker
from core.llm_service import LLMService
from models.document import Document, ChatMessage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipelineTester:
    def __init__(self):
        """Initialize RAG pipeline components"""
        self.processor = DocumentProcessor()
        self.retriever = HybridRetriever(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
            pinecone_index=os.getenv("PINECONE_INDEX"),
            redis_url=os.getenv("REDIS_URL")
        )
        self.reranker = Reranker(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            redis_url=os.getenv("REDIS_URL")
        )
        self.llm_service = LLMService(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            redis_url=os.getenv("REDIS_URL"),
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
            cache_ttl=int(os.getenv("CACHE_TTL", "3600"))
        )
        
    async def test_document_processing(self, pdf_path: str):
        """Test document processing pipeline"""
        logger.info("Testing document processing...")
        
        try:
            # Process document
            document = await self.processor.process_pdf(
                file_path=pdf_path,
                filename=Path(pdf_path).name
            )
            logger.info(f"Processed document: {document.filename}")
            
            # Extract metadata
            metadata = self.processor.extract_metadata(document)
            logger.info(f"Extracted metadata: {metadata}")
            
            # Create chunks
            chunks = self.processor.create_chunks(document)
            logger.info(f"Created {len(chunks)} chunks")
            
            return document, chunks
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
            
    async def test_retrieval(self, query: str, chunks):
        """Test retrieval pipeline"""
        logger.info("Testing retrieval...")
        
        try:
            # Add chunks to indices
            await self.retriever.add_documents(chunks)
            logger.info("Added chunks to retrieval indices")
            
            # Perform hybrid search
            results = await self.retriever.hybrid_search(
                query=query,
                top_k=10
            )
            logger.info(f"Retrieved {len(results)} results")
            
            # Log scores
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}:")
                logger.info(f"  Vector score: {result.vector_score:.3f}")
                logger.info(f"  BM25 score: {result.bm25_score:.3f}")
                logger.info(f"  TF-IDF score: {result.tfidf_score:.3f}")
                
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise
            
    async def test_reranking(self, query: str, results):
        """Test reranking pipeline"""
        logger.info("Testing reranking...")
        
        try:
            # Rerank with different models
            cohere_results = await self.reranker.rerank_cohere(query, results.copy())
            logger.info("Completed Cohere reranking")
            
            tart_results = await self.reranker.rerank_tart(query, results.copy())
            logger.info("Completed TART reranking")
            
            hybrid_results = await self.reranker.rerank_hybrid(query, results.copy())
            logger.info("Completed hybrid reranking")
            
            # Log scores
            logger.info("\nReranking Scores:")
            for i in range(min(3, len(results))):
                logger.info(f"\nResult {i+1}:")
                logger.info(f"  Cohere: {cohere_results[i].rerank_score:.3f}")
                logger.info(f"  TART: {tart_results[i].rerank_score:.3f}")
                logger.info(f"  Hybrid: {hybrid_results[i].rerank_score:.3f}")
                
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise
            
    async def test_llm_response(self, query: str, results):
        """Test LLM response generation"""
        logger.info("Testing LLM response...")
        
        try:
            # Test with caching enabled
            logger.info("Testing with caching enabled...")
            async for response in self.llm_service.generate_response(
                query=query,
                results=results[:3],  # Use top 3 results
                chat_history=[],
                use_cache=True
            ):
                if response.finished:
                    logger.info("\nFinal Response:")
                    logger.info(response.message)
                    logger.info("\nSources:")
                    for source in response.sources:
                        logger.info(f"- {source.content[:100]}...")
                    
                    # Extract citations
                    citations = self.llm_service._extract_citations(response.message)
                    logger.info(f"\nCitations found: {citations}")
            
            # Test with caching disabled
            logger.info("\nTesting with caching disabled...")
            async for response in self.llm_service.generate_response(
                query=query,
                results=results[:3],
                chat_history=[],
                use_cache=False
            ):
                if response.finished:
                    logger.info("\nFinal Response (no cache):")
                    logger.info(response.message)
                    
        except Exception as e:
            logger.error(f"LLM response failed: {str(e)}")
            raise

async def main():
    """Run complete pipeline test"""
    # Create temporary PDF for testing
    test_pdf = """
    Machine Learning Fundamentals
    
    Supervised Learning involves training a model on labeled data to make predictions.
    Common algorithms include linear regression, decision trees, and neural networks.
    
    Unsupervised Learning works with unlabeled data to find patterns and structure.
    Examples include clustering, dimensionality reduction, and anomaly detection.
    
    Model Evaluation uses metrics like accuracy, precision, recall, and F1-score.
    Cross-validation helps assess model performance on unseen data.
    
    Overfitting occurs when a model learns noise in training data.
    Underfitting happens when a model is too simple to capture patterns.
    """
    
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(test_pdf.encode())
        test_file = f.name
    
    try:
        tester = RAGPipelineTester()
        
        # Test complete pipeline
        document, chunks = await tester.test_document_processing(test_file)
        
        query = "What is the difference between overfitting and underfitting?"
        
        results = await tester.test_retrieval(query, chunks)
        reranked_results = await tester.test_reranking(query, results)
        await tester.test_llm_response(query, reranked_results)
        
    finally:
        # Cleanup
        os.unlink(test_file)

if __name__ == "__main__":
    asyncio.run(main())

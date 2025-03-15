import logging
import json
import sys
from pathlib import Path
from datetime import datetime
import traceback
from typing import Dict, Any, Optional
import asyncio
from pprint import pformat
import pandas as pd
import numpy as np

from core.document_processor import DocumentProcessor
from core.hybrid_retriever import HybridRetriever
from core.reranker import Reranker
from core.llm_service import LLMService
from models.document import Document, DocumentChunk, SearchResult

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_debug.log')
    ]
)
logger = logging.getLogger(__name__)

class RAGDebugger:
    def __init__(self, debug_dir: str = "debug_logs"):
        """Initialize debugger"""
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
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
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
    def log_document(
        self,
        document: Document,
        prefix: str = "document"
    ):
        """Log document details"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.debug_dir / f"{prefix}_{timestamp}.json"
        
        doc_data = {
            "id": str(document.id),
            "filename": document.filename,
            "content_preview": document.content[:500],
            "content_length": len(document.content),
            "metadata": document.metadata
        }
        
        with open(log_file, "w") as f:
            json.dump(doc_data, f, indent=2)
            
        logger.debug(f"Document logged to {log_file}")
        
    def log_chunks(
        self,
        chunks: List[DocumentChunk],
        prefix: str = "chunks"
    ):
        """Log chunk details"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.debug_dir / f"{prefix}_{timestamp}.json"
        
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata
            })
            
        with open(log_file, "w") as f:
            json.dump(chunks_data, f, indent=2)
            
        logger.debug(f"Chunks logged to {log_file}")
        
    def log_search_results(
        self,
        query: str,
        results: List[SearchResult],
        prefix: str = "search"
    ):
        """Log search results with scores"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.debug_dir / f"{prefix}_{timestamp}.json"
        
        results_data = {
            "query": query,
            "results": []
        }
        
        for result in results:
            result_data = {
                "chunk_id": str(result.chunk.id),
                "content": result.chunk.content,
                "scores": {
                    "vector": result.vector_score,
                    "bm25": result.bm25_score,
                    "tfidf": result.tfidf_score,
                    "rerank": result.rerank_score,
                    "combined": result.combined_score
                },
                "metadata": result.chunk.metadata
            }
            results_data["results"].append(result_data)
            
        with open(log_file, "w") as f:
            json.dump(results_data, f, indent=2)
            
        logger.debug(f"Search results logged to {log_file}")
        
    def analyze_scores(
        self,
        results: List[SearchResult]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze score distributions"""
        scores = {
            "vector": [],
            "bm25": [],
            "tfidf": [],
            "rerank": [],
            "combined": []
        }
        
        for result in results:
            if result.vector_score:
                scores["vector"].append(result.vector_score)
            if result.bm25_score:
                scores["bm25"].append(result.bm25_score)
            if result.tfidf_score:
                scores["tfidf"].append(result.tfidf_score)
            if result.rerank_score:
                scores["rerank"].append(result.rerank_score)
            if result.combined_score:
                scores["combined"].append(result.combined_score)
                
        analysis = {}
        for method, method_scores in scores.items():
            if method_scores:
                analysis[method] = {
                    "mean": np.mean(method_scores),
                    "std": np.std(method_scores),
                    "min": np.min(method_scores),
                    "max": np.max(method_scores),
                    "median": np.median(method_scores)
                }
                
        return analysis
        
    async def debug_pipeline(
        self,
        query: str,
        document_path: str
    ):
        """Debug complete RAG pipeline"""
        try:
            logger.info(f"Starting pipeline debug for query: {query}")
            
            # Process document
            logger.info("Processing document...")
            document = await self.processor.process_pdf(
                document_path,
                Path(document_path).name
            )
            self.log_document(document)
            
            # Create chunks
            logger.info("Creating chunks...")
            chunks = self.processor.create_chunks(document)
            self.log_chunks(chunks)
            
            # Add to retriever
            logger.info("Adding chunks to retriever...")
            await self.retriever.add_documents(chunks)
            
            # Perform search
            logger.info("Performing hybrid search...")
            search_results = await self.retriever.hybrid_search(
                query=query,
                top_k=10
            )
            self.log_search_results(query, search_results, "initial_search")
            
            # Analyze initial scores
            logger.info("Analyzing initial scores...")
            initial_analysis = self.analyze_scores(search_results)
            logger.info(f"Initial score analysis:\n{pformat(initial_analysis)}")
            
            # Rerank results
            logger.info("Reranking results...")
            reranked_results = await self.reranker.rerank_hybrid(
                query=query,
                results=search_results
            )
            self.log_search_results(query, reranked_results, "reranked")
            
            # Analyze reranked scores
            logger.info("Analyzing reranked scores...")
            reranked_analysis = self.analyze_scores(reranked_results)
            logger.info(f"Reranked score analysis:\n{pformat(reranked_analysis)}")
            
            # Generate response
            logger.info("Generating response...")
            responses = []
            async for response in self.llm_service.generate_response(
                query=query,
                results=reranked_results[:3]
            ):
                responses.append(response)
                if response.finished:
                    logger.info(f"Final response:\n{response.message}")
                    logger.info("Sources:")
                    for source in response.sources:
                        logger.info(f"- {source.content[:100]}...")
                        
            logger.info("Pipeline debug completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline debug failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    async def analyze_failure_cases(
        self,
        queries: List[str],
        document_path: str,
        threshold: float = 0.5
    ):
        """Analyze cases where retrieval/reranking might be failing"""
        failures = []
        
        for query in queries:
            try:
                # Run pipeline
                document = await self.processor.process_pdf(
                    document_path,
                    Path(document_path).name
                )
                chunks = self.processor.create_chunks(document)
                await self.retriever.add_documents(chunks)
                
                # Get results
                results = await self.retriever.hybrid_search(query, top_k=10)
                reranked = await self.reranker.rerank_hybrid(query, results)
                
                # Check scores
                top_score = reranked[0].combined_score
                if top_score < threshold:
                    failures.append({
                        "query": query,
                        "top_score": top_score,
                        "top_result": reranked[0].chunk.content,
                        "analysis": self.analyze_scores(reranked)
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing query '{query}': {str(e)}")
                
        if failures:
            # Log failures
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.debug_dir / f"failures_{timestamp}.json"
            
            with open(log_file, "w") as f:
                json.dump(failures, f, indent=2)
                
            logger.warning(f"Found {len(failures)} potential failure cases")
            logger.info(f"Failure analysis logged to {log_file}")
            
        return failures

async def main():
    """Run debugger"""
    debugger = RAGDebugger()
    
    # Example usage
    test_query = "What is machine learning?"
    test_doc = "path/to/document.pdf"  # Replace with actual path
    
    await debugger.debug_pipeline(test_query, test_doc)
    
    # Example failure analysis
    test_queries = [
        "What is supervised learning?",
        "Explain neural networks",
        "How does backpropagation work?"
    ]
    
    failures = await debugger.analyze_failure_cases(
        test_queries,
        test_doc,
        threshold=0.7
    )

if __name__ == "__main__":
    asyncio.run(main())

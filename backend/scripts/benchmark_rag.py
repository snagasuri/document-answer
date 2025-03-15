import asyncio
import logging
import time
import json
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

from core.document_processor import DocumentProcessor
from core.hybrid_retriever import HybridRetriever
from core.reranker import Reranker
from core.llm_service import LLMService
from scripts.generate_test_data import TestDataGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGBenchmark:
    def __init__(self):
        """Initialize benchmark components"""
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
        
    async def benchmark_retrieval(
        self,
        queries: List[str],
        num_runs: int = 5
    ) -> Dict:
        """Benchmark retrieval performance"""
        results = {
            "vector": [],
            "bm25": [],
            "tfidf": [],
            "hybrid": []
        }
        
        for query in queries:
            for _ in range(num_runs):
                # Vector search
                start = time.time()
                vector_results = await self.retriever.hybrid_search(
                    query=query,
                    top_k=10,
                    vector_weight=1.0,
                    bm25_weight=0.0,
                    tfidf_weight=0.0
                )
                results["vector"].append(time.time() - start)
                
                # BM25 search
                start = time.time()
                bm25_results = await self.retriever.hybrid_search(
                    query=query,
                    top_k=10,
                    vector_weight=0.0,
                    bm25_weight=1.0,
                    tfidf_weight=0.0
                )
                results["bm25"].append(time.time() - start)
                
                # TF-IDF search
                start = time.time()
                tfidf_results = await self.retriever.hybrid_search(
                    query=query,
                    top_k=10,
                    vector_weight=0.0,
                    bm25_weight=0.0,
                    tfidf_weight=1.0
                )
                results["tfidf"].append(time.time() - start)
                
                # Hybrid search
                start = time.time()
                hybrid_results = await self.retriever.hybrid_search(
                    query=query,
                    top_k=10,
                    vector_weight=0.7,
                    bm25_weight=0.15,
                    tfidf_weight=0.15
                )
                results["hybrid"].append(time.time() - start)
                
        return results
        
    async def benchmark_reranking(
        self,
        queries: List[str],
        num_runs: int = 5
    ) -> Dict:
        """Benchmark reranking performance"""
        results = {
            "cohere": [],
            "tart": [],
            "hybrid": []
        }
        
        for query in queries:
            # Get initial results
            search_results = await self.retriever.hybrid_search(query, top_k=10)
            
            for _ in range(num_runs):
                # Cohere reranking
                start = time.time()
                cohere_results = await self.reranker.rerank_cohere(
                    query=query,
                    results=search_results.copy()
                )
                results["cohere"].append(time.time() - start)
                
                # TART reranking
                start = time.time()
                tart_results = await self.reranker.rerank_tart(
                    query=query,
                    results=search_results.copy()
                )
                results["tart"].append(time.time() - start)
                
                # Hybrid reranking
                start = time.time()
                hybrid_results = await self.reranker.rerank_hybrid(
                    query=query,
                    results=search_results.copy()
                )
                results["hybrid"].append(time.time() - start)
                
        return results
        
    def plot_results(
        self,
        results: Dict,
        title: str,
        output_file: str
    ):
        """Plot benchmark results"""
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        data = []
        labels = []
        for method, times in results.items():
            data.append(times)
            labels.append(f"{method}\n(mean: {np.mean(times):.3f}s)")
        
        plt.boxplot(data, labels=labels)
        plt.title(title)
        plt.ylabel("Time (seconds)")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(output_file)
        plt.close()
        
    def save_results(
        self,
        results: Dict,
        output_file: str
    ):
        """Save benchmark results"""
        # Calculate statistics
        stats = {}
        for method, times in results.items():
            stats[method] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "p50": np.percentile(times, 50),
                "p95": np.percentile(times, 95),
                "p99": np.percentile(times, 99)
            }
        
        # Save to JSON
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=2)
            
    async def run_complete_benchmark(
        self,
        output_dir: str = "benchmark_results"
    ):
        """Run complete benchmark suite"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate test data
        test_gen = TestDataGenerator(os.getenv("OPENROUTER_API_KEY"))
        docs_df, qa_df = await test_gen.create_test_dataset(
            topics=[
                {
                    "name": "Machine Learning",
                    "sections": ["Supervised", "Unsupervised", "Evaluation"]
                },
                {
                    "name": "Deep Learning",
                    "sections": ["Neural Networks", "CNN", "RNN"]
                }
            ],
            output_dir=f"{output_dir}/test_data"
        )
        
        # Extract test queries
        test_queries = qa_df["question"].tolist()[:10]  # Use first 10 questions
        
        # Run benchmarks
        logger.info("Running retrieval benchmark...")
        retrieval_results = await self.benchmark_retrieval(test_queries)
        
        logger.info("Running reranking benchmark...")
        reranking_results = await self.benchmark_reranking(test_queries)
        
        # Save results
        logger.info("Saving results...")
        self.save_results(
            retrieval_results,
            f"{output_dir}/retrieval_stats_{timestamp}.json"
        )
        self.save_results(
            reranking_results,
            f"{output_dir}/reranking_stats_{timestamp}.json"
        )
        
        # Create plots
        logger.info("Creating plots...")
        self.plot_results(
            retrieval_results,
            "Retrieval Performance Comparison",
            f"{output_dir}/retrieval_benchmark_{timestamp}.png"
        )
        self.plot_results(
            reranking_results,
            "Reranking Performance Comparison",
            f"{output_dir}/reranking_benchmark_{timestamp}.png"
        )
        
        logger.info(f"Benchmark results saved to {output_dir}")
        
        return {
            "retrieval": retrieval_results,
            "reranking": reranking_results
        }

async def main():
    """Run benchmark suite"""
    benchmark = RAGBenchmark()
    await benchmark.run_complete_benchmark()

if __name__ == "__main__":
    asyncio.run(main())

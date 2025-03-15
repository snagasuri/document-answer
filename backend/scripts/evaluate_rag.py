import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import evaluate
from rouge_score import rouge_scorer
from dotenv import load_dotenv
import httpx

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

class RAGEvaluator:
    def __init__(self):
        """Initialize evaluator"""
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
        
        # Initialize evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.meteor = evaluate.load('meteor')
        self.bertscore = evaluate.load('bertscore')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def evaluate_retrieval(
        self,
        query: str,
        relevant_chunks: List[str],
        top_k: int = 10
    ) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        # Get retrieval results
        results = await self.retriever.hybrid_search(query, top_k=top_k)
        retrieved_chunks = [r.chunk.content for r in results]
        
        # Calculate metrics
        metrics = {}
        
        # Precision@K
        relevant_retrieved = len(set(retrieved_chunks) & set(relevant_chunks))
        metrics["precision"] = relevant_retrieved / len(retrieved_chunks)
        
        # Recall@K
        metrics["recall"] = relevant_retrieved / len(relevant_chunks)
        
        # F1 Score
        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
        else:
            metrics["f1"] = 0.0
            
        # MRR (Mean Reciprocal Rank)
        for i, chunk in enumerate(retrieved_chunks):
            if chunk in relevant_chunks:
                metrics["mrr"] = 1.0 / (i + 1)
                break
        else:
            metrics["mrr"] = 0.0
            
        return metrics
        
    async def evaluate_reranking(
        self,
        query: str,
        relevant_chunks: List[str],
        initial_results: List[Any]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate reranking performance"""
        metrics = {}
        
        # Evaluate different reranking methods
        for method in ["cohere", "tart", "hybrid"]:
            if method == "cohere":
                reranked = await self.reranker.rerank_cohere(query, initial_results)
            elif method == "tart":
                reranked = await self.reranker.rerank_tart(query, initial_results)
            else:
                reranked = await self.reranker.rerank_hybrid(query, initial_results)
                
            reranked_chunks = [r.chunk.content for r in reranked]
            
            # Calculate metrics
            relevant_reranked = len(set(reranked_chunks[:3]) & set(relevant_chunks))
            precision = relevant_reranked / 3
            recall = relevant_reranked / len(relevant_chunks)
            
            metrics[method] = {
                "precision@3": precision,
                "recall@3": recall,
                "f1@3": 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0,
                "ndcg@3": self._calculate_ndcg(reranked_chunks[:3], relevant_chunks)
            }
            
        return metrics
        
    def _calculate_ndcg(
        self,
        results: List[str],
        relevant: List[str],
        k: int = None
    ) -> float:
        """Calculate NDCG (Normalized Discounted Cumulative Gain)"""
        if k is None:
            k = len(results)
            
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, result in enumerate(results[:k]):
            rel = 1.0 if result in relevant else 0.0
            dcg += rel / np.log2(i + 2)
            
        # Calculate IDCG
        for i in range(min(len(relevant), k)):
            idcg += 1.0 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
        
    async def evaluate_response(
        self,
        generated: str,
        reference: str
    ) -> Dict[str, float]:
        """Evaluate response quality"""
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, generated)
        metrics.update({
            f"rouge_{key}": value.fmeasure
            for key, value in rouge_scores.items()
        })
        
        # METEOR score
        meteor_score = self.meteor.compute(
            predictions=[generated],
            references=[reference]
        )
        metrics["meteor"] = meteor_score["meteor"]
        
        # BERTScore
        bert_scores = self.bertscore.compute(
            predictions=[generated],
            references=[reference],
            lang="en"
        )
        metrics["bertscore_f1"] = np.mean(bert_scores["f1"])
        
        # Semantic similarity
        gen_emb = self.embedding_model.encode([generated])[0]
        ref_emb = self.embedding_model.encode([reference])[0]
        metrics["semantic_similarity"] = cosine_similarity(
            [gen_emb],
            [ref_emb]
        )[0][0]
        
        return metrics
        
    async def evaluate_pipeline(
        self,
        test_data: pd.DataFrame,
        output_dir: str = "evaluation_results"
    ):
        """Evaluate complete RAG pipeline"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "retrieval": [],
            "reranking": [],
            "response": []
        }
        
        for _, row in test_data.iterrows():
            query = row["question"]
            reference = row["answer"]
            relevant_chunks = row.get("relevant_chunks", [])
            
            try:
                # Evaluate retrieval
                retrieval_metrics = await self.evaluate_retrieval(
                    query,
                    relevant_chunks
                )
                results["retrieval"].append({
                    "query": query,
                    **retrieval_metrics
                })
                
                # Get initial results for reranking
                initial_results = await self.retriever.hybrid_search(
                    query,
                    top_k=10
                )
                
                # Evaluate reranking
                reranking_metrics = await self.evaluate_reranking(
                    query,
                    relevant_chunks,
                    initial_results
                )
                results["reranking"].append({
                    "query": query,
                    **reranking_metrics
                })
                
                # Generate and evaluate response
                generated = ""
                async for response in self.llm_service.generate_response(
                    query=query,
                    results=initial_results[:3]
                ):
                    if response.finished:
                        generated = response.message
                        
                response_metrics = await self.evaluate_response(
                    generated,
                    reference
                )
                results["response"].append({
                    "query": query,
                    "generated": generated,
                    "reference": reference,
                    **response_metrics
                })
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {str(e)}")
                
        # Save results
        for component, component_results in results.items():
            df = pd.DataFrame(component_results)
            df.to_csv(f"{output_dir}/{component}_results_{timestamp}.csv", index=False)
            
            # Calculate and log average metrics
            if component == "retrieval":
                avg_metrics = df[[
                    "precision", "recall", "f1", "mrr"
                ]].mean()
            elif component == "reranking":
                avg_metrics = pd.DataFrame([
                    result for result in df["reranking_metrics"]
                ]).mean()
            else:
                avg_metrics = df[[
                    col for col in df.columns
                    if col not in ["query", "generated", "reference"]
                ]].mean()
                
            logger.info(f"\nAverage {component} metrics:")
            for metric, value in avg_metrics.items():
                logger.info(f"{metric}: {value:.3f}")
                
        return results

async def main():
    """Run evaluation"""
    evaluator = RAGEvaluator()
    
    # Generate test data
    test_gen = TestDataGenerator(os.getenv("OPENROUTER_API_KEY"))
    docs_df, qa_df = await test_gen.create_test_dataset(
        topics=[
            {
                "name": "Machine Learning",
                "sections": ["Supervised", "Unsupervised", "Evaluation"]
            }
        ]
    )
    
    # Run evaluation
    results = await evaluator.evaluate_pipeline(qa_df)
    
    # Log summary
    logger.info("\nEvaluation Summary:")
    for component, metrics in results.items():
        logger.info(f"\n{component.upper()} Results:")
        if isinstance(metrics, list) and metrics:
            avg_metrics = pd.DataFrame(metrics).mean()
            for metric, value in avg_metrics.items():
                if isinstance(value, float):
                    logger.info(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    asyncio.run(main())

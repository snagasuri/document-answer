import os
import yaml
from pathlib import Path
from autorag.evaluator import Evaluator
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_autorag_config():
    """Create AutoRAG configuration for testing different RAG components"""
    config = {
        "node_lines": [
            {
                "node_line_name": "retrieval_line",
                "nodes": [
                    {
                        "node_type": "retrieval",
                        "strategy": {
                            "metrics": [
                                "retrieval_f1",
                                "retrieval_recall",
                                "retrieval_ndcg",
                                "retrieval_mrr"
                            ]
                        },
                        "top_k": 10,
                        "modules": [
                            {
                                "module_type": "vectordb",
                                "vectordb": "default"
                            },
                            {
                                "module_type": "bm25"
                            },
                            {
                                "module_type": "tfidf"
                            },
                            {
                                "module_type": "hybrid_rrf",
                                "weight_range": "(4,80)"
                            }
                        ]
                    }
                ]
            },
            {
                "node_line_name": "reranking_line",
                "nodes": [
                    {
                        "node_type": "reranker",
                        "strategy": {
                            "metrics": [
                                "rerank_f1",
                                "rerank_recall",
                                "rerank_ndcg"
                            ]
                        },
                        "modules": [
                            {
                                "module_type": "cohere_rerank",
                                "model": "rerank-english-v2.0"
                            },
                            {
                                "module_type": "tart_rerank",
                                "model_name": "TART-full"
                            },
                            {
                                "module_type": "colbert_rerank"
                            }
                        ]
                    }
                ]
            },
            {
                "node_line_name": "generation_line",
                "nodes": [
                    {
                        "node_type": "prompt_maker",
                        "strategy": {
                            "metrics": [
                                {
                                    "metric_name": "meteor"
                                },
                                {
                                    "metric_name": "rouge"
                                },
                                {
                                    "metric_name": "sem_score",
                                    "embedding_model": "openai"
                                }
                            ]
                        },
                        "modules": [
                            {
                                "module_type": "fstring",
                                "prompt": "Read the passages and answer the given question.\nQuestion: {query}\nPassage: {retrieved_contents}\nAnswer:"
                            }
                        ]
                    },
                    {
                        "node_type": "generator",
                        "strategy": {
                            "metrics": [
                                {
                                    "metric_name": "meteor"
                                },
                                {
                                    "metric_name": "rouge"
                                },
                                {
                                    "metric_name": "sem_score",
                                    "embedding_model": "openai"
                                }
                            ]
                        },
                        "modules": [
                            {
                                "module_type": "openai_llm",
                                "llm": "gpt-4",
                                "batch": 8
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    return config

def run_optimization(qa_data_path: str, corpus_data_path: str):
    """Run AutoRAG optimization"""
    try:
        # Create config
        config = create_autorag_config()
        
        # Save config
        config_path = Path("autorag_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
            
        # Initialize evaluator
        evaluator = Evaluator(
            qa_data_path=qa_data_path,
            corpus_data_path=corpus_data_path
        )
        
        # Run evaluation
        logger.info("Starting AutoRAG evaluation...")
        evaluator.start_trial(config_path)
        
        # Load and analyze results
        results_df = pd.read_csv("0/summary.csv")
        
        # Get best configurations
        best_retrieval = results_df.sort_values("retrieval_f1", ascending=False).iloc[0]
        best_reranking = results_df.sort_values("rerank_f1", ascending=False).iloc[0]
        
        logger.info("\nBest Retrieval Configuration:")
        logger.info(f"Method: {best_retrieval['retrieval_method']}")
        logger.info(f"F1 Score: {best_retrieval['retrieval_f1']:.3f}")
        logger.info(f"Recall: {best_retrieval['retrieval_recall']:.3f}")
        logger.info(f"NDCG: {best_retrieval['retrieval_ndcg']:.3f}")
        
        logger.info("\nBest Reranking Configuration:")
        logger.info(f"Method: {best_reranking['reranker_method']}")
        logger.info(f"F1 Score: {best_reranking['rerank_f1']:.3f}")
        logger.info(f"Recall: {best_reranking['rerank_recall']:.3f}")
        logger.info(f"NDCG: {best_reranking['rerank_ndcg']:.3f}")
        
        return {
            "best_retrieval": best_retrieval.to_dict(),
            "best_reranking": best_reranking.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    qa_data = "path/to/qa_data.parquet"
    corpus_data = "path/to/corpus.parquet"
    
    results = run_optimization(qa_data, corpus_data)
    print("\nOptimization Results:")
    print(results)

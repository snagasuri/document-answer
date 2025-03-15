import logging
import time
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func
import os
from dotenv import load_dotenv

from core.config import settings
from models.document import Document, DocumentChunk, ChatMessage, ChatSession

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
RETRIEVAL_LATENCY = Histogram(
    'rag_retrieval_latency_seconds',
    'Time spent in retrieval',
    ['method']  # vector, bm25, tfidf, hybrid
)

RERANKING_LATENCY = Histogram(
    'rag_reranking_latency_seconds',
    'Time spent in reranking',
    ['method']  # cohere, tart, hybrid
)

LLM_LATENCY = Histogram(
    'rag_llm_latency_seconds',
    'Time spent in LLM generation'
)

RETRIEVAL_SCORES = Histogram(
    'rag_retrieval_scores',
    'Distribution of retrieval scores',
    ['method']
)

RERANKING_SCORES = Histogram(
    'rag_reranking_scores',
    'Distribution of reranking scores',
    ['method']
)

CACHE_HITS = Counter(
    'rag_cache_hits_total',
    'Number of cache hits',
    ['cache_type']  # retrieval, reranking, response
)

CACHE_MISSES = Counter(
    'rag_cache_misses_total',
    'Number of cache misses',
    ['cache_type']
)

DOCUMENT_COUNT = Gauge(
    'rag_document_count',
    'Number of documents in the system'
)

CHUNK_COUNT = Gauge(
    'rag_chunk_count',
    'Number of document chunks in the system'
)

class RAGMonitor:
    def __init__(self):
        """Initialize monitoring system"""
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.engine = create_async_engine(settings.POSTGRES_URL)
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
    async def collect_retrieval_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """Collect retrieval performance metrics"""
        try:
            async with self.redis_client.pipeline() as pipe:
                # Get retrieval latencies
                pipe.zrangebyscore(
                    "retrieval_latencies",
                    start_time.timestamp(),
                    end_time.timestamp(),
                    withscores=True
                )
                # Get retrieval scores
                pipe.zrangebyscore(
                    "retrieval_scores",
                    "-inf",
                    "+inf",
                    withscores=True
                )
                
                results = await pipe.execute()
                latencies, scores = results
                
                metrics = {
                    "latency": {
                        "mean": np.mean([l[1] for l in latencies]) if latencies else 0,
                        "p95": np.percentile([l[1] for l in latencies], 95) if latencies else 0,
                        "p99": np.percentile([l[1] for l in latencies], 99) if latencies else 0
                    },
                    "scores": {
                        "mean": np.mean([s[1] for s in scores]) if scores else 0,
                        "median": np.median([s[1] for s in scores]) if scores else 0
                    }
                }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error collecting retrieval metrics: {str(e)}")
            return {}
            
    async def collect_reranking_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """Collect reranking performance metrics"""
        try:
            async with self.redis_client.pipeline() as pipe:
                # Get reranking latencies
                pipe.zrangebyscore(
                    "reranking_latencies",
                    start_time.timestamp(),
                    end_time.timestamp(),
                    withscores=True
                )
                # Get reranking scores
                pipe.zrangebyscore(
                    "reranking_scores",
                    "-inf",
                    "+inf",
                    withscores=True
                )
                
                results = await pipe.execute()
                latencies, scores = results
                
                metrics = {
                    "latency": {
                        "mean": np.mean([l[1] for l in latencies]) if latencies else 0,
                        "p95": np.percentile([l[1] for l in latencies], 95) if latencies else 0,
                        "p99": np.percentile([l[1] for l in latencies], 99) if latencies else 0
                    },
                    "scores": {
                        "mean": np.mean([s[1] for s in scores]) if scores else 0,
                        "median": np.median([s[1] for s in scores]) if scores else 0
                    }
                }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error collecting reranking metrics: {str(e)}")
            return {}
            
    async def collect_cache_metrics(self) -> Dict:
        """Collect cache performance metrics"""
        try:
            async with self.redis_client.pipeline() as pipe:
                # Get cache hits/misses
                pipe.get("cache_hits")
                pipe.get("cache_misses")
                pipe.get("cache_hit_ratio")
                
                results = await pipe.execute()
                hits, misses, ratio = results
                
                metrics = {
                    "hits": int(hits) if hits else 0,
                    "misses": int(misses) if misses else 0,
                    "hit_ratio": float(ratio) if ratio else 0
                }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error collecting cache metrics: {str(e)}")
            return {}
            
    async def collect_database_metrics(self) -> Dict:
        """Collect database statistics"""
        try:
            async with self.async_session() as session:
                # Get document counts
                doc_count = await session.scalar(
                    select(func.count()).select_from(Document)
                )
                
                # Get chunk counts
                chunk_count = await session.scalar(
                    select(func.count()).select_from(DocumentChunk)
                )
                
                # Get chat statistics
                chat_count = await session.scalar(
                    select(func.count()).select_from(ChatSession)
                )
                
                message_count = await session.scalar(
                    select(func.count()).select_from(ChatMessage)
                )
                
                metrics = {
                    "documents": doc_count,
                    "chunks": chunk_count,
                    "chat_sessions": chat_count,
                    "chat_messages": message_count
                }
                
                # Update Prometheus gauges
                DOCUMENT_COUNT.set(doc_count)
                CHUNK_COUNT.set(chunk_count)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error collecting database metrics: {str(e)}")
            return {}
            
    async def generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
        report_file: str = "rag_metrics_report.json"
    ):
        """Generate comprehensive performance report"""
        try:
            # Collect all metrics
            retrieval_metrics = await self.collect_retrieval_metrics(
                start_time,
                end_time
            )
            
            reranking_metrics = await self.collect_reranking_metrics(
                start_time,
                end_time
            )
            
            cache_metrics = await self.collect_cache_metrics()
            db_metrics = await self.collect_database_metrics()
            
            # Compile report
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "retrieval": retrieval_metrics,
                "reranking": reranking_metrics,
                "cache": cache_metrics,
                "database": db_metrics
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Performance report saved to {report_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

async def main():
    """Run monitoring system"""
    # Start Prometheus metrics server
    start_http_server(8001)
    logger.info("Started Prometheus metrics server on port 8001")
    
    monitor = RAGMonitor()
    
    while True:
        try:
            # Generate report for last hour
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            await monitor.generate_report(start_time, end_time)
            
            # Wait for next collection
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}")
            await asyncio.sleep(60)  # Wait before retry

if __name__ == "__main__":
    asyncio.run(main())

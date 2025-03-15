import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import openai
import os
from tqdm import tqdm
import logging
from pathlib import Path
import json
import asyncio
import httpx
from datetime import datetime
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataGenerator:
    def __init__(self, openrouter_api_key: str):
        """Initialize test data generator"""
        self.api_key = openrouter_api_key
        self.api_base = "https://openrouter.ai/api/v1"
        
    async def generate_synthetic_document(
        self,
        topic: str,
        sections: List[str]
    ) -> str:
        """Generate a synthetic document about a topic"""
        try:
            prompt = f"""Create a detailed document about {topic}. 
            Include the following sections: {', '.join(sections)}.
            Make it informative and factual, suitable for testing a RAG system.
            Include specific details, numbers, and examples that can be used for question-answering.
            Length should be around 1000-1500 words."""
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "localhost:3000",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "openai/gpt-4",
                        "messages": [
                            {"role": "system", "content": "You are a technical writer creating educational content."},
                            {"role": "user", "content": prompt}
                        ]
                    },
                    timeout=30.0
                )
                
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return content
                
        except Exception as e:
            logger.error(f"Error generating document: {str(e)}")
            raise
            
    async def generate_qa_pairs(
        self,
        document: str,
        num_pairs: int = 5
    ) -> List[Dict]:
        """Generate question-answer pairs from a document"""
        try:
            prompt = f"""Given this document, generate {num_pairs} diverse question-answer pairs.
            Include both simple factual questions and more complex analytical questions.
            Format as JSON array with "question" and "answer" fields.
            Document:
            {document}"""
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "localhost:3000",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "openai/gpt-4",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert at creating educational test questions."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    },
                    timeout=30.0
                )
                
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                
                # Extract JSON from response - more robust approach
                qa_pairs = []
                try:
                    # Try to find JSON array in the content
                    start_idx = content.find("[")
                    end_idx = content.rfind("]") + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        qa_pairs = json.loads(json_str)
                    else:
                        # Fallback: create a simple QA pair
                        logger.warning("Could not find JSON array in response, using fallback")
                        qa_pairs = [
                            {"question": "What is this document about?", 
                             "answer": "This document is about " + document[:50] + "..."}
                        ]
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from LLM response, using fallback")
                    qa_pairs = [
                        {"question": "What is this document about?", 
                         "answer": "This document is about " + document[:50] + "..."}
                    ]
                
                return qa_pairs
                
        except Exception as e:
            logger.error(f"Error generating QA pairs: {str(e)}")
            raise
            
    async def create_test_dataset(
        self,
        topics: List[Dict],
        output_dir: str = "test_data"
    ):
        """Create complete test dataset with documents and QA pairs"""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            all_documents = []
            all_qa_pairs = []
            
            for topic in tqdm(topics, desc="Generating documents"):
                # Generate document
                document = await self.generate_synthetic_document(
                    topic["name"],
                    topic["sections"]
                )
                
                doc_id = len(all_documents)
                all_documents.append({
                    "id": doc_id,
                    "topic": topic["name"],
                    "content": document,
                    "created_at": datetime.utcnow().isoformat()
                })
                
                # Generate QA pairs
                qa_pairs = await self.generate_qa_pairs(document)
                
                for qa in qa_pairs:
                    qa["document_id"] = doc_id
                    all_qa_pairs.append(qa)
                
                # Add delay to avoid rate limits
                await asyncio.sleep(2)
            
            # Save to parquet files
            docs_df = pd.DataFrame(all_documents)
            qa_df = pd.DataFrame(all_qa_pairs)
            
            docs_df.to_parquet(f"{output_dir}/corpus.parquet")
            qa_df.to_parquet(f"{output_dir}/qa.parquet")
            
            logger.info(f"Created {len(all_documents)} documents")
            logger.info(f"Created {len(all_qa_pairs)} QA pairs")
            
            return docs_df, qa_df
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            raise

async def main():
    # Example topics
    topics = [
        {
            "name": "Machine Learning Fundamentals",
            "sections": [
                "Supervised Learning",
                "Unsupervised Learning",
                "Model Evaluation",
                "Overfitting and Underfitting"
            ]
        },
        {
            "name": "Database Systems",
            "sections": [
                "ACID Properties",
                "Indexing",
                "Query Optimization",
                "Transaction Management"
            ]
        },
        {
            "name": "Computer Networks",
            "sections": [
                "TCP/IP Protocol",
                "OSI Model",
                "Network Security",
                "Routing Algorithms"
            ]
        }
    ]
    
    generator = TestDataGenerator(os.getenv("OPENROUTER_API_KEY"))
    await generator.create_test_dataset(topics)

if __name__ == "__main__":
    asyncio.run(main())

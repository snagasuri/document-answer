#!/usr/bin/env python
"""
Initialize MongoDB collections for chat history and session management.
"""

import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

import sys
import os
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

from core.config import settings, MONGODB_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def init_mongodb():
    """Initialize MongoDB collections"""
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(settings.MONGODB_URI)
        db = client[settings.MONGODB_DB]
        
        # Get collection names
        collections = MONGODB_CONFIG["collections"]
        
        # Create collections if they don't exist
        for collection_name in collections.values():
            # Check if collection exists
            collection_exists = collection_name in await db.list_collection_names()
            
            if not collection_exists:
                # Create collection
                await db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
            else:
                logger.info(f"Collection already exists: {collection_name}")
        
        # Create indexes
        # Users collection
        await db[collections["users"]].create_index("clerkUserId", unique=True)
        logger.info(f"Created index on clerkUserId for {collections['users']}")
        
        # Chat sessions collection
        await db[collections["chat_sessions"]].create_index("clerkUserId")
        await db[collections["chat_sessions"]].create_index("updatedAt")
        logger.info(f"Created indexes for {collections['chat_sessions']}")
        
        # Chat messages collection
        await db[collections["chat_messages"]].create_index("sessionId")
        await db[collections["chat_messages"]].create_index("createdAt")
        logger.info(f"Created indexes for {collections['chat_messages']}")
        
        # Token usage collection
        await db[collections["token_usage"]].create_index("sessionId")
        await db[collections["token_usage"]].create_index("messageId")
        logger.info(f"Created indexes for {collections['token_usage']}")
        
        # Create test user if needed
        test_user = await db[collections["users"]].find_one({"clerkUserId": "test_user"})
        if not test_user:
            await db[collections["users"]].insert_one({
                "clerkUserId": "test_user",
                "email": "test@example.com",
                "createdAt": datetime.utcnow(),
                "lastActive": datetime.utcnow()
            })
            logger.info("Created test user")
            
            # Create test session
            session_result = await db[collections["chat_sessions"]].insert_one({
                "clerkUserId": "test_user",
                "title": "Test Session",
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
                "isActive": True
            })
            session_id = session_result.inserted_id
            logger.info(f"Created test session with ID: {session_id}")
            
            # Add test messages
            await db[collections["chat_messages"]].insert_one({
                "sessionId": session_id,
                "role": "user",
                "content": "What is RAG?",
                "tokenCount": 4,
                "createdAt": datetime.utcnow()
            })
            
            await db[collections["chat_messages"]].insert_one({
                "sessionId": session_id,
                "role": "assistant",
                "content": "RAG stands for Retrieval-Augmented Generation. It's a technique that combines retrieval-based and generation-based approaches for natural language processing tasks. In RAG, a retrieval system first fetches relevant documents or passages from a knowledge base, and then a generative model uses this retrieved information to produce more accurate and informed responses.",
                "tokenCount": 64,
                "createdAt": datetime.utcnow()
            })
            logger.info("Added test messages to test session")
        else:
            logger.info("Test user already exists")
        
        logger.info("MongoDB initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error initializing MongoDB: {str(e)}")
        raise

async def main():
    """Main function"""
    await init_mongodb()

if __name__ == "__main__":
    asyncio.run(main())

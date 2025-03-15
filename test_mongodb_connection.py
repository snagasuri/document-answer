"""
Simple script to test MongoDB connection.
Run this with:
python test_mongodb_connection.py <your_mongodb_uri>
"""

import sys
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def test_mongodb_connection(uri: str):
    """Test connection to MongoDB."""
    print(f"Testing connection to MongoDB...")
    
    try:
        # Create a client
        client = AsyncIOMotorClient(uri)
        
        # Get database
        db_name = 'test_database'
        db = client[db_name]
        
        # Ping the database
        await client.admin.command('ping')
        print(f"✅ Successfully connected to MongoDB!")
        
        # List collections
        collections = await db.list_collection_names()
        print(f"Available collections: {collections or '(none yet)'}")
        
        # Check expected collections
        expected_collections = [
            "chat_sessions", 
            "chat_messages", 
            "token_usage", 
            "documents", 
            "document_chunks"
        ]
        
        for coll in expected_collections:
            if coll in collections:
                count = await db[coll].count_documents({})
                print(f"Collection '{coll}' exists with {count} documents")
            else:
                print(f"Collection '{coll}' doesn't exist yet (will be created automatically when used)")
        
        return True
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_mongodb_connection.py <mongodb_uri>")
        sys.exit(1)
    
    uri = sys.argv[1]
    success = asyncio.run(test_mongodb_connection(uri))
    
    if success:
        print("\nYour MongoDB connection is working correctly!")
        print("Now update your Railway environment variables as described in railway-setup-guide.md")
    else:
        print("\nMongoDB connection failed. Check that:")
        print("1. The connection string is correct")
        print("2. Network access is properly configured in MongoDB Atlas")
        print("3. Username and password are correct")

import motor.motor_asyncio
import asyncio
from datetime import datetime

async def test_mongodb():
    # Connect to MongoDB
    client = motor.motor_asyncio.AsyncIOMotorClient(
        'mongodb+srv://ramnag2003:QnsLyJvMohgPexIF@cluster0.wsbsm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    )
    db = client['rag']
    
    # List collections
    print("Collections:")
    collections = await db.list_collection_names()
    print(collections)
    
    # Try to create a chat session
    session = {
        'clerkUserId': 'test_user',
        'title': 'Test Session',
        'createdAt': datetime.utcnow(),
        'updatedAt': datetime.utcnow(),
        'isActive': True
    }
    
    try:
        result = await db.chat_sessions.insert_one(session)
        print(f"Inserted session with ID: {result.inserted_id}")
    except Exception as e:
        print(f"Error inserting session: {str(e)}")
    
    # List chat sessions
    print("\nChat Sessions:")
    cursor = db.chat_sessions.find({})
    async for doc in cursor:
        print(doc)

if __name__ == "__main__":
    asyncio.run(test_mongodb())

"""
MongoDB service for chat history, session management, and document storage.
"""

import motor.motor_asyncio
from bson import ObjectId
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

from core.config import settings, MONGODB_CONFIG

class MongoDBService:
    """Service for MongoDB operations"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URI)
        self.db = self.client[settings.MONGODB_DB]
        
        # Collections
        self.users = self.db[MONGODB_CONFIG["collections"]["users"]]
        self.chat_sessions = self.db[MONGODB_CONFIG["collections"]["chat_sessions"]]
        self.chat_messages = self.db[MONGODB_CONFIG["collections"]["chat_messages"]]
        self.token_usage = self.db[MONGODB_CONFIG["collections"]["token_usage"]]
        self.documents = self.db["documents"]
        self.document_chunks = self.db["document_chunks"]
        
    async def create_user(self, clerk_user_id: str, email: str) -> Dict:
        """Create a new user"""
        user = {
            "clerkUserId": clerk_user_id,
            "email": email,
            "createdAt": datetime.utcnow(),
            "lastActive": datetime.utcnow()
        }
        result = await self.users.insert_one(user)
        user["_id"] = str(result.inserted_id)
        return user
        
    async def get_user(self, clerk_user_id: str) -> Optional[Dict]:
        """Get a user by Clerk ID"""
        user = await self.users.find_one({"clerkUserId": clerk_user_id})
        if user:
            user["_id"] = str(user["_id"])
        return user
        
    async def update_user_last_active(self, clerk_user_id: str) -> None:
        """Update user's last active timestamp"""
        await self.users.update_one(
            {"clerkUserId": clerk_user_id},
            {"$set": {"lastActive": datetime.utcnow()}}
        )
        
    async def create_chat_session(self, clerk_user_id: str, title: Optional[str] = None) -> Dict:
        """Create a new chat session"""
        session = {
            "clerkUserId": clerk_user_id,
            "title": title or "New Chat",
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
            "isActive": True,
            "documents": []  # List of document IDs associated with this session
        }
        result = await self.chat_sessions.insert_one(session)
        session["_id"] = str(result.inserted_id)
        
        # Update user's last active timestamp
        await self.update_user_last_active(clerk_user_id)
        
        return session
        
    async def get_chat_session(self, session_id: str) -> Optional[Dict]:
        """Get a chat session by ID"""
        try:
            session = await self.chat_sessions.find_one({"_id": ObjectId(session_id)})
            if session:
                session["_id"] = str(session["_id"])
            return session
        except Exception:
            return None
        
    async def list_chat_sessions(self, clerk_user_id: str) -> List[Dict]:
        """List all chat sessions for a user"""
        cursor = self.chat_sessions.find({"clerkUserId": clerk_user_id}).sort("updatedAt", -1)
        sessions = []
        async for session in cursor:
            session["_id"] = str(session["_id"])
            sessions.append(session)
            
        # Update user's last active timestamp
        await self.update_user_last_active(clerk_user_id)
        
        return sessions
        
    async def update_chat_session(self, session_id: str, updates: Dict) -> Optional[Dict]:
        """Update a chat session"""
        try:
            result = await self.chat_sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {**updates, "updatedAt": datetime.utcnow()}}
            )
            
            if result.modified_count > 0:
                session = await self.get_chat_session(session_id)
                return session
            return None
        except Exception:
            return None
            
    async def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages and documents"""
        try:
            # Get session to retrieve document IDs
            session = await self.get_chat_session(session_id)
            if not session:
                return False
                
            # Delete session
            result = await self.chat_sessions.delete_one({"_id": ObjectId(session_id)})
            
            if result.deleted_count > 0:
                # Delete all messages for this session
                await self.chat_messages.delete_many({"sessionId": ObjectId(session_id)})
                
                # Delete all token usage records for this session
                await self.token_usage.delete_many({"sessionId": ObjectId(session_id)})
                
                # Delete all documents associated with this session
                if "documents" in session and session["documents"]:
                    for doc_id in session["documents"]:
                        # Delete document chunks
                        await self.document_chunks.delete_many({"documentId": doc_id})
                        # Delete document
                        await self.documents.delete_one({"_id": ObjectId(doc_id)})
                
                return True
            return False
        except Exception:
            return False
        
    async def add_chat_message(self, session_id: str, role: str, content: str) -> Dict:
        """Add a message to a chat session"""
        try:
            message = {
                "sessionId": ObjectId(session_id),
                "role": role,
                "content": content,
                "tokenCount": None,  # Will be updated later
                "createdAt": datetime.utcnow()
            }
            result = await self.chat_messages.insert_one(message)
            message["_id"] = str(result.inserted_id)
            message["sessionId"] = str(message["sessionId"])
            
            # Update session's updatedAt timestamp
            await self.chat_sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"updatedAt": datetime.utcnow()}}
            )
            
            return message
        except Exception as e:
            raise ValueError(f"Failed to add message: {str(e)}")
        
    async def get_chat_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get messages for a chat session"""
        try:
            cursor = self.chat_messages.find({"sessionId": ObjectId(session_id)}).sort("createdAt", 1)
            messages = []
            async for message in cursor:
                message["_id"] = str(message["_id"])
                message["sessionId"] = str(message["sessionId"])
                messages.append(message)
            return messages
        except Exception as e:
            raise ValueError(f"Failed to get messages: {str(e)}")
        
    async def update_token_usage(self, message_id: str, usage: Dict[str, int]) -> Dict:
        """Update token usage for a message"""
        try:
            # Get the message to get the session ID
            message = await self.chat_messages.find_one({"_id": ObjectId(message_id)})
            if not message:
                raise ValueError(f"Message not found: {message_id}")
                
            token_data = {
                "messageId": ObjectId(message_id),
                "sessionId": message["sessionId"],
                "promptTokens": usage.get("prompt_tokens", 0),
                "completionTokens": usage.get("completion_tokens", 0),
                "totalTokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                "createdAt": datetime.utcnow()
            }
            
            result = await self.token_usage.insert_one(token_data)
            token_data["_id"] = str(result.inserted_id)
            token_data["messageId"] = str(token_data["messageId"])
            token_data["sessionId"] = str(token_data["sessionId"])
            
            # Update the message with token count
            await self.chat_messages.update_one(
                {"_id": ObjectId(message_id)},
                {"$set": {"tokenCount": token_data["totalTokens"]}}
            )
            
            return token_data
        except Exception as e:
            raise ValueError(f"Failed to update token usage: {str(e)}")
        
    async def get_session_token_usage(self, session_id: str) -> Dict[str, int]:
        """Get total token usage for a session"""
        try:
            pipeline = [
                {"$match": {"sessionId": ObjectId(session_id)}},
                {"$group": {
                    "_id": None,
                    "totalPromptTokens": {"$sum": "$promptTokens"},
                    "totalCompletionTokens": {"$sum": "$completionTokens"},
                    "totalTokens": {"$sum": "$totalTokens"}
                }}
            ]
            
            result = await self.token_usage.aggregate(pipeline).to_list(1)
            if not result:
                return {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
                
            return {
                "prompt_tokens": result[0]["totalPromptTokens"],
                "completion_tokens": result[0]["totalCompletionTokens"],
                "total_tokens": result[0]["totalTokens"]
            }
        except Exception as e:
            raise ValueError(f"Failed to get token usage: {str(e)}")
            
    # Document management methods
    async def add_document(self, session_id: str, filename: str, content: str, metadata: Dict = None) -> Dict:
        """Add a document to a session"""
        try:
            # Create document
            document = {
                "_id": ObjectId(),
                "filename": filename,
                "content": content,
                "metadata": metadata or {},
                "processingStatus": "processing",  # Add processing status
                "createdAt": datetime.utcnow()
            }
            
            # Insert document
            await self.documents.insert_one(document)
            
            # Update session with document ID
            await self.chat_sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$push": {"documents": document["_id"]}, "$set": {"updatedAt": datetime.utcnow()}}
            )
            
            # Convert ObjectId to string for response
            document["_id"] = str(document["_id"])
            
            return document
        except Exception as e:
            raise ValueError(f"Failed to add document: {str(e)}")
            
    async def add_document_chunks(self, document_id: str, chunks: List[Dict]) -> List[Dict]:
        """Add document chunks to the database"""
        try:
            # Convert document_id to ObjectId
            doc_id = ObjectId(document_id)
            
            # Prepare chunks for insertion
            chunk_docs = []
            for chunk in chunks:
                chunk_doc = {
                    "_id": ObjectId(),
                    "documentId": doc_id,
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "chunkIndex": chunk["chunk_index"],
                    "createdAt": datetime.utcnow()
                }
                chunk_docs.append(chunk_doc)
            
            # Insert chunks
            if chunk_docs:
                await self.document_chunks.insert_many(chunk_docs)
            
            # Convert ObjectIds to strings for response
            for chunk in chunk_docs:
                chunk["_id"] = str(chunk["_id"])
                chunk["documentId"] = str(chunk["documentId"])
            
            return chunk_docs
        except Exception as e:
            raise ValueError(f"Failed to add document chunks: {str(e)}")
            
    async def update_document_processing_status(self, document_id: str, status: str) -> bool:
        """Update document processing status"""
        try:
            result = await self.documents.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": {"processingStatus": status}}
            )
            return result.modified_count > 0
        except Exception:
            return False
            
    async def get_session_documents(self, session_id: str) -> List[Dict]:
        """Get all documents for a session"""
        try:
            # Get session
            session = await self.get_chat_session(session_id)
            if not session or "documents" not in session or not session["documents"]:
                return []
            
            # Get documents
            documents = []
            for doc_id in session["documents"]:
                doc = await self.documents.find_one({"_id": ObjectId(doc_id)})
                if doc:
                    doc["_id"] = str(doc["_id"])
                    documents.append(doc)
            
            return documents
        except Exception as e:
            raise ValueError(f"Failed to get session documents: {str(e)}")
            
    async def get_document(self, document_id: str) -> Optional[Dict]:
        """Get a document by ID"""
        try:
            doc = await self.documents.find_one({"_id": ObjectId(document_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception:
            return None
            
    async def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Get all chunks for a document"""
        try:
            cursor = self.document_chunks.find({"documentId": ObjectId(document_id)}).sort("chunkIndex", 1)
            chunks = []
            async for chunk in cursor:
                chunk["_id"] = str(chunk["_id"])
                chunk["documentId"] = str(chunk["documentId"])
                chunks.append(chunk)
            return chunks
        except Exception as e:
            raise ValueError(f"Failed to get document chunks: {str(e)}")
            
    async def delete_document(self, document_id: str, session_id: str) -> bool:
        """Delete a document and its chunks"""
        try:
            # Delete document
            result = await self.documents.delete_one({"_id": ObjectId(document_id)})
            
            if result.deleted_count > 0:
                # Delete all chunks for this document
                await self.document_chunks.delete_many({"documentId": ObjectId(document_id)})
                
                # Remove document ID from session
                await self.chat_sessions.update_one(
                    {"_id": ObjectId(session_id)},
                    {"$pull": {"documents": ObjectId(document_id)}, "$set": {"updatedAt": datetime.utcnow()}}
                )
                
                return True
            return False
        except Exception:
            return False

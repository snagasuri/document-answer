"""
Authentication service for Clerk integration.
"""

import jwt
from fastapi import Depends, HTTPException, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List

from core.config import settings, AUTH_CONFIG
from core.mongodb_service import MongoDBService

security = HTTPBearer()

async def get_mongodb_service():
    """Get MongoDB service instance"""
    return MongoDBService()

async def verify_clerk_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
) -> Dict[str, Any]:
    """Verify Clerk JWT token and return user data"""
    token = credentials.credentials
    
    try:
        # Verify JWT with Clerk's JWKS
        import httpx
        
        # Fetch JWKS from Clerk
        async with httpx.AsyncClient() as client:
            jwks_response = await client.get('https://api.clerk.dev/v1/jwks')
            jwks = jwks_response.json()
            
        # Find the key that matches the token's key ID
        unverified_header = jwt.get_unverified_header(token)
        key_id = unverified_header.get('kid')
        
        if not key_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing key ID")
            
        matching_key = None
        for key in jwks['keys']:
            if key['kid'] == key_id:
                matching_key = key
                break
                
        if not matching_key:
            raise HTTPException(status_code=401, detail="Invalid token: key not found")
            
        # Convert JWK to PEM format
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import serialization
        import base64
        
        # Decode the modulus and exponent
        n = int.from_bytes(base64.urlsafe_b64decode(matching_key['n'] + '=' * (-len(matching_key['n']) % 4)), byteorder='big')
        e = int.from_bytes(base64.urlsafe_b64decode(matching_key['e'] + '=' * (-len(matching_key['e']) % 4)), byteorder='big')
        
        # Create RSA public key
        public_key = rsa.RSAPublicNumbers(e, n).public_key()
        
        # Convert to PEM
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Verify JWT
        payload = jwt.decode(
            token,
            pem,
            algorithms=['RS256'],
            options={"verify_aud": False}  # Clerk uses aud claim differently
        )
        
        # Extract user ID
        clerk_user_id = payload.get("sub")
        if not clerk_user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
            
        # Get or create user in our database
        user = await mongodb_service.get_user(clerk_user_id)
        if not user:
            # Create user if not exists
            email = payload.get("email", "")
            user = await mongodb_service.create_user(clerk_user_id, email)
            
        return {
            "clerk_user_id": clerk_user_id,
            "user": user
        }
        
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

async def get_current_user(auth_data: Dict = Depends(verify_clerk_jwt)) -> Dict:
    """Get current authenticated user"""
    return auth_data["user"]

async def get_clerk_user_id(auth_data: Dict = Depends(verify_clerk_jwt)) -> str:
    """Get current authenticated user's Clerk ID"""
    return auth_data["clerk_user_id"]

# Alternative authentication method using header
async def verify_clerk_header(
    authorization: str = Header(None),
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
) -> Dict[str, Any]:
    """Verify Clerk JWT token from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
        
    scheme, token = authorization.split()
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
    try:
        # Verify JWT with Clerk's JWKS
        import httpx
        
        # Fetch JWKS from Clerk
        async with httpx.AsyncClient() as client:
            jwks_response = await client.get('https://api.clerk.dev/v1/jwks')
            jwks = jwks_response.json()
            
        # Find the key that matches the token's key ID
        unverified_header = jwt.get_unverified_header(token)
        key_id = unverified_header.get('kid')
        
        if not key_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing key ID")
            
        matching_key = None
        for key in jwks['keys']:
            if key['kid'] == key_id:
                matching_key = key
                break
                
        if not matching_key:
            raise HTTPException(status_code=401, detail="Invalid token: key not found")
            
        # Convert JWK to PEM format
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import serialization
        import base64
        
        # Decode the modulus and exponent
        n = int.from_bytes(base64.urlsafe_b64decode(matching_key['n'] + '=' * (-len(matching_key['n']) % 4)), byteorder='big')
        e = int.from_bytes(base64.urlsafe_b64decode(matching_key['e'] + '=' * (-len(matching_key['e']) % 4)), byteorder='big')
        
        # Create RSA public key
        public_key = rsa.RSAPublicNumbers(e, n).public_key()
        
        # Convert to PEM
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Verify JWT
        payload = jwt.decode(
            token,
            pem,
            algorithms=['RS256'],
            options={"verify_aud": False}  # Clerk uses aud claim differently
        )
        
        # Extract user ID
        clerk_user_id = payload.get("sub")
        if not clerk_user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
            
        # Get or create user in our database
        user = await mongodb_service.get_user(clerk_user_id)
        if not user:
            # Create user if not exists
            email = payload.get("email", "")
            user = await mongodb_service.create_user(clerk_user_id, email)
            
        return {
            "clerk_user_id": clerk_user_id,
            "user": user
        }
        
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

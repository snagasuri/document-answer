#!/usr/bin/env python
"""
Install all required dependencies for the chat functionality.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies."""
    print("Installing all required dependencies...")
    
    # List of required packages
    packages = [
        # MongoDB dependencies
        "motor==3.3.2",
        "pymongo==4.6.1",
        
        # Pydantic dependencies
        "pydantic==2.6.1",
        "pydantic-settings==2.1.0",
        
        # RAG dependencies
        "sentence-transformers==2.3.1",
        "pinecone==6.0.1",
        "rank-bm25==0.2.2",
        "scikit-learn==1.4.0",
        "numpy==1.26.3",
        "pandas==2.2.0",
        "cohere>=5.0.0a9",
        "transformers==4.37.2",
        "torch==2.2.0",
        "redis==5.0.1",
        
        # Other dependencies
        "python-dotenv==1.0.1",
        "python-jose[cryptography]==3.3.0",
        "PyJWT==2.8.0",
        "tiktoken==0.5.2",
        "tenacity==8.2.3",
        "httpx==0.26.0",
        "aiohttp==3.9.3"
    ]
    
    # Install each package
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll dependencies installed successfully!")
    print("\nYou can now run the MongoDB initialization script:")
    print("python backend/scripts/init_mongodb.py")

if __name__ == "__main__":
    install_dependencies()

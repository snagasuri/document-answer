#!/usr/bin/env python
"""
Setup script for spaCy models.
This script downloads the required spaCy models for the document processor.
"""

import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_spacy_model(model_name):
    """Download a spaCy model"""
    logger.info(f"Downloading spaCy model: {model_name}")
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True
        )
        logger.info(f"Successfully downloaded {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading {model_name}: {e}")
        return False

def main():
    """Main function to download spaCy models"""
    logger.info("Setting up spaCy models...")
    
    # Primary model - medium size for better accuracy
    primary_success = download_spacy_model("en_core_web_md")
    
    # Fallback model - small size for faster processing
    if not primary_success:
        logger.warning("Failed to download primary model, trying fallback model")
        fallback_success = download_spacy_model("en_core_web_sm")
        
        if not fallback_success:
            logger.error("Failed to download any spaCy model. Document processing may not work correctly.")
            return False
    
    logger.info("spaCy setup completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

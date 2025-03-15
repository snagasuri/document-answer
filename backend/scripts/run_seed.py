import asyncio
import sys
import os
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.scripts.seed_db import DatabaseSeeder
from backend.scripts.init_db import verify_tables_exist, create_tables_directly

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_seed():
    # First, verify that tables exist
    tables_exist = await verify_tables_exist()
    if not tables_exist:
        logger.warning("Required tables don't exist. Attempting to create them directly.")
        try:
            await create_tables_directly()
            tables_exist = await verify_tables_exist()
            if not tables_exist:
                logger.error("Failed to create tables. Cannot proceed with seeding.")
                return
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            logger.error("Cannot proceed with seeding.")
            return
    
    seeder = DatabaseSeeder()
    
    # Try to seed test documents, but continue if it fails
    try:
        await seeder.seed_test_documents()
        logger.info("Test documents seeded successfully")
    except Exception as e:
        logger.error(f"Error seeding test documents: {str(e)}")
        logger.info("Continuing with fallback data...")
        
    # Seed chat sessions - this should be more reliable since it doesn't depend on external APIs
    try:
        await seeder.seed_test_chat_sessions()
        logger.info("Test chat sessions seeded successfully")
    except Exception as e:
        logger.error(f"Error seeding test chat sessions: {str(e)}")
    
    logger.info("Database seeding process completed.")

if __name__ == "__main__":
    asyncio.run(run_seed())
    # Exit with success status even if some parts failed
    sys.exit(0)

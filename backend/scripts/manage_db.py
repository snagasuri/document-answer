import asyncio
import logging
import typer
from pathlib import Path
import subprocess
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text
from datetime import datetime
import json
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer()

async def check_connection():
    """Check database connection"""
    engine = create_async_engine(settings.POSTGRES_URL)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False
    finally:
        await engine.dispose()

@app.command()
def create_db():
    """Create database if it doesn't exist"""
    try:
        # Extract database name from URL
        db_name = settings.POSTGRES_URL.split("/")[-1]
        base_url = settings.POSTGRES_URL.rsplit("/", 1)[0]
        
        # Connect to default database with AUTOCOMMIT isolation level to allow DROP DATABASE commands
        engine = create_async_engine(f"{base_url}/postgres", isolation_level="AUTOCOMMIT")
        
        async def _create_db():
            async with engine.connect() as conn:
                # Disconnect other users
                await conn.execute(
                    text(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{db_name}'")
                )
                # Drop database if exists with autocommit
                await conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}").execution_options(autocommit=True))
                # Create database with autocommit
                await conn.execute(text(f"CREATE DATABASE {db_name}").execution_options(autocommit=True))
                
        asyncio.run(_create_db())
        logger.info(f"Database '{db_name}' created successfully")
        
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        raise typer.Exit(1)

@app.command()
def migrate(revision: str = None):
    """Run database migrations"""
    try:
        # Check connection
        if not asyncio.run(check_connection()):
            raise typer.Exit(1)
            
        # Run migrations
        if revision:
            subprocess.run(["python", "-m", "alembic", "-c", "alembic.ini", "upgrade", revision], check=True, cwd=str(Path(__file__).parent.parent))
        else:
            subprocess.run(["python", "-m", "alembic", "-c", "alembic.ini", "upgrade", "head"], check=True, cwd=str(Path(__file__).parent.parent))
            
        logger.info("Migrations completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Migration failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def rollback(revision: str = "base"):
    """Rollback database migrations"""
    try:
        # Check connection
        if not asyncio.run(check_connection()):
            raise typer.Exit(1)
            
        # Run rollback
        subprocess.run(["python", "-m", "alembic", "-c", "alembic.ini", "downgrade", revision], check=True, cwd=str(Path(__file__).parent.parent))
        logger.info(f"Rolled back to {revision}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Rollback failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def create_migration(message: str):
    """Create new migration"""
    try:
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create migration
        subprocess.run(
            ["python", "-m", "alembic", "-c", "alembic.ini", "revision", "--autogenerate", "-m", message],
            check=True, cwd=str(Path(__file__).parent.parent)
        )
        logger.info(f"Created new migration: {message}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create migration: {str(e)}")
        raise typer.Exit(1)

@app.command()
def export_schema(output: str = "database_schema.json"):
    """Export database schema to JSON"""
    try:
        # Check connection
        if not asyncio.run(check_connection()):
            raise typer.Exit(1)
            
        async def _export_schema():
            engine = create_async_engine(settings.POSTGRES_URL)
            schema = {}
            
            async with engine.connect() as conn:
                # Get tables
                result = await conn.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                """))
                tables = result.fetchall()
                
                # Get columns for each table
                for table in tables:
                    table_name = table[0]
                    result = await conn.execute(text(f"""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_name = '{table_name}'
                    """))
                    columns = result.fetchall()
                    
                    # Get indexes
                    result = await conn.execute(text(f"""
                        SELECT indexname, indexdef
                        FROM pg_indexes
                        WHERE tablename = '{table_name}'
                    """))
                    indexes = result.fetchall()
                    
                    schema[table_name] = {
                        "columns": [
                            {
                                "name": col[0],
                                "type": col[1],
                                "nullable": col[2] == "YES",
                                "default": col[3]
                            }
                            for col in columns
                        ],
                        "indexes": [
                            {
                                "name": idx[0],
                                "definition": idx[1]
                            }
                            for idx in indexes
                        ]
                    }
                    
            await engine.dispose()
            return schema
            
        # Export schema
        schema = asyncio.run(_export_schema())
        
        # Save to file
        with open(output, "w") as f:
            json.dump(schema, f, indent=2)
            
        logger.info(f"Schema exported to {output}")
        
    except Exception as e:
        logger.error(f"Failed to export schema: {str(e)}")
        raise typer.Exit(1)

async def verify_and_create_tables():
    """Verify tables exist and create them if they don't"""
    from backend.scripts.init_db import verify_tables_exist, create_tables_directly
    
    # Check if tables exist
    tables_exist = await verify_tables_exist()
    
    # If tables don't exist, create them directly
    if not tables_exist:
        logger.warning("Tables don't exist after migrations. Using fallback method to create tables directly.")
        await create_tables_directly()
        
        # Verify again
        tables_exist = await verify_tables_exist()
        if not tables_exist:
            logger.error("Failed to create tables even with fallback method.")
            return False
        else:
            logger.info("Tables created successfully using fallback method.")
            return True
    else:
        logger.info("Tables exist after migrations.")
        return True

@app.command(name="reset_db")
def reset_db():
    """Reset database (drop and recreate)"""
    try:
        # Create fresh database
        create_db()
        
        # Run migrations
        migrate()
        
        # Verify tables exist and create them if they don't
        if not asyncio.run(verify_and_create_tables()):
            logger.error("Failed to ensure tables exist.")
            raise typer.Exit(1)
        
        logger.info("Database reset successfully")
        
    except Exception as e:
        logger.error(f"Failed to reset database: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app(prog_name="manage_db.py")

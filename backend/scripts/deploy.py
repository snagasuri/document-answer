import asyncio
import logging
import typer
import subprocess
import os
from pathlib import Path
import json
import time
from typing import Dict
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer()

class Deployer:
    def __init__(self):
        """Initialize deployer"""
        self.project_root = Path(__file__).parent.parent.parent
        self.backend_dir = self.project_root / "backend"
        
        # Vercel configuration
        self.vercel_token = os.getenv("VERCEL_TOKEN")
        self.vercel_org_id = os.getenv("VERCEL_ORG_ID")
        self.vercel_project_id = os.getenv("VERCEL_PROJECT_ID")
        
        # Railway configuration
        self.railway_token = os.getenv("RAILWAY_TOKEN")
        
    async def deploy_backend(self):
        """Deploy backend to Railway"""
        try:
            # Install Railway CLI if not present
            try:
                subprocess.run(["railway", "version"], check=True)
            except:
                logger.info("Installing Railway CLI...")
                subprocess.run(["npm", "i", "-g", "@railway/cli"], check=True)
            
            # Login to Railway
            subprocess.run(
                ["railway", "login", "--token", self.railway_token],
                check=True
            )
            
            # Deploy services
            logger.info("Deploying backend services...")
            subprocess.run(
                ["railway", "up"],
                cwd=str(self.backend_dir),
                check=True
            )
            
            logger.info("Backend deployment completed")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Backend deployment failed: {str(e)}")
            raise
            
    async def deploy_frontend(self):
        """Deploy frontend to Vercel"""
        try:
            # Install Vercel CLI if not present
            try:
                subprocess.run(["vercel", "--version"], check=True)
            except:
                logger.info("Installing Vercel CLI...")
                subprocess.run(["npm", "i", "-g", "vercel"], check=True)
            
            # Deploy to Vercel
            logger.info("Deploying frontend...")
            subprocess.run(
                [
                    "vercel",
                    "deploy",
                    "--prod",
                    "--token", self.vercel_token,
                    "--yes"
                ],
                cwd=str(self.project_root),
                check=True
            )
            
            logger.info("Frontend deployment completed")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Frontend deployment failed: {str(e)}")
            raise
            
    async def setup_infrastructure(self):
        """Set up cloud infrastructure"""
        try:
            # Create Pinecone index if not exists
            logger.info("Setting up Pinecone...")
            import pinecone
            
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT")
            )
            
            index_name = os.getenv("PINECONE_INDEX")
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index: {index_name}")
            
            # Set up Railway services
            logger.info("Setting up Railway services...")
            subprocess.run(
                [
                    "railway", "link",
                    "--environment", "production"
                ],
                cwd=str(self.backend_dir),
                check=True
            )
            
            # Set up environment variables
            env_vars = {
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
                "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
                "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
                "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT"),
                "PINECONE_INDEX": os.getenv("PINECONE_INDEX")
            }
            
            for key, value in env_vars.items():
                subprocess.run(
                    ["railway", "variables", "set", key, value],
                    check=True
                )
                
            logger.info("Infrastructure setup completed")
            
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {str(e)}")
            raise
            
    async def check_deployment_health(self):
        """Check health of deployed services"""
        try:
            # Get deployment URLs
            backend_url = subprocess.check_output(
                ["railway", "domain"],
                text=True
            ).strip()
            
            frontend_url = subprocess.check_output(
                ["vercel", "ls", "--prod", "--json"],
                text=True
            )
            frontend_url = json.loads(frontend_url)[0]["url"]
            
            # Check backend health
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{backend_url}/health")
                if response.status_code == 200:
                    logger.info("Backend health check passed")
                else:
                    logger.error("Backend health check failed")
                    
                # Check frontend
                response = await client.get(frontend_url)
                if response.status_code == 200:
                    logger.info("Frontend health check passed")
                else:
                    logger.error("Frontend health check failed")
                    
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise
            
    async def rollback(self):
        """Rollback deployments"""
        try:
            # Rollback Railway deployment
            subprocess.run(
                ["railway", "down"],
                cwd=str(self.backend_dir),
                check=True
            )
            
            # Rollback Vercel deployment
            subprocess.run(
                ["vercel", "rollback", "--yes"],
                cwd=str(self.project_root),
                check=True
            )
            
            logger.info("Rollback completed")
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            raise

@app.command()
def deploy(frontend: bool = True, backend: bool = True):
    """Deploy application"""
    async def _deploy():
        deployer = Deployer()
        
        try:
            # Set up infrastructure
            await deployer.setup_infrastructure()
            
            # Deploy services
            if backend:
                await deployer.deploy_backend()
            if frontend:
                await deployer.deploy_frontend()
                
            # Check health
            await deployer.check_deployment_health()
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            # Attempt rollback
            await deployer.rollback()
            raise typer.Exit(1)
            
    asyncio.run(_deploy())

@app.command()
def rollback():
    """Rollback deployment"""
    async def _rollback():
        deployer = Deployer()
        await deployer.rollback()
        
    asyncio.run(_rollback())

@app.command()
def setup_infra():
    """Set up cloud infrastructure"""
    async def _setup():
        deployer = Deployer()
        await deployer.setup_infrastructure()
        
    asyncio.run(_setup())

if __name__ == "__main__":
    app()

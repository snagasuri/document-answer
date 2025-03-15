import asyncio
import logging
import os
import subprocess
from pathlib import Path
import shutil
from typing import List, Dict
import yaml
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    def __init__(self):
        """Initialize environment setup"""
        self.project_root = Path(__file__).parent.parent.parent
        self.backend_dir = self.project_root / "backend"
        
    async def check_dependencies(self) -> Dict[str, bool]:
        """Check if required system dependencies are installed"""
        dependencies = {
            "python": "python --version",
            "docker": "docker --version",
            "docker-compose": "docker-compose --version",
            "node": "node --version",
            "npm": "npm --version",
            "redis-cli": "redis-cli --version",
            "psql": "psql --version"
        }
        
        results = {}
        for dep, command in dependencies.items():
            try:
                subprocess.run(
                    command.split(),
                    check=True,
                    capture_output=True,
                    text=True
                )
                results[dep] = True
                logger.info(f"✓ {dep} is installed")
            except subprocess.CalledProcessError:
                results[dep] = False
                logger.error(f"✗ {dep} is not installed")
                
        return results
        
    async def setup_python_environment(self):
        """Set up Python virtual environment and install dependencies"""
        try:
            venv_path = self.backend_dir / "venv"
            
            # Create virtual environment
            if not venv_path.exists():
                logger.info("Creating Python virtual environment...")
                subprocess.run(
                    ["python", "-m", "venv", str(venv_path)],
                    check=True
                )
            
            # Determine pip path and python path
            if os.name != "nt":  # Unix-like systems
                pip_path = venv_path / "bin" / "pip"
                python_path = venv_path / "bin" / "python"
            else:  # Windows
                pip_path = venv_path / "Scripts" / "pip"
                python_path = venv_path / "Scripts" / "python"
            
            # Install requirements
            logger.info("Installing Python dependencies...")
            subprocess.run(
                [str(pip_path), "install", "-r", str(self.backend_dir / "requirements.txt")],
                check=True
            )
            
            # Install development requirements
            dev_requirements = [
                "pytest",
                "pytest-asyncio",
                "pytest-cov",
                "black",
                "isort",
                "mypy",
                "pylint"
            ]
            
            logger.info("Installing development dependencies...")
            subprocess.run(
                [str(pip_path), "install"] + dev_requirements,
                check=True
            )
            
            # Set up spaCy models
            logger.info("Setting up spaCy models...")
            subprocess.run(
                [str(python_path), str(self.backend_dir / "scripts" / "setup_spacy.py")],
                check=True
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error setting up Python environment: {str(e)}")
            raise
            
    async def setup_environment_variables(self):
        """Set up environment variables from template"""
        try:
            env_template = self.backend_dir / ".env.template"
            env_file = self.backend_dir / ".env"
            
            if not env_file.exists():
                logger.info("Creating .env file from template...")
                shutil.copy(env_template, env_file)
                logger.info("Please update .env with your API keys and settings")
            else:
                logger.info(".env file already exists")
                
            # Validate required variables
            load_dotenv(env_file)
            required_vars = [
                "OPENROUTER_API_KEY",
                "COHERE_API_KEY",
                "PINECONE_API_KEY",
                "PINECONE_ENVIRONMENT",
                "PINECONE_INDEX"
            ]
            
            missing_vars = [
                var for var in required_vars
                if not os.getenv(var)
            ]
            
            if missing_vars:
                logger.warning(f"Missing required environment variables: {missing_vars}")
                
        except Exception as e:
            logger.error(f"Error setting up environment variables: {str(e)}")
            raise
            
    async def setup_docker_environment(self):
        """Set up Docker environment"""
        try:
            # Check if containers are running
            result = subprocess.run(
                ["docker-compose", "ps", "-q"],
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                logger.info("Stopping existing containers...")
                subprocess.run(
                    ["docker-compose", "down"],
                    check=True
                )
            
            # Build and start containers
            logger.info("Building and starting Docker containers...")
            subprocess.run(
                ["docker-compose", "up", "-d", "--build"],
                check=True
            )
            
            # Wait for services to be ready
            await self.wait_for_services()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error setting up Docker environment: {str(e)}")
            raise
            
    async def wait_for_services(self, timeout: int = 60):
        """Wait for services to be ready"""
        services = {
            "PostgreSQL": 5432,
            "Redis": 6379,
            "Backend API": 8000
        }
        
        for service, port in services.items():
            start_time = asyncio.get_event_loop().time()
            while True:
                try:
                    # Try to connect to service
                    proc = await asyncio.create_subprocess_exec(
                        "nc", "-z", "localhost", str(port),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    await proc.communicate()
                    
                    if proc.returncode == 0:
                        logger.info(f"✓ {service} is ready")
                        break
                        
                except Exception:
                    pass
                
                if asyncio.get_event_loop().time() - start_time > timeout:
                    logger.error(f"Timeout waiting for {service}")
                    raise TimeoutError(f"{service} did not become ready in {timeout} seconds")
                    
                await asyncio.sleep(1)
                
    async def initialize_services(self):
        """Initialize database and other services"""
        try:
            # Initialize database
            logger.info("Initializing database...")
            subprocess.run(
                ["python", str(self.backend_dir / "scripts" / "init_db.py")],
                check=True
            )
            
            # Initialize Pinecone
            logger.info("Initializing Pinecone...")
            subprocess.run(
                ["python", str(self.backend_dir / "scripts" / "init_pinecone.py")],
                check=True
            )
            
            # Run tests
            logger.info("Running tests...")
            subprocess.run(
                ["python", str(self.backend_dir / "scripts" / "test_rag_pipeline.py")],
                check=True
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error initializing services: {str(e)}")
            raise
            
    async def setup_monitoring(self):
        """Set up monitoring and logging"""
        try:
            # Start monitoring service
            logger.info("Starting monitoring service...")
            subprocess.Popen(
                ["python", str(self.backend_dir / "scripts" / "monitor_rag.py")],
                start_new_session=True
            )
            
            logger.info("Monitoring service started on port 8001")
            
        except Exception as e:
            logger.error(f"Error setting up monitoring: {str(e)}")
            raise

async def main():
    """Run complete environment setup"""
    setup = EnvironmentSetup()
    
    try:
        # Check dependencies
        deps = await setup.check_dependencies()
        if not all(deps.values()):
            logger.error("Please install missing dependencies before continuing")
            return
            
        # Setup steps
        await setup.setup_python_environment()
        await setup.setup_environment_variables()
        await setup.setup_docker_environment()
        await setup.initialize_services()
        await setup.setup_monitoring()
        
        logger.info("Environment setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

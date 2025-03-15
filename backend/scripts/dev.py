import asyncio
import logging
import typer
import subprocess
import os
import signal
import time
from pathlib import Path
import webbrowser
from typing import List, Dict
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer()

class DevEnvironment:
    def __init__(self):
        """Initialize development environment"""
        self.project_root = Path(__file__).parent.parent.parent
        self.backend_dir = self.project_root / "backend"
        self.processes: Dict[str, subprocess.Popen] = {}
        
    def start_service(self, name: str, command: List[str], cwd: str = None):
        """Start a service process"""
        try:
            logger.info(f"Starting {name}...")
            process = subprocess.Popen(
                command,
                cwd=cwd or str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.processes[name] = process
            logger.info(f"{name} started with PID {process.pid}")
            return process
        except Exception as e:
            logger.error(f"Failed to start {name}: {str(e)}")
            raise
            
    def stop_service(self, name: str):
        """Stop a service process"""
        if name in self.processes:
            process = self.processes[name]
            logger.info(f"Stopping {name} (PID {process.pid})...")
            
            # Kill process and children
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            parent.kill()
            
            process.wait()
            del self.processes[name]
            logger.info(f"{name} stopped")
            
    def stop_all(self):
        """Stop all services"""
        for name in list(self.processes.keys()):
            self.stop_service(name)
            
    def check_service_health(self, name: str, url: str, timeout: int = 30):
        """Check if a service is healthy"""
        import httpx
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(url)
                if response.status_code == 200:
                    logger.info(f"{name} is healthy")
                    return True
            except:
                pass
            time.sleep(1)
            
        logger.error(f"{name} failed to become healthy")
        return False

@app.command()
def setup():
    """Set up development environment"""
    dev = DevEnvironment()
    
    try:
        # Install Python dependencies
        logger.info("Installing Python dependencies...")
        subprocess.run(
            ["pip", "install", "-r", "requirements.txt"],
            cwd=str(dev.backend_dir),
            check=True
        )
        
        # Install Node.js dependencies
        logger.info("Installing Node.js dependencies...")
        subprocess.run(
            ["npm", "install"],
            cwd=str(dev.project_root),
            check=True
        )
        
        # Initialize environment
        logger.info("Initializing environment...")
        subprocess.run(
            ["python", "scripts/setup_environment.py"],
            cwd=str(dev.backend_dir),
            check=True
        )
        
        logger.info("Development environment setup complete")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Setup failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def start(frontend: bool = True, skip_seed: bool = False):
    """Start development servers"""
    dev = DevEnvironment()
    
    try:
        # Start PostgreSQL and Redis
        dev.start_service(
            "docker",
            ["docker-compose", "up", "-d", "postgres", "redis"]
        )
        
        # Wait for services
        time.sleep(5)
        
        # Initialize database
        subprocess.run(
            ["python", "scripts/manage_db.py", "reset_db"],
            cwd=str(dev.backend_dir),
            check=True
        )
        
        # Seed test data (if not skipped)
        if not skip_seed:
            logger.info("Seeding test data (use --skip-seed to skip this step)")
            subprocess.run(
                ["python", "scripts/run_seed.py"],
                cwd=str(dev.backend_dir),
                check=True
            )
        else:
            logger.info("Skipping test data seeding")
        
        # Start backend server
        dev.start_service(
            "backend",
            ["python", "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"],
            cwd=str(dev.backend_dir)
        )
        
        # Start monitoring
        dev.start_service(
            "monitor",
            ["python", "scripts/monitor_rag.py"],
            cwd=str(dev.backend_dir)
        )
        
        # Start frontend server
        if frontend:
            dev.start_service(
                "frontend",
                ["npm", "run", "dev"]
            )
            
        # Wait for servers to be ready
        if dev.check_service_health("backend", "http://localhost:8000/health"):
            logger.info("Backend API is ready")
            webbrowser.open("http://localhost:8000/docs")  # Open API docs
            
        if frontend and dev.check_service_health("frontend", "http://localhost:3000"):
            logger.info("Frontend is ready")
            webbrowser.open("http://localhost:3000")  # Open frontend
            
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            dev.stop_all()
            
    except Exception as e:
        logger.error(f"Failed to start development servers: {str(e)}")
        dev.stop_all()
        raise typer.Exit(1)

@app.command()
def test(coverage: bool = True):
    """Run tests"""
    dev = DevEnvironment()
    
    try:
        # Run Python tests
        cmd = ["pytest"]
        if coverage:
            cmd.extend(["--cov=.", "--cov-report=html"])
        subprocess.run(
            cmd,
            cwd=str(dev.backend_dir),
            check=True
        )
        
        # Run frontend tests
        subprocess.run(
            ["npm", "test"],
            cwd=str(dev.project_root),
            check=True
        )
        
        logger.info("All tests passed")
        
        if coverage:
            webbrowser.open(str(dev.backend_dir / "coverage_html" / "index.html"))
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def lint():
    """Run linters"""
    dev = DevEnvironment()
    
    try:
        # Run Python linters
        subprocess.run(
            ["black", "."],
            cwd=str(dev.backend_dir),
            check=True
        )
        subprocess.run(
            ["isort", "."],
            cwd=str(dev.backend_dir),
            check=True
        )
        subprocess.run(
            ["mypy", "."],
            cwd=str(dev.backend_dir),
            check=True
        )
        subprocess.run(
            ["pylint", "app", "core", "models", "scripts"],
            cwd=str(dev.backend_dir),
            check=True
        )
        
        # Run frontend linters
        subprocess.run(
            ["npm", "run", "lint"],
            cwd=str(dev.project_root),
            check=True
        )
        
        logger.info("Linting completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Linting failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def clean():
    """Clean development environment"""
    dev = DevEnvironment()
    
    try:
        # Stop all services
        subprocess.run(
            ["docker-compose", "down"],
            cwd=str(dev.project_root),
            check=True
        )
        
        # Clean Python cache
        subprocess.run(
            ["find", ".", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-r", "{}", "+"],
            cwd=str(dev.backend_dir)
        )
        subprocess.run(
            ["find", ".", "-type", "d", "-name", ".pytest_cache", "-exec", "rm", "-r", "{}", "+"],
            cwd=str(dev.backend_dir)
        )
        
        # Clean coverage reports
        subprocess.run(
            ["rm", "-rf", "coverage_html", ".coverage"],
            cwd=str(dev.backend_dir)
        )
        
        # Clean logs
        subprocess.run(
            ["rm", "-f", "*.log"],
            cwd=str(dev.backend_dir)
        )
        
        logger.info("Development environment cleaned")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()

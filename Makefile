# Development environment setup
.PHONY: setup
setup:
	# Create Python virtual environment
	python -m venv backend/venv
	# Install Python dependencies
	cd backend && . venv/bin/activate && pip install -r requirements.txt
	# Install spaCy model
	cd backend && . venv/bin/activate && python -m spacy download en_core_web_sm
	# Install Node.js dependencies
	npm install

# Environment initialization
.PHONY: init
init:
	# Initialize environment
	cd backend && python scripts/setup_environment.py
	# Initialize database
	cd backend && python scripts/init_db.py
	# Initialize Pinecone
	cd backend && python scripts/init_pinecone.py

# Development servers
.PHONY: dev
dev:
	# Start development servers
	docker-compose up -d postgres redis
	cd backend && . venv/bin/activate && uvicorn app.main:app --reload --port 8000 &
	npm run dev

# Testing
.PHONY: test
test:
	# Run Python tests
	cd backend && . venv/bin/activate && pytest
	# Run frontend tests
	npm test

# Linting and formatting
.PHONY: lint
lint:
	# Python linting
	cd backend && . venv/bin/activate && \
		black . && \
		isort . && \
		mypy . && \
		pylint app core models scripts
	# JavaScript/TypeScript linting
	npm run lint

# RAG pipeline evaluation
.PHONY: evaluate-rag
evaluate-rag:
	# Generate test data
	cd backend && . venv/bin/activate && python scripts/generate_test_data.py
	# Run AutoRAG optimization
	cd backend && . venv/bin/activate && python scripts/optimize_rag.py
	# Run benchmarks
	cd backend && . venv/bin/activate && python scripts/benchmark_rag.py
	# Run evaluation
	cd backend && . venv/bin/activate && python scripts/evaluate_rag.py

# Monitoring
.PHONY: monitor
monitor:
	# Start monitoring services
	cd backend && . venv/bin/activate && python scripts/monitor_rag.py &
	# Open Prometheus metrics
	open http://localhost:8001

# Deployment
.PHONY: deploy
deploy:
	# Build and push Docker images
	docker-compose build
	# Deploy to production
	railway up

# Cleanup
.PHONY: clean
clean:
	# Stop all services
	docker-compose down
	# Remove Python virtual environment
	rm -rf backend/venv
	# Remove Node modules
	rm -rf node_modules
	# Remove cached files
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	# Remove logs
	rm -f *.log
	rm -rf logs/
	rm -rf debug_logs/
	rm -rf evaluation_results/
	rm -rf benchmark_results/

# Help
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make setup          - Set up development environment"
	@echo "  make init          - Initialize services and databases"
	@echo "  make dev           - Start development servers"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Run linters and formatters"
	@echo "  make evaluate-rag  - Run RAG pipeline evaluation"
	@echo "  make monitor       - Start monitoring services"
	@echo "  make deploy        - Deploy to production"
	@echo "  make clean         - Clean up development environment"

# Default target
.DEFAULT_GOAL := help

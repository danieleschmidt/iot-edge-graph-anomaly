.PHONY: help install install-dev clean test test-unit test-integration test-e2e lint format type-check security-check build docker-build docker-run docker-stop dev-setup pre-commit docs docs-serve

# ── CONFIGURATION ──
PYTHON := python3
PIP := pip
VENV := venv
APP_NAME := iot-edge-graph-anomaly
IMAGE_NAME := $(APP_NAME)
CONTAINER_NAME := $(APP_NAME)-container

# ── COLORS ──
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)IoT Edge Graph Anomaly Detection - Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# ── ENVIRONMENT SETUP ──
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install -e .[dev]
	pre-commit install

dev-setup: ## Complete development environment setup
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip setuptools wheel
	./$(VENV)/bin/pip install -e .[dev]
	./$(VENV)/bin/pre-commit install
	@echo "$(GREEN)Development environment ready! Activate with: source $(VENV)/bin/activate$(RESET)"

# ── CLEANING ──
clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# ── TESTING ──
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	pytest

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	pytest tests/ -m "not integration and not e2e" -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest tests/ -m "integration" -v

test-e2e: ## Run end-to-end tests only
	@echo "$(BLUE)Running end-to-end tests...$(RESET)"
	pytest tests/ -m "e2e" -v

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	pytest-watch

# ── CODE QUALITY ──
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	black src/ tests/
	isort src/ tests/

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(RESET)"
	mypy src/

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	bandit -r src/
	safety check

pre-commit: ## Run all pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

# ── BUILD & PACKAGE ──
build: ## Build package
	@echo "$(BLUE)Building package...$(RESET)"
	$(PYTHON) -m build

# ── DOCKER ──
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(IMAGE_NAME):latest .

docker-build-arm64: ## Build Docker image for ARM64 (Raspberry Pi)
	@echo "$(BLUE)Building ARM64 Docker image...$(RESET)"
	docker buildx build --platform linux/arm64 -t $(IMAGE_NAME):arm64 .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run -d --name $(CONTAINER_NAME) -p 8000:8000 -p 9090:9090 $(IMAGE_NAME):latest

docker-stop: ## Stop and remove Docker container
	@echo "$(BLUE)Stopping Docker container...$(RESET)"
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

docker-logs: ## View Docker container logs
	docker logs -f $(CONTAINER_NAME)

# ── DOCUMENTATION ──
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(RESET)"
	cd docs/_build/html && $(PYTHON) -m http.server 8080

# ── DEVELOPMENT ──
run: ## Run the application locally
	@echo "$(BLUE)Starting application...$(RESET)"
	$(PYTHON) -m src.iot_edge_anomaly.main

run-dev: ## Run the application in development mode
	@echo "$(BLUE)Starting application in development mode...$(RESET)"
	DEBUG=true $(PYTHON) -m src.iot_edge_anomaly.main

# ── MONITORING ──
health-check: ## Check application health
	@echo "$(BLUE)Checking application health...$(RESET)"
	curl -f http://localhost:8080/health || echo "$(RED)Health check failed$(RESET)"

metrics: ## View application metrics
	@echo "$(BLUE)Fetching application metrics...$(RESET)"
	curl -s http://localhost:9090/metrics

# ── RELEASE ──
release-patch: ## Create a patch release
	@echo "$(BLUE)Creating patch release...$(RESET)"
	bumpversion patch

release-minor: ## Create a minor release
	@echo "$(BLUE)Creating minor release...$(RESET)"
	bumpversion minor

release-major: ## Create a major release
	@echo "$(BLUE)Creating major release...$(RESET)"
	bumpversion major

# ── UTILITIES ──
install-hooks: ## Install git hooks
	@echo "$(BLUE)Installing git hooks...$(RESET)"
	pre-commit install --hook-type pre-commit
	pre-commit install --hook-type pre-push

update-deps: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e .[dev]

check-deps: ## Check for dependency vulnerabilities
	@echo "$(BLUE)Checking dependencies for vulnerabilities...$(RESET)"
	safety check
	pip-audit
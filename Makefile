.PHONY: help install install-dev test test-cov lint format type-check security build clean docker-build docker-run docs serve-docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting (flake8)"
	@echo "  format       Format code (black, isort)"
	@echo "  type-check   Run type checking (mypy)"
	@echo "  security     Run security checks (bandit, safety)"
	@echo "  build        Build distribution packages"
	@echo "  clean        Clean build artifacts"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo "  docs         Build documentation"
	@echo "  serve-docs   Serve documentation locally"
	@echo "  all-checks   Run all quality checks (lint, type, security, test)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev,test,docs]
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term --cov-report=xml

test-watch:
	pytest-watch tests/ --runner "pytest --cov=src"

# Code Quality
lint:
	flake8 src tests
	isort --check-only --diff src tests

format:
	black src tests
	isort src tests

type-check:
	mypy src

security:
	bandit -r src
	safety check

# All quality checks
all-checks: lint type-check security test-cov

# Build
build: clean
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Docker
docker-build:
	docker build -t iot-edge-graph-anomaly:latest .

docker-run:
	docker run --rm -p 8080:8080 iot-edge-graph-anomaly:latest

docker-dev:
	docker build -f Dockerfile.dev -t iot-edge-graph-anomaly:dev .
	docker run --rm -it -v $(PWD):/app -p 8080:8080 iot-edge-graph-anomaly:dev

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs && make html && python -m http.server 8000 --directory _build/html

# Development workflow shortcuts
dev-setup: install-dev
	@echo "Development environment setup complete!"

quick-test:
	pytest tests/ -x --ff

# Production deployment helpers
deploy-check: all-checks
	@echo "All checks passed - ready for deployment"

version-bump-patch:
	bump2version patch

version-bump-minor:
	bump2version minor

version-bump-major:
	bump2version major
# Multi-stage Dockerfile for Sentiment Analyzer Pro
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r sentiment && useradd -r -g sentiment sentiment

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md LICENSE ./

# Install package in development mode for now
RUN pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R sentiment:sentiment /app

# Switch to non-root user
USER sentiment

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "sentiment_analyzer.api.server"]

# Production stage
FROM base as production

# Install production dependencies only
COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Remove development dependencies
RUN pip uninstall -y pytest pytest-cov black flake8 mypy

# Set production environment
ENV ENVIRONMENT=production \
    SENTIMENT_LOG_LEVEL=info \
    SENTIMENT_METRICS_ENABLED=true

# Production command with gunicorn
CMD ["gunicorn", "sentiment_analyzer.api.server:create_app", \
     "--factory", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--keep-alive", "30", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100"]
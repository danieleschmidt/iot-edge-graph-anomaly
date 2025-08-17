# Production Multi-stage Build for Terragon IoT Edge Anomaly Detection v4.0
# Optimized for edge deployment with advanced AI capabilities
# Stage 1: Builder - Install dependencies and build wheel
FROM --platform=linux/arm64 python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and build wheel
COPY src/ src/
COPY setup.py ./
RUN pip wheel --no-deps --wheel-dir /build/dist .

# Stage 2: Runtime - Minimal production image
FROM --platform=linux/arm64 python:3.12-slim

# Create non-root user for security
RUN groupadd -r iotuser && useradd -r -g iotuser iotuser

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install wheel from builder stage
COPY --from=builder /build/dist/*.whl ./
RUN pip install --no-cache-dir *.whl && rm *.whl

# Copy configuration files
COPY config/ config/

# Set resource limits for edge device compatibility
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=2

# Change to non-root user
USER iotuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import iot_edge_anomaly; print('OK')" || exit 1

# Expose port for metrics
EXPOSE 8080

# Default command - Use advanced main with full capability
CMD ["python", "-m", "iot_edge_anomaly.advanced_main", "--config", "config/default.yaml"]
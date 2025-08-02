# Build and Deployment Guide

This guide covers building, containerizing, and deploying the IoT Edge Graph Anomaly Detection system.

## 🏗️ Build Process

### Local Development Build

```bash
# Install development dependencies
make install-dev

# Run linting and type checking
make lint
make type-check

# Run tests
make test

# Build Python package
make build
```

### Container Build

#### Standard Build
```bash
# Build for current platform
make docker-build

# Build for ARM64 (Raspberry Pi)
make docker-build-arm64

# Build multi-architecture image
docker buildx build --platform linux/amd64,linux/arm64 -t iot-edge-anomaly:latest .
```

#### Development Build
```bash
# Build with development tools
docker build --target development -t iot-edge-anomaly:dev .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.9 -t iot-edge-anomaly:py39 .
```

## 📦 Container Architecture

### Multi-Stage Build

```dockerfile
# Stage 1: Builder
FROM python:3.10-slim as builder
# Install dependencies and build wheel

# Stage 2: Runtime  
FROM python:3.10-slim
# Minimal runtime environment
```

### Security Hardening

- **Non-root user**: Application runs as `iotuser`
- **Minimal base image**: Python slim image
- **Read-only filesystem**: Most directories are read-only
- **No package cache**: Reduces attack surface
- **Health checks**: Built-in container health monitoring

### Resource Optimization

- **Multi-stage build**: Reduces final image size by ~60%
- **ARM64 optimized**: Native support for Raspberry Pi 4
- **Memory limits**: <100MB RAM usage
- **CPU optimization**: Limited to 2 threads for edge devices

## 🚀 Deployment Options

### 1. Single Container Deployment

```bash
# Basic deployment
docker run -d \
  --name iot-edge-anomaly \
  -p 8000:8000 \
  -p 8080:8080 \
  -v ./models:/app/models:ro \
  -v ./data:/app/data:ro \
  iot-edge-anomaly:latest
```

### 2. Docker Compose Deployment

```bash
# Full stack with monitoring
docker-compose up -d

# Development stack
docker-compose --profile dev up -d

# With data simulation
docker-compose --profile simulation up -d

# With MQTT broker
docker-compose --profile mqtt up -d
```

### 3. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iot-edge-anomaly
spec:
  replicas: 1
  selector:
    matchLabels:
      app: iot-edge-anomaly
  template:
    metadata:
      labels:
        app: iot-edge-anomaly
    spec:
      containers:
      - name: iot-edge-anomaly
        image: iot-edge-anomaly:latest
        ports:
        - containerPort: 8000
        - containerPort: 8080
        resources:
          limits:
            memory: "256Mi"
            cpu: "500m"
          requests:
            memory: "128Mi"
            cpu: "250m"
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: ANOMALY_THRESHOLD
          value: "0.5"
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        - name: data
          mountPath: /app/data
          readOnly: true
      volumes:
      - name: models
        hostPath:
          path: /opt/iot-models
      - name: data
        hostPath:
          path: /opt/iot-data
```

## 🔧 Configuration Management

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `ANOMALY_THRESHOLD` | `0.5` | Anomaly detection threshold |
| `MODEL_PATH` | `/app/models/model.pth` | Path to model file |
| `OTLP_ENDPOINT` | `http://localhost:4317` | OpenTelemetry endpoint |
| `DEVICE_ID` | `edge-device-001` | Unique device identifier |
| `MEMORY_LIMIT` | `100` | Memory limit in MB |
| `CPU_LIMIT` | `0.25` | CPU limit (0.25 = 25%) |

### Volume Mounts

- `/app/models` - Model files (read-only)
- `/app/data` - Input data (read-only)  
- `/app/logs` - Application logs (read-write)
- `/app/config` - Configuration files (read-only)

## 📊 Monitoring Integration

### Health Checks

```bash
# Container health check
curl http://localhost:8080/health

# Detailed health information
curl http://localhost:8080/health/detailed
```

### Metrics Export

```bash
# Prometheus metrics
curl http://localhost:9090/metrics

# Application metrics
curl http://localhost:8000/metrics
```

### Log Aggregation

```yaml
# docker-compose.yml logging configuration
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## 🔒 Security Considerations

### Container Security

- **User privileges**: Non-root execution
- **Filesystem**: Read-only where possible
- **Network**: Minimal port exposure
- **Secrets**: External secret management
- **Scanning**: Regular vulnerability scans

### Image Security

```bash
# Scan for vulnerabilities
docker scout cves iot-edge-anomaly:latest

# Security benchmark
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image iot-edge-anomaly:latest
```

## 🎯 Performance Optimization

### Build Optimization

```dockerfile
# Use specific base image tags
FROM python:3.10.12-slim

# Combine RUN commands to reduce layers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir requirements

# Use .dockerignore to exclude unnecessary files
COPY .dockerignore ./
```

### Runtime Optimization

```bash
# Set resource limits
docker run --memory=256m --cpus="0.5" iot-edge-anomaly:latest

# Use read-only filesystem
docker run --read-only --tmpfs /tmp iot-edge-anomaly:latest

# Optimize for ARM64
docker run --platform linux/arm64 iot-edge-anomaly:arm64
```

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
name: Build and Deploy
on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: |
          iot-edge-anomaly:latest
          iot-edge-anomaly:${{ github.sha }}
```

### Automated Testing

```bash
# Test container build
docker build -t iot-edge-anomaly:test .

# Test container startup
docker run --rm -d --name test-container iot-edge-anomaly:test
sleep 30
docker exec test-container python -c "import iot_edge_anomaly; print('OK')"
docker stop test-container
```

## 🚨 Troubleshooting

### Common Build Issues

#### Issue: "No space left on device"
```bash
# Clean up Docker system
docker system prune -a

# Remove unused volumes
docker volume prune
```

#### Issue: "Layer does not exist"
```bash
# Clear build cache
docker builder prune

# Rebuild without cache
docker build --no-cache -t iot-edge-anomaly:latest .
```

#### Issue: "ARM64 build fails"
```bash
# Install QEMU emulation
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Use buildx for cross-platform builds
docker buildx create --use
docker buildx build --platform linux/arm64 -t iot-edge-anomaly:arm64 .
```

### Runtime Issues

#### Issue: "Permission denied"
```bash
# Check user permissions
docker exec -it container-name id

# Fix volume permissions
sudo chown -R 1000:1000 ./models ./data
```

#### Issue: "Container exits immediately"
```bash
# Check logs
docker logs container-name

# Run in interactive mode
docker run -it --entrypoint /bin/bash iot-edge-anomaly:latest
```

#### Issue: "Out of memory"
```bash
# Check resource usage
docker stats container-name

# Increase memory limit
docker run --memory=512m iot-edge-anomaly:latest
```

## 📈 Performance Benchmarks

### Build Performance

- **Full build time**: ~3-5 minutes
- **Incremental build**: ~30-60 seconds
- **Multi-arch build**: ~8-12 minutes
- **Image size**: ~200-300MB

### Runtime Performance

- **Startup time**: <30 seconds
- **Memory usage**: <100MB
- **CPU usage**: <25% on Raspberry Pi 4
- **Inference latency**: <10ms per sample

---

**Build Documentation Version**: 1.0  
**Last Updated**: 2025-01-27  
**Next Review**: 2025-04-27  
**Maintainer**: DevOps Team
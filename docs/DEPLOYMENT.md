# Deployment Guide

## Overview

This guide covers deployment of the IoT Edge Graph Anomaly Detection system across different environments, from development to production edge devices.

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/terragonlabs/iot-edge-graph-anomaly.git
cd iot-edge-graph-anomaly

# Local development with Docker
make docker-build
make docker-run

# Access application
curl http://localhost:8080/health
```

### Production Edge Device

```bash
# Single-command deployment
docker run -d \
  --name iot-edge-anomaly \
  --restart unless-stopped \
  -p 8000:8000 \
  -p 8080:8080 \
  -p 9090:9090 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/config:/app/config:ro \
  --memory=256m \
  --cpus=0.5 \
  iot-edge-anomaly:latest
```

## Deployment Environments

### Development Environment

Perfect for local development and testing:

```bash
# Full development stack
docker-compose --profile dev up -d

# Services included:
# - Main application
# - Jupyter notebook server
# - Full monitoring stack
# - Development tools
```

#### Development Services

| Service | Port | Purpose |
|---------|------|---------|
| Application | 8000 | Main anomaly detection service |
| Health Check | 8080 | Health monitoring endpoint |
| Metrics | 9090 | Prometheus metrics export |
| Jupyter | 8888 | Development notebook environment |
| Grafana | 3000 | Monitoring dashboards |
| Prometheus | 9091 | Metrics storage |

### Staging Environment

Production-like environment for testing:

```bash
# Staging deployment
docker-compose up -d

# Includes:
# - Application container
# - Monitoring stack
# - Simulated sensor data
```

### Production Edge Deployment

Optimized for resource-constrained edge devices:

```bash
# Minimal production deployment
docker run -d \
  --name iot-edge-anomaly \
  --restart unless-stopped \
  --memory=128m \
  --cpus=0.25 \
  --read-only \
  --tmpfs /tmp \
  --user 1001:1001 \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  -v logs:/app/logs \
  -e LOG_LEVEL=INFO \
  -e OTLP_ENDPOINT=https://monitoring.example.com:4317 \
  iot-edge-anomaly:latest
```

## Platform-Specific Deployments

### Raspberry Pi Deployment

#### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Enable Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Install Docker Compose
sudo pip3 install docker-compose
```

#### ARM64 Container

```bash
# Pull ARM64 image
docker pull iot-edge-anomaly:arm64

# Run optimized for Pi
docker run -d \
  --name iot-edge-anomaly \
  --restart unless-stopped \
  --memory=100m \
  --cpus=0.5 \
  --device-cgroup-rule='c 89:* rmw' \
  -p 8080:8080 \
  -v /opt/iot-models:/app/models:ro \
  -e DEVICE_ID=rpi-$(hostname) \
  -e LOG_LEVEL=WARN \
  iot-edge-anomaly:arm64
```

#### Pi-Specific Optimizations

```bash
# Increase GPU memory split
echo 'gpu_mem=16' | sudo tee -a /boot/config.txt

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-powersave

# Set CPU governor to performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Industrial Edge Gateway

For industrial IoT gateways with more resources:

```bash
# Full monitoring deployment
docker-compose -f docker-compose.yml -f docker-compose.industrial.yml up -d

# Industrial-grade configuration
version: '3.8'
services:
  iot-edge-anomaly:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - ANOMALY_THRESHOLD=0.3
      - BATCH_SIZE=64
      - MONITORING_ENABLED=true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 30s
```

### Cloud Edge Deployment

For cloud-managed edge deployments:

```bash
# AWS IoT Greengrass
# Azure IoT Edge
# Google Cloud IoT Edge
```

## Kubernetes Deployment

### Basic Kubernetes Manifest

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iot-edge-anomaly
  labels:
    app: iot-edge-anomaly
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
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: iot-edge-anomaly
        image: iot-edge-anomaly:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        - containerPort: 8080
        - containerPort: 9090
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: OTLP_ENDPOINT
          value: "http://otel-collector:4317"
        resources:
          limits:
            memory: "256Mi"
            cpu: "500m"
          requests:
            memory: "128Mi"
            cpu: "250m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: models
        configMap:
          name: iot-models
      - name: config
        configMap:
          name: iot-config
---
apiVersion: v1
kind: Service
metadata:
  name: iot-edge-anomaly-service
spec:
  selector:
    app: iot-edge-anomaly
  ports:
  - name: http
    protocol: TCP
    port: 8000
    targetPort: 8000
  - name: health
    protocol: TCP
    port: 8080
    targetPort: 8080
  - name: metrics
    protocol: TCP
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

### Helm Chart Deployment

```bash
# Create Helm chart
helm create iot-edge-anomaly

# Install with custom values
helm install iot-edge-anomaly ./charts/iot-edge-anomaly \
  --set image.tag=v0.1.0 \
  --set resources.limits.memory=256Mi \
  --set monitoring.enabled=true
```

## Configuration Management

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARN, ERROR) |
| `ANOMALY_THRESHOLD` | `0.5` | Anomaly detection threshold |
| `MODEL_PATH` | `./models/lstm_autoencoder.pth` | Path to trained model |
| `OTLP_ENDPOINT` | `http://localhost:4317` | OpenTelemetry endpoint |
| `DEVICE_ID` | `edge-device-001` | Unique device identifier |
| `BATCH_SIZE` | `32` | Inference batch size |
| `SEQUENCE_LENGTH` | `10` | Input sequence length |
| `MEMORY_LIMIT` | `100M` | Memory usage limit |
| `CPU_LIMIT` | `0.25` | CPU usage limit |

### Configuration File

```yaml
# config/application.yaml
model:
  path: "/app/models/lstm_autoencoder.pth"
  input_size: 51
  hidden_size: 64
  num_layers: 2
  sequence_length: 10
  dropout_rate: 0.2

detection:
  threshold: 0.5
  window_size: 60
  cooldown_period: 300

monitoring:
  otlp_endpoint: "http://otel-collector:4317"
  metrics_interval: 30
  health_check_interval: 10

edge:
  device_id: "edge-device-001"
  location: "factory-floor-1"
  memory_limit: "100M"
  cpu_limit: 0.25

logging:
  level: "INFO"
  format: "json"
  rotation: "daily"
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'iot-edge-anomaly'
    static_configs:
      - targets: ['iot-edge-anomaly:9090']
    scrape_interval: 10s
    metrics_path: /metrics

rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "IoT Edge Anomaly Detection",
    "panels": [
      {
        "title": "Anomaly Detection Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(anomaly_count_total[5m])"
          }
        ]
      },
      {
        "title": "Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, inference_duration_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

## Security Configuration

### Container Security

```bash
# Run with security hardening
docker run -d \
  --name iot-edge-anomaly \
  --restart unless-stopped \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=50m \
  --user 1001:1001 \
  --cap-drop ALL \
  --cap-add NET_BIND_SERVICE \
  --security-opt no-new-privileges:true \
  --security-opt seccomp=seccomp-profile.json \
  --memory=256m \
  --cpus=0.5 \
  iot-edge-anomaly:latest
```

### Network Security

```bash
# Create secure network
docker network create --driver bridge \
  --subnet=172.20.0.0/16 \
  --opt com.docker.network.bridge.enable_icc=false \
  iot-secure-network

# Deploy with network isolation
docker run -d \
  --name iot-edge-anomaly \
  --network iot-secure-network \
  --ip 172.20.0.10 \
  iot-edge-anomaly:latest
```

### TLS Configuration

```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Mount certificates
docker run -d \
  -v $(pwd)/certs:/app/certs:ro \
  -e TLS_ENABLED=true \
  -e TLS_CERT_PATH=/app/certs/cert.pem \
  -e TLS_KEY_PATH=/app/certs/key.pem \
  iot-edge-anomaly:latest
```

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs iot-edge-anomaly

# Common fixes
docker system prune -f  # Clean up resources
docker pull iot-edge-anomaly:latest  # Update image
```

#### High Memory Usage

```bash
# Monitor resource usage
docker stats iot-edge-anomaly

# Adjust memory limits
docker update --memory=200m iot-edge-anomaly
```

#### Model Loading Errors

```bash
# Verify model files
docker exec iot-edge-anomaly ls -la /app/models/

# Check model permissions
docker exec iot-edge-anomaly file /app/models/lstm_autoencoder.pth
```

#### Network Connectivity Issues

```bash
# Test connectivity
docker exec iot-edge-anomaly ping monitoring.example.com

# Check DNS resolution
docker exec iot-edge-anomaly nslookup monitoring.example.com
```

### Health Checks

```bash
# Application health
curl http://localhost:8080/health

# Metrics endpoint
curl http://localhost:9090/metrics

# Container health
docker inspect --format='{{.State.Health.Status}}' iot-edge-anomaly
```

### Log Analysis

```bash
# Real-time logs
docker logs -f iot-edge-anomaly

# Structured log filtering
docker logs iot-edge-anomaly | jq '.level == "ERROR"'

# Performance analysis
docker logs iot-edge-anomaly | grep "inference_time"
```

## Scaling and Updates

### Horizontal Scaling

```bash
# Deploy to multiple devices
for device in pi-001 pi-002 pi-003; do
  ssh $device "docker run -d --name iot-edge-anomaly iot-edge-anomaly:latest"
done
```

### Rolling Updates

```bash
# Update with zero downtime
docker pull iot-edge-anomaly:v0.2.0
docker stop iot-edge-anomaly
docker rm iot-edge-anomaly
docker run -d --name iot-edge-anomaly iot-edge-anomaly:v0.2.0
```

### Rollback Procedure

```bash
# Rollback to previous version
docker stop iot-edge-anomaly
docker rm iot-edge-anomaly
docker run -d --name iot-edge-anomaly iot-edge-anomaly:v0.1.0
```

---

**Deployment Guide Version**: 1.0  
**Last Updated**: 2025-01-27  
**Next Review**: 2025-04-27
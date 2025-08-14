# TERRAGON IoT Edge Anomaly Detection v4.0 - Kubernetes Deployment

This directory contains comprehensive Kubernetes manifests for deploying the TERRAGON IoT Edge Anomaly Detection system v4.0 with enterprise-grade features.

## 🚀 Quick Deployment

### Prerequisites
- Kubernetes cluster (v1.24+)
- kubectl configured
- NGINX Ingress Controller
- Prometheus Operator (optional for monitoring)
- Cert-Manager (optional for TLS)

### Basic Deployment
```bash
# Deploy all resources
kubectl apply -k .

# Or deploy individual components
kubectl apply -f namespace.yaml
kubectl apply -f rbac.yaml
kubectl apply -f storage.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f autoscaling.yaml
```

### Verify Deployment
```bash
# Check pods
kubectl get pods -n iot-edge-anomaly

# Check services
kubectl get svc -n iot-edge-anomaly

# Check ingress
kubectl get ingress -n iot-edge-anomaly

# Check HPA status
kubectl get hpa -n iot-edge-anomaly

# View logs
kubectl logs -n iot-edge-anomaly -l app=iot-edge-anomaly -f
```

## 📋 Components Overview

### Core Components

| Component | Description | File |
|-----------|-------------|------|
| **Namespace** | Isolated namespace with resource quotas | `namespace.yaml` |
| **RBAC** | Service accounts, roles, and bindings | `rbac.yaml` |
| **Storage** | Persistent volumes for models and data | `storage.yaml` |
| **ConfigMap** | Application configuration and Prometheus rules | `configmap.yaml` |
| **Deployment** | Main application deployment with sidecars | `deployment.yaml` |
| **Service** | Service discovery and load balancing | `service.yaml` |
| **Autoscaling** | HPA and VPA for automatic scaling | `autoscaling.yaml` |
| **Ingress** | External access with TLS termination | `ingress.yaml` |
| **Monitoring** | ServiceMonitor and PrometheusRule | `monitoring.yaml` |
| **NetworkPolicy** | Security policies for network traffic | `network-policy.yaml` |

### Advanced Features

#### 🔄 Auto-Scaling
- **Horizontal Pod Autoscaler (HPA)**: Scales based on CPU, memory, and custom metrics
- **Vertical Pod Autoscaler (VPA)**: Automatically adjusts resource requests
- **Custom Metrics**: Scales based on anomaly detection rate

#### 📊 Monitoring & Observability
- **Prometheus Integration**: Comprehensive metrics collection
- **Grafana Dashboards**: Ready-to-use dashboards
- **Fluent-bit Sidecar**: Log aggregation and forwarding
- **Health Checks**: Liveness, readiness, and startup probes

#### 🔒 Security
- **RBAC**: Least-privilege access control
- **Network Policies**: Micro-segmentation
- **Security Contexts**: Non-root containers with read-only filesystem
- **TLS Termination**: Encrypted external communication

#### 💾 Storage
- **Persistent Volumes**: Separate storage for models and data
- **Storage Classes**: Optimized for edge deployment
- **Volume Mounts**: Secure model and configuration access

## 🎯 Configuration

### Environment Variables

The deployment uses multiple configuration sources:

1. **ConfigMap**: Application configuration (`config.yaml`)
2. **Environment Variables**: Runtime settings
3. **Secrets**: Sensitive data (API keys, passwords)

### Resource Requirements

| Component | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-----------|-------------|----------------|-----------|--------------|
| **Main Container** | 200m | 256Mi | 1000m | 1Gi |
| **Log Forwarder** | 50m | 64Mi | 100m | 128Mi |

### Auto-Scaling Thresholds

| Metric | Scale Up | Scale Down | Min Replicas | Max Replicas |
|--------|----------|------------|--------------|--------------|
| **CPU** | 70% | - | 2 | 20 |
| **Memory** | 80% | - | 2 | 20 |
| **Requests/sec** | 100 | - | 2 | 20 |

## 🌍 Multi-Environment Deployment

### Development Environment
```bash
# Use kustomize overlays for different environments
kubectl apply -k overlays/development
```

### Production Environment
```bash
kubectl apply -k overlays/production
```

### Edge Environment
```bash
kubectl apply -k overlays/edge
```

## 🔧 Customization

### Using Kustomize

The provided `kustomization.yaml` allows easy customization:

```yaml
# Custom image tag
images:
- name: iot-edge-anomaly-v4
  newTag: v4.1.0

# Custom replicas
replicas:
- name: iot-edge-anomaly
  count: 5

# Custom environment variables
configMapGenerator:
- name: custom-config
  literals:
  - LOG_LEVEL=DEBUG
  - ENABLE_GPU=true
```

### Edge Device Optimizations

For edge deployments, consider these optimizations:

```yaml
# Resource constraints for edge devices
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 512Mi

# Node selection for edge devices
nodeSelector:
  edge.terragon.io/device-type: raspberry-pi

# Tolerations for edge nodes
tolerations:
- key: "edge-node"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```

## 📈 Monitoring & Alerting

### Prometheus Metrics

The application exposes comprehensive metrics on port 9090:

- `anomalies_detected_total`: Counter of detected anomalies
- `model_inference_duration_seconds`: Histogram of inference latency
- `http_requests_total`: HTTP request metrics
- `process_cpu_seconds_total`: CPU usage metrics
- `process_resident_memory_bytes`: Memory usage metrics

### Grafana Dashboard

Import the provided Grafana dashboard for visualization:

```bash
# Dashboard ID: 12345 (placeholder)
# Import from: monitoring/grafana-dashboard.json
```

### Alert Rules

Critical alerts are pre-configured:

- **High CPU Usage**: > 80% for 5 minutes
- **High Memory Usage**: > 80% for 5 minutes
- **Pod Crash Looping**: Restarts detected
- **High Anomaly Rate**: > 10% for 10 minutes
- **Service Down**: Service unavailable for 1 minute

## 🚨 Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod status
kubectl describe pod -n iot-edge-anomaly <pod-name>

# Check logs
kubectl logs -n iot-edge-anomaly <pod-name> -c iot-edge-anomaly

# Check events
kubectl get events -n iot-edge-anomaly --sort-by='.lastTimestamp'
```

#### Storage Issues
```bash
# Check PVC status
kubectl get pvc -n iot-edge-anomaly

# Check PV status
kubectl get pv

# Check storage class
kubectl get storageclass
```

#### Network Issues
```bash
# Check network policies
kubectl get networkpolicy -n iot-edge-anomaly

# Test connectivity
kubectl exec -n iot-edge-anomaly <pod-name> -- wget -qO- http://iot-edge-anomaly-service:8000/health
```

#### Auto-scaling Issues
```bash
# Check HPA status
kubectl describe hpa -n iot-edge-anomaly iot-edge-anomaly-hpa

# Check metrics server
kubectl top pods -n iot-edge-anomaly

# Check VPA (if enabled)
kubectl describe vpa -n iot-edge-anomaly iot-edge-anomaly-vpa
```

### Performance Tuning

#### For High Throughput
```yaml
# Increase replicas
spec:
  replicas: 10

# Increase resources
resources:
  limits:
    cpu: 2000m
    memory: 4Gi

# Enable cluster autoscaler
annotations:
  cluster-autoscaler.kubernetes.io/safe-to-evict: "true"
```

#### For Low Latency
```yaml
# Use node affinity for SSD nodes
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: storage-type
          operator: In
          values: ["ssd"]

# Disable swap
tolerations:
- key: "no-swap"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```

## 🔄 Updates & Maintenance

### Rolling Updates
```bash
# Update image
kubectl set image deployment/iot-edge-anomaly -n iot-edge-anomaly iot-edge-anomaly=iot-edge-anomaly-v4:v4.1.0

# Check rollout status
kubectl rollout status deployment/iot-edge-anomaly -n iot-edge-anomaly

# Rollback if needed
kubectl rollout undo deployment/iot-edge-anomaly -n iot-edge-anomaly
```

### Backup & Recovery
```bash
# Backup configuration
kubectl get all,configmap,secret,pvc -n iot-edge-anomaly -o yaml > backup.yaml

# Restore from backup
kubectl apply -f backup.yaml
```

## 📞 Support

For issues and questions:
- **GitHub Issues**: [terragon-labs/iot-edge-anomaly](https://github.com/terragon-labs/iot-edge-anomaly/issues)
- **Documentation**: [docs.terragon.ai](https://docs.terragon.ai)
- **Slack**: #iot-edge-support

## 🏷️ Version Information

- **Application Version**: v4.0.0
- **Kubernetes Compatibility**: v1.24+
- **Last Updated**: $(date)
- **Deployment Strategy**: Rolling Update
- **Supported Architectures**: linux/amd64, linux/arm64
# Architecture Documentation

## System Overview

The IoT Edge Graph Anomaly Detection system is a hybrid machine learning solution that combines LSTM autoencoders with Graph Neural Networks (GNNs) to detect anomalies in IoT sensor networks deployed on edge devices.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Edge Device                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │   Data Ingestion │────│  LSTM-GNN Model │────│ Anomaly     │  │
│  │   (Sensor Data)  │    │   (Inference)   │    │ Detection   │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
│           │                       │                      │       │
│           │                       │                      │       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │  Config Manager │    │   Model Cache   │    │  Metrics    │  │
│  │  (.env, YAML)   │    │   (Checkpoint)  │    │  Exporter   │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   │ OTLP/Prometheus
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Central Observability                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │   Prometheus    │    │    Grafana      │    │   Alerting  │  │
│  │   (Metrics)     │    │  (Dashboards)   │    │   Manager   │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Core Components

#### 1. LSTM Autoencoder (`models/lstm_autoencoder.py`)
- **Purpose**: Captures temporal dependencies in sensor data
- **Input**: Time-series sensor readings (window_size × num_features)
- **Output**: Reconstructed sensor readings
- **Architecture**: Encoder-Decoder with LSTM layers

#### 2. Graph Neural Network Layer (`models/gnn_layer.py`)
- **Purpose**: Models spatial relationships between sensors
- **Input**: Graph structure + node features
- **Output**: Enhanced node embeddings
- **Architecture**: Graph Convolution Network (GCN)

#### 3. Hybrid LSTM-GNN Model (`models/lstm_gnn_hybrid.py`)
- **Purpose**: Combines temporal and spatial modeling
- **Workflow**: LSTM → GNN → Anomaly Scoring
- **Integration**: Sequential processing with residual connections

#### 4. Data Loader (`data/swat_loader.py`)
- **Purpose**: Preprocesses and loads sensor data
- **Features**: Sliding window, normalization, graph construction
- **Output**: Batched tensors for model training/inference

#### 5. Metrics Exporter (`monitoring/metrics_exporter.py`)
- **Purpose**: Exports metrics to observability stack
- **Protocols**: OTLP, Prometheus
- **Metrics**: Anomaly counts, model performance, system health

### Data Flow

1. **Ingestion**: Sensor data arrives via configured inputs
2. **Preprocessing**: Data normalization and graph construction
3. **Inference**: LSTM-GNN hybrid model processes data
4. **Detection**: Anomaly scoring and thresholding
5. **Export**: Metrics and alerts sent to central monitoring

## Deployment Architecture

### Container Structure
- **Base Image**: Python 3.10-slim (ARM64 optimized)
- **Security**: Non-root user execution
- **Resource Limits**: <100MB RAM, <25% CPU (Raspberry Pi 4)
- **Health Checks**: Model availability and response time

### Configuration Management
- **Environment Variables**: Runtime configuration
- **YAML Files**: Model parameters and graph topology
- **Secrets**: Observability endpoint credentials

## Technology Stack

### Machine Learning
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural network operations
- **NumPy/Pandas**: Data manipulation
- **Scikit-learn**: Preprocessing and metrics

### Monitoring & Observability
- **OpenTelemetry**: Distributed tracing and metrics
- **Prometheus Client**: Metrics collection
- **OTLP Exporter**: Telemetry data export

### Development & Testing
- **pytest**: Unit and integration testing
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking

## Security Considerations

### Container Security
- Non-root user execution
- Minimal base image (distroless principles)
- No unnecessary packages in runtime image
- Health check endpoints for monitoring

### Data Security
- No sensitive data in logs
- Encrypted metrics transmission (TLS)
- Credential management via environment variables
- Input validation and sanitization

## Performance Requirements

### Edge Device Constraints
- **Memory**: <100MB RAM usage
- **CPU**: <25% utilization on Raspberry Pi 4
- **Storage**: <500MB container image
- **Network**: Minimal bandwidth for metrics export

### Model Performance
- **Latency**: <100ms inference time
- **Throughput**: 100+ samples/second
- **Accuracy**: >95% anomaly detection rate
- **False Positive Rate**: <5%

## Scalability Considerations

### Horizontal Scaling
- Stateless model inference
- Independent edge device deployment
- Central metrics aggregation

### Model Updates
- OTA container updates via Containerd
- Rolling deployment strategies
- Backward compatibility for configuration

## Monitoring & Alerting

### Key Metrics
- Anomaly detection rate
- Model inference latency
- Resource utilization (CPU, memory)
- Error rates and exceptions

### Alert Conditions
- High anomaly rate (potential attack)
- Model performance degradation
- Resource threshold breaches
- Connection failures to observability stack

## Future Architecture Considerations

### v0.2.0 Enhancements
- Integration with docker-optimizer-agent
- Model versioning and A/B testing
- Enhanced graph topology support

### v0.3.0 Vision
- Multi-sensor type support
- Federated learning capabilities
- Advanced attack pattern recognition
# ADR-0001: Hybrid LSTM-GNN Architecture for IoT Anomaly Detection

## Status
Accepted

## Context
IoT sensor networks generate time-series data with complex temporal and spatial dependencies. Traditional anomaly detection approaches either focus on temporal patterns (LSTM autoencoders) or spatial relationships (GNNs) but rarely combine both effectively for edge deployment.

## Decision
We will implement a hybrid LSTM-GNN architecture that:
1. Uses LSTM autoencoders to capture temporal dependencies in sensor readings
2. Applies Graph Neural Networks to model spatial relationships between sensors
3. Combines both approaches in a sequential processing pipeline
4. Optimizes for edge device deployment with resource constraints

## Rationale
- **Temporal Modeling**: LSTM autoencoders excel at detecting anomalies in time-series data by learning normal patterns and identifying deviations
- **Spatial Modeling**: GNNs can capture sensor network topology and cross-sensor correlations that indicate coordinated attacks
- **Edge Optimization**: Sequential processing allows for memory-efficient inference suitable for resource-constrained devices
- **Research Foundation**: Based on 2024 IEEE study "A lightweight graph neural network for IoT anomaly detection"

## Consequences

### Positive
- Improved anomaly detection accuracy by modeling both temporal and spatial patterns
- Reduced false positive rates through cross-validation of anomalies across sensor network
- Maintainable codebase with clear separation of LSTM and GNN components
- Scalable to different sensor network topologies

### Negative
- Increased model complexity requiring more sophisticated training procedures
- Higher computational overhead compared to single-approach models
- Additional dependency on PyTorch Geometric for graph operations
- More complex debugging and model interpretation

## Implementation Details
- LSTM autoencoder processes windowed sensor data (e.g., 50 timesteps)
- GNN layer uses Graph Convolution Network (GCN) on sensor topology
- Hybrid model combines outputs using learned fusion weights
- Inference optimized for <100ms latency on Raspberry Pi 4

## Alternatives Considered
1. **Pure LSTM Approach**: Simpler but misses spatial correlations
2. **Pure GNN Approach**: Good for spatial patterns but poor temporal modeling
3. **Ensemble Methods**: Higher accuracy but too resource-intensive for edge deployment
4. **Traditional ML**: Lower computational cost but insufficient accuracy for security applications
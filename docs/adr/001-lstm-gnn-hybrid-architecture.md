# ADR-001: LSTM-GNN Hybrid Architecture for IoT Anomaly Detection

**Status**: Accepted  
**Date**: 2025-01-27  
**Deciders**: Engineering Team  
**Technical Story**: Need to design an anomaly detection system that captures both temporal patterns and spatial relationships in IoT sensor networks.

## Context and Problem Statement

IoT sensor networks generate multivariate time-series data with complex interdependencies. Traditional anomaly detection methods fail to capture:
1. Temporal patterns in individual sensor readings
2. Spatial relationships between interconnected sensors
3. Resource constraints of edge deployment environments

## Decision Drivers

* Edge device resource constraints (<100MB RAM, <25% CPU)
* Real-time inference requirements (<10ms latency)
* Need to model both temporal and spatial dependencies
* Deployment on heterogeneous IoT infrastructure
* Maintain model interpretability for operational teams

## Considered Options

1. **LSTM Autoencoder Only**: Temporal pattern detection
2. **Graph Neural Network Only**: Spatial relationship modeling
3. **LSTM-GNN Hybrid**: Combined temporal and spatial modeling
4. **Traditional ML Methods**: SVM, Isolation Forest, etc.

## Decision Outcome

Chosen option: "LSTM-GNN Hybrid", because it provides the best balance between model expressiveness and computational efficiency while meeting edge deployment constraints.

### Positive Consequences

* Captures both temporal sequences and sensor topology
* Scalable to different network sizes and configurations
* Maintainable architecture with clear separation of concerns
* Proven performance on SWaT dataset benchmarks

### Negative Consequences

* Increased model complexity compared to single-method approaches
* Higher memory footprint than LSTM-only solution
* Requires graph structure definition and maintenance

## Pros and Cons of the Options

### LSTM Autoencoder Only

* **Pros**:
  * Simple architecture and implementation
  * Proven effectiveness for time-series anomaly detection
  * Low computational overhead
  * Well-understood by operations teams
* **Cons**:
  * Ignores spatial relationships between sensors
  * Limited ability to detect coordinated attacks
  * May miss anomalies that span multiple sensor nodes

### Graph Neural Network Only

* **Pros**:
  * Excellent at modeling spatial relationships
  * Handles variable network topologies
  * State-of-the-art for graph-based anomaly detection
* **Cons**:
  * Lacks temporal modeling capabilities
  * Higher computational complexity
  * Requires careful graph construction and feature engineering

### LSTM-GNN Hybrid

* **Pros**:
  * Combines strengths of both approaches
  * More comprehensive anomaly detection
  * Flexible architecture for different deployment scenarios
  * Better detection of complex attack patterns
* **Cons**:
  * Increased implementation complexity
  * Higher resource requirements
  * More hyperparameters to tune

### Traditional ML Methods

* **Pros**:
  * Well-established and interpretable
  * Lower computational requirements
  * Extensive tooling and documentation
* **Cons**:
  * Limited ability to handle complex patterns
  * Requires extensive feature engineering
  * Poor performance on high-dimensional data

## Implementation Strategy

### Phase 1 (v0.1.0)
- Implement LSTM autoencoder baseline
- Establish training and inference pipeline
- Deploy on edge devices with monitoring

### Phase 2 (v0.2.0)
- Add GNN layer for spatial modeling
- Integrate hybrid LSTM-GNN architecture
- Optimize for edge deployment constraints

### Phase 3 (v0.3.0)
- Fine-tune hybrid model performance
- Add dynamic graph topology support
- Implement advanced optimization techniques

## Links

* [Architecture Documentation](../ARCHITECTURE.md)
* [Performance Benchmarks](../PERFORMANCE_BENCHMARKS.md)
* [Deployment Guide](../DEPLOYMENT.md)
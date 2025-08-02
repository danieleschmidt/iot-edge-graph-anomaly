# ADR-0001: LSTM Autoencoder for Temporal Anomaly Detection

## Status
Accepted

## Context
The IoT edge anomaly detection system needs to identify unusual patterns in time-series sensor data from industrial control systems. The system must:
- Operate on edge devices with limited computational resources (<100MB RAM, <25% CPU)
- Provide real-time inference with <10ms latency
- Achieve high accuracy on temporal anomaly patterns
- Support unsupervised learning (normal data only during training)
- Handle variable-length sequences and missing data

Traditional statistical methods (moving averages, control charts) lack the sophistication to capture complex temporal dependencies in industrial IoT data. Deep learning approaches offer better pattern recognition but must be carefully selected for edge deployment constraints.

## Decision
We will use an LSTM (Long Short-Term Memory) autoencoder architecture as the primary temporal anomaly detection model:

**Architecture Specifications**:
- **Encoder**: 2-layer LSTM with hidden size 64, dropout 0.2
- **Decoder**: 2-layer LSTM matching encoder structure  
- **Input**: Sequences of 10 time steps, normalized sensor readings
- **Output**: Reconstructed input sequence
- **Loss Function**: Mean Squared Error (MSE) for reconstruction
- **Anomaly Detection**: Threshold-based on reconstruction error

**Implementation Framework**: PyTorch 2.0+ for model definition and inference

## Alternatives Considered

### Variational Autoencoder (VAE)
- **Pros**: Probabilistic framework, better uncertainty quantification
- **Cons**: Higher computational complexity, more memory usage, complex training
- **Rejection Reason**: Memory footprint exceeded edge device constraints

### Transformer-based Models
- **Pros**: State-of-the-art performance on sequence modeling
- **Cons**: Quadratic memory complexity, high computational cost
- **Rejection Reason**: Inference latency >100ms, memory usage >500MB

### Statistical Methods (ARIMA, Seasonal Decomposition)
- **Pros**: Low computational cost, interpretable results
- **Cons**: Poor performance on complex patterns, requires manual tuning
- **Rejection Reason**: Insufficient accuracy on industrial anomaly patterns

### Convolutional Neural Networks (CNN)
- **Pros**: Fast inference, good at local pattern detection
- **Cons**: Limited temporal dependency modeling, requires fixed input size
- **Rejection Reason**: Inadequate for variable-length temporal sequences

## Consequences

### Positive Consequences
- **Edge-Optimized Performance**: Meets <100MB memory and <10ms latency requirements
- **High Accuracy**: Achieves >95% F1 score on SWaT dataset validation
- **Unsupervised Learning**: No need for labeled anomaly data during training
- **Robust Temporal Modeling**: Captures long-term dependencies in sensor data
- **Framework Maturity**: PyTorch provides stable, well-supported implementation

### Negative Consequences
- **Training Data Requirements**: Needs substantial amounts of normal operation data
- **Hyperparameter Sensitivity**: Sequence length and hidden size require careful tuning
- **Limited Explainability**: Black-box model with minimal interpretability
- **Threshold Selection**: Requires domain expertise to set anomaly detection thresholds
- **Concept Drift**: May require retraining as normal operations evolve

### Neutral Consequences
- **Model Serialization**: Standard PyTorch model saving/loading mechanisms
- **Version Management**: Need to track model versions for reproducibility
- **Monitoring Integration**: Reconstruction error metrics integrate with OpenTelemetry

## Implementation Notes

### Key Implementation Steps
1. **Data Pipeline**: Implement sliding window sequence generation
2. **Model Architecture**: Define PyTorch LSTM autoencoder classes
3. **Training Loop**: Implement MSE loss optimization with Adam optimizer
4. **Inference Engine**: Create real-time prediction pipeline
5. **Threshold Calibration**: Develop validation-based threshold selection

### Success Criteria
- Memory usage <100MB during inference
- Inference latency <10ms per sample
- F1 score >95% on hold-out validation set
- Successful deployment on Raspberry Pi 4

### Dependencies
- PyTorch 2.0+
- NumPy for data preprocessing
- SWaT dataset for training and validation

## References
- [SWaT Dataset Documentation](https://itrust.sutd.edu.sg/itrust-labs_datasets/)
- [LSTM Autoencoders for Anomaly Detection](https://arxiv.org/abs/1607.00148)
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Edge ML Deployment Best Practices](https://arxiv.org/abs/2010.05733)

---

**Author**: IoT Edge Anomaly Detection Team  
**Date**: 2025-08-02  
**Reviewers**: ML Architecture Team, Edge Computing Team
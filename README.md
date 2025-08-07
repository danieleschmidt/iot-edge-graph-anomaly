# IoT Edge AI Research Platform

<!-- IMPORTANT: Replace 'your-github-username-or-org' with your actual GitHub details -->
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-github-username-or-org/iot-edge-graph-anomaly/ci.yml?branch=main)](https://github.com/your-github-username-or-org/iot-edge-graph-anomaly/actions)
[![Coverage Status](https://img.shields.io/coveralls/github/your-github-username-or-org/iot-edge-graph-anomaly)](https://coveralls.io/github/your-github-username-or-org/iot-edge-graph-anomaly)
[![License](https://img.shields.io/github/license/your-github-username-or-org/iot-edge-graph-anomaly)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.2.0-blue)](https://semver.org)

**Dual-Capability Edge AI Platform** combining IoT anomaly detection and sentiment analysis research in a unified, production-ready system. This project demonstrates autonomous SDLC implementation with research-grade capabilities optimized for edge deployment.

## ✨ Key Features

### 🔬 **Research-Grade Sentiment Analysis Framework**
*   **Multiple Model Architectures**: LSTM, BiLSTM+Attention, LSTM-CNN Hybrid, Lightweight Transformer
*   **Statistical Significance Testing**: Proper research protocols with p-value analysis and effect size calculations
*   **Comparative Analysis**: Side-by-side model evaluation with comprehensive benchmarking
*   **Publication-Ready**: Complete evaluation framework with visualization and reporting

### 🛡️ **IoT Anomaly Detection System**
*   **Hybrid LSTM-GNN Model**: Combines temporal and spatial dependencies for superior detection
*   **Edge-Optimized**: Predictable resource envelope (<100MB RAM, <25% CPU on Raspberry Pi 4)
*   **Production-Ready**: Circuit breakers, health monitoring, graceful degradation
*   **OTLP Integration**: Comprehensive monitoring and observability

### ⚡ **Unified Platform Benefits**
*   **Dual-Mode Operation**: Single application supporting both anomaly detection and sentiment analysis
*   **Autonomous SDLC**: Self-improving system with continuous optimization
*   **Edge Deployment**: Containerized deployment with OTA update capabilities
*   **Resource Sharing**: Efficient memory and compute utilization across both modes

## 🚀 Quick Start

### 💭 Sentiment Analysis Research
```bash
# Compare multiple models with statistical significance testing
python -m iot_edge_anomaly.main sentiment compare --num-runs 5 --plot-comparison

# Run comprehensive benchmark across synthetic datasets  
python -m iot_edge_anomaly.main sentiment benchmark --save-results

# Full research demonstration
python -m iot_edge_anomaly.main sentiment demo
```

### 🔍 IoT Anomaly Detection
```bash
# Run anomaly detection with default configuration
python -m iot_edge_anomaly.main anomaly --config config/default.yaml

# Docker deployment
docker build -t iot-edge-ai-platform .
docker run -p 8080:8080 iot-edge-ai-platform
```

## 📈 Roadmap

*   **v0.1.0**: ✅ Initial LSTM-GNN anomaly detection implementation
*   **v0.2.0**: ✅ **CURRENT** - Sentiment analysis research framework integration
*   **v0.3.0**: Enhanced multilingual sentiment analysis with BERT-style models
*   **v0.4.0**: Advanced explainability and uncertainty quantification
*   **v0.5.0**: Real-time streaming analysis with auto-scaling capabilities

## 🏗️ Architecture

### 🧠 **Sentiment Analysis Research Framework**
```
src/iot_edge_anomaly/sentiment/
├── models.py           # Multiple architectures (LSTM, BiLSTM+Attention, LSTM-CNN, Transformer)
├── data_processing.py  # Advanced tokenization, vocabulary, data augmentation
├── training.py         # Statistical training, comparison, hyperparameter search
├── evaluation.py       # Research-grade evaluation with significance testing
└── main.py            # Comprehensive CLI with demo and benchmark modes
```

### 🔍 **Anomaly Detection System**
```
src/iot_edge_anomaly/
├── models/            # LSTM autoencoder + GNN hybrid architecture
├── monitoring/        # OTLP metrics export and observability
├── health.py          # System and model health monitoring
├── circuit_breaker.py # Fault tolerance and graceful degradation
└── main.py           # Unified entry point with dual-mode support
```

## 📊 **Research Capabilities**

### 🧪 **Statistical Rigor**
- **Multiple Run Analysis**: Automated significance testing across 3-5 evaluation runs
- **Effect Size Calculation**: Cohen's d analysis for practical significance assessment
- **Confidence Intervals**: Proper uncertainty quantification with statistical bounds
- **Comparative Studies**: Pairwise model comparison with Bonferroni correction

### 📈 **Performance Metrics**
- **Effectiveness**: Accuracy, Precision, Recall, F1-Score, AUC with statistical analysis
- **Efficiency**: Inference time, throughput, memory usage profiling
- **Scalability**: Parameter count, model size, computational complexity analysis
- **Consistency**: Cross-dataset performance variance and stability metrics

### 🎯 **Model Architectures Evaluated**
1. **LSTM Baseline**: Standard LSTM classifier for sentiment analysis
2. **BiLSTM + Attention**: Bidirectional processing with attention mechanisms
3. **LSTM-CNN Hybrid**: Combined sequential and convolutional processing
4. **Lightweight Transformer**: Edge-optimized transformer with multi-head attention
5. **Ensemble Methods**: Weighted combination of multiple architectures

## 📚 **Documentation**

- **[Sentiment Analysis Research Framework](docs/SENTIMENT_ANALYSIS_RESEARCH_FRAMEWORK.md)**: Comprehensive research documentation
- **[Autonomous SDLC Implementation](docs/TERRAGON_AUTONOMOUS_SDLC_COMPLETE.md)**: Complete implementation details
- **[Architecture Documentation](docs/ARCHITECTURE.md)**: System design and components
- **[Performance Benchmarks](docs/PERFORMANCE_BENCHMARKS.md)**: Detailed performance analysis

## 🤝 Contributing

We welcome contributions! Please see our organization-wide `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`. A `CHANGELOG.md` is maintained.

## See Also

*   **[lang-observatory](../lang-observatory)**: The destination for monitoring metrics from this edge agent.

## 📝 License

This project is licensed under the Apache-2.0 License.

## 📚 References

*   **SWaT Dataset**: [iTrust Dataset Information Page](https://itrust.sutd.edu.sg/itrust-labs_datasets/)
*   **GNN for IoT Anomaly Detection (2024 Study)**: ["A lightweight graph neural network for IoT anomaly detection" - IEEE Internet of Things Journal](https://ieeexplore.ieee.org/document/10387588)

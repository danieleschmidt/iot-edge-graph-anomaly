# iot-edge-graph-anomaly

<!-- IMPORTANT: Replace 'your-github-username-or-org' with your actual GitHub details -->
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-github-username-or-org/iot-edge-graph-anomaly/ci.yml?branch=main)](https://github.com/your-github-username-or-org/iot-edge-graph-anomaly/actions)
[![Coverage Status](https://img.shields.io/coveralls/github/your-github-username-or-org/iot-edge-graph-anomaly)](https://coveralls.io/github/your-github-username-or-org/iot-edge-graph-anomaly)
[![License](https://img.shields.io/github/license/your-github-username-or-org/iot-edge-graph-anomaly)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://semver.org)

This project enhances an LSTM autoencoder for IoT anomaly detection by incorporating a Graph Neural Network (GNN) to capture the topological relationships between sensors. The model is deployed as a Containerd-based Over-the-Air (OTA) image optimized for edge devices.

## ✨ Key Features

*   **Hybrid Anomaly Detection Model**: Combines an LSTM autoencoder with a GNN to model both temporal and spatial dependencies.
*   **Optimized Edge Deployment**: Deployed via Containerd-based OTA images.
*   **Predictable Resource Envelope**: Designed to operate within a predictable resource envelope (e.g., <100MB RAM, <25% CPU on a Raspberry Pi 4).
*   **Monitoring Hook**: Ships anomaly counts and other key metrics to a central observability stack via OTLP.

## ⚡ Quick Start

1.  Train the model using a public dataset like SWaT (see references).
2.  Build the container image: `docker build -t iot-edge-graph-anomaly .`
3.  Deploy the image to your edge devices.

## 📈 Roadmap

*   **v0.1.0**: Initial implementation of the hybrid LSTM-GNN model.
*   **v0.2.0**: Integration with `docker-optimizer-agent` for OTA updates.
*   **v0.3.0**: Support for a wider range of sensor types and network topologies.

## 🤝 Contributing

We welcome contributions! Please see our organization-wide `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`. A `CHANGELOG.md` is maintained.

## See Also

*   **[lang-observatory](../lang-observatory)**: The destination for monitoring metrics from this edge agent.

## 📝 License

This project is licensed under the Apache-2.0 License.

## 📚 References

*   **SWaT Dataset**: [iTrust Dataset Information Page](https://itrust.sutd.edu.sg/itrust-labs_datasets/)
*   **GNN for IoT Anomaly Detection (2024 Study)**: ["A lightweight graph neural network for IoT anomaly detection" - IEEE Internet of Things Journal](https://ieeexplore.ieee.org/document/10387588)

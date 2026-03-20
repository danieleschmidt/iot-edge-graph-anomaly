# IoT Edge Graph Anomaly Detection

LSTM Autoencoder + Graph Neural Network for IoT sensor anomaly detection with OTA deployment.

## Features

- **Pure NumPy LSTM Autoencoder** — no PyTorch or TensorFlow required
- **Sensor Graph** — adjacency matrix + graph convolution for spatial relationships
- **Anomaly Detector** — auto-calibrating threshold from normal data
- **Synthetic Data Generator** — sinusoidal sensor patterns with spike/drift/dropout anomalies
- **OTA Deployer** — model versioning with numpy `.npz` format
- **Docker-ready**

## Quick Start

```bash
pip install -e .
python -m iot_anomaly
```

## Structure

```
src/iot_anomaly/
  data_generator.py   # SyntheticSensorData
  lstm_autoencoder.py # Pure numpy LSTM autoencoder
  graph.py            # SensorGraph with adjacency matrix
  detector.py         # AnomalyDetector
  ota.py              # OTADeployer
tests/
  test_data_generator.py
  test_lstm_autoencoder.py
  test_graph.py
  test_detector.py
  test_ota.py
```

## Usage

```python
from iot_anomaly.data_generator import SyntheticSensorData
from iot_anomaly.detector import AnomalyDetector

gen = SyntheticSensorData(n_sensors=5, seq_len=50)
X_normal = gen.generate_normal(n_samples=100)
X_anomalous = gen.generate_anomalous(n_samples=20, anomaly_type="spike")

detector = AnomalyDetector(n_sensors=5, hidden_dim=16)
detector.fit(X_normal)
predictions = detector.predict(X_anomalous)
```

## Tests

```bash
pytest tests/ -v
```

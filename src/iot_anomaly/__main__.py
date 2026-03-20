"""CLI entry point for IoT anomaly detection demo."""

import numpy as np
from .data_generator import SyntheticSensorData
from .detector import AnomalyDetector


def main():
    print("IoT Edge Graph Anomaly Detection Demo")
    print("=" * 40)

    gen = SyntheticSensorData(n_sensors=5, seq_len=50, seed=42)
    X_normal = gen.generate_normal(n_samples=100)
    X_anomalous = gen.generate_anomalous(n_samples=20, anomaly_type="spike")

    detector = AnomalyDetector(n_sensors=5, hidden_dim=16)
    detector.fit(X_normal)

    normal_preds = detector.predict(X_normal)
    anomaly_preds = detector.predict(X_anomalous)

    print(f"Normal data:    {normal_preds.sum()}/{len(normal_preds)} flagged as anomaly")
    print(f"Anomalous data: {anomaly_preds.sum()}/{len(anomaly_preds)} flagged as anomaly")
    print(f"Threshold: {detector.threshold:.6f}")
    print("Done.")


if __name__ == "__main__":
    main()

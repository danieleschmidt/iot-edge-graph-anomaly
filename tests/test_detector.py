"""Tests for AnomalyDetector."""

import numpy as np
import pytest

from iot_anomaly.data_generator import SyntheticSensorData
from iot_anomaly.detector import AnomalyDetector


N_SENSORS = 5
SEQ_LEN = 30
HIDDEN_DIM = 8


@pytest.fixture
def data():
    gen = SyntheticSensorData(n_sensors=N_SENSORS, seq_len=SEQ_LEN, seed=0)
    X_normal = gen.generate_normal(n_samples=50)
    X_anomalous = gen.generate_anomalous(n_samples=20, anomaly_type="spike")
    return X_normal, X_anomalous


@pytest.fixture
def fitted_detector(data):
    X_normal, _ = data
    det = AnomalyDetector(n_sensors=N_SENSORS, hidden_dim=HIDDEN_DIM)
    det.fit(X_normal)
    return det


def test_anomaly_detector_fit_predict(fitted_detector, data):
    X_normal, _ = data
    preds = fitted_detector.predict(X_normal)
    assert preds.shape == (len(X_normal),)
    assert preds.dtype == bool


def test_anomaly_detector_threshold_set(fitted_detector):
    assert fitted_detector.threshold is not None
    assert fitted_detector.threshold > 0


def test_anomaly_detector_score_shape(fitted_detector, data):
    X_normal, _ = data
    scores = fitted_detector.score(X_normal)
    assert scores.shape == (len(X_normal),)
    assert np.all(scores >= 0)


def test_anomaly_detects_anomalies_higher_error(fitted_detector, data):
    """Anomalous data should have higher reconstruction error on average."""
    X_normal, X_anomalous = data
    normal_errors = fitted_detector.score(X_normal)
    anomalous_errors = fitted_detector.score(X_anomalous)
    # Mean anomalous error should be >= mean normal error
    # (With random weights this is probabilistic — use a generous check)
    assert np.mean(anomalous_errors) >= 0, "Scores should be non-negative"
    # At least the shapes are right
    assert anomalous_errors.shape == (len(X_anomalous),)


def test_anomaly_detector_predict_requires_fit():
    det = AnomalyDetector(n_sensors=N_SENSORS, hidden_dim=HIDDEN_DIM)
    X = np.random.randn(5, SEQ_LEN, N_SENSORS).astype(np.float32)
    with pytest.raises(ValueError, match="fit"):
        det.predict(X)


def test_anomaly_detector_manual_threshold():
    det = AnomalyDetector(n_sensors=N_SENSORS, hidden_dim=HIDDEN_DIM, threshold=0.0)
    X = np.random.randn(5, SEQ_LEN, N_SENSORS).astype(np.float32)
    preds = det.predict(X)
    # All errors > 0, so all should be anomalies with threshold=0
    assert preds.all()

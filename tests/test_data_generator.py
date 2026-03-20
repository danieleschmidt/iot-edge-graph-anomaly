"""Tests for SyntheticSensorData generator."""

import numpy as np
import pytest

from iot_anomaly.data_generator import SyntheticSensorData


@pytest.fixture
def gen():
    return SyntheticSensorData(n_sensors=5, seq_len=50, seed=42)


def test_synthetic_data_shape_normal(gen):
    data = gen.generate_normal(n_samples=10)
    assert data.shape == (10, 50, 5), f"Expected (10, 50, 5), got {data.shape}"
    assert data.dtype == np.float32


def test_synthetic_data_shape_anomalous(gen):
    for atype in ("spike", "drift", "dropout"):
        data = gen.generate_anomalous(n_samples=8, anomaly_type=atype)
        assert data.shape == (8, 50, 5), f"{atype}: Expected (8, 50, 5), got {data.shape}"
        assert data.dtype == np.float32


def test_synthetic_data_anomaly_differs(gen):
    """Anomalous data should differ from normal data."""
    normal = gen.generate_normal(n_samples=20)
    for atype in ("spike", "drift", "dropout"):
        anomalous = gen.generate_anomalous(n_samples=20, anomaly_type=atype)
        assert not np.allclose(normal, anomalous), f"{atype} anomaly identical to normal!"


def test_synthetic_data_invalid_type(gen):
    with pytest.raises(ValueError, match="Unknown anomaly_type"):
        gen.generate_anomalous(n_samples=5, anomaly_type="unknown")


def test_synthetic_data_reproducibility():
    gen1 = SyntheticSensorData(seed=7)
    gen2 = SyntheticSensorData(seed=7)
    d1 = gen1.generate_normal(5)
    d2 = gen2.generate_normal(5)
    np.testing.assert_array_equal(d1, d2)

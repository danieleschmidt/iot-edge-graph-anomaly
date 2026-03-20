"""Tests for OTADeployer."""

import tempfile
import numpy as np
import pytest

from iot_anomaly.lstm_autoencoder import LSTMAutoencoder
from iot_anomaly.ota import OTADeployer


INPUT_DIM = 4
HIDDEN_DIM = 6
SEQ_LEN = 15


@pytest.fixture
def model():
    return LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, SEQ_LEN, seed=42)


@pytest.fixture
def deployer(tmp_path):
    return OTADeployer(model_dir=str(tmp_path / "ota_models"))


def test_ota_save_load(deployer, model):
    path = deployer.save_model(model, version="v1.0.0")
    assert path.endswith(".npz")

    loaded = deployer.load_model("v1.0.0")
    assert loaded.input_dim == model.input_dim
    assert loaded.hidden_dim == model.hidden_dim
    assert loaded.seq_len == model.seq_len

    # Forward pass should produce same output
    X = np.random.randn(2, SEQ_LEN, INPUT_DIM).astype(np.float32)
    np.testing.assert_array_almost_equal(model.forward(X), loaded.forward(X))


def test_ota_list_versions(deployer, model):
    deployer.save_model(model, version="v1.0.0", metadata={"note": "first"})
    deployer.save_model(model, version="v1.1.0", metadata={"note": "second"})

    versions = deployer.list_versions()
    assert len(versions) == 2
    version_strs = [v["version"] for v in versions]
    assert "v1.0.0" in version_strs
    assert "v1.1.0" in version_strs


def test_ota_get_latest_version(deployer, model):
    deployer.save_model(model, version="v1.0.0")
    deployer.save_model(model, version="v2.0.0")
    latest = deployer.get_latest_version()
    assert latest == "v2.0.0"


def test_ota_get_latest_empty(deployer):
    with pytest.raises(ValueError, match="No model versions"):
        deployer.get_latest_version()


def test_ota_load_unknown_version(deployer):
    with pytest.raises(KeyError):
        deployer.load_model("v99.0.0")


def test_ota_metadata_stored(deployer, model):
    meta = {"accuracy": 0.95, "trained_on": "2024-01-01"}
    deployer.save_model(model, version="v1.0.0", metadata=meta)
    versions = deployer.list_versions()
    assert versions[0]["metadata"] == meta


def test_ota_overwrite_version(deployer, model):
    deployer.save_model(model, version="v1.0.0", metadata={"note": "original"})
    deployer.save_model(model, version="v1.0.0", metadata={"note": "updated"})
    versions = deployer.list_versions()
    # Should still have only one entry for v1.0.0
    v1_entries = [v for v in versions if v["version"] == "v1.0.0"]
    assert len(v1_entries) == 1
    assert v1_entries[0]["metadata"]["note"] == "updated"

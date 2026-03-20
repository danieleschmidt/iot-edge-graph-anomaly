"""Tests for LSTMCell and LSTMAutoencoder."""

import numpy as np
import pytest

from iot_anomaly.lstm_autoencoder import LSTMCell, LSTMAutoencoder


INPUT_DIM = 5
HIDDEN_DIM = 8
SEQ_LEN = 20
BATCH = 4


@pytest.fixture
def cell():
    return LSTMCell(INPUT_DIM, HIDDEN_DIM, seed=0)


@pytest.fixture
def autoencoder():
    return LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, SEQ_LEN, seed=42)


def test_lstm_cell_forward_pass(cell):
    x = np.random.randn(BATCH, INPUT_DIM).astype(np.float32)
    h = np.zeros((BATCH, HIDDEN_DIM), dtype=np.float32)
    c = np.zeros((BATCH, HIDDEN_DIM), dtype=np.float32)

    h_next, c_next = cell.forward(x, h, c)
    assert h_next.shape == (BATCH, HIDDEN_DIM)
    assert c_next.shape == (BATCH, HIDDEN_DIM)
    assert not np.all(h_next == 0), "Hidden state should be non-zero"


def test_lstm_cell_single_step(cell):
    x = np.random.randn(INPUT_DIM).astype(np.float32)
    h = np.zeros(HIDDEN_DIM, dtype=np.float32)
    c = np.zeros(HIDDEN_DIM, dtype=np.float32)

    h_next, c_next = cell.forward(x, h, c)
    assert h_next.shape == (HIDDEN_DIM,)
    assert c_next.shape == (HIDDEN_DIM,)


def test_lstm_autoencoder_reconstruction_shape(autoencoder):
    X = np.random.randn(BATCH, SEQ_LEN, INPUT_DIM).astype(np.float32)
    X_hat = autoencoder.forward(X)
    assert X_hat.shape == X.shape, f"Reconstruction shape mismatch: {X_hat.shape} != {X.shape}"


def test_lstm_autoencoder_reconstruction_error_positive(autoencoder):
    X = np.random.randn(BATCH, SEQ_LEN, INPUT_DIM).astype(np.float32)
    errors = autoencoder.reconstruction_error(X)
    assert errors.shape == (BATCH,)
    assert np.all(errors >= 0), "Reconstruction errors must be non-negative"
    assert np.any(errors > 0), "At least some reconstruction error expected"


def test_lstm_autoencoder_encode_shape(autoencoder):
    X = np.random.randn(BATCH, SEQ_LEN, INPUT_DIM).astype(np.float32)
    latent = autoencoder.encode(X)
    assert latent.shape == (BATCH, HIDDEN_DIM)


def test_lstm_autoencoder_decode_shape(autoencoder):
    latent = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)
    X_hat = autoencoder.decode(latent, SEQ_LEN)
    assert X_hat.shape == (BATCH, SEQ_LEN, INPUT_DIM)


def test_lstm_autoencoder_single_sample(autoencoder):
    x = np.random.randn(SEQ_LEN, INPUT_DIM).astype(np.float32)
    x_hat = autoencoder.forward(x)
    assert x_hat.shape == x.shape
    err = autoencoder.reconstruction_error(x)
    assert isinstance(err, float)
    assert err >= 0


def test_lstm_autoencoder_weights_roundtrip(autoencoder):
    weights = autoencoder.get_weights()
    # Create fresh model and load weights
    model2 = LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, SEQ_LEN, seed=99)
    model2.set_weights(weights)
    X = np.random.randn(3, SEQ_LEN, INPUT_DIM).astype(np.float32)
    np.testing.assert_array_almost_equal(
        autoencoder.forward(X), model2.forward(X)
    )

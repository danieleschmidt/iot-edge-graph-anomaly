"""Anomaly Detector combining LSTM autoencoder with sensor graph."""

import numpy as np
from .lstm_autoencoder import LSTMAutoencoder
from .graph import SensorGraph


class AnomalyDetector:
    """IoT anomaly detector using LSTM autoencoder + graph smoothing.
    
    Fits a threshold on normal data (mean + 2*std of reconstruction errors),
    then flags samples exceeding the threshold as anomalies.
    
    Args:
        n_sensors: Number of input sensor channels.
        hidden_dim: LSTM hidden dimensionality.
        threshold: Optional fixed threshold. If None, auto-calibrated via fit().
        seq_len: Sequence length (inferred from data if None).
        seed: Random seed.
    """

    def __init__(
        self,
        n_sensors: int,
        hidden_dim: int = 16,
        threshold: float = None,
        seq_len: int = None,
        seed: int = 42,
    ):
        self.n_sensors = n_sensors
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.seq_len = seq_len
        self.seed = seed
        self._model: LSTMAutoencoder = None
        self._graph: SensorGraph = None

    def _build_model(self, seq_len: int):
        self.seq_len = seq_len
        self._model = LSTMAutoencoder(
            input_dim=self.n_sensors,
            hidden_dim=self.hidden_dim,
            seq_len=seq_len,
            seed=self.seed,
        )
        self._graph = SensorGraph(n_nodes=self.n_sensors)
        # Connect adjacent sensors by default
        for i in range(self.n_sensors - 1):
            self._graph.add_edge(i, i + 1)

    def fit(self, X_normal: np.ndarray) -> "AnomalyDetector":
        """Calibrate the anomaly threshold using normal data.
        
        Args:
            X_normal: Normal samples of shape (n_samples, seq_len, n_sensors).
        
        Returns:
            self
        """
        n_samples, seq_len, n_sensors = X_normal.shape
        if self._model is None:
            self._build_model(seq_len)

        errors = self.score(X_normal)
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))
        self.threshold = mean_err + 2.0 * std_err
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error for each sample.
        
        Args:
            X: Samples of shape (n_samples, seq_len, n_sensors).
        
        Returns:
            np.ndarray of shape (n_samples,) with reconstruction errors.
        """
        if self._model is None:
            _, seq_len, _ = X.shape
            self._build_model(seq_len)

        return self._model.reconstruction_error(X)  # (n_samples,)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels for each sample.
        
        Args:
            X: Samples of shape (n_samples, seq_len, n_sensors).
        
        Returns:
            np.ndarray of bool shape (n_samples,): True = anomaly.
        """
        if self.threshold is None:
            raise ValueError("Call fit() first or provide a threshold at construction.")

        errors = self.score(X)
        return errors > self.threshold

    def get_model(self) -> LSTMAutoencoder:
        """Return the underlying LSTM autoencoder."""
        return self._model

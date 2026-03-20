"""Synthetic sensor data generator for IoT anomaly detection."""

import numpy as np


class SyntheticSensorData:
    """Generate synthetic IoT sensor data with sinusoidal patterns and noise.
    
    Args:
        n_sensors: Number of sensors/features per timestep.
        seq_len: Length of each time series sequence.
        seed: Random seed for reproducibility.
    """

    def __init__(self, n_sensors: int = 5, seq_len: int = 50, seed: int = 42):
        self.n_sensors = n_sensors
        self.seq_len = seq_len
        self.rng = np.random.RandomState(seed)
        # Each sensor has its own frequency and phase
        self.frequencies = self.rng.uniform(0.05, 0.3, n_sensors)
        self.phases = self.rng.uniform(0, 2 * np.pi, n_sensors)
        self.amplitudes = self.rng.uniform(0.5, 2.0, n_sensors)

    def _base_signal(self, n_samples: int) -> np.ndarray:
        """Generate base sinusoidal signal for all samples."""
        t = np.arange(self.seq_len)  # (seq_len,)
        # shape: (n_sensors, seq_len)
        signals = np.array([
            self.amplitudes[i] * np.sin(2 * np.pi * self.frequencies[i] * t + self.phases[i])
            for i in range(self.n_sensors)
        ])  # (n_sensors, seq_len)
        # Broadcast to (n_samples, seq_len, n_sensors)
        base = np.tile(signals.T[np.newaxis, :, :], (n_samples, 1, 1))
        return base

    def generate_normal(self, n_samples: int) -> np.ndarray:
        """Generate normal (non-anomalous) sensor data.
        
        Returns:
            np.ndarray of shape (n_samples, seq_len, n_sensors)
        """
        base = self._base_signal(n_samples)
        noise = self.rng.normal(0, 0.05, base.shape)
        return (base + noise).astype(np.float32)

    def generate_anomalous(
        self,
        n_samples: int,
        anomaly_type: str = "spike",
    ) -> np.ndarray:
        """Generate anomalous sensor data.
        
        Args:
            n_samples: Number of samples to generate.
            anomaly_type: One of "spike", "drift", "dropout".
        
        Returns:
            np.ndarray of shape (n_samples, seq_len, n_sensors)
        """
        base = self._base_signal(n_samples)
        noise = self.rng.normal(0, 0.05, base.shape)
        data = (base + noise).astype(np.float32)

        if anomaly_type == "spike":
            # Random spike at a random time position
            spike_time = self.rng.randint(5, self.seq_len - 5)
            spike_sensor = self.rng.randint(0, self.n_sensors)
            data[:, spike_time, spike_sensor] += self.rng.uniform(5.0, 10.0, n_samples).astype(np.float32)

        elif anomaly_type == "drift":
            # Linear drift added to all sensors
            drift = np.linspace(0, 3.0, self.seq_len).astype(np.float32)
            data += drift[np.newaxis, :, np.newaxis]

        elif anomaly_type == "dropout":
            # Zero out random sensor for part of the sequence
            dropout_sensor = self.rng.randint(0, self.n_sensors)
            start = self.rng.randint(0, self.seq_len // 2)
            end = start + self.seq_len // 3
            data[:, start:end, dropout_sensor] = 0.0

        else:
            raise ValueError(f"Unknown anomaly_type: {anomaly_type!r}. Use 'spike', 'drift', or 'dropout'.")

        return data

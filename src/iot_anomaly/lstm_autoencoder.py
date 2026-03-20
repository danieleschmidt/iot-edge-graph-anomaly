"""Pure numpy LSTM Autoencoder for time-series reconstruction."""

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -500, 500))


class LSTMCell:
    """Single LSTM cell with numpy forward pass.
    
    Args:
        input_dim: Dimensionality of input.
        hidden_dim: Dimensionality of hidden state.
        seed: Random seed for weight initialization.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)
        scale = 0.1

        # Combined weight matrix for [input_gate, forget_gate, cell_gate, output_gate]
        # W: (4*hidden_dim, input_dim), U: (4*hidden_dim, hidden_dim), b: (4*hidden_dim,)
        self.W = rng.randn(4 * hidden_dim, input_dim).astype(np.float32) * scale
        self.U = rng.randn(4 * hidden_dim, hidden_dim).astype(np.float32) * scale
        self.b = np.zeros(4 * hidden_dim, dtype=np.float32)
        # Forget gate bias initialized to 1 for better gradient flow
        self.b[hidden_dim:2 * hidden_dim] = 1.0

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ):
        """Forward pass for a single timestep.
        
        Args:
            x: Input at current timestep, shape (input_dim,) or (batch, input_dim).
            h_prev: Previous hidden state, shape (hidden_dim,) or (batch, hidden_dim).
            c_prev: Previous cell state, shape (hidden_dim,) or (batch, hidden_dim).
        
        Returns:
            Tuple of (h_next, c_next).
        """
        # gates: (batch, 4*hidden_dim) or (4*hidden_dim,)
        gates = x @ self.W.T + h_prev @ self.U.T + self.b

        hd = self.hidden_dim
        i_gate = _sigmoid(gates[..., :hd])
        f_gate = _sigmoid(gates[..., hd:2 * hd])
        g_gate = _tanh(gates[..., 2 * hd:3 * hd])
        o_gate = _sigmoid(gates[..., 3 * hd:])

        c_next = f_gate * c_prev + i_gate * g_gate
        h_next = o_gate * _tanh(c_next)
        return h_next, c_next

    def get_weights(self) -> dict:
        return {"W": self.W, "U": self.U, "b": self.b}

    def set_weights(self, weights: dict):
        self.W = weights["W"]
        self.U = weights["U"]
        self.b = weights["b"]


class LSTMAutoencoder:
    """LSTM Autoencoder for time-series reconstruction using pure numpy.
    
    Architecture:
      Encoder: LSTM over input sequence -> latent (last hidden state)
      Decoder: repeat latent seq_len times -> LSTM -> linear projection -> output
    
    Args:
        input_dim: Number of input features per timestep.
        hidden_dim: LSTM hidden state dimensionality.
        seq_len: Length of input sequences.
        seed: Random seed for weight initialization.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        rng = np.random.RandomState(seed)

        self.encoder_cell = LSTMCell(input_dim, hidden_dim, seed=seed)
        self.decoder_cell = LSTMCell(hidden_dim, hidden_dim, seed=seed + 1)

        # Output projection: hidden_dim -> input_dim
        self.W_out = rng.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        self.b_out = np.zeros(input_dim, dtype=np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input sequence to latent representation.
        
        Args:
            x: Input of shape (seq_len, input_dim) or (batch, seq_len, input_dim).
        
        Returns:
            Latent vector: last hidden state, shape (hidden_dim,) or (batch, hidden_dim).
        """
        batched = x.ndim == 3
        if not batched:
            x = x[np.newaxis]  # (1, seq_len, input_dim)

        batch = x.shape[0]
        h = np.zeros((batch, self.hidden_dim), dtype=np.float32)
        c = np.zeros((batch, self.hidden_dim), dtype=np.float32)

        for t in range(self.seq_len):
            h, c = self.encoder_cell.forward(x[:, t, :], h, c)

        return h if batched else h[0]

    def decode(self, latent: np.ndarray, seq_len: int) -> np.ndarray:
        """Decode latent vector back to sequence.
        
        Args:
            latent: Shape (hidden_dim,) or (batch, hidden_dim).
            seq_len: Number of timesteps to reconstruct.
        
        Returns:
            Reconstructed sequence of shape (seq_len, input_dim) or (batch, seq_len, input_dim).
        """
        batched = latent.ndim == 2
        if not batched:
            latent = latent[np.newaxis]  # (1, hidden_dim)

        batch = latent.shape[0]
        h = latent.copy()
        c = np.zeros((batch, self.hidden_dim), dtype=np.float32)
        outputs = []

        for _ in range(seq_len):
            h, c = self.decoder_cell.forward(h, h, c)
            out = h @ self.W_out.T + self.b_out  # (batch, input_dim)
            outputs.append(out)

        result = np.stack(outputs, axis=1)  # (batch, seq_len, input_dim)
        return result if batched else result[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass: encode then decode.
        
        Args:
            x: Shape (seq_len, input_dim) or (batch, seq_len, input_dim).
        
        Returns:
            Reconstructed x, same shape as input.
        """
        latent = self.encode(x)
        return self.decode(latent, self.seq_len)

    def reconstruction_error(self, x: np.ndarray) -> float:
        """Compute mean squared reconstruction error.
        
        Args:
            x: Shape (seq_len, input_dim) or (batch, seq_len, input_dim).
        
        Returns:
            If batched: np.ndarray of shape (batch,). 
            If single: float scalar.
        """
        batched = x.ndim == 3
        x_hat = self.forward(x)
        diff = x - x_hat
        if batched:
            return np.mean(diff ** 2, axis=(1, 2))
        return float(np.mean(diff ** 2))

    def fit(self, X: np.ndarray, epochs: int = 5, lr: float = 0.01) -> "LSTMAutoencoder":
        """Minimal gradient-free fitting using random perturbation search.
        
        This is a simplified implementation focused on demonstrating the interface.
        Uses random weight perturbations and keeps improvements.
        
        Args:
            X: Training data of shape (n_samples, seq_len, input_dim).
            epochs: Number of training epochs.
            lr: Controls perturbation scale (learning rate analog).
        
        Returns:
            self
        """
        rng = np.random.RandomState(0)
        best_error = float(np.mean(self.reconstruction_error(X)))

        for _ in range(epochs):
            # Try small random perturbations to W_out only (fast to evaluate)
            perturbation = rng.randn(*self.W_out.shape).astype(np.float32) * lr
            old_W = self.W_out.copy()
            self.W_out += perturbation
            new_error = float(np.mean(self.reconstruction_error(X)))
            if new_error < best_error:
                best_error = new_error
            else:
                self.W_out = old_W  # revert

        return self

    def get_weights(self) -> dict:
        """Return all model weights as a dict."""
        return {
            "encoder_W": self.encoder_cell.W,
            "encoder_U": self.encoder_cell.U,
            "encoder_b": self.encoder_cell.b,
            "decoder_W": self.decoder_cell.W,
            "decoder_U": self.decoder_cell.U,
            "decoder_b": self.decoder_cell.b,
            "W_out": self.W_out,
            "b_out": self.b_out,
        }

    def set_weights(self, weights: dict):
        """Load weights from a dict."""
        self.encoder_cell.W = weights["encoder_W"]
        self.encoder_cell.U = weights["encoder_U"]
        self.encoder_cell.b = weights["encoder_b"]
        self.decoder_cell.W = weights["decoder_W"]
        self.decoder_cell.U = weights["decoder_U"]
        self.decoder_cell.b = weights["decoder_b"]
        self.W_out = weights["W_out"]
        self.b_out = weights["b_out"]

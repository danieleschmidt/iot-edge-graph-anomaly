"""
LSTM Autoencoder for temporal anomaly detection in IoT sensor data.

This module implements a Long Short-Term Memory (LSTM) based autoencoder
for detecting anomalies in time-series IoT sensor data. The autoencoder
learns to reconstruct normal patterns and identifies anomalies based on
reconstruction error.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for time-series anomaly detection.
    
    The model consists of an encoder-decoder architecture:
    - Encoder: LSTM layers that compress the input sequence into a latent representation
    - Decoder: LSTM layers that reconstruct the input from the latent representation
    
    Args:
        input_size (int): Number of input features per timestep
        hidden_size (int): Number of hidden units in LSTM layers
        num_layers (int): Number of LSTM layers in encoder/decoder
        dropout (float): Dropout probability for regularization
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Encoder: compress sequence to latent representation
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder: reconstruct sequence from latent representation
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer: map hidden state back to input dimensions
        self.output_layer = nn.Linear(hidden_size, input_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Encode input sequence to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            encoded: Encoded representation (batch_size, seq_len, hidden_size)
            hidden: Final hidden state tuple (h_n, c_n)
        """
        encoded, hidden = self.encoder(x)
        return encoded, hidden
    
    def decode(self, encoded: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation back to original sequence.
        
        Args:
            encoded: Encoded representation (batch_size, seq_len, hidden_size)
            seq_len: Length of sequence to reconstruct
            
        Returns:
            decoded: Reconstructed sequence (batch_size, seq_len, input_size)
        """
        # Use the last encoded state as initial hidden state for decoder
        batch_size = encoded.size(0)
        
        # Initialize decoder input with zeros
        decoder_input = torch.zeros(batch_size, seq_len, self.hidden_size).to(encoded.device)
        
        # For reconstruction, we'll use the encoded representation directly
        # In a more sophisticated version, we might use teacher forcing
        decoded_hidden, _ = self.decoder(encoded)
        
        # Map back to input dimensions
        decoded = self.output_layer(decoded_hidden)
        
        return decoded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            reconstruction: Reconstructed input (batch_size, seq_len, input_size)
        """
        # Encode input sequence
        encoded, _ = self.encode(x)
        
        # Decode to reconstruct original sequence
        reconstruction = self.decode(encoded, x.size(1))
        
        return reconstruction
    
    def compute_reconstruction_error(
        self, 
        x: torch.Tensor, 
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute reconstruction error (anomaly score).
        
        Args:
            x: Input tensor
            reduction: How to reduce the error ('mean', 'sum', 'none')
            
        Returns:
            error: Reconstruction error
        """
        reconstruction = self.forward(x)
        
        if reduction == 'none':
            # Return error per sample and timestep
            error = torch.mean((x - reconstruction) ** 2, dim=-1)
        elif reduction == 'sum':
            error = torch.sum((x - reconstruction) ** 2)
        else:  # mean
            error = torch.mean((x - reconstruction) ** 2)
            
        return error
    
    def detect_anomalies(
        self, 
        x: torch.Tensor, 
        threshold: float
    ) -> torch.Tensor:
        """
        Detect anomalies based on reconstruction error threshold.
        
        Args:
            x: Input tensor
            threshold: Anomaly detection threshold
            
        Returns:
            anomalies: Boolean tensor indicating anomalies
        """
        with torch.no_grad():
            errors = self.compute_reconstruction_error(x, reduction='none')
            # Average error across sequence for each sample
            sample_errors = torch.mean(errors, dim=1)
            anomalies = sample_errors > threshold
            
        return anomalies
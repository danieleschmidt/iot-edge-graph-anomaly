"""
Advanced Transformer-based Variational Autoencoder for IoT Anomaly Detection.

This module implements a novel Transformer-VAE hybrid architecture that combines
transformer attention mechanisms with probabilistic VAE framework for superior
temporal modeling in IoT anomaly detection scenarios.

Key Features:
- Multi-head self-attention for long-range temporal dependencies
- Variational latent space for uncertainty quantification  
- Edge-optimized architecture with quantization support
- Integration with existing LSTM-GNN hybrid system
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer architecture.
    
    Adds position information to input embeddings using sinusoidal functions.
    Optimized for variable sequence lengths in IoT time series data.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:x.size(0), :]


class MultiHeadSelfAttention(nn.Module):
    """
    Optimized multi-head self-attention for IoT time series.
    
    Features:
    - Sparse attention patterns for efficiency
    - Temperature-scaled attention for better convergence
    - Dropout for regularization
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        temperature: float = 1.0,
        sparse_factor: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        self.sparse_factor = sparse_factor
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with sparse attention computation.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor with same shape as input
        """
        seq_len, batch_size, d_model = x.size()
        residual = x
        
        # Apply layer normalization (pre-norm architecture)
        x = self.layer_norm(x)
        
        # Compute Q, K, V
        q = self.w_q(x).view(seq_len, batch_size, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(seq_len, batch_size, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(seq_len, batch_size, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        
        # Apply sparse attention pattern for efficiency
        if self.sparse_factor < 1.0:
            scores = self._apply_sparse_pattern(scores)
        
        # Apply mask if provided
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(seq_len, batch_size, d_model)
        output = self.w_o(context)
        
        # Residual connection
        return output + residual
    
    def _apply_sparse_pattern(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply sparse attention pattern to reduce computational complexity."""
        seq_len = scores.size(-1)
        
        # Create sparse mask - keep only top-k scores per row
        k = max(1, int(seq_len * self.sparse_factor))
        
        # Get top-k indices
        _, top_indices = torch.topk(scores, k=k, dim=-1)
        
        # Create sparse mask
        sparse_mask = torch.zeros_like(scores)
        sparse_mask.scatter_(-1, top_indices, 1.0)
        
        # Apply sparse mask
        return scores.masked_fill(sparse_mask == 0, -1e9)


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with feed-forward network.
    
    Architecture:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Residual connections and layer normalization
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        d_ff: int = 2048, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU activation for better performance
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer layer."""
        # Self-attention with residual connection
        x = self.self_attn(x, mask)
        
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        return x + residual


class TransformerEncoder(nn.Module):
    """
    Multi-layer transformer encoder for temporal modeling.
    
    Stacks multiple transformer layers with optional gradient checkpointing
    for memory efficiency during training.
    """
    
    def __init__(
        self, 
        input_size: int,
        d_model: int = 256, 
        num_layers: int = 6, 
        num_heads: int = 8, 
        d_ff: int = 1024, 
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Input projection and embedding
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Optional attention mask
            
        Returns:
            Encoded tensor of shape (batch_size, seq_len, d_model)
        """
        # Reshape to (seq_len, batch_size, input_size)
        x = x.transpose(0, 1)
        
        # Input projection and positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.embedding_dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, mask)
            else:
                x = layer(x, mask)
        
        # Final layer norm and reshape back
        x = self.layer_norm(x)
        return x.transpose(0, 1)


class VariationalLayer(nn.Module):
    """
    Variational layer for latent space modeling.
    
    Implements reparameterization trick for differentiable sampling
    from Gaussian latent distribution.
    """
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Separate networks for mean and log variance
        self.mu_layer = nn.Linear(input_dim, latent_dim)
        self.logvar_layer = nn.Linear(input_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with reparameterization trick.
        
        Returns:
            z: Sampled latent variables
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Use mean during inference
        
        return z, mu, logvar


class TransformerVAE(nn.Module):
    """
    Transformer-based Variational Autoencoder for IoT Anomaly Detection.
    
    Combines transformer temporal modeling with variational latent space
    for superior anomaly detection with uncertainty quantification.
    
    Architecture:
    - Transformer encoder for temporal feature extraction
    - Variational bottleneck for latent space modeling
    - Transformer decoder for reconstruction
    - Anomaly scoring based on reconstruction error and KL divergence
    """
    
    def __init__(
        self,
        input_size: int = 5,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        latent_dim: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        beta: float = 1.0,  # KL divergence weight
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            input_size=input_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Variational layer
        self.variational_layer = VariationalLayer(d_model, latent_dim)
        
        # Decoder (reverse of encoder)
        self.decoder = TransformerEncoder(
            input_size=latent_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_size)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input sequence to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            z: Latent variables
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Pass through transformer encoder
        encoded = self.encoder(x)  # (batch_size, seq_len, d_model)
        
        # Global average pooling for sequence-level representation
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # Variational layer
        z, mu, logvar = self.variational_layer(pooled)
        
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent variables to reconstruct input.
        
        Args:
            z: Latent variables of shape (batch_size, latent_dim)
            seq_len: Target sequence length
            
        Returns:
            Reconstructed sequence of shape (batch_size, seq_len, input_size)
        """
        batch_size = z.size(0)
        
        # Expand latent variables to sequence length
        z_expanded = z.unsqueeze(1).expand(batch_size, seq_len, self.latent_dim)
        
        # Pass through transformer decoder
        decoded = self.decoder(z_expanded)  # (batch_size, seq_len, d_model)
        
        # Project to output space
        output = self.output_projection(decoded)  # (batch_size, seq_len, input_size)
        
        return output
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed input
                - z: Latent variables
                - mu: Mean of latent distribution
                - logvar: Log variance of latent distribution
        """
        seq_len = x.size(1)
        
        # Encode
        z, mu, logvar = self.encode(x)
        
        # Decode
        reconstruction = self.decode(z, seq_len)
        
        return {
            'reconstruction': reconstruction,
            'z': z,
            'mu': mu,
            'logvar': logvar
        }
    
    def compute_anomaly_score(
        self, 
        x: torch.Tensor, 
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute anomaly score combining reconstruction error and KL divergence.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            reduction: Reduction method ('mean', 'sum', 'none')
            
        Returns:
            Anomaly scores
        """
        outputs = self.forward(x)
        reconstruction = outputs['reconstruction']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction error (MSE)
        recon_error = F.mse_loss(reconstruction, x, reduction='none')
        if reduction == 'mean':
            recon_error = recon_error.mean(dim=[1, 2])
        elif reduction == 'sum':
            recon_error = recon_error.sum(dim=[1, 2])
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Combined anomaly score
        anomaly_score = recon_error + self.beta * kl_div
        
        return anomaly_score
    
    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss with reconstruction and KL divergence terms.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Dictionary containing loss components
        """
        outputs = self.forward(x)
        reconstruction = outputs['reconstruction']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation for downstream tasks."""
        z, _, _ = self.encode(x)
        return z
    
    def sample_from_latent(self, num_samples: int, seq_len: int) -> torch.Tensor:
        """
        Generate samples from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            seq_len: Sequence length for generated samples
            
        Returns:
            Generated samples
        """
        device = next(self.parameters()).device
        
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode to generate samples
        with torch.no_grad():
            samples = self.decode(z, seq_len)
        
        return samples


class OptimizedTransformerVAE(TransformerVAE):
    """
    Edge-optimized version of TransformerVAE with quantization support.
    
    Features:
    - Reduced model complexity for edge deployment
    - Dynamic quantization support
    - Pruning-friendly architecture
    - Memory-efficient attention patterns
    """
    
    def __init__(self, *args, **kwargs):
        # Reduce default parameters for edge deployment
        kwargs.setdefault('d_model', 128)
        kwargs.setdefault('num_layers', 3)
        kwargs.setdefault('num_heads', 4)
        kwargs.setdefault('d_ff', 512)
        kwargs.setdefault('latent_dim', 32)
        
        super().__init__(*args, **kwargs)
        
        # Enable memory-efficient features
        self.encoder.use_gradient_checkpointing = True
        self.decoder.use_gradient_checkpointing = True
    
    def quantize_model(self, dtype=torch.qint8) -> 'OptimizedTransformerVAE':
        """Apply dynamic quantization for edge deployment."""
        quantized_model = torch.quantization.quantize_dynamic(
            self, {nn.Linear}, dtype=dtype
        )
        logger.info(f"Model quantized to {dtype}")
        return quantized_model


# Factory function for easy model creation
def create_transformer_vae(
    config: Dict[str, Any], 
    optimized: bool = True
) -> TransformerVAE:
    """
    Factory function to create TransformerVAE models based on configuration.
    
    Args:
        config: Model configuration dictionary
        optimized: Whether to create optimized version for edge deployment
        
    Returns:
        TransformerVAE model instance
    """
    model_class = OptimizedTransformerVAE if optimized else TransformerVAE
    
    return model_class(
        input_size=config.get('input_size', 5),
        d_model=config.get('d_model', 128 if optimized else 256),
        num_layers=config.get('num_layers', 3 if optimized else 4),
        num_heads=config.get('num_heads', 4 if optimized else 8),
        d_ff=config.get('d_ff', 512 if optimized else 1024),
        latent_dim=config.get('latent_dim', 32 if optimized else 64),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 500),
        beta=config.get('beta', 1.0)
    )
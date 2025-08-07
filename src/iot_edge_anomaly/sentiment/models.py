"""
Sentiment Analysis Models using LSTM-based architectures.

Implements multiple sentiment analysis approaches for comparative research:
1. Simple LSTM classifier
2. BiLSTM with attention
3. LSTM-CNN hybrid
4. Transformer-based approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np

class SentimentLSTM(nn.Module):
    """
    Simple LSTM-based sentiment classifier.
    
    This model serves as the baseline for comparative analysis.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,  # negative, neutral, positive
        dropout: float = 0.2,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif 'embedding' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the sentiment LSTM.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size = input_ids.size(0)
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the final hidden state for classification
        # Take the last layer's hidden state
        final_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Apply dropout and classify
        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        return logits


class BiLSTMAttentionSentiment(nn.Module):
    """
    Bidirectional LSTM with attention mechanism for sentiment analysis.
    
    This model represents a more advanced approach for comparative analysis.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name or 'bilstm' in name:
                    nn.init.xavier_uniform_(param)
                elif 'embedding' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with bidirectional LSTM and attention.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention mechanism
        attention_weights = torch.tanh(self.attention(lstm_out))  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=-1)  # (batch_size, seq_len)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_weights = attention_weights * attention_mask.float()
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted sum of LSTM outputs
        attended_output = torch.bmm(attention_weights.unsqueeze(1), lstm_out)  # (batch_size, 1, hidden_dim * 2)
        attended_output = attended_output.squeeze(1)  # (batch_size, hidden_dim * 2)
        
        # Classification
        output = self.dropout(attended_output)
        logits = self.classifier(output)
        
        return logits


class LSTMCNNHybridSentiment(nn.Module):
    """
    LSTM-CNN hybrid model for sentiment analysis.
    
    Combines sequential modeling (LSTM) with local pattern detection (CNN).
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        cnn_filters: int = 100,
        filter_sizes: List[int] = [2, 3, 4, 5],
        num_classes: int = 3,
        dropout: float = 0.2,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.cnn_filters = cnn_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        # LSTM component
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden_dim,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # CNN components
        self.convs = nn.ModuleList([
            nn.Conv1d(lstm_hidden_dim, cnn_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Combined classifier
        combined_features = len(filter_sizes) * cnn_filters
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_features, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif 'conv' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'embedding' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through LSTM-CNN hybrid.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, lstm_hidden_dim)
        
        # Apply attention mask to LSTM output if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            lstm_out = lstm_out * mask
        
        # CNN processing - transpose for Conv1d
        lstm_out = lstm_out.transpose(1, 2)  # (batch_size, lstm_hidden_dim, seq_len)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(lstm_out))  # (batch_size, cnn_filters, new_seq_len)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch_size, cnn_filters, 1)
            conv_outputs.append(pooled.squeeze(-1))  # (batch_size, cnn_filters)
        
        # Concatenate CNN features
        combined_features = torch.cat(conv_outputs, dim=-1)  # (batch_size, len(filter_sizes) * cnn_filters)
        
        # Classification
        output = self.dropout(combined_features)
        logits = self.classifier(output)
        
        return logits


class TransformerSentiment(nn.Module):
    """
    Lightweight Transformer model for sentiment analysis.
    
    Uses multi-head self-attention for capturing long-range dependencies.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 256,
        max_seq_len: int = 512,
        num_classes: int = 3,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        embeddings = token_emb + pos_emb
        
        # Create padding mask for transformer
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        
        # Transformer forward pass
        transformer_out = self.transformer(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Use [CLS]-like token (first token) for classification
        cls_output = transformer_out[:, 0, :]  # (batch_size, embedding_dim)
        
        # Classification
        output = self.layer_norm(cls_output)
        output = self.dropout(output)
        logits = self.classifier(output)
        
        return logits


class SentimentEnsemble(nn.Module):
    """
    Ensemble model combining multiple sentiment analysis approaches.
    
    Provides robust predictions by combining different model architectures.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        num_classes: int = 3
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_classes = num_classes
        
        # Set ensemble weights
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            assert len(weights) == self.num_models, "Number of weights must match number of models"
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        self.register_buffer('weight_tensor', torch.tensor(self.weights, dtype=torch.float32))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Weighted average logits of shape (batch_size, num_classes)
        """
        predictions = []
        
        for model in self.models:
            logits = model(input_ids, attention_mask)
            predictions.append(logits)
        
        # Stack predictions and compute weighted average
        stacked_preds = torch.stack(predictions, dim=0)  # (num_models, batch_size, num_classes)
        weights = self.weight_tensor.view(-1, 1, 1)  # (num_models, 1, 1)
        
        ensemble_logits = torch.sum(stacked_preds * weights, dim=0)  # (batch_size, num_classes)
        
        return ensemble_logits
    
    def update_weights(self, new_weights: List[float]):
        """Update ensemble weights based on validation performance."""
        assert len(new_weights) == self.num_models
        total_weight = sum(new_weights)
        self.weights = [w / total_weight for w in new_weights]
        self.weight_tensor.data = torch.tensor(self.weights, dtype=torch.float32)


def create_sentiment_model(model_type: str, vocab_size: int, **kwargs) -> nn.Module:
    """
    Factory function to create sentiment analysis models.
    
    Args:
        model_type: Type of model ('lstm', 'bilstm_attention', 'lstm_cnn', 'transformer', 'ensemble')
        vocab_size: Size of vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        Initialized sentiment analysis model
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return SentimentLSTM(vocab_size, **kwargs)
    elif model_type == 'bilstm_attention':
        return BiLSTMAttentionSentiment(vocab_size, **kwargs)
    elif model_type == 'lstm_cnn':
        return LSTMCNNHybridSentiment(vocab_size, **kwargs)
    elif model_type == 'transformer':
        return TransformerSentiment(vocab_size, **kwargs)
    elif model_type == 'ensemble':
        # Create ensemble with all models
        models = [
            SentimentLSTM(vocab_size, **kwargs),
            BiLSTMAttentionSentiment(vocab_size, **kwargs),
            LSTMCNNHybridSentiment(vocab_size, **kwargs),
            TransformerSentiment(vocab_size, **kwargs)
        ]
        return SentimentEnsemble(models, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
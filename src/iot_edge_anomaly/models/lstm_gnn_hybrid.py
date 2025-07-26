"""
LSTM-GNN Hybrid Model for IoT Edge Anomaly Detection.

This module combines the LSTM autoencoder (for temporal patterns) with
a Graph Neural Network (for spatial relationships) to create a hybrid
model that captures both temporal and spatial dependencies in IoT sensor data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging

from .lstm_autoencoder import LSTMAutoencoder
from .gnn_layer import GraphNeuralNetworkLayer, create_sensor_graph

logger = logging.getLogger(__name__)


class FeatureFusionLayer(nn.Module):
    """
    Feature fusion layer for combining LSTM and GNN outputs.
    
    Supports multiple fusion strategies:
    - Concatenate: Simple concatenation of features
    - Add: Element-wise addition (requires same dimensions)
    - Attention: Learned attention weighting between modalities
    - Gate: Gated fusion with learned mixing weights
    """
    
    def __init__(
        self,
        lstm_dim: int,
        gnn_dim: int,
        output_dim: int,
        fusion_method: str = 'concatenate'
    ):
        super(FeatureFusionLayer, self).__init__()
        
        self.lstm_dim = lstm_dim
        self.gnn_dim = gnn_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        if fusion_method == 'concatenate':
            self.fusion_layer = nn.Linear(lstm_dim + gnn_dim, output_dim)
            
        elif fusion_method == 'add':
            if lstm_dim != gnn_dim:
                raise ValueError("LSTM and GNN dimensions must match for 'add' fusion")
            self.fusion_layer = nn.Linear(lstm_dim, output_dim)
            
        elif fusion_method == 'attention':
            self.lstm_attention = nn.Linear(lstm_dim, 1)
            self.gnn_attention = nn.Linear(gnn_dim, 1)
            self.fusion_layer = nn.Linear(max(lstm_dim, gnn_dim), output_dim)
            
        elif fusion_method == 'gate':
            self.gate_layer = nn.Sequential(
                nn.Linear(lstm_dim + gnn_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1)
            )
            self.lstm_proj = nn.Linear(lstm_dim, output_dim)
            self.gnn_proj = nn.Linear(gnn_dim, output_dim)
            
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self, 
        lstm_features: torch.Tensor, 
        gnn_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse LSTM and GNN features.
        
        Args:
            lstm_features: LSTM output features
            gnn_features: GNN output features
            
        Returns:
            Fused features
        """
        if self.fusion_method == 'concatenate':
            combined = torch.cat([lstm_features, gnn_features], dim=-1)
            return self.fusion_layer(combined)
            
        elif self.fusion_method == 'add':
            combined = lstm_features + gnn_features
            return self.fusion_layer(combined)
            
        elif self.fusion_method == 'attention':
            # Compute attention weights
            lstm_att = torch.sigmoid(self.lstm_attention(lstm_features))
            gnn_att = torch.sigmoid(self.gnn_attention(gnn_features))
            
            # Normalize attention weights
            total_att = lstm_att + gnn_att
            lstm_weight = lstm_att / total_att
            gnn_weight = gnn_att / total_att
            
            # Weighted combination (assuming same dimensions)
            if lstm_features.size(-1) == gnn_features.size(-1):
                combined = lstm_weight * lstm_features + gnn_weight * gnn_features
            else:
                # Use the larger dimension
                if lstm_features.size(-1) > gnn_features.size(-1):
                    combined = lstm_features
                else:
                    combined = gnn_features
                    
            return self.fusion_layer(combined)
            
        elif self.fusion_method == 'gate':
            # Learn gating weights
            gate_input = torch.cat([lstm_features, gnn_features], dim=-1)
            gate_weights = self.gate_layer(gate_input)  # (batch, 2)
            
            # Project features and apply gates
            lstm_proj = self.lstm_proj(lstm_features)
            gnn_proj = self.gnn_proj(gnn_features)
            
            # Weighted combination
            combined = (gate_weights[:, 0:1] * lstm_proj + 
                       gate_weights[:, 1:2] * gnn_proj)
            
            return combined


class LSTMGNNHybridModel(nn.Module):
    """
    Hybrid LSTM-GNN model for IoT anomaly detection.
    
    Combines temporal modeling (LSTM autoencoder) with spatial modeling (GNN)
    to capture both temporal sequences and sensor network topology for
    enhanced anomaly detection performance.
    
    Args:
        config: Configuration dictionary with 'lstm', 'gnn', and 'fusion' sections
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(LSTMGNNHybridModel, self).__init__()
        
        self.config = config
        
        # Extract configurations
        lstm_config = config.get('lstm', {})
        gnn_config = config.get('gnn', {})
        fusion_config = config.get('fusion', {})
        
        # Initialize LSTM autoencoder
        self.lstm_autoencoder = LSTMAutoencoder(
            input_size=lstm_config.get('input_size', 5),
            hidden_size=lstm_config.get('hidden_size', 64),
            num_layers=lstm_config.get('num_layers', 2),
            dropout=lstm_config.get('dropout', 0.1)
        )
        
        # Initialize GNN layer
        self.gnn_layer = GraphNeuralNetworkLayer(
            input_dim=gnn_config.get('input_dim', 64),
            hidden_dim=gnn_config.get('hidden_dim', 32),
            output_dim=gnn_config.get('output_dim', 64),
            num_layers=gnn_config.get('num_layers', 2),
            dropout=gnn_config.get('dropout', 0.1)
        )
        
        # Initialize feature fusion
        self.fusion_layer = FeatureFusionLayer(
            lstm_dim=lstm_config.get('hidden_size', 64),
            gnn_dim=gnn_config.get('output_dim', 64),
            output_dim=fusion_config.get('output_dim', 128),
            fusion_method=fusion_config.get('method', 'concatenate')
        )
        
        # Output reconstruction layer
        self.reconstruction_layer = nn.Linear(
            fusion_config.get('output_dim', 128),
            lstm_config.get('input_size', 5)
        )
        
        # Graph construction parameters
        self.graph_method = config.get('graph', {}).get('method', 'correlation')
        self.graph_threshold = config.get('graph', {}).get('threshold', 0.5)
        
        logger.info(f"Initialized LSTM-GNN hybrid model with fusion: {fusion_config.get('method', 'concatenate')}")
    
    def forward(
        self, 
        time_series_data: torch.Tensor,
        graph_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through hybrid model.
        
        Args:
            time_series_data: Time-series data (batch, seq_len, num_sensors)
            graph_data: Optional pre-computed graph data
            
        Returns:
            Reconstructed time-series data
        """
        batch_size, seq_len, num_sensors = time_series_data.shape
        
        # LSTM path: Process temporal sequences
        lstm_encoded, _ = self.lstm_autoencoder.encode(time_series_data)
        
        # Take the last timestep encoding for spatial processing
        lstm_features = lstm_encoded[:, -1, :]  # (batch, hidden_size)
        
        # GNN path: Process spatial relationships
        if graph_data is None:
            # Create graph from time-series data
            graph_data = create_sensor_graph(
                time_series_data,
                method=self.graph_method,
                threshold=self.graph_threshold
            )
        
        # Prepare node features for GNN
        # Use LSTM encodings as node features
        node_features = lstm_features.mean(dim=0).unsqueeze(0).repeat(num_sensors, 1)
        if graph_data['x'].size(0) != num_sensors:
            # Adjust if graph has different number of nodes
            node_features = graph_data['x']
        
        # GNN forward pass
        gnn_features = self.gnn_layer(node_features, graph_data['edge_index'])
        
        # Aggregate GNN features to match batch dimension
        gnn_aggregated = torch.mean(gnn_features, dim=0).unsqueeze(0).repeat(batch_size, 1)
        
        # Feature fusion
        fused_features = self.fusion_layer(lstm_features, gnn_aggregated)
        
        # Reconstruct time-series
        # Expand fused features to sequence length
        fused_expanded = fused_features.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Final reconstruction
        reconstruction = self.reconstruction_layer(fused_expanded)
        
        return reconstruction
    
    def encode(
        self, 
        time_series_data: torch.Tensor,
        graph_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode time-series data using both LSTM and GNN.
        
        Args:
            time_series_data: Input time-series
            graph_data: Optional graph data
            
        Returns:
            Tuple of (lstm_encoding, gnn_encoding)
        """
        batch_size, seq_len, num_sensors = time_series_data.shape
        
        # LSTM encoding
        lstm_encoded, _ = self.lstm_autoencoder.encode(time_series_data)
        lstm_features = lstm_encoded[:, -1, :]
        
        # GNN encoding
        if graph_data is None:
            graph_data = create_sensor_graph(
                time_series_data,
                method=self.graph_method,
                threshold=self.graph_threshold
            )
        
        node_features = lstm_features.mean(dim=0).unsqueeze(0).repeat(num_sensors, 1)
        if graph_data['x'].size(0) != num_sensors:
            node_features = graph_data['x']
        
        gnn_features = self.gnn_layer(node_features, graph_data['edge_index'])
        gnn_aggregated = torch.mean(gnn_features, dim=0).unsqueeze(0).repeat(batch_size, 1)
        
        return lstm_features, gnn_aggregated
    
    def compute_hybrid_anomaly_score(
        self,
        time_series_data: torch.Tensor,
        graph_data: Optional[Dict[str, torch.Tensor]] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute anomaly score using hybrid model.
        
        Args:
            time_series_data: Input data
            graph_data: Optional graph data
            reduction: How to reduce the error
            
        Returns:
            Anomaly scores
        """
        reconstruction = self.forward(time_series_data, graph_data)
        
        # Compute reconstruction error
        error = torch.mean((time_series_data - reconstruction) ** 2, dim=-1)
        
        if reduction == 'mean':
            return torch.mean(error)
        elif reduction == 'sum':
            return torch.sum(error)
        else:  # 'none'
            return error
    
    def detect_anomalies(
        self,
        time_series_data: torch.Tensor,
        threshold: float,
        graph_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Detect anomalies using hybrid model.
        
        Args:
            time_series_data: Input data
            threshold: Anomaly threshold
            graph_data: Optional graph data
            
        Returns:
            Boolean tensor indicating anomalies
        """
        with torch.no_grad():
            scores = self.compute_hybrid_anomaly_score(
                time_series_data, graph_data, reduction='none'
            )
            # Average across sequence for per-sample score
            sample_scores = torch.mean(scores, dim=1)
            return sample_scores > threshold
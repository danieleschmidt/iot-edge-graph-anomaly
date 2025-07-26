"""
Graph Neural Network Layer for IoT Edge Anomaly Detection.

This module implements a Graph Neural Network (GNN) to capture spatial 
relationships between IoT sensors. The GNN processes sensor topology graphs
to learn structural patterns that complement the temporal patterns learned
by the LSTM autoencoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GraphTopologyBuilder:
    """
    Builder class for creating graph topologies from sensor networks.
    
    Supports various methods for building edge connectivity:
    - Correlation-based: Connect sensors with high temporal correlation
    - Distance-based: Connect sensors based on physical proximity
    - Fully-connected: Dense connectivity for small sensor networks
    """
    
    def __init__(self):
        self.threshold = 0.5
    
    def build_from_correlation(
        self, 
        correlation_matrix: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Build graph edges from sensor correlation matrix.
        
        Args:
            correlation_matrix: Symmetric correlation matrix (n_sensors x n_sensors)
            threshold: Minimum correlation to create edge
            
        Returns:
            edge_index: Edge indices of shape (2, num_edges)
        """
        n_sensors = correlation_matrix.size(0)
        
        # Get indices where correlation > threshold (excluding diagonal)
        mask = (correlation_matrix > threshold) & (torch.eye(n_sensors) == 0)
        source_nodes, target_nodes = torch.where(mask)
        
        # Create bidirectional edges
        edge_index = torch.stack([
            torch.cat([source_nodes, target_nodes]),
            torch.cat([target_nodes, source_nodes])
        ], dim=0)
        
        return edge_index
    
    def build_fully_connected(self, num_nodes: int) -> torch.Tensor:
        """Build fully connected graph for small sensor networks."""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edges.append([i, j])
        
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def build_from_distance(
        self, 
        positions: torch.Tensor, 
        max_distance: float = 1.0
    ) -> torch.Tensor:
        """
        Build graph from sensor positions (for future use).
        
        Args:
            positions: Sensor positions (n_sensors x 2 or 3)
            max_distance: Maximum distance to create edge
            
        Returns:
            edge_index: Edge indices
        """
        n_sensors = positions.size(0)
        
        # Compute pairwise distances
        distances = torch.cdist(positions, positions)
        
        # Create edges for sensors within max_distance
        mask = (distances <= max_distance) & (distances > 0)
        source_nodes, target_nodes = torch.where(mask)
        
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        return edge_index


class GraphNeuralNetworkLayer(nn.Module):
    """
    Graph Neural Network layer for processing IoT sensor topology.
    
    Uses Graph Convolutional Networks (GCN) to learn spatial relationships
    between sensors and enhance anomaly detection by incorporating
    structural information.
    
    Args:
        input_dim: Input feature dimension per node
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension per node
        num_layers: Number of GCN layers
        dropout: Dropout probability for regularization
        activation: Activation function ('relu', 'elu', 'tanh')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super(GraphNeuralNetworkLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        else:
            # Single layer case
            self.convs[0] = GCNConv(input_dim, output_dim)
        
        # Batch normalization for stability
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(output_dim))
            else:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized GNN with {num_layers} layers: {input_dim} -> {hidden_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN layers.
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            
        Returns:
            Node embeddings (num_nodes, output_dim)
        """
        # Handle empty graphs (single node, no edges)
        if edge_index.size(1) == 0:
            # For isolated nodes, use simple linear transformation
            for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                if i == 0:
                    # For GCN with no edges, manually apply weight transformation
                    x = torch.mm(x, conv.lin.weight.t()) + conv.bias
                else:
                    x = torch.mm(x, conv.lin.weight.t()) + conv.bias
                
                x = bn(x)
                if i < len(self.convs) - 1:  # No activation on last layer
                    x = self.activation(x)
                    x = self.dropout_layer(x)
            return x
        
        # Normal GCN forward pass
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            
            if i < len(self.convs) - 1:  # No activation on last layer
                x = self.activation(x)
                x = self.dropout_layer(x)
        
        return x
    
    def get_node_embeddings(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Get intermediate node embeddings from specific layer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            layer_idx: Layer index (-1 for final layer)
            
        Returns:
            Node embeddings from specified layer
        """
        if layer_idx == -1:
            return self.forward(x, edge_index)
        
        # Forward pass up to specified layer
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if edge_index.size(1) == 0:
                x = torch.mm(x, conv.lin.weight.t()) + conv.bias
            else:
                x = conv(x, edge_index)
            
            x = bn(x)
            
            if i == layer_idx:
                return x
            
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)
        
        return x


class SpatialAttentionMechanism(nn.Module):
    """
    Spatial attention mechanism for weighting sensor importance.
    
    Learns to focus on the most relevant sensors for anomaly detection
    based on their spatial relationships and current feature values.
    """
    
    def __init__(self, feature_dim: int, attention_dim: int = 16):
        super(SpatialAttentionMechanism, self).__init__()
        
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.attention_fc = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spatial attention weights.
        
        Args:
            node_features: Node features (num_nodes, feature_dim)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Compute attention scores
        attention_scores = self.attention_fc(node_features)  # (num_nodes, 1)
        attention_weights = self.softmax(attention_scores.squeeze(-1))  # (num_nodes,)
        
        # Apply attention weights
        attended_features = node_features * attention_weights.unsqueeze(-1)
        
        return attended_features, attention_weights


def create_sensor_graph(
    sensor_data: torch.Tensor,
    method: str = 'correlation',
    threshold: float = 0.5,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Create graph representation from sensor data.
    
    Args:
        sensor_data: Time-series sensor data (batch, seq_len, num_sensors)
        method: Graph construction method ('correlation', 'fully_connected')
        threshold: Threshold for edge creation
        **kwargs: Additional arguments for graph construction
        
    Returns:
        Dict containing 'x' (node features) and 'edge_index' (edges)
    """
    batch_size, seq_len, num_sensors = sensor_data.shape
    
    builder = GraphTopologyBuilder()
    
    if method == 'correlation':
        # Compute correlation matrix from time-series data
        # Average over batch dimension for consistent graph structure
        data_flat = sensor_data.view(-1, num_sensors)  # (batch*seq_len, num_sensors)
        correlation_matrix = torch.corrcoef(data_flat.t())
        
        # Handle NaN values (can occur with constant sensors)
        correlation_matrix = torch.nan_to_num(correlation_matrix, nan=0.0)
        
        edge_index = builder.build_from_correlation(correlation_matrix, threshold)
        
    elif method == 'fully_connected':
        edge_index = builder.build_fully_connected(num_sensors)
        
    else:
        raise ValueError(f"Unknown graph construction method: {method}")
    
    # Use latest timestep as node features (or could use statistical features)
    node_features = sensor_data[:, -1, :].t()  # (num_sensors, batch_size)
    
    # For now, use mean across batch as node features
    if batch_size > 1:
        node_features = torch.mean(node_features, dim=1, keepdim=True)  # (num_sensors, 1)
    
    # Ensure proper shape for GNN input
    if node_features.dim() == 1:
        node_features = node_features.unsqueeze(-1)
    
    return {
        'x': node_features.squeeze(-1) if node_features.size(-1) == 1 else node_features,
        'edge_index': edge_index
    }
"""
Self-Supervised Temporal-Spatial Registration Learning for Few-Shot Anomaly Detection.

This module implements a novel self-supervised learning approach that uses temporal-spatial
registration as a proxy task to learn robust representations for anomaly detection without
requiring labeled anomaly data.

Key Features:
- Temporal registration for time-series alignment and feature learning
- Spatial registration for sensor network topology understanding
- Few-shot anomaly detection with minimal labeled data
- Self-supervised pretraining on normal operation data
- Contrastive learning for robust feature representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import math
import random
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class TemporalRegistrationModule(nn.Module):
    """
    Temporal registration module for learning temporal alignments in time series data.
    
    Uses registration as a self-supervised proxy task to learn meaningful temporal
    representations that are robust to temporal shifts and variations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_shift: int = 10,
        registration_loss_weight: float = 1.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_shift = max_shift
        self.registration_loss_weight = registration_loss_weight
        
        # Temporal feature encoder
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Registration predictor network
        self.registration_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_shift * 2 + 1)  # Predict shift amount
        )
        
        # Temporal alignment network
        self.alignment_net = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Feature projection for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        return_alignment: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through temporal registration module.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            return_alignment: Whether to return alignment information
            
        Returns:
            Dictionary containing temporal features and registration info
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Temporal feature encoding: (batch_size, seq_len, input_dim) -> (batch_size, hidden_dim, seq_len)
        x_conv = x.transpose(1, 2)
        temporal_features = self.temporal_encoder(x_conv)  # (batch_size, hidden_dim, seq_len)
        
        # LSTM processing: (batch_size, seq_len, hidden_dim)
        temporal_features_lstm = temporal_features.transpose(1, 2)
        lstm_output, (h_n, c_n) = self.alignment_net(temporal_features_lstm)
        
        # Project features for contrastive learning
        projected_features = self.projection_head(lstm_output)
        
        outputs = {
            'temporal_features': lstm_output,
            'projected_features': projected_features,
            'raw_features': temporal_features
        }
        
        if return_alignment:
            # Predict registration parameters
            reg_params = self.registration_net(temporal_features)
            outputs['registration_params'] = reg_params
        
        return outputs
    
    def compute_registration_loss(
        self,
        features_anchor: torch.Tensor,
        features_shifted: torch.Tensor,
        true_shift: int
    ) -> torch.Tensor:
        """
        Compute registration loss for self-supervised learning.
        
        Args:
            features_anchor: Features from anchor sequence
            features_shifted: Features from temporally shifted sequence
            true_shift: Ground truth temporal shift amount
            
        Returns:
            Registration loss
        """
        # Predict shift from features
        predicted_shift = self.registration_net(features_shifted.transpose(1, 2))
        
        # Convert true shift to one-hot target
        shift_target = torch.zeros_like(predicted_shift)
        shift_idx = true_shift + self.max_shift  # Center around max_shift
        shift_target[:, shift_idx] = 1.0
        
        # Cross-entropy loss for shift prediction
        registration_loss = F.cross_entropy(predicted_shift, shift_target.argmax(dim=1))
        
        return registration_loss
    
    def apply_temporal_shift(
        self, 
        x: torch.Tensor, 
        shift_range: Tuple[int, int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Apply random temporal shift for data augmentation.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            shift_range: Range of possible shifts
            
        Returns:
            Tuple of (original, shifted, shift_amount)
        """
        if shift_range is None:
            shift_range = (-self.max_shift, self.max_shift)
        
        # Random shift amount
        shift_amount = random.randint(shift_range[0], shift_range[1])
        
        if shift_amount == 0:
            return x, x, 0
        
        batch_size, seq_len, input_dim = x.shape
        
        if shift_amount > 0:
            # Positive shift - remove from beginning, pad at end
            x_shifted = torch.cat([
                x[:, shift_amount:, :],
                torch.zeros(batch_size, shift_amount, input_dim, device=x.device)
            ], dim=1)
        else:
            # Negative shift - pad at beginning, remove from end
            x_shifted = torch.cat([
                torch.zeros(batch_size, -shift_amount, input_dim, device=x.device),
                x[:, :shift_amount, :]
            ], dim=1)
        
        return x, x_shifted, shift_amount


class SpatialRegistrationModule(nn.Module):
    """
    Spatial registration module for learning sensor network topology alignments.
    
    Learns to align and register different views of the sensor network topology
    for robust spatial relationship understanding.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        registration_temperature: float = 0.1
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.registration_temperature = registration_temperature
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph attention layers for spatial encoding
        from torch_geometric.nn import GATConv
        self.gnn_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim if i > 0 else hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=False
            )
            for i in range(num_gnn_layers)
        ])
        
        # Spatial transformation predictor
        self.spatial_transform_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 2D affine transformation parameters
        )
        
        # Graph embedding aggregator
        self.graph_pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through spatial registration module.
        
        Args:
            node_features: Node features (num_nodes, node_feature_dim)
            edge_index: Graph connectivity (2, num_edges)
            batch: Batch assignment for multiple graphs
            
        Returns:
            Dictionary containing spatial features and embeddings
        """
        # Node feature encoding
        x = self.node_encoder(node_features)
        
        # Graph neural network processing
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, edge_index))
        
        # Graph-level embedding
        if batch is not None:
            from torch_geometric.utils import scatter
            graph_embedding = scatter(x, batch, dim=0, reduce='mean')
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        # Spatial transformation parameters
        transform_params = self.spatial_transform_net(graph_embedding)
        
        # Projected features for contrastive learning
        projected_features = self.projection_head(graph_embedding)
        
        return {
            'node_features': x,
            'graph_embedding': graph_embedding,
            'transform_params': transform_params,
            'projected_features': projected_features
        }
    
    def apply_spatial_augmentation(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        augmentation_type: str = 'random'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial augmentation to create different views of the graph.
        
        Args:
            node_features: Original node features
            edge_index: Original graph connectivity
            augmentation_type: Type of augmentation ('dropout', 'permute', 'noise')
            
        Returns:
            Tuple of (augmented_features, augmented_edge_index)
        """
        if augmentation_type == 'dropout':
            # Random node feature dropout
            dropout_mask = torch.rand_like(node_features) > 0.2
            aug_features = node_features * dropout_mask.float()
            aug_edge_index = edge_index
            
        elif augmentation_type == 'permute':
            # Random node permutation
            num_nodes = node_features.size(0)
            perm = torch.randperm(num_nodes)
            aug_features = node_features[perm]
            
            # Update edge indices
            perm_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(perm)}
            aug_edge_index = edge_index.clone()
            for i in range(edge_index.size(1)):
                aug_edge_index[0, i] = perm_map[edge_index[0, i].item()]
                aug_edge_index[1, i] = perm_map[edge_index[1, i].item()]
                
        elif augmentation_type == 'noise':
            # Add Gaussian noise to features
            noise = torch.randn_like(node_features) * 0.1
            aug_features = node_features + noise
            aug_edge_index = edge_index
            
        else:  # random
            # Randomly choose augmentation type
            aug_type = random.choice(['dropout', 'permute', 'noise'])
            return self.apply_spatial_augmentation(node_features, edge_index, aug_type)
        
        return aug_features, aug_edge_index
    
    def compute_spatial_registration_loss(
        self,
        features_view1: torch.Tensor,
        features_view2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial registration loss using contrastive learning.
        
        Args:
            features_view1: Features from first view of the graph
            features_view2: Features from second view of the graph
            
        Returns:
            Spatial registration loss
        """
        # Normalize features
        features_view1 = F.normalize(features_view1, dim=-1)
        features_view2 = F.normalize(features_view2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features_view1, features_view2.t())
        similarity_matrix = similarity_matrix / self.registration_temperature
        
        # Labels for contrastive learning (diagonal should be positive pairs)
        batch_size = features_view1.size(0)
        labels = torch.arange(batch_size, device=features_view1.device)
        
        # Cross-entropy loss for both directions
        loss_v1_to_v2 = F.cross_entropy(similarity_matrix, labels)
        loss_v2_to_v1 = F.cross_entropy(similarity_matrix.t(), labels)
        
        return (loss_v1_to_v2 + loss_v2_to_v1) / 2


class ContrastiveLearningModule(nn.Module):
    """
    Contrastive learning module for self-supervised representation learning.
    
    Implements InfoNCE loss and other contrastive learning techniques for
    learning robust representations from temporal-spatial registrations.
    """
    
    def __init__(
        self,
        feature_dim: int,
        temperature: float = 0.1,
        negative_sampling_ratio: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.negative_sampling_ratio = negative_sampling_ratio
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features for contrastive learning."""
        return self.projection_head(features)
    
    def compute_infonce_loss(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            anchor_features: Anchor sample features
            positive_features: Positive sample features
            negative_features: Negative sample features (optional)
            
        Returns:
            InfoNCE loss
        """
        batch_size = anchor_features.size(0)
        
        # Project features
        anchor_proj = self.projection_head(anchor_features)
        positive_proj = self.projection_head(positive_features)
        
        # Normalize features
        anchor_proj = F.normalize(anchor_proj, dim=-1)
        positive_proj = F.normalize(positive_proj, dim=-1)
        
        # Positive similarity
        pos_sim = torch.sum(anchor_proj * positive_proj, dim=-1) / self.temperature
        
        if negative_features is not None:
            # Use provided negative samples
            negative_proj = self.projection_head(negative_features)
            negative_proj = F.normalize(negative_proj, dim=-1)
            
            neg_sim = torch.matmul(anchor_proj, negative_proj.t()) / self.temperature
        else:
            # Use in-batch negatives
            all_features = torch.cat([anchor_proj, positive_proj], dim=0)
            neg_sim = torch.matmul(anchor_proj, all_features.t()) / self.temperature
            
            # Remove self-similarity
            mask = torch.eye(batch_size, device=anchor_features.device)
            neg_sim = neg_sim * (1 - mask) - 1e9 * mask
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_features.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SelfSupervisedRegistrationLearner(nn.Module):
    """
    Complete Self-Supervised Registration Learning system for IoT anomaly detection.
    
    Combines temporal and spatial registration modules with contrastive learning
    for few-shot anomaly detection without labeled data requirements.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        temporal_layers: int = 3,
        spatial_layers: int = 3,
        dropout: float = 0.1,
        registration_weight: float = 1.0,
        contrastive_temperature: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.registration_weight = registration_weight
        
        # Temporal registration module
        self.temporal_registration = TemporalRegistrationModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=temporal_layers,
            dropout=dropout
        )
        
        # Spatial registration module
        self.spatial_registration = SpatialRegistrationModule(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=spatial_layers,
            dropout=dropout,
            registration_temperature=contrastive_temperature
        )
        
        # Contrastive learning module
        self.contrastive_learner = ContrastiveLearningModule(
            feature_dim=hidden_dim,
            temperature=contrastive_temperature
        )
        
        # Feature fusion and anomaly detection head
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Prototype memory for few-shot learning
        self.register_buffer('normal_prototypes', torch.randn(10, hidden_dim))
        self.register_buffer('anomaly_prototypes', torch.randn(5, hidden_dim))
        self.prototype_momentum = 0.9
        
    def forward(
        self,
        temporal_data: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_registration_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through self-supervised registration learner.
        
        Args:
            temporal_data: Temporal sensor data (batch_size, seq_len, input_dim)
            node_features: Node features for spatial modeling
            edge_index: Graph connectivity
            batch: Batch assignment for multiple graphs
            return_registration_info: Whether to return registration details
            
        Returns:
            Dictionary containing anomaly scores and learned features
        """
        # Temporal registration
        temporal_output = self.temporal_registration(temporal_data)
        temporal_features = temporal_output['temporal_features'].mean(dim=1)  # Average over sequence
        
        # Spatial registration
        spatial_output = self.spatial_registration(node_features, edge_index, batch)
        spatial_features = spatial_output['graph_embedding']
        
        # Handle dimension mismatch
        if temporal_features.size(0) != spatial_features.size(0):
            min_batch = min(temporal_features.size(0), spatial_features.size(0))
            temporal_features = temporal_features[:min_batch]
            spatial_features = spatial_features[:min_batch]
        
        # Feature fusion
        fused_features = torch.cat([temporal_features, spatial_features], dim=-1)
        fused_features = self.feature_fusion(fused_features)
        
        # Anomaly detection
        anomaly_scores = self.anomaly_detector(fused_features)
        
        outputs = {
            'anomaly_scores': anomaly_scores,
            'fused_features': fused_features,
            'temporal_features': temporal_features,
            'spatial_features': spatial_features
        }
        
        if return_registration_info:
            outputs['temporal_registration'] = temporal_output
            outputs['spatial_registration'] = spatial_output
        
        return outputs
    
    def pretrain_with_registration(
        self,
        temporal_data: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        Self-supervised pretraining using temporal-spatial registration.
        
        Args:
            temporal_data: Unlabeled temporal data for pretraining
            node_features: Node features
            edge_index: Graph connectivity
            num_epochs: Number of pretraining epochs
            
        Returns:
            Dictionary containing training history
        """
        self.train()
        history = {'temporal_loss': [], 'spatial_loss': [], 'contrastive_loss': []}
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Temporal registration task
            temporal_anchor, temporal_shifted, shift_amount = self.temporal_registration.apply_temporal_shift(temporal_data)
            
            temporal_output_anchor = self.temporal_registration(temporal_anchor)
            temporal_output_shifted = self.temporal_registration(temporal_shifted, return_alignment=True)
            
            temporal_loss = self.temporal_registration.compute_registration_loss(
                temporal_output_anchor['raw_features'],
                temporal_output_shifted['raw_features'],
                shift_amount
            )
            
            # Spatial registration task
            spatial_aug1, edge_aug1 = self.spatial_registration.apply_spatial_augmentation(
                node_features, edge_index, 'dropout'
            )
            spatial_aug2, edge_aug2 = self.spatial_registration.apply_spatial_augmentation(
                node_features, edge_index, 'noise'
            )
            
            spatial_output1 = self.spatial_registration(spatial_aug1, edge_aug1)
            spatial_output2 = self.spatial_registration(spatial_aug2, edge_aug2)
            
            spatial_loss = self.spatial_registration.compute_spatial_registration_loss(
                spatial_output1['projected_features'],
                spatial_output2['projected_features']
            )
            
            # Contrastive learning between temporal and spatial features
            temporal_proj = self.contrastive_learner(temporal_output_anchor['temporal_features'].mean(dim=1))
            spatial_proj = self.contrastive_learner(spatial_output1['graph_embedding'])
            
            # Handle batch size mismatch
            min_batch = min(temporal_proj.size(0), spatial_proj.size(0))
            temporal_proj = temporal_proj[:min_batch]
            spatial_proj = spatial_proj[:min_batch]
            
            contrastive_loss = self.contrastive_learner.compute_infonce_loss(
                temporal_proj, spatial_proj
            )
            
            # Total loss
            total_loss = temporal_loss + spatial_loss + contrastive_loss
            total_loss.backward()
            optimizer.step()
            
            # Record history
            history['temporal_loss'].append(temporal_loss.item())
            history['spatial_loss'].append(spatial_loss.item())
            history['contrastive_loss'].append(contrastive_loss.item())
            
            if (epoch + 1) % 20 == 0:
                logger.info(f'Epoch {epoch+1}/{num_epochs} - '
                          f'Temporal: {temporal_loss.item():.4f}, '
                          f'Spatial: {spatial_loss.item():.4f}, '
                          f'Contrastive: {contrastive_loss.item():.4f}')
        
        return history
    
    def few_shot_adaptation(
        self,
        support_data: Dict[str, torch.Tensor],
        support_labels: torch.Tensor,
        num_adaptation_steps: int = 10
    ):
        """
        Few-shot adaptation using limited labeled examples.
        
        Args:
            support_data: Dictionary with temporal_data, node_features, edge_index
            support_labels: Labels for support examples (0: normal, 1: anomaly)
            num_adaptation_steps: Number of adaptation steps
        """
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        for step in range(num_adaptation_steps):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward(
                support_data['temporal_data'],
                support_data['node_features'],
                support_data['edge_index']
            )
            
            # Supervised loss
            supervised_loss = F.binary_cross_entropy(
                outputs['anomaly_scores'].squeeze(),
                support_labels.float()
            )
            
            # Update prototypes
            self._update_prototypes(outputs['fused_features'], support_labels)
            
            supervised_loss.backward()
            optimizer.step()
    
    def _update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """Update prototype memory with exponential moving average."""
        with torch.no_grad():
            normal_mask = (labels == 0)
            anomaly_mask = (labels == 1)
            
            if normal_mask.any():
                normal_features = features[normal_mask].mean(dim=0)
                self.normal_prototypes[0] = (
                    self.prototype_momentum * self.normal_prototypes[0] +
                    (1 - self.prototype_momentum) * normal_features
                )
            
            if anomaly_mask.any():
                anomaly_features = features[anomaly_mask].mean(dim=0)
                self.anomaly_prototypes[0] = (
                    self.prototype_momentum * self.anomaly_prototypes[0] +
                    (1 - self.prototype_momentum) * anomaly_features
                )
    
    def predict_with_prototypes(
        self,
        temporal_data: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Make predictions using prototype-based few-shot learning.
        
        Returns:
            Anomaly scores based on distance to prototypes
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(temporal_data, node_features, edge_index)
            features = outputs['fused_features']
            
            # Compute distances to prototypes
            normal_distances = torch.cdist(features, self.normal_prototypes)
            anomaly_distances = torch.cdist(features, self.anomaly_prototypes)
            
            # Min distances to each prototype set
            min_normal_dist = normal_distances.min(dim=1)[0]
            min_anomaly_dist = anomaly_distances.min(dim=1)[0]
            
            # Anomaly score based on relative distances
            anomaly_scores = min_normal_dist / (min_normal_dist + min_anomaly_dist + 1e-8)
            
        return anomaly_scores


# Factory function for creating self-supervised registration learners
def create_self_supervised_learner(config: Dict[str, Any]) -> SelfSupervisedRegistrationLearner:
    """
    Factory function to create self-supervised registration learners.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SelfSupervisedRegistrationLearner instance
    """
    return SelfSupervisedRegistrationLearner(
        input_dim=config.get('input_dim', 5),
        node_feature_dim=config.get('node_feature_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        temporal_layers=config.get('temporal_layers', 3),
        spatial_layers=config.get('spatial_layers', 3),
        dropout=config.get('dropout', 0.1),
        registration_weight=config.get('registration_weight', 1.0),
        contrastive_temperature=config.get('contrastive_temperature', 0.1)
    )
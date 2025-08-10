"""
Physics-Informed LSTM-GNN Hybrid Model for Industrial IoT Anomaly Detection.

This module implements a novel physics-informed neural network that combines
LSTM temporal modeling, GNN spatial relationships, and physical constraints
from industrial process knowledge for superior anomaly detection.

Key Features:
- Integration of physical laws (conservation, thermodynamics, fluid dynamics)
- Physics-constrained loss functions for better generalization
- Domain knowledge injection through physics equations
- Enhanced interpretability through physics-based reasoning
- Robust performance with limited labeled data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
import logging
import math

from .lstm_gnn_hybrid import LSTMGNNHybridModel
from .transformer_vae import TransformerVAE
from .sparse_graph_attention import SparseGraphAttentionNetwork

logger = logging.getLogger(__name__)


class PhysicsConstraints:
    """
    Collection of physics-based constraint functions for industrial IoT systems.
    
    Implements common physical laws and constraints that govern sensor behavior
    in industrial environments (SWaT dataset and similar industrial control systems).
    """
    
    @staticmethod
    def mass_conservation_constraint(
        flow_in: torch.Tensor, 
        flow_out: torch.Tensor, 
        tank_level_change: torch.Tensor,
        dt: float = 1.0,
        tolerance: float = 0.1
    ) -> torch.Tensor:
        """
        Mass conservation constraint for tank systems.
        
        Physical law: mass_in - mass_out = d(mass_stored)/dt
        
        Args:
            flow_in: Inflow rate (batch_size, seq_len)
            flow_out: Outflow rate (batch_size, seq_len)
            tank_level_change: Change in tank level (batch_size, seq_len)
            dt: Time step
            tolerance: Acceptable deviation from constraint
            
        Returns:
            Constraint violation penalty
        """
        # Compute mass balance
        net_flow = flow_in - flow_out
        expected_level_change = net_flow * dt
        
        # Constraint violation
        violation = torch.abs(tank_level_change - expected_level_change)
        
        # Penalize violations beyond tolerance
        penalty = F.relu(violation - tolerance)
        
        return penalty.mean()
    
    @staticmethod
    def energy_conservation_constraint(
        temperature: torch.Tensor,
        heat_input: torch.Tensor,
        heat_output: torch.Tensor,
        mass_flow: torch.Tensor,
        specific_heat: float = 4186.0,  # Water specific heat J/(kg·K)
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Energy conservation constraint for thermal systems.
        
        Physical law: dE/dt = heat_in - heat_out - heat_convection
        
        Args:
            temperature: System temperature (batch_size, seq_len)
            heat_input: Heat input rate (batch_size, seq_len)
            heat_output: Heat output rate (batch_size, seq_len)
            mass_flow: Mass flow rate (batch_size, seq_len)
            specific_heat: Specific heat capacity
            dt: Time step
            
        Returns:
            Energy constraint violation penalty
        """
        # Compute temperature change
        temp_change = temperature[:, 1:] - temperature[:, :-1]
        
        # Net heat input
        net_heat = (heat_input - heat_output)[:, :-1]
        
        # Expected temperature change from energy balance
        # dT = Q_net / (m * c_p)
        expected_temp_change = net_heat / (mass_flow[:, :-1] * specific_heat + 1e-8)
        
        # Constraint violation
        violation = torch.abs(temp_change - expected_temp_change)
        
        return violation.mean()
    
    @staticmethod
    def pressure_drop_constraint(
        pressure_upstream: torch.Tensor,
        pressure_downstream: torch.Tensor,
        flow_rate: torch.Tensor,
        pipe_resistance: float = 1.0
    ) -> torch.Tensor:
        """
        Pressure drop constraint for pipe flow systems.
        
        Physical law: ΔP = k * Q² (turbulent flow)
        
        Args:
            pressure_upstream: Upstream pressure (batch_size, seq_len)
            pressure_downstream: Downstream pressure (batch_size, seq_len)
            flow_rate: Flow rate through pipe (batch_size, seq_len)
            pipe_resistance: Pipe resistance coefficient
            
        Returns:
            Pressure drop constraint violation penalty
        """
        # Actual pressure drop
        actual_dp = pressure_upstream - pressure_downstream
        
        # Expected pressure drop from flow rate
        expected_dp = pipe_resistance * flow_rate ** 2
        
        # Constraint violation
        violation = torch.abs(actual_dp - expected_dp)
        
        return violation.mean()
    
    @staticmethod
    def pump_characteristic_constraint(
        flow_rate: torch.Tensor,
        pump_head: torch.Tensor,
        pump_speed: torch.Tensor,
        pump_coefficients: Tuple[float, float, float] = (100.0, -0.01, -0.0001)
    ) -> torch.Tensor:
        """
        Pump characteristic curve constraint.
        
        Physical law: H = a + b*Q + c*Q² (pump head-flow relationship)
        
        Args:
            flow_rate: Pump flow rate (batch_size, seq_len)
            pump_head: Pump head (batch_size, seq_len)
            pump_speed: Pump speed (batch_size, seq_len)
            pump_coefficients: Coefficients (a, b, c) for pump curve
            
        Returns:
            Pump characteristic constraint violation penalty
        """
        a, b, c = pump_coefficients
        
        # Normalize by pump speed (affinity laws)
        normalized_flow = flow_rate / (pump_speed + 1e-8)
        normalized_head = pump_head / ((pump_speed + 1e-8) ** 2)
        
        # Expected head from pump characteristic
        expected_head = a + b * normalized_flow + c * (normalized_flow ** 2)
        
        # Constraint violation
        violation = torch.abs(normalized_head - expected_head)
        
        return violation.mean()
    
    @staticmethod
    def valve_flow_constraint(
        flow_rate: torch.Tensor,
        pressure_drop: torch.Tensor,
        valve_opening: torch.Tensor,
        valve_cv: float = 10.0
    ) -> torch.Tensor:
        """
        Valve flow characteristic constraint.
        
        Physical law: Q = Cv * sqrt(ΔP) * valve_opening
        
        Args:
            flow_rate: Flow through valve (batch_size, seq_len)
            pressure_drop: Pressure drop across valve (batch_size, seq_len)
            valve_opening: Valve opening percentage (batch_size, seq_len)
            valve_cv: Valve flow coefficient
            
        Returns:
            Valve flow constraint violation penalty
        """
        # Expected flow from valve equation
        expected_flow = valve_cv * torch.sqrt(torch.abs(pressure_drop) + 1e-8) * valve_opening
        
        # Constraint violation
        violation = torch.abs(flow_rate - expected_flow)
        
        return violation.mean()


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function that combines data-driven loss with physics constraints.
    
    Implements adaptive weighting between data loss and physics constraints based on
    training progress and constraint satisfaction levels.
    """
    
    def __init__(
        self,
        constraint_weights: Dict[str, float] = None,
        adaptive_weighting: bool = True,
        constraint_tolerance: float = 0.01,
        physics_weight_schedule: str = 'cosine'  # 'constant', 'linear', 'cosine'
    ):
        super().__init__()
        
        self.constraint_weights = constraint_weights or {
            'mass_conservation': 1.0,
            'energy_conservation': 1.0,
            'pressure_drop': 0.5,
            'pump_characteristic': 0.5,
            'valve_flow': 0.5
        }
        
        self.adaptive_weighting = adaptive_weighting
        self.constraint_tolerance = constraint_tolerance
        self.physics_weight_schedule = physics_weight_schedule
        
        # Track constraint satisfaction history
        self.register_buffer('constraint_history', torch.zeros(100, len(self.constraint_weights)))
        self.history_ptr = 0
        
        # Physics constraints instance
        self.physics = PhysicsConstraints()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sensor_data: Dict[str, torch.Tensor],
        epoch: int = 0,
        max_epochs: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss combining data loss and constraint violations.
        
        Args:
            predictions: Model predictions (batch_size, seq_len, num_features)
            targets: Target values (batch_size, seq_len, num_features)
            sensor_data: Dictionary containing sensor readings by type
            epoch: Current training epoch
            max_epochs: Total number of training epochs
            
        Returns:
            Dictionary containing loss components
        """
        # Base data loss (MSE)
        data_loss = F.mse_loss(predictions, targets)
        
        # Physics constraint violations
        constraint_losses = self._compute_constraint_losses(sensor_data, predictions)
        
        # Adaptive physics weights
        if self.adaptive_weighting:
            physics_weights = self._compute_adaptive_weights(constraint_losses, epoch, max_epochs)
        else:
            physics_weights = self.constraint_weights
        
        # Weighted constraint loss
        total_constraint_loss = 0.0
        constraint_components = {}
        
        for constraint_name, loss_value in constraint_losses.items():
            weight = physics_weights.get(constraint_name, 0.0)
            weighted_loss = weight * loss_value
            total_constraint_loss += weighted_loss
            constraint_components[f'constraint_{constraint_name}'] = weighted_loss
        
        # Total physics-informed loss
        total_loss = data_loss + total_constraint_loss
        
        # Update constraint history
        self._update_constraint_history(constraint_losses)
        
        # Return comprehensive loss breakdown
        loss_dict = {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'constraint_loss': total_constraint_loss,
            **constraint_components,
            'physics_weights': physics_weights
        }
        
        return loss_dict
    
    def _compute_constraint_losses(
        self, 
        sensor_data: Dict[str, torch.Tensor],
        predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute individual physics constraint violations."""
        constraint_losses = {}
        
        try:
            # Mass conservation constraint (tanks/vessels)
            if all(key in sensor_data for key in ['flow_in', 'flow_out', 'level']):
                flow_in = sensor_data['flow_in']
                flow_out = sensor_data['flow_out']
                
                # Compute level change from consecutive measurements
                level_change = sensor_data['level'][:, 1:] - sensor_data['level'][:, :-1]
                
                constraint_losses['mass_conservation'] = self.physics.mass_conservation_constraint(
                    flow_in[:, :-1], flow_out[:, :-1], level_change
                )
            
            # Energy conservation constraint (heat exchangers, reactors)
            if all(key in sensor_data for key in ['temperature', 'heat_input', 'flow_rate']):
                constraint_losses['energy_conservation'] = self.physics.energy_conservation_constraint(
                    sensor_data['temperature'],
                    sensor_data.get('heat_input', torch.zeros_like(sensor_data['temperature'])),
                    sensor_data.get('heat_output', torch.zeros_like(sensor_data['temperature'])),
                    sensor_data['flow_rate']
                )
            
            # Pressure drop constraint (pipes, valves)
            if all(key in sensor_data for key in ['pressure_up', 'pressure_down', 'flow_rate']):
                constraint_losses['pressure_drop'] = self.physics.pressure_drop_constraint(
                    sensor_data['pressure_up'],
                    sensor_data['pressure_down'],
                    sensor_data['flow_rate']
                )
            
            # Pump characteristic constraint
            if all(key in sensor_data for key in ['pump_flow', 'pump_head', 'pump_speed']):
                constraint_losses['pump_characteristic'] = self.physics.pump_characteristic_constraint(
                    sensor_data['pump_flow'],
                    sensor_data['pump_head'],
                    sensor_data['pump_speed']
                )
            
            # Valve flow constraint
            if all(key in sensor_data for key in ['valve_flow', 'valve_dp', 'valve_opening']):
                constraint_losses['valve_flow'] = self.physics.valve_flow_constraint(
                    sensor_data['valve_flow'],
                    sensor_data['valve_dp'],
                    sensor_data['valve_opening']
                )
                
        except Exception as e:
            logger.warning(f"Error computing physics constraints: {e}")
            # Return zero losses if constraint computation fails
            for name in self.constraint_weights.keys():
                constraint_losses[name] = torch.tensor(0.0, device=predictions.device)
        
        return constraint_losses
    
    def _compute_adaptive_weights(
        self, 
        constraint_losses: Dict[str, torch.Tensor],
        epoch: int,
        max_epochs: int
    ) -> Dict[str, float]:
        """Compute adaptive weights for physics constraints."""
        adaptive_weights = {}
        
        # Base schedule factor
        if self.physics_weight_schedule == 'linear':
            schedule_factor = min(1.0, epoch / (max_epochs * 0.5))
        elif self.physics_weight_schedule == 'cosine':
            schedule_factor = 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
        else:  # constant
            schedule_factor = 1.0
        
        for constraint_name, base_weight in self.constraint_weights.items():
            if constraint_name in constraint_losses:
                # Get recent constraint satisfaction
                constraint_value = constraint_losses[constraint_name].item()
                
                # Adapt based on constraint satisfaction
                if constraint_value > self.constraint_tolerance:
                    # Increase weight for poorly satisfied constraints
                    adaptation_factor = min(2.0, 1.0 + constraint_value / self.constraint_tolerance)
                else:
                    # Decrease weight for well-satisfied constraints
                    adaptation_factor = max(0.1, constraint_value / self.constraint_tolerance)
                
                # Combine schedule and adaptation factors
                adaptive_weights[constraint_name] = base_weight * schedule_factor * adaptation_factor
            else:
                adaptive_weights[constraint_name] = 0.0
        
        return adaptive_weights
    
    def _update_constraint_history(self, constraint_losses: Dict[str, torch.Tensor]):
        """Update constraint satisfaction history."""
        constraint_values = []
        for name in self.constraint_weights.keys():
            value = constraint_losses.get(name, torch.tensor(0.0))
            constraint_values.append(value.item() if hasattr(value, 'item') else 0.0)
        
        # Update history buffer
        self.constraint_history[self.history_ptr] = torch.tensor(constraint_values)
        self.history_ptr = (self.history_ptr + 1) % 100
    
    def get_constraint_stats(self) -> Dict[str, float]:
        """Get statistics on constraint satisfaction."""
        stats = {}
        
        for i, constraint_name in enumerate(self.constraint_weights.keys()):
            constraint_history = self.constraint_history[:, i]
            
            stats[f'{constraint_name}_mean'] = constraint_history.mean().item()
            stats[f'{constraint_name}_std'] = constraint_history.std().item()
            stats[f'{constraint_name}_min'] = constraint_history.min().item()
            stats[f'{constraint_name}_max'] = constraint_history.max().item()
        
        return stats


class PhysicsInformedHybridModel(nn.Module):
    """
    Physics-Informed LSTM-GNN Hybrid Model for IoT Anomaly Detection.
    
    Combines temporal modeling (LSTM/Transformer), spatial relationships (GNN),
    and physics constraints for superior anomaly detection in industrial systems.
    
    Key Features:
    - Multi-modal architecture combining data-driven and physics-based approaches
    - Adaptive physics constraint weighting during training
    - Enhanced interpretability through physics-based reasoning
    - Robust performance with limited labeled data
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        gnn_layers: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.1,
        use_transformer: bool = True,
        use_sparse_attention: bool = True,
        physics_weight: float = 1.0,
        constraint_config: Dict[str, Any] = None
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.physics_weight = physics_weight
        self.use_transformer = use_transformer
        self.use_sparse_attention = use_sparse_attention
        
        # Temporal modeling component
        if use_transformer:
            self.temporal_model = TransformerVAE(
                input_size=input_size,
                d_model=hidden_size,
                num_layers=lstm_layers,
                latent_dim=latent_dim,
                dropout=dropout
            )
        else:
            self.temporal_model = LSTMGNNHybridModel(
                input_size=input_size,
                lstm_hidden_size=hidden_size,
                lstm_num_layers=lstm_layers,
                gnn_hidden_size=hidden_size,
                gnn_num_layers=gnn_layers,
                output_size=latent_dim,
                dropout=dropout
            )
        
        # Spatial relationship modeling
        if use_sparse_attention:
            self.spatial_model = SparseGraphAttentionNetwork(
                input_dim=latent_dim,
                hidden_dim=hidden_size,
                output_dim=latent_dim,
                num_layers=gnn_layers,
                dropout=dropout
            )
        else:
            # Fallback to standard GNN
            from torch_geometric.nn import GCNConv
            self.spatial_model = nn.ModuleList([
                GCNConv(latent_dim if i == 0 else hidden_size, 
                       hidden_size if i < gnn_layers - 1 else latent_dim)
                for i in range(gnn_layers)
            ])
        
        # Physics-informed components
        self.physics_loss = PhysicsInformedLoss(
            constraint_weights=constraint_config.get('weights', {}) if constraint_config else {},
            adaptive_weighting=constraint_config.get('adaptive_weighting', True) if constraint_config else True,
            constraint_tolerance=constraint_config.get('tolerance', 0.01) if constraint_config else 0.01
        )
        
        # Feature fusion and output layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, latent_dim)
        )
        
        # Anomaly scoring head
        self.anomaly_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Physics interpretability module
        self.physics_interpreter = PhysicsInterpretabilityModule(
            feature_dim=latent_dim,
            num_constraints=len(self.physics_loss.constraint_weights)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sensor_metadata: Dict[str, torch.Tensor] = None,
        return_physics_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through physics-informed hybrid model.
        
        Args:
            x: Input sensor data (batch_size, seq_len, input_size)
            edge_index: Graph connectivity (2, num_edges)
            sensor_metadata: Additional sensor information for physics constraints
            return_physics_info: Whether to return physics interpretability information
            
        Returns:
            Dictionary containing model outputs and physics information
        """
        batch_size, seq_len, _ = x.shape
        
        # Temporal feature extraction
        if self.use_transformer:
            temporal_output = self.temporal_model(x)
            temporal_features = temporal_output['z']  # Latent representation
        else:
            temporal_features = self.temporal_model.encode_sequence(x)
        
        # Spatial relationship modeling
        if self.use_sparse_attention:
            spatial_output = self.spatial_model(
                temporal_features, 
                edge_index,
                batch=None
            )
            spatial_features = spatial_output['graph_embedding']
        else:
            # Standard GNN processing
            spatial_features = temporal_features
            for layer in self.spatial_model:
                spatial_features = F.relu(layer(spatial_features, edge_index))
        
        # Feature fusion
        if temporal_features.dim() == 3:
            temporal_features = temporal_features.mean(dim=1)  # Average over sequence
        if spatial_features.dim() == 3:
            spatial_features = spatial_features.mean(dim=1)
        
        # Ensure compatible dimensions
        if temporal_features.size(0) != spatial_features.size(0):
            # Handle batch size mismatch
            min_batch = min(temporal_features.size(0), spatial_features.size(0))
            temporal_features = temporal_features[:min_batch]
            spatial_features = spatial_features[:min_batch]
        
        # Concatenate temporal and spatial features
        combined_features = torch.cat([temporal_features, spatial_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # Anomaly scoring
        anomaly_scores = self.anomaly_head(fused_features)
        
        # Physics interpretability
        physics_info = {}
        if return_physics_info:
            physics_info = self.physics_interpreter(fused_features, sensor_metadata)
        
        return {
            'anomaly_scores': anomaly_scores,
            'temporal_features': temporal_features,
            'spatial_features': spatial_features,
            'fused_features': fused_features,
            'physics_info': physics_info
        }
    
    def compute_physics_informed_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sensor_data: Dict[str, torch.Tensor],
        epoch: int = 0,
        max_epochs: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """Compute physics-informed loss."""
        return self.physics_loss(predictions, targets, sensor_data, epoch, max_epochs)
    
    def get_physics_insights(
        self, 
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sensor_metadata: Dict[str, torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Get physics-based insights for anomaly explanation.
        
        Returns interpretable physics-based explanations for detected anomalies.
        """
        with torch.no_grad():
            outputs = self.forward(x, edge_index, sensor_metadata, return_physics_info=True)
            
            physics_insights = {
                'constraint_violations': self.physics_loss.get_constraint_stats(),
                'anomaly_scores': outputs['anomaly_scores'].cpu().numpy(),
                'physics_interpretability': outputs['physics_info'],
                'temporal_attention': getattr(self.temporal_model, 'attention_weights', None),
                'spatial_attention': outputs.get('spatial_features', {}).get('attention_weights', None)
            }
        
        return physics_insights
    
    def validate_physics_constraints(
        self, 
        sensor_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Validate current sensor data against physics constraints."""
        constraint_violations = {}
        
        try:
            # Check each constraint individually
            physics = PhysicsConstraints()
            
            if all(key in sensor_data for key in ['flow_in', 'flow_out', 'level']):
                violation = physics.mass_conservation_constraint(
                    sensor_data['flow_in'],
                    sensor_data['flow_out'],
                    sensor_data['level'][:, 1:] - sensor_data['level'][:, :-1]
                )
                constraint_violations['mass_conservation'] = violation.item()
            
            # Add other constraint validations...
            
        except Exception as e:
            logger.warning(f"Physics constraint validation error: {e}")
        
        return constraint_violations


class PhysicsInterpretabilityModule(nn.Module):
    """
    Module for physics-based interpretability and explanation generation.
    
    Provides human-readable explanations for anomalies based on physics
    constraint violations and domain knowledge.
    """
    
    def __init__(self, feature_dim: int, num_constraints: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_constraints = num_constraints
        
        # Physics explanation network
        self.explanation_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_constraints),
            nn.Softmax(dim=-1)
        )
        
        # Constraint importance predictor
        self.importance_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_constraints)
        )
    
    def forward(
        self, 
        features: torch.Tensor,
        sensor_metadata: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate physics-based interpretability information."""
        
        # Predict constraint violation probabilities
        constraint_probs = self.explanation_net(features)
        
        # Predict constraint importance scores
        importance_scores = self.importance_net(features)
        
        return {
            'constraint_probabilities': constraint_probs,
            'constraint_importance': importance_scores,
            'explanation_features': features
        }


# Factory function for creating physics-informed models
def create_physics_informed_model(config: Dict[str, Any]) -> PhysicsInformedHybridModel:
    """
    Factory function to create physics-informed hybrid models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PhysicsInformedHybridModel instance
    """
    return PhysicsInformedHybridModel(
        input_size=config.get('input_size', 5),
        hidden_size=config.get('hidden_size', 128),
        lstm_layers=config.get('lstm_layers', 2),
        gnn_layers=config.get('gnn_layers', 2),
        latent_dim=config.get('latent_dim', 64),
        dropout=config.get('dropout', 0.1),
        use_transformer=config.get('use_transformer', True),
        use_sparse_attention=config.get('use_sparse_attention', True),
        physics_weight=config.get('physics_weight', 1.0),
        constraint_config=config.get('constraint_config', {})
    )
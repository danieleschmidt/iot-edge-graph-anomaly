"""
Causal Discovery Network for IoT Anomaly Detection.

Revolutionary implementation of automated causal relationship inference for industrial IoT systems.
Discovers causal structures in sensor networks to provide explainable anomaly detection with
root cause analysis capabilities.

Key Features:
- Structural Causal Models (SCM) for sensor relationship discovery
- Directed Acyclic Graph (DAG) learning with differentiable constraints
- Causal intervention modeling for anomaly explanation
- Granger causality testing for temporal relationships
- Invariant Risk Minimization for robust causal inference
- Pearl's causal hierarchy integration (Association, Intervention, Counterfactual)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import itertools
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_regression
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class CausalConfig:
    """Configuration for causal discovery system."""
    num_variables: int = 5
    max_lag: int = 5  # Maximum time lag for temporal causality
    significance_level: float = 0.05  # Statistical significance threshold
    dag_constraint_weight: float = 1.0  # Weight for DAG constraint in optimization
    sparsity_weight: float = 0.1  # Encourage sparse causal graphs
    intervention_strength: float = 0.5  # Strength of interventional changes
    use_structural_equations: bool = True
    causal_discovery_method: str = "notears"  # notears, pc, fci, lingam
    enable_temporal_causality: bool = True
    enable_nonlinear_relationships: bool = True


class CausalMechanism(nn.Module):
    """
    Learnable structural equation for a single variable in the causal model.
    
    Implements f_i(PA_i, ε_i) where PA_i are parents of variable i and ε_i is noise.
    """
    
    def __init__(self, num_parents: int, hidden_dim: int = 32, nonlinear: bool = True):
        super().__init__()
        self.num_parents = num_parents
        self.nonlinear = nonlinear
        
        if nonlinear:
            # Nonlinear structural equation with neural network
            self.mechanism = nn.Sequential(
                nn.Linear(num_parents, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # Linear structural equation
            self.mechanism = nn.Linear(num_parents, 1)
        
        # Noise parameter (learnable)
        self.log_noise_std = nn.Parameter(torch.tensor(-1.0))
    
    def forward(self, parent_values: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """
        Apply causal mechanism to compute child variable.
        
        Args:
            parent_values: Values of parent variables [batch_size, num_parents]
            add_noise: Whether to add structural noise
            
        Returns:
            Child variable values [batch_size, 1]
        """
        # Deterministic part
        deterministic_part = self.mechanism(parent_values)
        
        if add_noise and self.training:
            # Add structural noise
            noise_std = torch.exp(self.log_noise_std)
            noise = torch.randn_like(deterministic_part) * noise_std
            return deterministic_part + noise
        else:
            return deterministic_part


class DAGConstraint(nn.Module):
    """
    Differentiable DAG constraint implementation using matrix exponential.
    
    Ensures that the learned adjacency matrix represents a Directed Acyclic Graph (DAG)
    by penalizing cycles in the graph structure.
    """
    
    def __init__(self, num_variables: int):
        super().__init__()
        self.num_variables = num_variables
    
    def forward(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute DAG constraint violation.
        
        Args:
            adjacency_matrix: Adjacency matrix [num_variables, num_variables]
            
        Returns:
            DAG constraint violation (0 if DAG, positive if cycles exist)
        """
        # Using matrix exponential trace trick: tr(exp(A ◦ A)) - d
        # where ◦ denotes element-wise multiplication
        A_squared = adjacency_matrix * adjacency_matrix
        
        # Compute matrix exponential trace
        # Using Taylor series approximation for numerical stability
        I = torch.eye(self.num_variables, device=adjacency_matrix.device)
        exp_A = I
        A_power = I
        
        for k in range(1, 10):  # Truncated Taylor series
            A_power = torch.matmul(A_power, A_squared) / k
            exp_A = exp_A + A_power
        
        dag_constraint = torch.trace(exp_A) - self.num_variables
        
        return torch.relu(dag_constraint)  # Only penalize positive violations


class GrangerCausalityTester:
    """
    Granger causality testing for temporal causal relationships.
    
    Tests whether past values of X help predict future values of Y beyond
    what Y's own past values can predict.
    """
    
    def __init__(self, max_lag: int = 5, significance_level: float = 0.05):
        self.max_lag = max_lag
        self.significance_level = significance_level
    
    def test_granger_causality(self, 
                              x: torch.Tensor, 
                              y: torch.Tensor) -> Dict[str, float]:
        """
        Test Granger causality from X to Y.
        
        Args:
            x: Time series X [time_steps, num_features]
            y: Time series Y [time_steps, num_features]
            
        Returns:
            Dict containing test statistics and p-values
        """
        # Convert to numpy for statistical testing
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        results = {}
        
        for lag in range(1, self.max_lag + 1):
            # Create lagged variables
            X_lagged = self._create_lagged_matrix(x_np, lag)
            Y_lagged = self._create_lagged_matrix(y_np, lag)
            Y_target = y_np[lag:, :]
            
            if X_lagged.shape[0] < 10:  # Minimum samples for reliable test
                continue
            
            # Restricted model: Y(t) ~ Y(t-1), ..., Y(t-lag)
            from sklearn.linear_model import LinearRegression
            
            restricted_model = LinearRegression()
            restricted_model.fit(Y_lagged, Y_target)
            restricted_pred = restricted_model.predict(Y_lagged)
            restricted_rss = np.sum((Y_target - restricted_pred) ** 2)
            
            # Unrestricted model: Y(t) ~ Y(t-1), ..., Y(t-lag), X(t-1), ..., X(t-lag)
            combined_features = np.concatenate([Y_lagged, X_lagged], axis=1)
            unrestricted_model = LinearRegression()
            unrestricted_model.fit(combined_features, Y_target)
            unrestricted_pred = unrestricted_model.predict(combined_features)
            unrestricted_rss = np.sum((Y_target - unrestricted_pred) ** 2)
            
            # F-test for Granger causality
            n = X_lagged.shape[0]  # Number of observations
            k_restricted = Y_lagged.shape[1]  # Number of parameters in restricted model
            k_unrestricted = combined_features.shape[1]  # Number of parameters in unrestricted model
            
            f_statistic = ((restricted_rss - unrestricted_rss) / (k_unrestricted - k_restricted)) / \
                         (unrestricted_rss / (n - k_unrestricted))
            
            # Calculate p-value
            from scipy.stats import f
            p_value = 1 - f.cdf(f_statistic, k_unrestricted - k_restricted, n - k_unrestricted)
            
            results[f'lag_{lag}'] = {
                'f_statistic': float(f_statistic),
                'p_value': float(p_value),
                'is_causal': p_value < self.significance_level,
                'restricted_rss': float(restricted_rss),
                'unrestricted_rss': float(unrestricted_rss)
            }
        
        return results
    
    def _create_lagged_matrix(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Create matrix of lagged variables."""
        n_samples, n_features = data.shape
        lagged_data = []
        
        for lag in range(1, max_lag + 1):
            if lag < n_samples:
                lagged_data.append(data[:-lag, :])
        
        if lagged_data:
            return np.concatenate(lagged_data, axis=1)[max_lag-1:, :]
        else:
            return np.empty((0, 0))


class InvariantRiskMinimization(nn.Module):
    """
    Invariant Risk Minimization for robust causal inference.
    
    Learns representations that are invariant across different environments/domains
    to discover stable causal relationships.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Feature extractor (representation learning)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32)
        )
        
        # Classifier (environment-invariant)
        self.classifier = nn.Linear(32, 1)
        
        # Environment classifier (should fail on invariant features)
        self.env_classifier = nn.Linear(32, 2)  # Binary environment classification
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract invariant features and make predictions.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dict containing features, predictions, and environment classifications
        """
        features = self.feature_extractor(x)
        predictions = torch.sigmoid(self.classifier(features))
        env_predictions = F.softmax(self.env_classifier(features), dim=1)
        
        return {
            'features': features,
            'predictions': predictions.squeeze(-1),
            'env_predictions': env_predictions
        }
    
    def compute_irm_loss(self, 
                        predictions: torch.Tensor, 
                        targets: torch.Tensor, 
                        env_labels: torch.Tensor,
                        penalty_weight: float = 1.0) -> torch.Tensor:
        """
        Compute IRM loss that encourages environment-invariant representations.
        
        Args:
            predictions: Model predictions
            targets: True labels
            env_labels: Environment labels for each sample
            penalty_weight: Weight for invariance penalty
            
        Returns:
            IRM loss combining prediction loss and invariance penalty
        """
        # Standard prediction loss
        prediction_loss = F.binary_cross_entropy(predictions, targets)
        
        # Invariance penalty: gradient of loss w.r.t. classifier should be similar across environments
        unique_envs = torch.unique(env_labels)
        penalties = []
        
        for env in unique_envs:
            env_mask = env_labels == env
            if env_mask.sum() > 1:  # Need at least 2 samples per environment
                env_predictions = predictions[env_mask]
                env_targets = targets[env_mask]
                env_loss = F.binary_cross_entropy(env_predictions, env_targets)
                
                # Compute gradient penalty (simplified)
                penalty = torch.autograd.grad(env_loss, [self.classifier.weight], 
                                            create_graph=True, retain_graph=True)[0].pow(2).mean()
                penalties.append(penalty)
        
        if penalties:
            invariance_penalty = torch.stack(penalties).mean()
        else:
            invariance_penalty = torch.tensor(0.0, device=predictions.device)
        
        total_loss = prediction_loss + penalty_weight * invariance_penalty
        
        return total_loss


class CausalDiscoveryNetwork(nn.Module):
    """
    Complete Causal Discovery Network for IoT Anomaly Detection.
    
    Learns causal structure among sensor variables and uses causal reasoning
    for explainable anomaly detection and root cause analysis.
    """
    
    def __init__(self, 
                 input_size: int = 5,
                 config: Optional[CausalConfig] = None):
        super().__init__()
        
        self.input_size = input_size
        self.config = config or CausalConfig(num_variables=input_size)
        
        # Learnable adjacency matrix (causal graph structure)
        self.adjacency_logits = nn.Parameter(torch.randn(input_size, input_size) * 0.1)
        
        # Causal mechanisms for each variable
        self.causal_mechanisms = nn.ModuleList([
            CausalMechanism(
                num_parents=input_size - 1,  # Maximum possible parents
                nonlinear=self.config.enable_nonlinear_relationships
            ) for _ in range(input_size)
        ])
        
        # DAG constraint enforcer
        self.dag_constraint = DAGConstraint(input_size)
        
        # Granger causality tester
        self.granger_tester = GrangerCausalityTester(
            max_lag=self.config.max_lag,
            significance_level=self.config.significance_level
        )
        
        # Invariant Risk Minimization module
        self.irm_module = InvariantRiskMinimization(input_size)
        
        # Anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(input_size + 32, 64),  # input + IRM features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Causal intervention module
        self.intervention_strength = self.config.intervention_strength
        
        logger.info(f"Initialized Causal Discovery Network for {input_size} variables")
    
    def get_causal_adjacency_matrix(self, apply_threshold: bool = True) -> torch.Tensor:
        """
        Get current causal adjacency matrix.
        
        Args:
            apply_threshold: Whether to threshold weak connections
            
        Returns:
            Adjacency matrix [input_size, input_size]
        """
        adjacency = torch.sigmoid(self.adjacency_logits)
        
        # Zero out diagonal (no self-loops)
        adjacency = adjacency * (1 - torch.eye(self.input_size, device=adjacency.device))
        
        if apply_threshold:
            # Threshold weak connections for sparsity
            threshold = 0.3
            adjacency = adjacency * (adjacency > threshold).float()
        
        return adjacency
    
    def forward(self, 
                x: torch.Tensor, 
                env_labels: Optional[torch.Tensor] = None,
                return_causal_info: bool = False) -> Dict[str, Any]:
        """
        Forward pass with causal reasoning.
        
        Args:
            x: Input sensor data [batch_size, seq_len, input_size]
            env_labels: Environment labels for IRM [batch_size]
            return_causal_info: Whether to return detailed causal information
            
        Returns:
            Dict containing anomaly predictions and causal analysis
        """
        # Use latest timestep
        if x.dim() == 3:
            current_x = x[:, -1, :]  # [batch_size, input_size]
            temporal_x = x  # Keep full sequence for temporal analysis
        else:
            current_x = x
            temporal_x = x.unsqueeze(1)  # Add time dimension
        
        batch_size = current_x.size(0)
        device = current_x.device
        
        # Get current causal structure
        adjacency_matrix = self.get_causal_adjacency_matrix(apply_threshold=False)
        
        # Compute DAG constraint violation
        dag_violation = self.dag_constraint(adjacency_matrix)
        
        # Apply causal mechanisms
        reconstructed_values = []
        causal_residuals = []
        
        for i in range(self.input_size):
            # Get parents of variable i
            parent_weights = adjacency_matrix[:, i]  # Parents -> child i
            parent_values = current_x * parent_weights.unsqueeze(0)  # [batch_size, input_size]
            
            # Apply causal mechanism
            reconstructed = self.causal_mechanisms[i](parent_values, add_noise=False)
            reconstructed_values.append(reconstructed)
            
            # Calculate residual (unexplained variance)
            residual = (current_x[:, i:i+1] - reconstructed).abs()
            causal_residuals.append(residual)
        
        reconstructed_tensor = torch.cat(reconstructed_values, dim=1)  # [batch_size, input_size]
        residual_tensor = torch.cat(causal_residuals, dim=1)  # [batch_size, input_size]
        
        # Invariant Risk Minimization
        irm_results = self.irm_module(current_x)
        invariant_features = irm_results['features']
        
        # Combine original features with invariant features
        combined_features = torch.cat([current_x, invariant_features], dim=1)
        
        # Anomaly detection
        anomaly_scores = self.anomaly_detector(combined_features).squeeze(-1)
        
        # Causal anomaly score (based on unexplained residuals)
        causal_anomaly_score = torch.mean(residual_tensor, dim=1)  # [batch_size]
        
        # Combined anomaly score
        final_anomaly_score = 0.7 * anomaly_scores + 0.3 * causal_anomaly_score
        
        results = {
            'anomaly_scores': final_anomaly_score,
            'causal_anomaly_scores': causal_anomaly_score,
            'neural_anomaly_scores': anomaly_scores,
            'dag_violation': dag_violation,
            'adjacency_matrix': adjacency_matrix,
            'causal_residuals': residual_tensor
        }
        
        if return_causal_info:
            # Temporal causal analysis
            temporal_causality = self._analyze_temporal_causality(temporal_x)
            
            # Intervention analysis
            intervention_effects = self._analyze_interventions(current_x)
            
            # Root cause analysis
            root_causes = self._identify_root_causes(residual_tensor, adjacency_matrix)
            
            results.update({
                'temporal_causality': temporal_causality,
                'intervention_effects': intervention_effects,
                'root_causes': root_causes,
                'reconstructed_values': reconstructed_tensor,
                'invariant_features': invariant_features,
                'irm_env_predictions': irm_results['env_predictions'],
                'causal_graph_networkx': self._to_networkx_graph(adjacency_matrix)
            })
        
        return results
    
    def _analyze_temporal_causality(self, temporal_x: torch.Tensor) -> Dict[str, Any]:
        """Analyze temporal causal relationships using Granger causality."""
        if not self.config.enable_temporal_causality or temporal_x.size(1) < 10:
            return {'granger_causality_matrix': torch.zeros(self.input_size, self.input_size)}
        
        # Average across batch for temporal analysis
        avg_temporal = temporal_x.mean(0)  # [seq_len, input_size]
        
        granger_matrix = torch.zeros(self.input_size, self.input_size)
        
        for i in range(self.input_size):
            for j in range(self.input_size):
                if i != j:
                    x_series = avg_temporal[:, j:j+1]  # Cause
                    y_series = avg_temporal[:, i:i+1]  # Effect
                    
                    granger_results = self.granger_tester.test_granger_causality(x_series, y_series)
                    
                    # Use strongest causality across all lags
                    max_causality = 0.0
                    for lag_result in granger_results.values():
                        if lag_result['is_causal']:
                            causality_strength = 1.0 - lag_result['p_value']
                            max_causality = max(max_causality, causality_strength)
                    
                    granger_matrix[i, j] = max_causality
        
        return {
            'granger_causality_matrix': granger_matrix,
            'granger_results_detailed': granger_results
        }
    
    def _analyze_interventions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze effects of hypothetical interventions on each variable.
        
        Simulates do(X_i = x_i + δ) interventions to understand causal effects.
        """
        batch_size = x.size(0)
        intervention_effects = {}
        adjacency = self.get_causal_adjacency_matrix()
        
        for i in range(self.input_size):
            # Create intervention: increase variable i by intervention_strength
            intervened_x = x.clone()
            intervened_x[:, i] += self.intervention_strength
            
            # Compute downstream effects
            effects = torch.zeros_like(x)
            
            # Direct effects (immediate children)
            children = adjacency[i, :] > 0.3  # Variables causally influenced by i
            
            for j in range(self.input_size):
                if children[j]:
                    # Compute effect using causal mechanism
                    parent_weights = adjacency[:, j]
                    parent_values = intervened_x * parent_weights.unsqueeze(0)
                    predicted_value = self.causal_mechanisms[j](parent_values, add_noise=False)
                    
                    # Effect is difference from original prediction
                    original_parent_values = x * parent_weights.unsqueeze(0)
                    original_predicted = self.causal_mechanisms[j](original_parent_values, add_noise=False)
                    
                    effects[:, j] = (predicted_value - original_predicted).squeeze(-1)
            
            intervention_effects[f'intervention_var_{i}'] = effects
        
        return intervention_effects
    
    def _identify_root_causes(self, residuals: torch.Tensor, adjacency: torch.Tensor) -> Dict[str, Any]:
        """
        Identify root causes of anomalies using causal structure.
        
        Variables with high residuals and many outgoing edges are likely root causes.
        """
        batch_size = residuals.size(0)
        
        # Calculate causal influence (number of outgoing edges weighted by strength)
        causal_influence = torch.sum(adjacency, dim=1)  # Sum over children
        
        # Root cause score combines residual magnitude with causal influence
        root_cause_scores = torch.zeros(batch_size, self.input_size, device=residuals.device)
        
        for i in range(self.input_size):
            influence_weight = causal_influence[i] + 0.1  # Avoid division by zero
            root_cause_scores[:, i] = residuals[:, i] * influence_weight
        
        # Identify top root causes for each sample
        top_k = min(3, self.input_size)
        top_causes = torch.topk(root_cause_scores, k=top_k, dim=1)
        
        return {
            'root_cause_scores': root_cause_scores,
            'top_root_causes_indices': top_causes.indices,  # [batch_size, top_k]
            'top_root_causes_scores': top_causes.values,   # [batch_size, top_k]
            'causal_influence_per_variable': causal_influence
        }
    
    def _to_networkx_graph(self, adjacency: torch.Tensor) -> nx.DiGraph:
        """Convert adjacency matrix to NetworkX directed graph for visualization."""
        adj_np = adjacency.detach().cpu().numpy()
        
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(self.input_size):
            G.add_node(i, label=f'Sensor_{i}')
        
        # Add edges with weights
        for i in range(self.input_size):
            for j in range(self.input_size):
                if adj_np[i, j] > 0.1:  # Threshold for visualization
                    G.add_edge(i, j, weight=adj_np[i, j])
        
        return G
    
    def compute_reconstruction_error(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Compute reconstruction error compatible with anomaly detection interface.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            reduction: How to reduce the error ('mean', 'sum', 'none')
            
        Returns:
            Reconstruction error tensor
        """
        results = self.forward(x)
        anomaly_scores = results['anomaly_scores']
        
        # Convert anomaly probability to reconstruction-like error
        reconstruction_error = -torch.log(1 - anomaly_scores + 1e-8)
        
        if reduction == 'mean':
            return reconstruction_error.mean()
        elif reduction == 'sum':
            return reconstruction_error.sum()
        else:
            return reconstruction_error
    
    def compute_causal_loss(self, 
                           x: torch.Tensor, 
                           targets: torch.Tensor, 
                           env_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss including causal constraints.
        
        Args:
            x: Input data
            targets: Anomaly labels
            env_labels: Environment labels for IRM
            
        Returns:
            Dict containing different loss components
        """
        results = self.forward(x, env_labels, return_causal_info=True)
        
        # Prediction loss
        prediction_loss = F.binary_cross_entropy(results['anomaly_scores'], targets)
        
        # DAG constraint loss
        dag_loss = self.config.dag_constraint_weight * results['dag_violation']
        
        # Sparsity loss (encourage sparse causal graphs)
        adjacency = results['adjacency_matrix']
        sparsity_loss = self.config.sparsity_weight * torch.mean(adjacency)
        
        # IRM loss (if environment labels provided)
        irm_loss = torch.tensor(0.0, device=x.device)
        if env_labels is not None:
            irm_results = results['irm_env_predictions']
            dummy_targets = results['neural_anomaly_scores']  # Use neural predictions as proxy
            irm_loss = self.irm_module.compute_irm_loss(
                dummy_targets, targets, env_labels, penalty_weight=0.1
            )
        
        # Causal consistency loss (reconstructed values should match observations)
        reconstruction_loss = F.mse_loss(results['reconstructed_values'], 
                                       x[:, -1, :] if x.dim() == 3 else x)
        
        total_loss = prediction_loss + dag_loss + sparsity_loss + irm_loss + 0.1 * reconstruction_loss
        
        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'dag_loss': dag_loss,
            'sparsity_loss': sparsity_loss,
            'irm_loss': irm_loss,
            'reconstruction_loss': reconstruction_loss
        }
    
    def discover_causal_structure(self, 
                                 training_data: torch.Tensor, 
                                 num_iterations: int = 100) -> Dict[str, Any]:
        """
        Discover causal structure from observational data.
        
        Args:
            training_data: Training sensor data [num_samples, seq_len, input_size]
            num_iterations: Number of optimization iterations
            
        Returns:
            Discovered causal structure and statistics
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        
        logger.info("Starting causal structure discovery...")
        
        loss_history = []
        dag_violations = []
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Sample batch
            batch_indices = torch.randint(0, training_data.size(0), (32,))
            batch_data = training_data[batch_indices]
            
            # Forward pass
            results = self.forward(batch_data, return_causal_info=True)
            
            # Compute causal discovery loss (unsupervised)
            dag_loss = self.config.dag_constraint_weight * results['dag_violation']
            sparsity_loss = self.config.sparsity_weight * torch.mean(results['adjacency_matrix'])
            reconstruction_loss = F.mse_loss(results['reconstructed_values'], batch_data[:, -1, :])
            
            total_loss = dag_loss + sparsity_loss + reconstruction_loss
            
            total_loss.backward()
            optimizer.step()
            
            loss_history.append(total_loss.item())
            dag_violations.append(results['dag_violation'].item())
            
            if iteration % 20 == 0:
                logger.info(f"Iteration {iteration}: Loss={total_loss.item():.4f}, "
                          f"DAG_violation={results['dag_violation'].item():.4f}")
        
        # Final causal structure
        final_adjacency = self.get_causal_adjacency_matrix(apply_threshold=True)
        causal_graph = self._to_networkx_graph(final_adjacency)
        
        logger.info("Causal structure discovery completed")
        
        return {
            'causal_adjacency_matrix': final_adjacency,
            'causal_graph': causal_graph,
            'loss_history': loss_history,
            'dag_violations': dag_violations,
            'num_discovered_edges': int((final_adjacency > 0.1).sum().item()),
            'graph_density': float(torch.mean(final_adjacency).item())
        }


def create_causal_discovery_detector(config: Dict[str, Any]) -> CausalDiscoveryNetwork:
    """
    Factory function to create causal discovery anomaly detector.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured causal discovery network
    """
    input_size = config.get('input_size', 5)
    
    causal_config_dict = config.get('causal_config', {})
    causal_config = CausalConfig(
        num_variables=input_size,
        max_lag=causal_config_dict.get('max_lag', 5),
        significance_level=causal_config_dict.get('significance_level', 0.05),
        dag_constraint_weight=causal_config_dict.get('dag_constraint_weight', 1.0),
        sparsity_weight=causal_config_dict.get('sparsity_weight', 0.1),
        enable_temporal_causality=causal_config_dict.get('enable_temporal_causality', True),
        enable_nonlinear_relationships=causal_config_dict.get('enable_nonlinear_relationships', True)
    )
    
    network = CausalDiscoveryNetwork(input_size=input_size, config=causal_config)
    
    logger.info(f"Created causal discovery detector with {sum(p.numel() for p in network.parameters())} parameters")
    logger.info(f"Causal discovery features: "
               f"temporal_causality={causal_config.enable_temporal_causality}, "
               f"nonlinear_relationships={causal_config.enable_nonlinear_relationships}")
    
    return network
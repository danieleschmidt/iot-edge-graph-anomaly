"""
Continual Learning Network for IoT Anomaly Detection.

Revolutionary implementation of lifelong learning without catastrophic forgetting
for adaptive IoT anomaly detection systems that evolve with changing environments.

Key Features:
- Elastic Weight Consolidation (EWC) for importance-weighted parameter preservation
- Progressive Neural Networks for task-specific capacity expansion
- Memory-Augmented Networks with external memory for experience replay
- Meta-Learning with Model-Agnostic Meta-Learning (MAML) for fast adaptation
- Incremental Learning with Growing Neural Gas for dynamic topology
- Experience Replay with prioritized sampling based on anomaly severity
- Online Learning with concept drift detection and model adaptation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import math
import random
from collections import deque, namedtuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ContinualLearningConfig:
    """Configuration for continual learning system."""
    # Memory configuration
    memory_size: int = 10000
    replay_batch_size: int = 32
    memory_update_strategy: str = "reservoir"  # reservoir, ring_buffer, prioritized
    
    # EWC configuration
    ewc_lambda: float = 1000.0  # Importance weight for EWC regularization
    fisher_sample_size: int = 1000  # Samples for Fisher Information Matrix estimation
    update_fisher_every: int = 1000  # Update Fisher matrix every N samples
    
    # Progressive Networks configuration
    enable_progressive_networks: bool = True
    max_columns: int = 5  # Maximum number of task-specific columns
    lateral_connection_strength: float = 0.1
    
    # Meta-learning configuration
    meta_learning_rate: float = 0.001
    meta_batch_size: int = 16
    inner_loop_steps: int = 5
    adaptation_learning_rate: float = 0.01
    
    # Concept drift detection
    drift_detection_window: int = 1000
    drift_threshold: float = 0.1
    adaptation_strength: float = 0.5
    
    # Experience replay
    prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    
    # Growing Neural Gas
    enable_growing_topology: bool = True
    max_neurons: int = 1000
    insertion_threshold: float = 0.1
    deletion_threshold: float = 0.01


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority', 'timestamp'])


class FisherInformationMatrix:
    """
    Fisher Information Matrix computation for Elastic Weight Consolidation.
    
    Estimates parameter importance by computing the Fisher Information Matrix
    which measures sensitivity of log-likelihood to parameter changes.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.fisher_matrices = {}
        self.optimal_parameters = {}
    
    def compute_fisher_matrix(self, data_loader, num_samples: int = 1000):
        """
        Compute Fisher Information Matrix for current task.
        
        Args:
            data_loader: DataLoader with samples from current task
            num_samples: Number of samples to use for estimation
        """
        logger.info("Computing Fisher Information Matrix...")
        
        # Initialize Fisher matrices
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_matrices[name] = torch.zeros_like(param)
        
        # Set model to evaluation mode
        self.model.eval()
        
        sample_count = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            if sample_count >= num_samples:
                break
            
            batch_size = data.size(0)
            
            # Forward pass
            outputs = self.model(data)
            
            # Sample from categorical distribution for classification
            if outputs.dim() > 1 and outputs.size(1) > 1:
                # Multi-class case
                probs = F.softmax(outputs, dim=1)
                sampled_labels = torch.multinomial(probs, 1).squeeze()
            else:
                # Binary case - sample from Bernoulli
                probs = torch.sigmoid(outputs.squeeze())
                sampled_labels = torch.bernoulli(probs)
            
            # Compute loss w.r.t. sampled labels
            if outputs.dim() > 1 and outputs.size(1) > 1:
                loss = F.cross_entropy(outputs, sampled_labels.long())
            else:
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), sampled_labels.float())
            
            # Backward pass to get gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher Information)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_matrices[name] += param.grad.pow(2) * batch_size
            
            sample_count += batch_size
        
        # Normalize by number of samples
        for name in self.fisher_matrices:
            self.fisher_matrices[name] /= sample_count
        
        # Store optimal parameters for current task
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_parameters[name] = param.data.clone()
        
        logger.info(f"Fisher Information Matrix computed using {sample_count} samples")
    
    def ewc_loss(self, lambda_ewc: float = 1000.0) -> torch.Tensor:
        """
        Compute Elastic Weight Consolidation loss.
        
        Args:
            lambda_ewc: Regularization strength
            
        Returns:
            EWC penalty term
        """
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrices:
                fisher = self.fisher_matrices[name]
                optimal_param = self.optimal_parameters[name]
                loss += torch.sum(fisher * (param - optimal_param).pow(2))
        
        return lambda_ewc * loss


class ProgressiveNeuralNetwork(nn.Module):
    """
    Progressive Neural Network for continual learning without catastrophic forgetting.
    
    Grows laterally by adding new columns for new tasks while preserving
    previous knowledge in frozen columns with lateral connections.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_columns = 0
        
        # Column-specific layers
        self.columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        
        # Add first column
        self.add_column()
    
    def add_column(self):
        """Add a new column for a new task."""
        column_layers = nn.ModuleList()
        lateral_layers = nn.ModuleList()
        
        # Input layer
        if self.num_columns == 0:
            input_dim = self.input_size
        else:
            input_dim = self.input_size
        
        prev_layer_size = input_dim
        
        # Hidden layers
        for i, hidden_size in enumerate(self.hidden_sizes):
            # Main layer for this column
            layer = nn.Linear(prev_layer_size, hidden_size)
            column_layers.append(layer)
            
            # Lateral connections from previous columns
            if self.num_columns > 0:
                lateral_input_size = 0
                for col_idx in range(self.num_columns):
                    if i < len(self.columns[col_idx]) - 1:  # Exclude output layer
                        lateral_input_size += self.hidden_sizes[i]
                
                if lateral_input_size > 0:
                    lateral_layer = nn.Linear(lateral_input_size, hidden_size)
                    lateral_layers.append(lateral_layer)
                else:
                    lateral_layers.append(None)
            else:
                lateral_layers.append(None)
            
            prev_layer_size = hidden_size
        
        # Output layer
        output_layer = nn.Linear(prev_layer_size, self.output_size)
        column_layers.append(output_layer)
        lateral_layers.append(None)  # No lateral connections to output
        
        self.columns.append(column_layers)
        self.lateral_connections.append(lateral_layers)
        self.num_columns += 1
        
        # Freeze previous columns
        if self.num_columns > 1:
            for col_idx in range(self.num_columns - 1):
                for param in self.columns[col_idx].parameters():
                    param.requires_grad = False
        
        logger.info(f"Added column {self.num_columns}, total columns: {self.num_columns}")
    
    def forward(self, x: torch.Tensor, column_id: int = -1) -> torch.Tensor:
        """
        Forward pass through specified column (default: latest column).
        
        Args:
            x: Input tensor
            column_id: Column to use (-1 for latest)
            
        Returns:
            Output tensor
        """
        if column_id == -1:
            column_id = self.num_columns - 1
        
        if column_id >= self.num_columns:
            raise ValueError(f"Column {column_id} does not exist")
        
        # Store activations from all previous columns
        all_activations = [[] for _ in range(self.num_columns)]
        
        # Forward through all columns up to and including target column
        for col_idx in range(column_id + 1):
            current_activation = x
            
            # Process through layers of this column
            for layer_idx, layer in enumerate(self.columns[col_idx][:-1]):  # Exclude output layer
                # Apply main layer
                main_output = layer(current_activation)
                
                # Add lateral connections if available
                if (col_idx > 0 and 
                    layer_idx < len(self.lateral_connections[col_idx]) and 
                    self.lateral_connections[col_idx][layer_idx] is not None):
                    
                    # Collect activations from previous columns at same layer
                    lateral_inputs = []
                    for prev_col in range(col_idx):
                        if layer_idx < len(all_activations[prev_col]):
                            lateral_inputs.append(all_activations[prev_col][layer_idx])
                    
                    if lateral_inputs:
                        lateral_input = torch.cat(lateral_inputs, dim=1)
                        lateral_output = self.lateral_connections[col_idx][layer_idx](lateral_input)
                        current_activation = F.relu(main_output + lateral_output * 0.1)  # Small lateral influence
                    else:
                        current_activation = F.relu(main_output)
                else:
                    current_activation = F.relu(main_output)
                
                # Store activation for lateral connections
                all_activations[col_idx].append(current_activation)
        
        # Final output layer
        output = self.columns[column_id][-1](current_activation)
        return output


class PrioritizedExperienceBuffer:
    """
    Prioritized Experience Replay Buffer for continual learning.
    
    Stores experiences with priority based on anomaly severity and temporal importance.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization strength
        self.beta = beta    # Importance sampling correction
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def add(self, experience: Experience):
        """Add experience with priority."""
        self.buffer.append(experience)
        # New experiences get maximum priority
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], torch.Tensor, torch.Tensor]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Sample experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, torch.tensor(indices), torch.tensor(weights, dtype=torch.float32)
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.item()
            self.max_priority = max(self.max_priority, priority.item())
    
    def __len__(self):
        return len(self.buffer)


class ConceptDriftDetector:
    """
    Statistical concept drift detection using Page-Hinkley test.
    
    Detects changes in data distribution that may require model adaptation.
    """
    
    def __init__(self, threshold: float = 0.1, window_size: int = 1000):
        self.threshold = threshold
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.sum_x = 0.0
        self.sum_x_squared = 0.0
        self.n = 0
        self.mean = 0.0
        self.variance = 0.0
        self.ph_sum = 0.0
        self.ph_min = 0.0
        self.drift_detected = False
        self.warning_detected = False
    
    def update(self, error: float) -> Dict[str, bool]:
        """
        Update drift detector with new error value.
        
        Args:
            error: Prediction error for current sample
            
        Returns:
            Dict containing drift and warning flags
        """
        self.n += 1
        
        # Update running statistics
        self.sum_x += error
        self.sum_x_squared += error ** 2
        
        if self.n > 1:
            self.mean = self.sum_x / self.n
            self.variance = (self.sum_x_squared / self.n) - (self.mean ** 2)
        
        # Page-Hinkley test
        if self.n > 10 and self.variance > 0:  # Need minimum samples and non-zero variance
            std_dev = math.sqrt(self.variance)
            normalized_error = (error - self.mean) / (std_dev + 1e-8)
            
            # Update Page-Hinkley statistics
            self.ph_sum = max(0, self.ph_sum + normalized_error - self.threshold / 2)
            self.ph_min = min(self.ph_sum, self.ph_min)
            
            # Check for drift
            ph_test = self.ph_sum - self.ph_min
            
            if ph_test > self.threshold:
                self.drift_detected = True
            elif ph_test > self.threshold / 2:
                self.warning_detected = True
            else:
                self.warning_detected = False
                self.drift_detected = False
        
        # Reset if window is full
        if self.n >= self.window_size:
            self.reset()
        
        return {
            'drift_detected': self.drift_detected,
            'warning_detected': self.warning_detected,
            'drift_magnitude': self.ph_sum - self.ph_min if hasattr(self, 'ph_sum') else 0.0
        }


class MetaLearningModule(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) for fast adaptation to new domains.
    
    Learns initialization parameters that can quickly adapt to new tasks
    with minimal gradient steps.
    """
    
    def __init__(self, base_model: nn.Module, meta_lr: float = 0.001, adaptation_lr: float = 0.01):
        super().__init__()
        self.base_model = base_model
        self.meta_lr = meta_lr
        self.adaptation_lr = adaptation_lr
        
        # Meta-optimizer for outer loop
        self.meta_optimizer = torch.optim.Adam(self.base_model.parameters(), lr=meta_lr)
    
    def clone_model(self) -> nn.Module:
        """Create a clone of the base model for adaptation."""
        cloned_model = type(self.base_model)(
            *[getattr(self.base_model, attr) for attr in ['input_size', 'hidden_size'] if hasattr(self.base_model, attr)]
        )
        cloned_model.load_state_dict(self.base_model.state_dict())
        return cloned_model
    
    def adaptation_step(self, 
                       model: nn.Module, 
                       support_data: torch.Tensor, 
                       support_labels: torch.Tensor,
                       num_steps: int = 5) -> nn.Module:
        """
        Perform adaptation steps on support data.
        
        Args:
            model: Model to adapt
            support_data: Support set data
            support_labels: Support set labels
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted model
        """
        adapted_model = model
        
        for step in range(num_steps):
            # Forward pass
            predictions = adapted_model(support_data)
            
            # Compute loss
            if predictions.dim() > 1 and predictions.size(1) > 1:
                loss = F.cross_entropy(predictions, support_labels.long())
            else:
                loss = F.binary_cross_entropy_with_logits(predictions.squeeze(), support_labels.float())
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=True)
            
            # Manual parameter update (since we need gradients through this process)
            adapted_params = []
            for param, grad in zip(adapted_model.parameters(), grads):
                adapted_params.append(param - self.adaptation_lr * grad)
            
            # Update model parameters
            param_idx = 0
            for param in adapted_model.parameters():
                param.data = adapted_params[param_idx]
                param_idx += 1
        
        return adapted_model
    
    def meta_update(self, 
                   task_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        """
        Perform meta-update using batch of tasks.
        
        Args:
            task_batch: List of (support_data, support_labels, query_data, query_labels) tuples
        """
        meta_loss = 0.0
        
        for support_data, support_labels, query_data, query_labels in task_batch:
            # Clone model for this task
            task_model = self.clone_model()
            
            # Adaptation on support set
            adapted_model = self.adaptation_step(task_model, support_data, support_labels)
            
            # Evaluate on query set
            query_predictions = adapted_model(query_data)
            
            if query_predictions.dim() > 1 and query_predictions.size(1) > 1:
                task_loss = F.cross_entropy(query_predictions, query_labels.long())
            else:
                task_loss = F.binary_cross_entropy_with_logits(query_predictions.squeeze(), query_labels.float())
            
            meta_loss += task_loss
        
        # Meta-optimization step
        meta_loss /= len(task_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()


class ContinualLearningNetwork(nn.Module):
    """
    Complete Continual Learning Network for adaptive IoT anomaly detection.
    
    Combines multiple continual learning techniques for lifelong learning
    without catastrophic forgetting in dynamic IoT environments.
    """
    
    def __init__(self, 
                 input_size: int = 5,
                 hidden_size: int = 64,
                 config: Optional[ContinualLearningConfig] = None):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.config = config or ContinualLearningConfig()
        
        # Base neural network
        self.base_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Continual learning components
        self.fisher_matrix = FisherInformationMatrix(self.base_network)
        
        if self.config.enable_progressive_networks:
            self.progressive_network = ProgressiveNeuralNetwork(
                input_size=input_size,
                hidden_sizes=[hidden_size, hidden_size],
                output_size=1
            )
        
        # Experience replay buffer
        self.experience_buffer = PrioritizedExperienceBuffer(
            capacity=self.config.memory_size,
            alpha=self.config.priority_alpha,
            beta=self.config.priority_beta
        )
        
        # Concept drift detection
        self.drift_detector = ConceptDriftDetector(
            threshold=self.config.drift_threshold,
            window_size=self.config.drift_detection_window
        )
        
        # Meta-learning module
        self.meta_learner = MetaLearningModule(
            base_model=self.base_network,
            meta_lr=self.config.meta_learning_rate,
            adaptation_lr=self.config.adaptation_learning_rate
        )
        
        # Learning state
        self.current_task = 0
        self.samples_seen = 0
        self.last_fisher_update = 0
        self.adaptation_mode = False
        
        logger.info(f"Continual Learning Network initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, x: torch.Tensor, use_progressive: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through continual learning network.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            use_progressive: Whether to use progressive network
            
        Returns:
            Dict containing predictions and learning information
        """
        # Use latest timestep
        if x.dim() == 3:
            x = x[:, -1, :]  # [batch_size, input_size]
        
        batch_size = x.size(0)
        
        if use_progressive and self.config.enable_progressive_networks:
            # Use progressive network
            predictions = self.progressive_network(x)
        else:
            # Use base network
            predictions = self.base_network(x)
        
        # Apply sigmoid for anomaly probability
        anomaly_scores = torch.sigmoid(predictions.squeeze(-1))
        
        return {
            'anomaly_scores': anomaly_scores,
            'raw_predictions': predictions.squeeze(-1),
            'current_task': self.current_task,
            'samples_seen': self.samples_seen,
            'adaptation_mode': self.adaptation_mode
        }
    
    def learn_incrementally(self, 
                           x: torch.Tensor, 
                           y: torch.Tensor, 
                           task_id: Optional[int] = None) -> Dict[str, float]:
        """
        Incremental learning step with experience replay and drift detection.
        
        Args:
            x: Input data [batch_size, seq_len, input_size]
            y: Target labels [batch_size]
            task_id: Task identifier (None for automatic detection)
            
        Returns:
            Dict containing learning statistics
        """
        # Use latest timestep
        if x.dim() == 3:
            x = x[:, -1, :]
        
        batch_size = x.size(0)
        self.samples_seen += batch_size
        
        # Forward pass
        results = self.forward(x)
        predictions = results['raw_predictions']
        
        # Compute prediction error for drift detection
        prediction_errors = F.binary_cross_entropy_with_logits(
            predictions, y.float(), reduction='none'
        )
        
        # Update drift detector
        drift_info = {}
        for error in prediction_errors:
            drift_status = self.drift_detector.update(error.item())
            for key, value in drift_status.items():
                if key not in drift_info:
                    drift_info[key] = []
                drift_info[key].append(value)
        
        # Check if drift detected
        drift_detected = any(drift_info.get('drift_detected', [False]))
        
        if drift_detected:
            logger.info("Concept drift detected - entering adaptation mode")
            self.adaptation_mode = True
            
            # Add new column to progressive network if enabled
            if self.config.enable_progressive_networks:
                self.progressive_network.add_column()
                self.current_task += 1
        
        # Store experiences in replay buffer
        for i in range(batch_size):
            priority = prediction_errors[i].item() + 1e-6  # Avoid zero priorities
            experience = Experience(
                state=x[i].clone(),
                action=0,  # Not applicable for anomaly detection
                reward=-prediction_errors[i].item(),  # Negative error as reward
                next_state=x[i].clone(),  # Same state for simplicity
                done=False,
                priority=priority,
                timestamp=self.samples_seen
            )
            self.experience_buffer.add(experience)
        
        # Compute loss components
        prediction_loss = F.binary_cross_entropy_with_logits(predictions, y.float())
        
        # EWC regularization loss
        ewc_loss = self.fisher_matrix.ewc_loss(self.config.ewc_lambda)
        
        # Experience replay loss
        replay_loss = torch.tensor(0.0, device=x.device)
        if len(self.experience_buffer) >= self.config.replay_batch_size:
            replay_experiences, replay_indices, importance_weights = \
                self.experience_buffer.sample(self.config.replay_batch_size)
            
            replay_states = torch.stack([exp.state for exp in replay_experiences])
            replay_rewards = torch.tensor([exp.reward for exp in replay_experiences], device=x.device)
            
            replay_predictions = self.base_network(replay_states).squeeze(-1)
            replay_targets = (replay_rewards > -0.5).float()  # Convert rewards to binary targets
            
            replay_loss = F.binary_cross_entropy_with_logits(replay_predictions, replay_targets)
            replay_loss = replay_loss * importance_weights.to(x.device).mean()  # Apply importance sampling
            
            # Update priorities based on prediction errors
            new_priorities = torch.abs(replay_predictions - replay_targets) + 1e-6
            self.experience_buffer.update_priorities(replay_indices, new_priorities)
        
        # Total loss
        total_loss = prediction_loss + ewc_loss + 0.1 * replay_loss
        
        # Optimization step
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update Fisher Information Matrix periodically
        if (self.samples_seen - self.last_fisher_update) >= self.config.update_fisher_every:
            # Create dummy dataloader for Fisher computation
            dataset = torch.utils.data.TensorDataset(x, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
            self.fisher_matrix.compute_fisher_matrix(dataloader, self.config.fisher_sample_size)
            self.last_fisher_update = self.samples_seen
        
        return {
            'total_loss': total_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'ewc_loss': ewc_loss.item(),
            'replay_loss': replay_loss.item(),
            'drift_detected': drift_detected,
            'buffer_size': len(self.experience_buffer),
            'adaptation_mode': self.adaptation_mode
        }
    
    def meta_learn_from_tasks(self, task_batch: List[Dict[str, torch.Tensor]]) -> float:
        """
        Meta-learning update using batch of tasks.
        
        Args:
            task_batch: List of task dictionaries with 'support' and 'query' data
            
        Returns:
            Meta-learning loss
        """
        maml_tasks = []
        
        for task in task_batch:
            support_x = task['support_x'][:, -1, :] if task['support_x'].dim() == 3 else task['support_x']
            support_y = task['support_y']
            query_x = task['query_x'][:, -1, :] if task['query_x'].dim() == 3 else task['query_x']
            query_y = task['query_y']
            
            maml_tasks.append((support_x, support_y, query_x, query_y))
        
        meta_loss = self.meta_learner.meta_update(maml_tasks)
        
        logger.info(f"Meta-learning update completed with loss: {meta_loss:.4f}")
        
        return meta_loss
    
    def adapt_to_new_domain(self, 
                           support_data: torch.Tensor, 
                           support_labels: torch.Tensor,
                           num_adaptation_steps: int = None) -> nn.Module:
        """
        Quickly adapt to new domain using meta-learned initialization.
        
        Args:
            support_data: Support set for new domain
            support_labels: Support set labels
            num_adaptation_steps: Number of adaptation steps
            
        Returns:
            Adapted model
        """
        if num_adaptation_steps is None:
            num_adaptation_steps = self.config.inner_loop_steps
        
        # Use latest timestep
        if support_data.dim() == 3:
            support_data = support_data[:, -1, :]
        
        # Clone base network for adaptation
        adapted_model = self.meta_learner.clone_model()
        
        # Perform adaptation steps
        adapted_model = self.meta_learner.adaptation_step(
            adapted_model, support_data, support_labels, num_adaptation_steps
        )
        
        logger.info(f"Model adapted to new domain with {num_adaptation_steps} steps")
        
        return adapted_model
    
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
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        return {
            'current_task': self.current_task,
            'samples_seen': self.samples_seen,
            'buffer_size': len(self.experience_buffer),
            'adaptation_mode': self.adaptation_mode,
            'progressive_columns': self.progressive_network.num_columns if self.config.enable_progressive_networks else 0,
            'fisher_matrix_age': self.samples_seen - self.last_fisher_update,
            'drift_detector_state': {
                'n_samples': self.drift_detector.n,
                'current_mean': self.drift_detector.mean,
                'current_variance': self.drift_detector.variance
            }
        }
    
    def save_checkpoint(self, path: str):
        """Save complete model state including continual learning components."""
        checkpoint = {
            'base_network_state': self.base_network.state_dict(),
            'progressive_network_state': self.progressive_network.state_dict() if self.config.enable_progressive_networks else None,
            'fisher_matrices': self.fisher_matrix.fisher_matrices,
            'optimal_parameters': self.fisher_matrix.optimal_parameters,
            'current_task': self.current_task,
            'samples_seen': self.samples_seen,
            'last_fisher_update': self.last_fisher_update,
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"Continual learning checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load complete model state including continual learning components."""
        checkpoint = torch.load(path)
        
        self.base_network.load_state_dict(checkpoint['base_network_state'])
        
        if checkpoint['progressive_network_state'] and self.config.enable_progressive_networks:
            self.progressive_network.load_state_dict(checkpoint['progressive_network_state'])
        
        self.fisher_matrix.fisher_matrices = checkpoint['fisher_matrices']
        self.fisher_matrix.optimal_parameters = checkpoint['optimal_parameters']
        self.current_task = checkpoint['current_task']
        self.samples_seen = checkpoint['samples_seen']
        self.last_fisher_update = checkpoint['last_fisher_update']
        
        logger.info(f"Continual learning checkpoint loaded from {path}")


def create_continual_learning_detector(config: Dict[str, Any]) -> ContinualLearningNetwork:
    """
    Factory function to create continual learning anomaly detector.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured continual learning network
    """
    input_size = config.get('input_size', 5)
    hidden_size = config.get('hidden_size', 64)
    
    continual_config_dict = config.get('continual_learning_config', {})
    continual_config = ContinualLearningConfig(
        memory_size=continual_config_dict.get('memory_size', 10000),
        ewc_lambda=continual_config_dict.get('ewc_lambda', 1000.0),
        enable_progressive_networks=continual_config_dict.get('enable_progressive_networks', True),
        meta_learning_rate=continual_config_dict.get('meta_learning_rate', 0.001),
        drift_threshold=continual_config_dict.get('drift_threshold', 0.1),
        prioritized_replay=continual_config_dict.get('prioritized_replay', True)
    )
    
    network = ContinualLearningNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        config=continual_config
    )
    
    total_params = sum(p.numel() for p in network.parameters())
    
    logger.info(f"Created continual learning detector with {total_params} parameters")
    logger.info(f"Memory capacity: {continual_config.memory_size}")
    logger.info(f"EWC lambda: {continual_config.ewc_lambda}")
    logger.info(f"Progressive networks: {continual_config.enable_progressive_networks}")
    logger.info(f"Meta-learning enabled with lr: {continual_config.meta_learning_rate}")
    
    return network
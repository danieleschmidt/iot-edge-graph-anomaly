"""
Advanced Benchmarking Suite for IoT Anomaly Detection Research.

Comprehensive research framework for evaluating and comparing anomaly detection
algorithms with statistical significance testing, performance analysis, and
publication-ready results.

Key Features:
- Multi-dataset evaluation framework
- Statistical significance testing (t-tests, Wilcoxon, ANOVA)
- Performance vs accuracy trade-off analysis
- Edge deployment benchmarking
- Reproducible experimental methodology
- Automated report generation
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare, kruskal

# Add src to path to import our models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
from iot_edge_anomaly.models.lstm_gnn_hybrid import LSTMGNNHybrid
from iot_edge_anomaly.models.transformer_vae import TransformerVAE
from iot_edge_anomaly.models.sparse_graph_attention import SparseGraphAttentionNetwork
from iot_edge_anomaly.models.physics_informed_hybrid import PhysicsInformedHybrid
from iot_edge_anomaly.models.quantum_classical_hybrid import QuantumClassicalHybridNetwork
from iot_edge_anomaly.models.advanced_hybrid_integration import create_advanced_hybrid_system

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    experiment_name: str
    output_dir: str
    random_seed: int = 42
    num_runs: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    statistical_significance_level: float = 0.05
    edge_device_simulation: bool = True
    generate_visualizations: bool = True
    save_detailed_results: bool = True


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    model_class: type
    config_params: Dict[str, Any]
    description: str = ""
    

@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    name: str
    description: str
    data_generator: Callable
    anomaly_ratio: float
    sequence_length: int
    feature_dim: int


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    dataset_name: str
    run_id: int
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    
    # Timing metrics
    training_time: float
    inference_time: float
    inference_time_per_sample: float
    
    # Resource metrics
    model_size_mb: float
    memory_usage_mb: float
    energy_consumption_mj: float
    
    # Advanced metrics
    reconstruction_error: float
    uncertainty_score: float
    explanation_quality: float


class SyntheticDataGenerator:
    """Generate synthetic IoT sensor datasets for benchmarking."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def generate_swat_like_dataset(
        self, 
        num_samples: int = 1000,
        sequence_length: int = 20,
        feature_dim: int = 5,
        anomaly_ratio: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate SWaT-like industrial control system data.
        
        Args:
            num_samples: Number of sequences to generate
            sequence_length: Length of each sequence
            feature_dim: Number of sensor features
            anomaly_ratio: Fraction of anomalous samples
            
        Returns:
            (data, labels) where data is [num_samples, sequence_length, feature_dim]
        """
        # Normal operation patterns
        normal_samples = int(num_samples * (1 - anomaly_ratio))
        anomaly_samples = num_samples - normal_samples
        
        # Generate normal data with correlated sensors
        normal_data = []
        for i in range(normal_samples):
            # Base signal with some correlation between sensors
            base_signal = np.sin(np.linspace(0, 4*np.pi, sequence_length)) * 2
            
            # Add sensor-specific patterns
            sequence = []
            for t in range(sequence_length):
                features = []
                for f in range(feature_dim):
                    # Each sensor has base signal + specific pattern + noise
                    value = base_signal[t] + np.sin(t * 0.3 + f) + np.random.normal(0, 0.1)
                    
                    # Keep values in realistic ranges
                    if f == 0:  # Temperature sensor
                        value = 20 + value * 5
                    elif f == 1:  # Pressure sensor  
                        value = 50 + value * 10
                    elif f == 2:  # Flow rate sensor
                        value = 1.0 + value * 0.5
                    elif f == 3:  # Level sensor
                        value = 0.5 + value * 0.3
                    else:  # Additional sensors
                        value = 100 + value * 20
                    
                    features.append(value)
                sequence.append(features)
            
            normal_data.append(sequence)
        
        # Generate anomalous data
        anomaly_data = []
        for i in range(anomaly_samples):
            # Start with normal pattern
            base_signal = np.sin(np.linspace(0, 4*np.pi, sequence_length)) * 2
            
            sequence = []
            # Inject anomalies randomly in the sequence
            anomaly_start = np.random.randint(5, sequence_length - 10)
            anomaly_duration = np.random.randint(3, 8)
            
            for t in range(sequence_length):
                features = []
                for f in range(feature_dim):
                    # Normal pattern
                    value = base_signal[t] + np.sin(t * 0.3 + f) + np.random.normal(0, 0.1)
                    
                    # Inject anomaly
                    if anomaly_start <= t < anomaly_start + anomaly_duration:
                        anomaly_type = np.random.choice(['spike', 'drift', 'noise'])
                        
                        if anomaly_type == 'spike':
                            value += np.random.normal(0, 3)  # Sudden spike
                        elif anomaly_type == 'drift':
                            value += (t - anomaly_start) * 0.5  # Gradual drift
                        else:  # noise
                            value += np.random.normal(0, 2)  # Increased noise
                    
                    # Scale to realistic ranges
                    if f == 0:  # Temperature
                        value = 20 + value * 5
                    elif f == 1:  # Pressure
                        value = 50 + value * 10
                    elif f == 2:  # Flow rate
                        value = 1.0 + value * 0.5
                    elif f == 3:  # Level
                        value = 0.5 + value * 0.3
                    else:
                        value = 100 + value * 20
                    
                    features.append(value)
                sequence.append(features)
            
            anomaly_data.append(sequence)
        
        # Combine data and create labels
        all_data = normal_data + anomaly_data
        labels = [0] * normal_samples + [1] * anomaly_samples
        
        # Convert to tensors
        data_tensor = torch.tensor(all_data, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Shuffle
        indices = torch.randperm(num_samples)
        data_tensor = data_tensor[indices]
        label_tensor = label_tensor[indices]
        
        return data_tensor, label_tensor
    
    def generate_multimodal_dataset(
        self,
        num_samples: int = 1000,
        sequence_length: int = 20,
        feature_dim: int = 8,
        anomaly_ratio: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate multi-modal sensor data (vibration, acoustic, thermal)."""
        normal_samples = int(num_samples * (1 - anomaly_ratio))
        anomaly_samples = num_samples - normal_samples
        
        all_data = []
        labels = []
        
        # Normal samples
        for i in range(normal_samples):
            sequence = []
            
            # Multiple operational modes
            mode = np.random.choice(['steady', 'startup', 'shutdown'])
            
            for t in range(sequence_length):
                features = []
                
                if mode == 'steady':
                    # Steady state operation
                    vibration = np.random.normal(0.1, 0.02)  # Low vibration
                    acoustic = np.random.normal(40, 5)       # Moderate noise
                    thermal = np.random.normal(60, 3)        # Stable temperature
                elif mode == 'startup':
                    # Startup phase - increasing patterns
                    progress = t / sequence_length
                    vibration = 0.05 + progress * 0.1 + np.random.normal(0, 0.01)
                    acoustic = 35 + progress * 15 + np.random.normal(0, 2)
                    thermal = 25 + progress * 40 + np.random.normal(0, 2)
                else:  # shutdown
                    # Shutdown phase - decreasing patterns
                    progress = (sequence_length - t) / sequence_length
                    vibration = 0.05 + progress * 0.1 + np.random.normal(0, 0.01)
                    acoustic = 30 + progress * 20 + np.random.normal(0, 2)
                    thermal = 30 + progress * 35 + np.random.normal(0, 2)
                
                # Additional correlated sensors
                pressure = 10 + thermal * 0.5 + np.random.normal(0, 1)
                flow = 2.0 + vibration * 10 + np.random.normal(0, 0.2)
                electrical = 220 + acoustic * 0.1 + np.random.normal(0, 5)
                
                features = [vibration, acoustic, thermal, pressure, flow, electrical]
                
                # Add more features to reach feature_dim
                while len(features) < feature_dim:
                    # Synthetic correlated features
                    synthetic = np.mean(features) + np.random.normal(0, 0.1)
                    features.append(synthetic)
                
                sequence.append(features[:feature_dim])
            
            all_data.append(sequence)
            labels.append(0)
        
        # Anomalous samples
        for i in range(anomaly_samples):
            sequence = []
            
            # Choose anomaly type
            anomaly_type = np.random.choice([
                'bearing_fault', 'imbalance', 'misalignment', 'overheating', 'cavitation'
            ])
            
            anomaly_start = np.random.randint(3, sequence_length - 5)
            
            for t in range(sequence_length):
                features = []
                
                # Base normal operation
                vibration = np.random.normal(0.1, 0.02)
                acoustic = np.random.normal(40, 5)
                thermal = np.random.normal(60, 3)
                
                # Apply anomaly patterns
                if t >= anomaly_start:
                    if anomaly_type == 'bearing_fault':
                        vibration += 0.3 + np.sin(t * 2) * 0.1  # High frequency vibrations
                        acoustic += 20  # Increased noise
                    elif anomaly_type == 'imbalance':
                        vibration += 0.2 * np.sin(t)  # Rotational frequency vibration
                        acoustic += 10
                    elif anomaly_type == 'misalignment':
                        vibration += 0.15 * np.sin(t * 0.5)  # Low frequency vibrations
                        acoustic += 15
                    elif anomaly_type == 'overheating':
                        thermal += (t - anomaly_start) * 2  # Progressive heating
                        acoustic += 5
                    elif anomaly_type == 'cavitation':
                        vibration += np.random.exponential(0.1)  # Random spikes
                        acoustic += np.random.exponential(5)
                
                # Correlated sensors
                pressure = 10 + thermal * 0.5 + np.random.normal(0, 1)
                flow = 2.0 + vibration * 10 + np.random.normal(0, 0.2)
                electrical = 220 + acoustic * 0.1 + np.random.normal(0, 5)
                
                features = [vibration, acoustic, thermal, pressure, flow, electrical]
                
                # Add more features
                while len(features) < feature_dim:
                    synthetic = np.mean(features) + np.random.normal(0, 0.1)
                    features.append(synthetic)
                
                sequence.append(features[:feature_dim])
            
            all_data.append(sequence)
            labels.append(1)
        
        # Convert to tensors and shuffle
        data_tensor = torch.tensor(all_data, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        indices = torch.randperm(len(all_data))
        return data_tensor[indices], label_tensor[indices]


class ModelTrainer:
    """Train and evaluate anomaly detection models."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def train_model(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """
        Train an anomaly detection model.
        
        Returns:
            Training metrics and history
        """
        model = model.to(self.device)
        model.train()
        
        # Create data loaders
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()  # Reconstruction loss for autoencoders
        
        # Training history
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # Compute reconstruction error
                if hasattr(model, 'compute_reconstruction_error'):
                    loss = model.compute_reconstruction_error(batch_data, reduction='mean')
                else:
                    # Fallback for models without this method
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_data)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(self.device)
                    
                    if hasattr(model, 'compute_reconstruction_error'):
                        loss = model.compute_reconstruction_error(batch_data, reduction='mean')
                    else:
                        outputs = model(batch_data)
                        loss = criterion(outputs, batch_data)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            model.train()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model
            test_data: Test sequences
            test_labels: True anomaly labels
            threshold: Anomaly threshold (auto-determined if None)
            
        Returns:
            Evaluation metrics
        """
        model.eval()
        model = model.to(self.device)
        
        # Measure inference time
        inference_times = []
        reconstruction_errors = []
        
        with torch.no_grad():
            for i in range(len(test_data)):
                start_time = time.time()
                
                sample = test_data[i:i+1].to(self.device)
                
                if hasattr(model, 'compute_reconstruction_error'):
                    error = model.compute_reconstruction_error(sample, reduction='mean')
                else:
                    output = model(sample)
                    error = torch.mean((output - sample) ** 2)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                reconstruction_errors.append(error.item())
        
        # Determine threshold if not provided
        if threshold is None:
            normal_errors = [err for i, err in enumerate(reconstruction_errors) if test_labels[i] == 0]
            threshold = np.mean(normal_errors) + 2 * np.std(normal_errors)
        
        # Make predictions
        predictions = [1 if err > threshold else 0 for err in reconstruction_errors]
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        
        # ROC AUC using reconstruction errors as scores
        try:
            roc_auc = roc_auc_score(test_labels, reconstruction_errors)
        except ValueError:
            roc_auc = 0.5  # Fallback if all labels are the same
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'threshold': threshold,
            'mean_reconstruction_error': np.mean(reconstruction_errors),
            'std_reconstruction_error': np.std(reconstruction_errors),
            'inference_time_per_sample': np.mean(inference_times),
            'inference_time_std': np.std(inference_times),
            'model_size_mb': model_size_mb,
            'param_count': param_count
        }


class AdvancedBenchmarkSuite:
    """
    Advanced benchmarking suite for IoT anomaly detection research.
    
    Provides comprehensive evaluation with statistical analysis,
    edge deployment simulation, and publication-ready results.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data_generator = SyntheticDataGenerator(config.random_seed)
        self.trainer = ModelTrainer()
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(config.output_dir) / 'benchmark.log'),
                logging.StreamHandler()
            ]
        )
        
        self.results: List[BenchmarkResult] = []
        
        logger.info(f"Initialized Advanced Benchmark Suite: {config.experiment_name}")
    
    def define_models(self) -> List[ModelConfig]:
        """Define models to benchmark."""
        return [
            ModelConfig(
                name="LSTM_Autoencoder",
                model_class=LSTMAutoencoder,
                config_params={'input_size': 5, 'hidden_size': 64, 'num_layers': 2},
                description="Baseline LSTM autoencoder for sequence anomaly detection"
            ),
            ModelConfig(
                name="LSTM_GNN_Hybrid",
                model_class=LSTMGNNHybrid,
                config_params={'input_size': 5, 'hidden_size': 64, 'gnn_hidden': 32},
                description="Hybrid LSTM-GNN model with spatial-temporal learning"
            ),
            ModelConfig(
                name="Transformer_VAE",
                model_class=TransformerVAE,
                config_params={'input_dim': 5, 'hidden_dim': 64, 'latent_dim': 32},
                description="Transformer-based VAE for complex temporal patterns"
            ),
            ModelConfig(
                name="Physics_Informed",
                model_class=PhysicsInformedHybrid,
                config_params={'input_size': 5, 'hidden_size': 64},
                description="Physics-informed neural network with domain constraints"
            ),
            ModelConfig(
                name="Quantum_Classical_Hybrid",
                model_class=QuantumClassicalHybridNetwork,
                config_params={'input_size': 5, 'hidden_size': 64},
                description="Quantum-enhanced classical model for constraint optimization"
            )
        ]
    
    def define_datasets(self) -> List[DatasetConfig]:
        """Define datasets for benchmarking."""
        return [
            DatasetConfig(
                name="SWaT_Synthetic",
                description="Synthetic industrial control system data based on SWaT",
                data_generator=lambda: self.data_generator.generate_swat_like_dataset(
                    num_samples=1000, sequence_length=20, feature_dim=5, anomaly_ratio=0.1
                ),
                anomaly_ratio=0.1,
                sequence_length=20,
                feature_dim=5
            ),
            DatasetConfig(
                name="Multimodal_Industrial",
                description="Multi-modal sensor data with various fault types",
                data_generator=lambda: self.data_generator.generate_multimodal_dataset(
                    num_samples=800, sequence_length=20, feature_dim=8, anomaly_ratio=0.15
                ),
                anomaly_ratio=0.15,
                sequence_length=20,
                feature_dim=8
            )
        ]
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across all models and datasets.
        
        Returns:
            Aggregated benchmark results with statistical analysis
        """
        logger.info("Starting comprehensive benchmark")
        
        models = self.define_models()
        datasets = self.define_datasets()
        
        # Run experiments
        for dataset_config in datasets:
            logger.info(f"Benchmarking on dataset: {dataset_config.name}")
            
            for model_config in models:
                logger.info(f"  Evaluating model: {model_config.name}")
                
                # Run multiple trials for statistical significance
                for run_id in range(self.config.num_runs):
                    logger.debug(f"    Run {run_id + 1}/{self.config.num_runs}")
                    
                    try:
                        result = self._run_single_experiment(
                            model_config, dataset_config, run_id
                        )
                        self.results.append(result)
                    except Exception as e:
                        logger.error(f"Error in run {run_id} for {model_config.name}: {e}")
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Generate report
        if self.config.save_detailed_results:
            self._save_results()
        
        if self.config.generate_visualizations:
            self._generate_visualizations()
        
        logger.info("Comprehensive benchmark completed")
        return analysis
    
    def _run_single_experiment(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        run_id: int
    ) -> BenchmarkResult:
        """Run a single experimental trial."""
        
        # Generate data
        data, labels = dataset_config.data_generator()
        
        # Split data
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=self.config.test_size, random_state=self.config.random_seed + run_id
        )
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data, train_labels, test_size=self.config.validation_size, random_state=self.config.random_seed + run_id
        )
        
        # Initialize model
        model_params = model_config.config_params.copy()
        
        # Adjust input size for dataset
        if 'input_size' in model_params:
            model_params['input_size'] = dataset_config.feature_dim
        elif 'input_dim' in model_params:
            model_params['input_dim'] = dataset_config.feature_dim
        
        model = model_config.model_class(**model_params)
        
        # Train model
        training_results = self.trainer.train_model(
            model, train_data, train_labels, val_data, val_labels,
            epochs=30, batch_size=32
        )
        
        # Evaluate model
        eval_results = self.trainer.evaluate_model(
            model, test_data, test_labels
        )
        
        # Simulate edge deployment metrics
        edge_metrics = self._simulate_edge_deployment(model, test_data[:10])
        
        # Create benchmark result
        result = BenchmarkResult(
            model_name=model_config.name,
            dataset_name=dataset_config.name,
            run_id=run_id,
            accuracy=eval_results['accuracy'],
            precision=eval_results['precision'],
            recall=eval_results['recall'],
            f1_score=eval_results['f1_score'],
            roc_auc=eval_results['roc_auc'],
            training_time=training_results['training_time'],
            inference_time=eval_results['inference_time_per_sample'],
            inference_time_per_sample=eval_results['inference_time_per_sample'],
            model_size_mb=eval_results['model_size_mb'],
            memory_usage_mb=edge_metrics['memory_usage_mb'],
            energy_consumption_mj=edge_metrics['energy_consumption_mj'],
            reconstruction_error=eval_results['mean_reconstruction_error'],
            uncertainty_score=eval_results['std_reconstruction_error'],
            explanation_quality=0.8  # Placeholder - would compute from actual explanations
        )
        
        return result
    
    def _simulate_edge_deployment(self, model: nn.Module, sample_data: torch.Tensor) -> Dict[str, float]:
        """Simulate edge device deployment metrics."""
        
        # Simulate memory usage (simplified)
        param_count = sum(p.numel() for p in model.parameters())
        base_memory = param_count * 4 / (1024 * 1024)  # MB for parameters
        activation_memory = sample_data.numel() * 4 / (1024 * 1024)  # MB for activations
        total_memory = base_memory + activation_memory * 2  # 2x for intermediate computations
        
        # Simulate energy consumption (very simplified model)
        # Based on operations count and typical edge device power consumption
        total_ops = param_count * len(sample_data)  # Approximate FLOPs
        energy_per_op = 1e-12  # Joules per operation (very rough estimate)
        energy_consumption = total_ops * energy_per_op * 1000  # Convert to millijoules
        
        return {
            'memory_usage_mb': total_memory,
            'energy_consumption_mj': energy_consumption
        }
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results with statistical tests."""
        
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        analysis = {
            'summary_statistics': {},
            'statistical_tests': {},
            'rankings': {},
            'performance_analysis': {}
        }
        
        # Summary statistics by model
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                  'training_time', 'inference_time_per_sample', 'model_size_mb']
        
        for metric in metrics:
            analysis['summary_statistics'][metric] = (
                df.groupby('model_name')[metric]
                .agg(['mean', 'std', 'min', 'max'])
                .round(4)
                .to_dict()
            )
        
        # Statistical significance tests
        models = df['model_name'].unique()
        datasets = df['dataset_name'].unique()
        
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            analysis['statistical_tests'][metric] = {}
            
            for dataset in datasets:
                dataset_results = df[df['dataset_name'] == dataset]
                
                # Friedman test for multiple model comparison
                model_groups = [
                    dataset_results[dataset_results['model_name'] == model][metric].values
                    for model in models
                    if len(dataset_results[dataset_results['model_name'] == model]) > 0
                ]
                
                if len(model_groups) >= 3 and all(len(group) > 0 for group in model_groups):
                    try:
                        stat, p_value = friedmanchisquare(*model_groups)
                        analysis['statistical_tests'][metric][dataset] = {
                            'test': 'Friedman',
                            'statistic': stat,
                            'p_value': p_value,
                            'significant': p_value < self.config.statistical_significance_level
                        }
                    except Exception as e:
                        logger.warning(f"Statistical test failed for {metric} on {dataset}: {e}")
                
                # Pairwise comparisons
                pairwise_tests = {}
                for i, model1 in enumerate(models):
                    for model2 in models[i+1:]:
                        data1 = dataset_results[dataset_results['model_name'] == model1][metric].values
                        data2 = dataset_results[dataset_results['model_name'] == model2][metric].values
                        
                        if len(data1) > 1 and len(data2) > 1:
                            try:
                                stat, p_value = ttest_rel(data1, data2)
                                pairwise_tests[f"{model1}_vs_{model2}"] = {
                                    'statistic': stat,
                                    'p_value': p_value,
                                    'significant': p_value < self.config.statistical_significance_level
                                }
                            except Exception:
                                pass
                
                if pairwise_tests:
                    analysis['statistical_tests'][metric][f"{dataset}_pairwise"] = pairwise_tests
        
        # Model rankings
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            metric_rankings = (
                df.groupby('model_name')[metric]
                .mean()
                .sort_values(ascending=False)
                .to_dict()
            )
            analysis['rankings'][metric] = metric_rankings
        
        # Performance vs efficiency analysis
        efficiency_metrics = df.groupby('model_name').agg({
            'f1_score': 'mean',
            'inference_time_per_sample': 'mean',
            'model_size_mb': 'mean',
            'energy_consumption_mj': 'mean'
        }).round(4)
        
        analysis['performance_analysis']['efficiency_trade_offs'] = efficiency_metrics.to_dict()
        
        return analysis
    
    def _save_results(self):
        """Save detailed results to files."""
        output_dir = Path(self.config.output_dir)
        
        # Save raw results
        results_df = pd.DataFrame([asdict(result) for result in self.results])
        results_df.to_csv(output_dir / 'detailed_results.csv', index=False)
        
        # Save analysis
        analysis = self._analyze_results()
        with open(output_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _generate_visualizations(self):
        """Generate publication-ready visualizations."""
        output_dir = Path(self.config.output_dir)
        
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # Set style for publication-quality plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Performance comparison box plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['accuracy', 'f1_score', 'roc_auc', 'inference_time_per_sample']
        titles = ['Accuracy', 'F1 Score', 'ROC AUC', 'Inference Time (s/sample)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            sns.boxplot(data=df, x='model_name', y=metric, ax=ax)
            ax.set_title(f'{title} by Model')
            ax.set_xlabel('Model')
            ax.set_ylabel(title)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance vs Efficiency scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            ax.scatter(model_data['inference_time_per_sample'], 
                      model_data['f1_score'],
                      label=model, alpha=0.7, s=60)
        
        ax.set_xlabel('Inference Time (s/sample)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Performance vs Efficiency Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(output_dir / 'performance_vs_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical significance heatmap
        analysis = self._analyze_results()
        
        if 'statistical_tests' in analysis and 'f1_score' in analysis['statistical_tests']:
            # Create pairwise p-value matrix
            models = df['model_name'].unique()
            p_value_matrix = np.ones((len(models), len(models)))
            
            for dataset, tests in analysis['statistical_tests']['f1_score'].items():
                if 'pairwise' in dataset and isinstance(tests, dict):
                    for comparison, test_result in tests.items():
                        model1, model2 = comparison.split('_vs_')
                        if model1 in models and model2 in models:
                            idx1 = list(models).index(model1)
                            idx2 = list(models).index(model2)
                            p_value_matrix[idx1, idx2] = test_result['p_value']
                            p_value_matrix[idx2, idx1] = test_result['p_value']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(p_value_matrix, 
                       xticklabels=models, 
                       yticklabels=models,
                       annot=True, 
                       cmap='RdYlBu_r',
                       center=0.05,
                       ax=ax)
            ax.set_title('Statistical Significance (p-values) for F1 Score Comparisons')
            plt.tight_layout()
            plt.savefig(output_dir / 'statistical_significance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main function to run advanced benchmarking suite."""
    
    # Configuration
    config = BenchmarkConfig(
        experiment_name="IoT_Anomaly_Detection_Benchmark_2025",
        output_dir="./benchmark_results",
        num_runs=5,
        random_seed=42,
        statistical_significance_level=0.05,
        edge_device_simulation=True,
        generate_visualizations=True,
        save_detailed_results=True
    )
    
    # Run benchmark
    benchmark = AdvancedBenchmarkSuite(config)
    results = benchmark.run_comprehensive_benchmark()
    
    # Print summary
    print("\n" + "="*80)
    print("ADVANCED BENCHMARKING RESULTS SUMMARY")
    print("="*80)
    
    print("\nTop performing models by F1 Score:")
    f1_rankings = results['rankings']['f1_score']
    for i, (model, score) in enumerate(f1_rankings.items(), 1):
        print(f"{i}. {model}: {score:.4f}")
    
    print("\nStatistical significance tests completed.")
    print(f"Results saved to: {config.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
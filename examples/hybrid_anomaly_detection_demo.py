#!/usr/bin/env python3
"""
Hybrid LSTM-GNN Anomaly Detection Demonstration.

This example demonstrates the complete LSTM-GNN hybrid model for anomaly detection
on IoT sensor networks, showing both temporal and spatial pattern learning.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import time
from typing import Dict, Any, Tuple, List

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from iot_edge_anomaly.models.lstm_gnn_hybrid import LSTMGNNHybridModel
from iot_edge_anomaly.models.gnn_layer import create_sensor_graph
from iot_edge_anomaly.data.swat_loader import SWaTDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridAnomalyDetectionDemo:
    """Demonstration of hybrid LSTM-GNN anomaly detection."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = self._create_demo_config()
        logger.info(f"Using device: {self.device}")
    
    def _create_demo_config(self) -> Dict[str, Any]:
        """Create configuration for hybrid model."""
        return {
            'lstm': {
                'input_size': 5,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.1
            },
            'gnn': {
                'input_dim': 64,
                'hidden_dim': 32,
                'output_dim': 64,
                'num_layers': 2,
                'dropout': 0.1
            },
            'fusion': {
                'output_dim': 128,
                'method': 'concatenate'  # Options: 'concatenate', 'attention', 'gate'
            },
            'graph': {
                'method': 'correlation',  # Options: 'correlation', 'fully_connected'
                'threshold': 0.3
            }
        }
    
    def generate_synthetic_data(self, n_samples: int = 1000, n_sensors: int = 5, 
                               seq_len: int = 20, anomaly_ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic IoT sensor data with anomalies."""
        logger.info(f"Generating {n_samples} samples with {n_sensors} sensors")
        
        # Normal data: correlated sine waves with noise
        t = torch.linspace(0, 4 * np.pi, seq_len)
        normal_data = []
        
        for i in range(n_samples):
            # Base patterns with different frequencies and phases
            base_signal = torch.zeros(seq_len, n_sensors)
            for sensor in range(n_sensors):
                frequency = 1 + sensor * 0.2
                phase = sensor * np.pi / 4
                amplitude = 2 + sensor * 0.5
                
                base_signal[:, sensor] = amplitude * torch.sin(frequency * t + phase)
                
                # Add correlation between adjacent sensors
                if sensor > 0:
                    correlation_strength = 0.3
                    base_signal[:, sensor] += correlation_strength * base_signal[:, sensor-1]
                
                # Add noise
                noise_level = 0.2
                base_signal[:, sensor] += noise_level * torch.randn(seq_len)
            
            normal_data.append(base_signal)
        
        # Create anomalous samples
        n_anomalies = int(n_samples * anomaly_ratio)
        anomaly_indices = torch.randperm(n_samples)[:n_anomalies]
        
        labels = torch.zeros(n_samples, dtype=torch.bool)
        labels[anomaly_indices] = True
        
        # Inject different types of anomalies
        for idx in anomaly_indices:
            anomaly_type = torch.randint(0, 3, (1,)).item()
            
            if anomaly_type == 0:
                # Spike anomaly: sudden large values
                spike_sensor = torch.randint(0, n_sensors, (1,)).item()
                spike_time = torch.randint(5, seq_len-5, (1,)).item()
                normal_data[idx][spike_time-2:spike_time+3, spike_sensor] *= 5
                
            elif anomaly_type == 1:
                # Drift anomaly: gradual change
                drift_sensor = torch.randint(0, n_sensors, (1,)).item()
                drift_magnitude = 3.0 * torch.randn(1).item()
                drift_pattern = torch.linspace(0, drift_magnitude, seq_len)
                normal_data[idx][:, drift_sensor] += drift_pattern
                
            else:
                # Correlation anomaly: break sensor correlations
                sensor_pair = torch.randperm(n_sensors)[:2]
                normal_data[idx][:, sensor_pair[1]] = torch.randn_like(normal_data[idx][:, sensor_pair[1]])
        
        data = torch.stack(normal_data)
        logger.info(f"Generated data shape: {data.shape}, anomalies: {labels.sum().item()}/{n_samples}")
        
        return data, labels
    
    def train_model(self, train_data: torch.Tensor, epochs: int = 50) -> List[float]:
        """Train the hybrid LSTM-GNN model."""
        logger.info(f"Training hybrid model for {epochs} epochs")
        
        # Initialize model
        self.model = LSTMGNNHybridModel(self.config).to(self.device)
        
        # Only train on normal data (unsupervised)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses = []
        self.model.train()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Mini-batch training
            batch_size = 32
            n_batches = (len(train_data) + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_data))
                batch_data = train_data[start_idx:end_idx].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                reconstruction = self.model(batch_data)
                
                # Reconstruction loss
                loss = nn.MSELoss()(reconstruction, batch_data)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        logger.info("Training completed")
        return train_losses
    
    def evaluate_model(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        logger.info("Evaluating model performance")
        
        self.model.eval()
        
        with torch.no_grad():
            # Compute anomaly scores
            reconstruction_errors = []
            batch_size = 32
            
            for i in range(0, len(test_data), batch_size):
                batch_data = test_data[i:i+batch_size].to(self.device)
                scores = self.model.compute_hybrid_anomaly_score(
                    batch_data, reduction='none'
                )
                # Average across sequence dimension
                sample_scores = torch.mean(scores, dim=1)
                reconstruction_errors.extend(sample_scores.cpu().numpy())
            
            reconstruction_errors = np.array(reconstruction_errors)
        
        # Find optimal threshold using ROC curve
        from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
        
        # ROC Analysis
        fpr, tpr, thresholds = roc_curve(test_labels.numpy(), reconstruction_errors)
        roc_auc = roc_auc_score(test_labels.numpy(), reconstruction_errors)
        
        # Find threshold that maximizes Youden's J statistic
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_threshold_idx]
        
        # Precision-Recall Analysis
        precision, recall, pr_thresholds = precision_recall_curve(test_labels.numpy(), reconstruction_errors)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        
        # Classification metrics at optimal threshold
        predictions = reconstruction_errors > best_threshold
        tp = np.sum((predictions == 1) & (test_labels.numpy() == 1))
        fp = np.sum((predictions == 1) & (test_labels.numpy() == 0))
        tn = np.sum((predictions == 0) & (test_labels.numpy() == 0))
        fn = np.sum((predictions == 0) & (test_labels.numpy() == 1))
        
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision_final = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_final = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_final = 2 * (precision_final * recall_final) / (precision_final + recall_final) if (precision_final + recall_final) > 0 else 0
        
        results = {
            'roc_auc': roc_auc,
            'best_threshold': best_threshold,
            'accuracy': accuracy,
            'precision': precision_final,
            'recall': recall_final,
            'f1_score': f1_final,
            'best_f1': best_f1
        }
        
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"Best F1: {best_f1:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        return results
    
    def visualize_results(self, test_data: torch.Tensor, test_labels: torch.Tensor, 
                         results: Dict[str, float]):
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations")
        
        # Create output directory
        output_dir = Path("hybrid_demo_results")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Graph structure visualization
        self._visualize_graph_structure(test_data[:10], output_dir)
        
        # 2. Anomaly detection results
        self._visualize_anomaly_detection(test_data, test_labels, results, output_dir)
        
        # 3. Feature analysis
        self._visualize_feature_analysis(test_data[:100], test_labels[:100], output_dir)
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def _visualize_graph_structure(self, sample_data: torch.Tensor, output_dir: Path):
        """Visualize the learned graph structure."""
        import networkx as nx
        
        # Create graph from sample data
        graph_data = create_sensor_graph(
            sample_data, 
            method=self.config['graph']['method'],
            threshold=self.config['graph']['threshold']
        )
        
        # Create NetworkX graph
        G = nx.Graph()
        n_sensors = sample_data.size(-1)
        G.add_nodes_from(range(n_sensors))
        
        if graph_data['edge_index'].size(1) > 0:
            edges = graph_data['edge_index'].t().numpy()
            # Remove duplicate edges for undirected graph
            unique_edges = set()
            for edge in edges:
                unique_edges.add(tuple(sorted(edge)))
            G.add_edges_from(list(unique_edges))
        
        # Plot graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Node colors based on sensor index
        node_colors = plt.cm.Set3(np.linspace(0, 1, n_sensors))
        
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=800,
                with_labels=True,
                font_size=12,
                font_weight='bold',
                edge_color='gray',
                width=2)
        
        plt.title(f"Sensor Network Graph Structure\n"
                 f"Method: {self.config['graph']['method']}, "
                 f"Threshold: {self.config['graph']['threshold']}\n"
                 f"Nodes: {n_sensors}, Edges: {G.number_of_edges()}")
        
        plt.savefig(output_dir / "graph_structure.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_anomaly_detection(self, test_data: torch.Tensor, test_labels: torch.Tensor, 
                                   results: Dict[str, float], output_dir: Path):
        """Visualize anomaly detection results."""
        self.model.eval()
        
        with torch.no_grad():
            # Compute scores for all test data
            scores = []
            batch_size = 32
            
            for i in range(0, len(test_data), batch_size):
                batch_data = test_data[i:i+batch_size].to(self.device)
                batch_scores = self.model.compute_hybrid_anomaly_score(
                    batch_data, reduction='none'
                )
                sample_scores = torch.mean(batch_scores, dim=1)
                scores.extend(sample_scores.cpu().numpy())
            
            scores = np.array(scores)
        
        # Plot 1: Score distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Score distribution
        axes[0, 0].hist(scores[~test_labels.numpy()], bins=50, alpha=0.7, 
                       label='Normal', color='blue', density=True)
        axes[0, 0].hist(scores[test_labels.numpy()], bins=50, alpha=0.7,
                       label='Anomaly', color='red', density=True)
        axes[0, 0].axvline(results['best_threshold'], color='black', 
                          linestyle='--', label=f'Threshold: {results["best_threshold"]:.3f}')
        axes[0, 0].set_xlabel('Reconstruction Error')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Anomaly Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(test_labels.numpy(), scores)
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC AUC: {results["roc_auc"]:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(test_labels.numpy(), scores)
        axes[1, 0].plot(recall, precision, linewidth=2, label=f'Best F1: {results["best_f1"]:.3f}')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time series examples
        sample_indices = torch.where(test_labels)[0][:3]  # First 3 anomalies
        
        for i, idx in enumerate(sample_indices):
            if i >= 3:
                break
            sample = test_data[idx].numpy()
            
            # Plot first 3 sensors
            for sensor in range(min(3, sample.shape[1])):
                axes[1, 1].plot(sample[:, sensor], 
                               label=f'Anomaly {i+1}, Sensor {sensor+1}', 
                               alpha=0.7)
        
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Sensor Value')
        axes[1, 1].set_title('Example Anomalous Patterns')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "anomaly_detection_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_feature_analysis(self, test_data: torch.Tensor, test_labels: torch.Tensor, 
                                   output_dir: Path):
        """Analyze learned features."""
        self.model.eval()
        
        with torch.no_grad():
            # Extract features from both LSTM and GNN
            lstm_features_list = []
            gnn_features_list = []
            
            batch_size = 32
            for i in range(0, len(test_data), batch_size):
                batch_data = test_data[i:i+batch_size].to(self.device)
                lstm_feat, gnn_feat = self.model.encode(batch_data)
                lstm_features_list.append(lstm_feat.cpu())
                gnn_features_list.append(gnn_feat.cpu())
            
            lstm_features = torch.cat(lstm_features_list, dim=0).numpy()
            gnn_features = torch.cat(gnn_features_list, dim=0).numpy()
        
        # Dimensionality reduction for visualization
        from sklearn.decomposition import PCA
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # LSTM features PCA
        pca_lstm = PCA(n_components=2)
        lstm_2d = pca_lstm.fit_transform(lstm_features)
        
        normal_mask = ~test_labels[:len(lstm_2d)].numpy()
        anomaly_mask = test_labels[:len(lstm_2d)].numpy()
        
        axes[0].scatter(lstm_2d[normal_mask, 0], lstm_2d[normal_mask, 1], 
                       c='blue', alpha=0.6, label='Normal', s=30)
        axes[0].scatter(lstm_2d[anomaly_mask, 0], lstm_2d[anomaly_mask, 1], 
                       c='red', alpha=0.8, label='Anomaly', s=30)
        axes[0].set_title('LSTM Features (PCA)')
        axes[0].set_xlabel(f'PC1 ({pca_lstm.explained_variance_ratio_[0]:.2%} var)')
        axes[0].set_ylabel(f'PC2 ({pca_lstm.explained_variance_ratio_[1]:.2%} var)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # GNN features PCA
        pca_gnn = PCA(n_components=2)
        gnn_2d = pca_gnn.fit_transform(gnn_features)
        
        axes[1].scatter(gnn_2d[normal_mask, 0], gnn_2d[normal_mask, 1], 
                       c='blue', alpha=0.6, label='Normal', s=30)
        axes[1].scatter(gnn_2d[anomaly_mask, 0], gnn_2d[anomaly_mask, 1], 
                       c='red', alpha=0.8, label='Anomaly', s=30)
        axes[1].set_title('GNN Features (PCA)')
        axes[1].set_xlabel(f'PC1 ({pca_gnn.explained_variance_ratio_[0]:.2%} var)')
        axes[1].set_ylabel(f'PC2 ({pca_gnn.explained_variance_ratio_[1]:.2%} var)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Combined features correlation
        combined_features = np.concatenate([lstm_features, gnn_features], axis=1)
        correlation_matrix = np.corrcoef(combined_features.T)
        
        n_lstm = lstm_features.shape[1]
        im = axes[2].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[2].set_title('LSTM-GNN Feature Correlations')
        axes[2].axvline(n_lstm-0.5, color='black', linewidth=2)
        axes[2].axhline(n_lstm-0.5, color='black', linewidth=2)
        axes[2].set_xlabel('Feature Index')
        axes[2].set_ylabel('Feature Index')
        
        # Add text annotations for regions
        axes[2].text(n_lstm//2, -2, 'LSTM', ha='center', fontweight='bold')
        axes[2].text(n_lstm + gnn_features.shape[1]//2, -2, 'GNN', ha='center', fontweight='bold')
        
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def benchmark_performance(self, test_data: torch.Tensor) -> Dict[str, float]:
        """Benchmark model performance metrics."""
        logger.info("Benchmarking model performance")
        
        self.model.eval()
        
        # Inference time benchmark
        inference_times = []
        batch_size = 1
        
        with torch.no_grad():
            for i in range(min(100, len(test_data))):  # Test on first 100 samples
                sample = test_data[i:i+1].to(self.device)
                
                start_time = time.time()
                _ = self.model(sample)
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
        
        # Memory usage (approximate)
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2  # MB
        
        performance_metrics = {
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'p95_inference_time_ms': np.percentile(inference_times, 95) * 1000,
            'model_size_mb': model_size,
            'parameters_count': sum(p.numel() for p in self.model.parameters())
        }
        
        logger.info(f"Average inference time: {performance_metrics['avg_inference_time_ms']:.2f}ms")
        logger.info(f"P95 inference time: {performance_metrics['p95_inference_time_ms']:.2f}ms")
        logger.info(f"Model size: {performance_metrics['model_size_mb']:.2f}MB")
        logger.info(f"Parameters: {performance_metrics['parameters_count']:,}")
        
        return performance_metrics
    
    def run_complete_demo(self):
        """Run the complete hybrid anomaly detection demonstration."""
        logger.info("=== Starting Hybrid LSTM-GNN Anomaly Detection Demo ===")
        
        # Generate data
        all_data, all_labels = self.generate_synthetic_data(n_samples=2000, n_sensors=5, seq_len=20)
        
        # Split data
        train_size = int(0.7 * len(all_data))
        val_size = int(0.15 * len(all_data))
        
        train_data = all_data[:train_size]
        val_data = all_data[train_size:train_size+val_size]
        test_data = all_data[train_size+val_size:]
        
        train_labels = all_labels[:train_size]
        val_labels = all_labels[train_size:train_size+val_size]
        test_labels = all_labels[train_size+val_size:]
        
        # Filter training data to only normal samples
        normal_mask = ~train_labels
        train_normal_data = train_data[normal_mask]
        
        logger.info(f"Data split - Train: {len(train_normal_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Train model
        train_losses = self.train_model(train_normal_data, epochs=30)
        
        # Evaluate model
        results = self.evaluate_model(test_data, test_labels)
        
        # Benchmark performance
        perf_metrics = self.benchmark_performance(test_data)
        
        # Create visualizations
        self.visualize_results(test_data, test_labels, results)
        
        # Save results
        output_dir = Path("hybrid_demo_results")
        results_summary = {
            **results,
            **perf_metrics,
            'config': self.config
        }
        
        import json
        with open(output_dir / "results_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info("=== Demo Completed Successfully ===")
        logger.info(f"Results saved to {output_dir}")
        
        return results_summary


if __name__ == "__main__":
    demo = HybridAnomalyDetectionDemo()
    results = demo.run_complete_demo()
    
    print("\n" + "="*50)
    print("HYBRID LSTM-GNN DEMO RESULTS")
    print("="*50)
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Average Inference Time: {results['avg_inference_time_ms']:.2f}ms")
    print(f"Model Size: {results['model_size_mb']:.2f}MB")
    print("="*50)
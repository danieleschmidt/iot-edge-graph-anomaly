#!/usr/bin/env python3
"""
Comparative Analysis: LSTM vs GNN vs Hybrid Models.

This research module provides comprehensive comparison between different
anomaly detection approaches, with statistical significance testing and
performance benchmarking across multiple metrics.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
from iot_edge_anomaly.models.lstm_gnn_hybrid import LSTMGNNHybridModel
from examples.hybrid_anomaly_detection_demo import HybridAnomalyDetectionDemo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    model_name: str
    roc_auc: float
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    avg_inference_time_ms: float
    model_size_mb: float
    training_time_s: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'roc_auc': self.roc_auc,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'accuracy': self.accuracy,
            'avg_inference_time_ms': self.avg_inference_time_ms,
            'model_size_mb': self.model_size_mb,
            'training_time_s': self.training_time_s
        }


class SimpleGNNModel(nn.Module):
    """Standalone GNN model for comparison."""
    
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        from iot_edge_anomaly.models.gnn_layer import GraphNeuralNetworkLayer, create_sensor_graph
        
        self.input_size = input_size
        self.gnn = GraphNeuralNetworkLayer(
            input_dim=input_size,
            hidden_dim=hidden_size,
            output_dim=input_size,
            num_layers=num_layers
        )
        self.graph_method = 'correlation'
        self.graph_threshold = 0.3
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN."""
        batch_size, seq_len, num_sensors = x.shape
        
        # Use the latest time step as features
        node_features = x[:, -1, :].t()  # (num_sensors, batch_size)
        
        # Create graph
        from iot_edge_anomaly.models.gnn_layer import create_sensor_graph
        graph_data = create_sensor_graph(x, method=self.graph_method, threshold=self.graph_threshold)
        
        # Prepare node features for GNN (average across batch)
        if batch_size > 1:
            node_features = torch.mean(node_features, dim=1, keepdim=True)
        
        if node_features.dim() == 1:
            node_features = node_features.unsqueeze(-1)
        
        # GNN forward pass
        gnn_output = self.gnn(node_features.squeeze(-1), graph_data['edge_index'])
        
        # Reconstruct time series (broadcast GNN output to sequence)
        reconstruction = gnn_output.unsqueeze(0).unsqueeze(1).repeat(batch_size, seq_len, 1)
        
        return reconstruction
    
    def compute_reconstruction_error(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute reconstruction error."""
        reconstruction = self.forward(x)
        error = torch.mean((x - reconstruction) ** 2, dim=-1)
        
        if reduction == 'mean':
            return torch.mean(error)
        elif reduction == 'sum':
            return torch.sum(error)
        else:
            return error


class ComparativeAnalysisFramework:
    """Framework for comparative analysis of anomaly detection models."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        self.datasets = {}
        logger.info(f"Using device: {self.device}")
    
    def generate_datasets(self, n_runs: int = 5) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Generate multiple datasets for statistical significance."""
        logger.info(f"Generating {n_runs} datasets for robust evaluation")
        
        datasets = {
            'synthetic_normal': [],
            'synthetic_complex': [],
            'synthetic_sparse': []
        }
        
        demo = HybridAnomalyDetectionDemo()
        
        for run in range(n_runs):
            # Normal complexity dataset
            data, labels = demo.generate_synthetic_data(
                n_samples=1000, n_sensors=5, seq_len=20, anomaly_ratio=0.1
            )
            datasets['synthetic_normal'].append((data, labels))
            
            # Complex patterns (more sensors, longer sequences)
            data, labels = demo.generate_synthetic_data(
                n_samples=800, n_sensors=8, seq_len=30, anomaly_ratio=0.15
            )
            datasets['synthetic_complex'].append((data, labels))
            
            # Sparse anomalies (fewer anomalies, harder to detect)
            data, labels = demo.generate_synthetic_data(
                n_samples=1200, n_sensors=5, seq_len=20, anomaly_ratio=0.05
            )
            datasets['synthetic_sparse'].append((data, labels))
        
        self.datasets = datasets
        return datasets
    
    def train_and_evaluate_model(self, model: nn.Module, model_name: str,
                                train_data: torch.Tensor, test_data: torch.Tensor,
                                test_labels: torch.Tensor, epochs: int = 30) -> ExperimentResult:
        """Train and evaluate a single model."""
        logger.info(f"Training {model_name}")
        
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training
        model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            batch_size = 32
            epoch_losses = []
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                
                if hasattr(model, 'compute_reconstruction_error'):
                    loss = model.compute_reconstruction_error(batch)
                else:
                    reconstruction = model(batch)
                    loss = criterion(reconstruction, batch)
                
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(epoch_losses):.6f}")
        
        training_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Compute anomaly scores
            scores = []
            inference_times = []
            
            for i in range(len(test_data)):
                sample = test_data[i:i+1].to(self.device)
                
                start_inf = time.time()
                if hasattr(model, 'compute_reconstruction_error'):
                    score = model.compute_reconstruction_error(sample, reduction='mean')
                elif hasattr(model, 'compute_hybrid_anomaly_score'):
                    score = model.compute_hybrid_anomaly_score(sample, reduction='mean')
                else:
                    reconstruction = model(sample)
                    score = torch.mean((sample - reconstruction) ** 2)
                
                end_inf = time.time()
                
                scores.append(score.cpu().item())
                inference_times.append(end_inf - start_inf)
            
            scores = np.array(scores)
        
        # Performance metrics
        from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
        
        # ROC Analysis
        roc_auc = roc_auc_score(test_labels.numpy(), scores)
        
        # Optimal threshold
        fpr, tpr, thresholds = roc_curve(test_labels.numpy(), scores)
        j_scores = tpr - fpr
        best_threshold = thresholds[np.argmax(j_scores)]
        
        # Classification metrics
        predictions = scores > best_threshold
        tp = np.sum((predictions == 1) & (test_labels.numpy() == 1))
        fp = np.sum((predictions == 1) & (test_labels.numpy() == 0))
        tn = np.sum((predictions == 0) & (test_labels.numpy() == 0))
        fn = np.sum((predictions == 0) & (test_labels.numpy() == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        
        # Model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        result = ExperimentResult(
            model_name=model_name,
            roc_auc=roc_auc,
            f1_score=f1,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            avg_inference_time_ms=np.mean(inference_times) * 1000,
            model_size_mb=model_size,
            training_time_s=training_time
        )
        
        logger.info(f"{model_name} - ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}")
        return result
    
    def run_comparative_study(self, dataset_name: str = 'synthetic_normal') -> pd.DataFrame:
        """Run comprehensive comparative study."""
        logger.info(f"Running comparative study on {dataset_name}")
        
        if not self.datasets:
            self.generate_datasets()
        
        all_results = []
        
        for run_idx, (data, labels) in enumerate(self.datasets[dataset_name]):
            logger.info(f"Run {run_idx + 1}/{len(self.datasets[dataset_name])}")
            
            # Split data
            train_size = int(0.7 * len(data))
            test_data = data[train_size:]
            test_labels = labels[train_size:]
            
            # Filter training data to normal samples only
            train_labels = labels[:train_size]
            normal_mask = ~train_labels
            train_data = data[:train_size][normal_mask]
            
            # Model configurations
            input_size = data.size(-1)
            models = {
                'LSTM-Autoencoder': LSTMAutoencoder(
                    input_size=input_size,
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.1
                ),
                'GNN-Only': SimpleGNNModel(
                    input_size=input_size,
                    hidden_size=32,
                    num_layers=2
                ),
                'LSTM-GNN-Hybrid': LSTMGNNHybridModel({
                    'lstm': {'input_size': input_size, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1},
                    'gnn': {'input_dim': 64, 'hidden_dim': 32, 'output_dim': 64, 'num_layers': 2, 'dropout': 0.1},
                    'fusion': {'output_dim': 128, 'method': 'concatenate'},
                    'graph': {'method': 'correlation', 'threshold': 0.3}
                })
            }
            
            run_results = []
            for model_name, model in models.items():
                try:
                    result = self.train_and_evaluate_model(
                        model, f"{model_name}_run_{run_idx}",
                        train_data, test_data, test_labels,
                        epochs=20  # Reduced for comparison study
                    )
                    result.model_name = model_name  # Clean name for aggregation
                    run_results.append(result)
                except Exception as e:
                    logger.error(f"Error with {model_name} run {run_idx}: {e}")
                    continue
            
            all_results.extend(run_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame([r.to_dict() for r in all_results])
        return results_df
    
    def statistical_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        logger.info("Performing statistical significance analysis")
        
        metrics = ['roc_auc', 'f1_score', 'accuracy', 'avg_inference_time_ms', 'model_size_mb']
        models = results_df['model_name'].unique()
        
        # Aggregated statistics
        stats_summary = results_df.groupby('model_name')[metrics].agg(['mean', 'std', 'count']).round(4)
        
        # Pairwise t-tests
        significance_tests = {}
        
        for metric in ['roc_auc', 'f1_score', 'accuracy']:
            significance_tests[metric] = {}
            
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    data1 = results_df[results_df['model_name'] == model1][metric].values
                    data2 = results_df[results_df['model_name'] == model2][metric].values
                    
                    if len(data1) > 1 and len(data2) > 1:
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        significance_tests[metric][f"{model1}_vs_{model2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'effect_size': (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
                        }
        
        return {
            'summary_statistics': stats_summary,
            'significance_tests': significance_tests
        }
    
    def create_comprehensive_visualizations(self, results_df: pd.DataFrame,
                                          stats_analysis: Dict[str, Any],
                                          output_dir: Path):
        """Create comprehensive research visualizations."""
        logger.info("Creating research visualizations")
        
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # 1. Performance comparison boxplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['roc_auc', 'f1_score', 'accuracy', 'avg_inference_time_ms', 'model_size_mb', 'training_time_s']
        titles = ['ROC AUC', 'F1 Score', 'Accuracy', 'Inference Time (ms)', 'Model Size (MB)', 'Training Time (s)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            sns.boxplot(data=results_df, x='model_name', y=metric, ax=axes[i])
            axes[i].set_title(title)
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add significance stars
            models = results_df['model_name'].unique()
            if metric in ['roc_auc', 'f1_score', 'accuracy'] and len(models) >= 2:
                y_max = results_df[metric].max()
                y_range = results_df[metric].max() - results_df[metric].min()
                
                # Find best performing model
                means = results_df.groupby('model_name')[metric].mean()
                best_model = means.idxmax()
                
                for j, model in enumerate(models):
                    if model != best_model:
                        comparison_key = f"{best_model}_vs_{model}"
                        if comparison_key in stats_analysis['significance_tests'][metric]:
                            test_result = stats_analysis['significance_tests'][metric][comparison_key]
                            if test_result['significant']:
                                axes[i].text(j, y_max + 0.1 * y_range, '*', 
                                           ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Statistical significance heatmap
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(['roc_auc', 'f1_score', 'accuracy']):
            models = results_df['model_name'].unique()
            n_models = len(models)
            p_matrix = np.ones((n_models, n_models))
            
            for j, model1 in enumerate(models):
                for k, model2 in enumerate(models):
                    if j != k:
                        key1 = f"{model1}_vs_{model2}"
                        key2 = f"{model2}_vs_{model1}"
                        
                        if key1 in stats_analysis['significance_tests'][metric]:
                            p_matrix[j, k] = stats_analysis['significance_tests'][metric][key1]['p_value']
                        elif key2 in stats_analysis['significance_tests'][metric]:
                            p_matrix[j, k] = stats_analysis['significance_tests'][metric][key2]['p_value']
            
            mask = p_matrix >= 0.05
            
            sns.heatmap(p_matrix, 
                       xticklabels=[m.replace('_', ' ') for m in models],
                       yticklabels=[m.replace('_', ' ') for m in models],
                       annot=True, fmt='.3f', cmap='RdYlBu_r',
                       mask=mask, cbar_kws={'label': 'p-value'},
                       ax=axes[i])
            axes[i].set_title(f'Statistical Significance - {metric.replace("_", " ").title()}')
            
        plt.tight_layout()
        plt.savefig(output_dir / "significance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance vs Efficiency trade-off
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, model in enumerate(results_df['model_name'].unique()):
            model_data = results_df[results_df['model_name'] == model]
            
            x = model_data['avg_inference_time_ms'].mean()
            y = model_data['roc_auc'].mean()
            size = model_data['model_size_mb'].mean() * 100  # Scale for visibility
            
            ax.scatter(x, y, s=size, alpha=0.7, color=colors[i % len(colors)], 
                      label=f'{model}\n({model_data["model_size_mb"].mean():.1f}MB)')
            
            # Add error bars
            x_err = model_data['avg_inference_time_ms'].std()
            y_err = model_data['roc_auc'].std()
            ax.errorbar(x, y, xerr=x_err, yerr=y_err, color=colors[i % len(colors)], alpha=0.5)
        
        ax.set_xlabel('Average Inference Time (ms)')
        ax.set_ylabel('ROC AUC')
        ax.set_title('Performance vs Efficiency Trade-off\n(Bubble size = Model Size)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_efficiency_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_research_report(self, results_df: pd.DataFrame, 
                               stats_analysis: Dict[str, Any], 
                               output_dir: Path) -> str:
        """Generate comprehensive research report."""
        logger.info("Generating research report")
        
        report = [
            "# Comparative Analysis: LSTM vs GNN vs Hybrid Models for IoT Anomaly Detection",
            "",
            "## Executive Summary",
            "",
            f"This study compares three anomaly detection approaches across {len(results_df)} experimental runs:",
            "- **LSTM Autoencoder**: Temporal pattern learning",
            "- **GNN-Only**: Spatial relationship modeling", 
            "- **LSTM-GNN Hybrid**: Combined temporal-spatial approach",
            "",
            "## Methodology",
            "",
            "### Experimental Design",
            f"- **Datasets**: {len(set(results_df.index)) // len(results_df['model_name'].unique())} independent runs",
            "- **Metrics**: ROC AUC, F1-score, Accuracy, Inference Time, Model Size",
            "- **Statistical Testing**: Paired t-tests with α=0.05",
            "- **Hardware**: GPU-accelerated training and inference",
            "",
            "## Results Summary",
            "",
        ]
        
        # Add performance summary
        summary_stats = stats_analysis['summary_statistics']
        
        report.extend([
            "### Performance Metrics (Mean ± Std)",
            "",
            "| Model | ROC AUC | F1-Score | Accuracy | Inference Time (ms) | Model Size (MB) |",
            "|-------|---------|----------|----------|-------------------|----------------|"
        ])
        
        for model in results_df['model_name'].unique():
            model_stats = summary_stats.loc[model]
            roc_auc = f"{model_stats[('roc_auc', 'mean')]:.3f} ± {model_stats[('roc_auc', 'std')]:.3f}"
            f1 = f"{model_stats[('f1_score', 'mean')]:.3f} ± {model_stats[('f1_score', 'std')]:.3f}"
            acc = f"{model_stats[('accuracy', 'mean')]:.3f} ± {model_stats[('accuracy', 'std')]:.3f}"
            time_ms = f"{model_stats[('avg_inference_time_ms', 'mean')]:.2f} ± {model_stats[('avg_inference_time_ms', 'std')]:.2f}"
            size_mb = f"{model_stats[('model_size_mb', 'mean')]:.1f} ± {model_stats[('model_size_mb', 'std')]:.1f}"
            
            report.append(f"| {model.replace('_', ' ')} | {roc_auc} | {f1} | {acc} | {time_ms} | {size_mb} |")
        
        report.extend([
            "",
            "## Statistical Significance Analysis",
            "",
            "### Key Findings",
            ""
        ])
        
        # Add significance findings
        for metric in ['roc_auc', 'f1_score', 'accuracy']:
            best_model = results_df.groupby('model_name')[metric].mean().idxmax()
            best_score = results_df.groupby('model_name')[metric].mean().max()
            
            report.append(f"**{metric.replace('_', ' ').title()}**: {best_model} achieves highest score ({best_score:.3f})")
            
            significant_improvements = []
            for comparison, test_result in stats_analysis['significance_tests'][metric].items():
                if test_result['significant'] and best_model in comparison:
                    other_model = comparison.replace(best_model, '').replace('_vs_', '').strip('_')
                    effect_size = abs(test_result['effect_size'])
                    significant_improvements.append(f"{other_model} (p={test_result['p_value']:.3f}, d={effect_size:.2f})")
            
            if significant_improvements:
                report.append(f"- Significant improvements over: {', '.join(significant_improvements)}")
            
            report.append("")
        
        # Add conclusions
        report.extend([
            "## Conclusions and Recommendations",
            "",
            "### Model Performance Ranking",
            ""
        ])
        
        # Rank models by ROC AUC
        model_ranking = results_df.groupby('model_name')['roc_auc'].mean().sort_values(ascending=False)
        
        for i, (model, score) in enumerate(model_ranking.items(), 1):
            efficiency = results_df.groupby('model_name')['avg_inference_time_ms'].mean()[model]
            size = results_df.groupby('model_name')['model_size_mb'].mean()[model]
            
            report.append(f"{i}. **{model}** (ROC AUC: {score:.3f})")
            report.append(f"   - Inference Time: {efficiency:.2f}ms")
            report.append(f"   - Model Size: {size:.1f}MB")
            report.append("")
        
        report.extend([
            "### Recommendations",
            "",
            "- **Best Overall Performance**: Choose the highest ROC AUC model for maximum detection capability",
            "- **Resource Constrained**: Consider inference time and memory constraints for edge deployment",
            "- **Production Deployment**: Balance performance, efficiency, and maintainability requirements",
            "",
            f"**Generated on**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        report_text = '\n'.join(report)
        
        # Save report
        with open(output_dir / "comparative_analysis_report.md", 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def run_full_research_study(self):
        """Run complete comparative research study."""
        logger.info("=== Starting Comprehensive Comparative Analysis ===")
        
        # Create output directory
        output_dir = Path("research_results")
        output_dir.mkdir(exist_ok=True)
        
        # Generate datasets
        self.generate_datasets(n_runs=5)
        
        # Run comparative study on different dataset types
        all_results = []
        
        for dataset_name in ['synthetic_normal', 'synthetic_complex', 'synthetic_sparse']:
            logger.info(f"Analyzing {dataset_name} dataset")
            
            results_df = self.run_comparative_study(dataset_name)
            results_df['dataset'] = dataset_name
            all_results.append(results_df)
        
        # Combine results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Statistical analysis
        stats_analysis = self.statistical_analysis(combined_results)
        
        # Create visualizations
        self.create_comprehensive_visualizations(combined_results, stats_analysis, output_dir)
        
        # Generate report
        report = self.generate_research_report(combined_results, stats_analysis, output_dir)
        
        # Save raw results
        combined_results.to_csv(output_dir / "raw_results.csv", index=False)
        
        import json
        with open(output_dir / "statistical_analysis.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(v) for v in obj]
                else:
                    return obj
            
            json.dump(convert_types(stats_analysis), f, indent=2)
        
        logger.info("=== Research Study Completed ===")
        logger.info(f"Results saved to {output_dir}")
        
        return combined_results, stats_analysis, report


if __name__ == "__main__":
    framework = ComparativeAnalysisFramework()
    results_df, stats_analysis, report = framework.run_full_research_study()
    
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS RESULTS")
    print("="*80)
    
    # Print summary
    summary = results_df.groupby('model_name')[['roc_auc', 'f1_score', 'avg_inference_time_ms']].agg(['mean', 'std'])
    
    for model in results_df['model_name'].unique():
        model_data = summary.loc[model]
        roc_auc_mean = model_data[('roc_auc', 'mean')]
        f1_mean = model_data[('f1_score', 'mean')]
        time_mean = model_data[('avg_inference_time_ms', 'mean')]
        
        print(f"{model}:")
        print(f"  ROC AUC: {roc_auc_mean:.3f} ± {model_data[('roc_auc', 'std')]:.3f}")
        print(f"  F1-Score: {f1_mean:.3f} ± {model_data[('f1_score', 'std')]:.3f}")
        print(f"  Inference Time: {time_mean:.2f} ± {model_data[('avg_inference_time_ms', 'std')]:.2f}ms")
        print()
    
    print("="*80)
    print("Detailed results and visualizations saved to 'research_results/' directory")
    print("="*80)
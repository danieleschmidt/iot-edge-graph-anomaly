"""
Comprehensive evaluation and benchmarking for sentiment analysis models.

Provides research-grade evaluation with statistical significance testing,
ablation studies, and comparative analysis capabilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import logging
from pathlib import Path
import json
from collections import defaultdict, OrderedDict
import warnings
from dataclasses import dataclass
import copy

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    inference_time: float
    memory_usage: float
    model_size: int
    training_time: Optional[float] = None
    energy_consumption: Optional[float] = None
    

class SentimentEvaluator:
    """
    Comprehensive evaluation framework for sentiment analysis models.
    
    Provides research-grade evaluation with multiple metrics, statistical tests,
    and detailed analysis capabilities.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        num_classes: int = 3,
        class_names: Optional[List[str]] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        if len(self.class_names) != num_classes:
            raise ValueError(f"Number of class names ({len(self.class_names)}) must match num_classes ({num_classes})")
    
    def evaluate_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        model_name: str = "Model",
        return_predictions: bool = False,
        compute_feature_importance: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation
            model_name: Name for the model
            return_predictions: Whether to return predictions and labels
            compute_feature_importance: Whether to compute feature importance
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        model.eval()
        model.to(self.device)
        
        all_predictions = []
        all_labels = []
        all_logits = []
        inference_times = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Measure inference time
                start_time = time.time()
                logits = model(input_ids, attention_mask)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Store results
                predictions = torch.argmax(logits, dim=-1)
                probabilities = torch.softmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        logits = np.array(all_logits)
        
        # Calculate basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # AUC scores
        try:
            if self.num_classes == 2:
                auc = roc_auc_score(labels, logits[:, 1])
                avg_precision = average_precision_score(labels, logits[:, 1])
            else:
                auc = roc_auc_score(labels, logits, multi_class='ovr', average='weighted')
                avg_precision = average_precision_score(labels, logits, average='weighted')
        except ValueError:
            auc = 0.0
            avg_precision = 0.0
        
        # Performance metrics
        avg_inference_time = np.mean(inference_times)
        total_inference_time = np.sum(inference_times)
        
        # Model size estimation
        model_size = sum(p.numel() for p in model.parameters())
        
        # Memory usage estimation
        memory_usage = self._estimate_memory_usage(model)
        
        # Classification report
        classification_rep = classification_report(
            labels, predictions,
            target_names=self.class_names[:len(np.unique(labels))],
            output_dict=True,
            zero_division=0
        )
        
        # Detailed results
        results = {
            'model_name': model_name,
            'overall_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc,
                'average_precision': avg_precision
            },
            'per_class_metrics': {
                'precision': per_class_precision,
                'recall': per_class_recall,
                'f1_score': per_class_f1,
                'support': per_class_support
            },
            'confusion_matrix': cm,
            'classification_report': classification_rep,
            'performance_metrics': {
                'avg_inference_time': avg_inference_time,
                'total_inference_time': total_inference_time,
                'throughput': len(labels) / total_inference_time,
                'model_size': model_size,
                'memory_usage_mb': memory_usage
            },
            'data_statistics': {
                'num_samples': len(labels),
                'class_distribution': np.bincount(labels),
                'class_proportions': np.bincount(labels) / len(labels)
            }
        }
        
        # Add predictions if requested
        if return_predictions:
            results['predictions'] = {
                'labels': labels,
                'predictions': predictions,
                'logits': logits,
                'probabilities': logits
            }
        
        # Add feature importance if requested
        if compute_feature_importance:
            results['feature_importance'] = self._compute_feature_importance(
                model, data_loader
            )
        
        return results
    
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate model memory usage in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        return model_size_mb
    
    def _compute_feature_importance(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Compute feature importance using gradient-based methods.
        
        Note: This is a simplified implementation. More sophisticated methods
        like integrated gradients would be used in practice.
        """
        logger.info("Computing feature importance...")
        
        model.eval()
        model.requires_grad_(True)
        
        importance_scores = []
        sample_count = 0
        
        for batch in data_loader:
            if sample_count >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            input_ids.requires_grad_(True)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Compute gradients with respect to input
            target_class = torch.argmax(logits, dim=-1)
            loss = torch.gather(logits, 1, target_class.unsqueeze(1)).sum()
            loss.backward()
            
            # Store gradients as importance scores
            if input_ids.grad is not None:
                importance = torch.abs(input_ids.grad).mean(dim=0).cpu().numpy()
                importance_scores.append(importance)
            
            sample_count += input_ids.size(0)
            
            # Clear gradients
            model.zero_grad()
            input_ids.grad = None
        
        if importance_scores:
            avg_importance = np.mean(importance_scores, axis=0)
            return {
                'token_importance': avg_importance,
                'max_importance_positions': np.argsort(avg_importance)[-10:],
                'importance_statistics': {
                    'mean': np.mean(avg_importance),
                    'std': np.std(avg_importance),
                    'max': np.max(avg_importance),
                    'min': np.min(avg_importance)
                }
            }
        else:
            return {}
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data_loader: DataLoader,
        num_runs: int = 3,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare multiple models with statistical significance testing.
        
        Args:
            models: Dictionary of model name -> model instance
            data_loader: DataLoader for evaluation
            num_runs: Number of evaluation runs for statistical testing
            significance_level: P-value threshold for significance
            
        Returns:
            Comprehensive comparison results
        """
        logger.info(f"Comparing {len(models)} models with {num_runs} runs each...")
        
        all_results = defaultdict(list)
        detailed_results = {}
        
        # Run multiple evaluations for each model
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            
            model_results = []
            for run in range(num_runs):
                logger.info(f"  Run {run + 1}/{num_runs}")
                
                # Create a fresh copy of the model for each run
                model_copy = copy.deepcopy(model)
                
                # Evaluate
                result = self.evaluate_model(
                    model_copy, data_loader, f"{model_name}_run_{run}"
                )
                
                model_results.append(result)
                
                # Store key metrics for statistical analysis
                all_results[model_name].append({
                    'accuracy': result['overall_metrics']['accuracy'],
                    'precision': result['overall_metrics']['precision'],
                    'recall': result['overall_metrics']['recall'],
                    'f1_score': result['overall_metrics']['f1_score'],
                    'auc_score': result['overall_metrics']['auc_score'],
                    'inference_time': result['performance_metrics']['avg_inference_time'],
                    'throughput': result['performance_metrics']['throughput']
                })
            
            # Aggregate results for this model
            detailed_results[model_name] = self._aggregate_model_results(model_results)
        
        # Perform statistical comparisons
        statistical_results = self._perform_statistical_comparisons(
            all_results, significance_level
        )
        
        # Create rankings
        rankings = self._create_model_rankings(detailed_results)
        
        # Generate summary
        summary = self._generate_comparison_summary(
            detailed_results, statistical_results, rankings
        )
        
        return {
            'detailed_results': detailed_results,
            'statistical_comparisons': statistical_results,
            'rankings': rankings,
            'summary': summary,
            'num_runs': num_runs,
            'significance_level': significance_level
        }
    
    def _aggregate_model_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple runs."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        performance_metrics = ['avg_inference_time', 'throughput']
        
        aggregated = {'overall_metrics': {}, 'performance_metrics': {}}
        
        # Aggregate overall metrics
        for metric in metrics:
            values = [r['overall_metrics'][metric] for r in results]
            aggregated['overall_metrics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Aggregate performance metrics
        for metric in performance_metrics:
            values = [r['performance_metrics'][metric] for r in results]
            aggregated['performance_metrics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Use first run for non-statistical metrics
        aggregated['confusion_matrix'] = results[0]['confusion_matrix']
        aggregated['classification_report'] = results[0]['classification_report']
        aggregated['model_size'] = results[0]['performance_metrics']['model_size']
        aggregated['memory_usage_mb'] = results[0]['performance_metrics']['memory_usage_mb']
        
        return aggregated
    
    def _perform_statistical_comparisons(
        self,
        all_results: Dict[str, List[Dict[str, float]]],
        significance_level: float
    ) -> Dict[str, Dict[str, Any]]:
        """Perform pairwise statistical comparisons."""
        model_names = list(all_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        
        comparisons = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                comparisons[comparison_key] = {}
                
                for metric in metrics:
                    values1 = [run[metric] for run in all_results[model1]]
                    values2 = [run[metric] for run in all_results[model2]]
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        ((len(values1) - 1) * np.var(values1, ddof=1) + 
                         (len(values2) - 1) * np.var(values2, ddof=1)) /
                        (len(values1) + len(values2) - 2)
                    )
                    
                    cohens_d = ((np.mean(values1) - np.mean(values2)) / pooled_std 
                               if pooled_std > 0 else 0)
                    
                    # Determine significance
                    is_significant = p_value < significance_level
                    
                    # Determine practical significance (effect size)
                    if abs(cohens_d) < 0.2:
                        effect_magnitude = "negligible"
                    elif abs(cohens_d) < 0.5:
                        effect_magnitude = "small"
                    elif abs(cohens_d) < 0.8:
                        effect_magnitude = "medium"
                    else:
                        effect_magnitude = "large"
                    
                    comparisons[comparison_key][metric] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'is_significant': is_significant,
                        'effect_size': cohens_d,
                        'effect_magnitude': effect_magnitude,
                        'mean_difference': np.mean(values1) - np.mean(values2),
                        'confidence_interval': stats.t.interval(
                            1 - significance_level,
                            len(values1) + len(values2) - 2,
                            np.mean(values1) - np.mean(values2),
                            pooled_std * np.sqrt(1/len(values1) + 1/len(values2))
                        ) if pooled_std > 0 else (0, 0)
                    }
        
        return comparisons
    
    def _create_model_rankings(self, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create model rankings for different metrics."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        performance_metrics = ['avg_inference_time', 'throughput']
        
        rankings = {}
        
        # Rank by effectiveness metrics (higher is better)
        for metric in metrics:
            model_scores = {
                model: results['overall_metrics'][metric]['mean']
                for model, results in detailed_results.items()
            }
            rankings[metric] = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Rank by efficiency metrics
        rankings['inference_speed'] = sorted(
            [(model, results['performance_metrics']['avg_inference_time']['mean'])
             for model, results in detailed_results.items()],
            key=lambda x: x[1]  # Lower is better for inference time
        )
        
        rankings['throughput'] = sorted(
            [(model, results['performance_metrics']['throughput']['mean'])
             for model, results in detailed_results.items()],
            key=lambda x: x[1], reverse=True  # Higher is better for throughput
        )
        
        rankings['model_size'] = sorted(
            [(model, results['model_size'])
             for model, results in detailed_results.items()],
            key=lambda x: x[1]  # Smaller is better
        )
        
        rankings['memory_usage'] = sorted(
            [(model, results['memory_usage_mb'])
             for model, results in detailed_results.items()],
            key=lambda x: x[1]  # Lower is better
        )
        
        # Create overall ranking (weighted combination)
        overall_scores = {}
        for model in detailed_results.keys():
            # Weighted score: 40% accuracy, 30% f1, 20% inference speed, 10% model size
            acc_score = detailed_results[model]['overall_metrics']['accuracy']['mean']
            f1_score = detailed_results[model]['overall_metrics']['f1_score']['mean']
            
            # Normalize inference time (convert to score where higher is better)
            inf_times = [r['performance_metrics']['avg_inference_time']['mean'] 
                        for r in detailed_results.values()]
            max_time = max(inf_times)
            inf_score = (max_time - detailed_results[model]['performance_metrics']['avg_inference_time']['mean']) / max_time
            
            # Normalize model size (convert to score where higher is better)
            sizes = [r['model_size'] for r in detailed_results.values()]
            max_size = max(sizes)
            size_score = (max_size - detailed_results[model]['model_size']) / max_size
            
            overall_score = (0.4 * acc_score + 0.3 * f1_score + 
                           0.2 * inf_score + 0.1 * size_score)
            overall_scores[model] = overall_score
        
        rankings['overall'] = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _generate_comparison_summary(
        self,
        detailed_results: Dict[str, Any],
        statistical_results: Dict[str, Dict[str, Any]],
        rankings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a comprehensive comparison summary."""
        
        best_models = {
            'accuracy': rankings['accuracy'][0][0],
            'f1_score': rankings['f1_score'][0][0],
            'inference_speed': rankings['inference_speed'][0][0],
            'model_size': rankings['model_size'][0][0],
            'overall': rankings['overall'][0][0]
        }
        
        # Count significant differences
        significant_comparisons = 0
        total_comparisons = 0
        
        for comparison_data in statistical_results.values():
            for metric_data in comparison_data.values():
                total_comparisons += 1
                if metric_data['is_significant']:
                    significant_comparisons += 1
        
        # Find models with consistent performance
        consistency_scores = {}
        for model, results in detailed_results.items():
            # Calculate coefficient of variation for key metrics
            acc_cv = results['overall_metrics']['accuracy']['std'] / results['overall_metrics']['accuracy']['mean']
            f1_cv = results['overall_metrics']['f1_score']['std'] / results['overall_metrics']['f1_score']['mean']
            consistency_scores[model] = (acc_cv + f1_cv) / 2  # Lower is better
        
        most_consistent = min(consistency_scores, key=consistency_scores.get)
        
        return {
            'best_models': best_models,
            'most_consistent_model': most_consistent,
            'statistical_significance': {
                'significant_comparisons': significant_comparisons,
                'total_comparisons': total_comparisons,
                'significance_rate': significant_comparisons / total_comparisons if total_comparisons > 0 else 0
            },
            'key_findings': self._generate_key_findings(detailed_results, rankings),
            'recommendations': self._generate_recommendations(detailed_results, best_models)
        }
    
    def _generate_key_findings(
        self,
        detailed_results: Dict[str, Any],
        rankings: Dict[str, Any]
    ) -> List[str]:
        """Generate key findings from the comparison."""
        findings = []
        
        # Best performing model
        best_overall = rankings['overall'][0]
        findings.append(f"Best overall model: {best_overall[0]} (score: {best_overall[1]:.4f})")
        
        # Performance spread
        acc_scores = [results['overall_metrics']['accuracy']['mean'] 
                     for results in detailed_results.values()]
        acc_spread = max(acc_scores) - min(acc_scores)
        findings.append(f"Accuracy spread: {acc_spread:.4f} ({min(acc_scores):.4f} - {max(acc_scores):.4f})")
        
        # Speed comparison
        best_speed = rankings['inference_speed'][0]
        worst_speed = rankings['inference_speed'][-1]
        speedup = worst_speed[1] / best_speed[1]
        findings.append(f"Fastest model: {best_speed[0]} ({best_speed[1]:.4f}ms, {speedup:.1f}x faster than slowest)")
        
        # Size comparison
        smallest = rankings['model_size'][0]
        largest = rankings['model_size'][-1]
        size_ratio = largest[1] / smallest[1]
        findings.append(f"Smallest model: {smallest[0]} ({smallest[1]/1e6:.1f}M parameters, {size_ratio:.1f}x smaller than largest)")
        
        return findings
    
    def _generate_recommendations(
        self,
        detailed_results: Dict[str, Any],
        best_models: Dict[str, str]
    ) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        # Overall recommendation
        overall_best = best_models['overall']
        recommendations.append(f"For general use: {overall_best} provides the best balance of accuracy and efficiency")
        
        # Speed-critical applications
        speed_best = best_models['inference_speed']
        recommendations.append(f"For speed-critical applications: {speed_best} offers fastest inference")
        
        # Resource-constrained environments
        size_best = best_models['model_size']
        recommendations.append(f"For resource-constrained environments: {size_best} has the smallest footprint")
        
        # Accuracy-critical applications
        acc_best = best_models['accuracy']
        recommendations.append(f"For accuracy-critical applications: {acc_best} provides highest accuracy")
        
        return recommendations
    
    def plot_comparison_results(
        self,
        comparison_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Create comprehensive visualization of comparison results."""
        detailed_results = comparison_results['detailed_results']
        rankings = comparison_results['rankings']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        model_names = list(detailed_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        
        # Plot 1: Accuracy comparison with error bars
        ax = axes[0]
        acc_means = [detailed_results[m]['overall_metrics']['accuracy']['mean'] for m in model_names]
        acc_stds = [detailed_results[m]['overall_metrics']['accuracy']['std'] for m in model_names]
        
        bars = ax.bar(model_names, acc_means, yerr=acc_stds, capsize=5, alpha=0.7)
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        # Color bars by rank
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(model_names)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: F1 Score comparison
        ax = axes[1]
        f1_means = [detailed_results[m]['overall_metrics']['f1_score']['mean'] for m in model_names]
        f1_stds = [detailed_results[m]['overall_metrics']['f1_score']['std'] for m in model_names]
        
        ax.bar(model_names, f1_means, yerr=f1_stds, capsize=5, alpha=0.7, color='skyblue')
        ax.set_title('F1 Score Comparison')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Inference time comparison
        ax = axes[2]
        time_means = [detailed_results[m]['performance_metrics']['avg_inference_time']['mean'] * 1000 
                     for m in model_names]  # Convert to ms
        time_stds = [detailed_results[m]['performance_metrics']['avg_inference_time']['std'] * 1000 
                    for m in model_names]
        
        ax.bar(model_names, time_means, yerr=time_stds, capsize=5, alpha=0.7, color='salmon')
        ax.set_title('Inference Time Comparison')
        ax.set_ylabel('Inference Time (ms)')
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: Model size comparison
        ax = axes[3]
        sizes = [detailed_results[m]['model_size'] / 1e6 for m in model_names]  # Convert to millions
        
        ax.bar(model_names, sizes, alpha=0.7, color='lightgreen')
        ax.set_title('Model Size Comparison')
        ax.set_ylabel('Parameters (Millions)')
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 5: Radar chart for multiple metrics
        ax = axes[4]
        
        # Prepare data for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        ax = plt.subplot(2, 3, 5, projection='polar')
        
        for i, model in enumerate(model_names):
            values = [detailed_results[model]['overall_metrics'][metric]['mean'] 
                     for metric in metrics]
            values += values[:1]  # Close the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, alpha=0.7)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Performance Radar')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Plot 6: Performance vs Efficiency scatter
        ax = axes[5]
        
        f1_scores = [detailed_results[m]['overall_metrics']['f1_score']['mean'] for m in model_names]
        inf_times = [detailed_results[m]['performance_metrics']['avg_inference_time']['mean'] * 1000 
                    for m in model_names]
        sizes = [detailed_results[m]['model_size'] / 1e6 for m in model_names]
        
        scatter = ax.scatter(inf_times, f1_scores, s=[s*10 for s in sizes], alpha=0.7, 
                           c=range(len(model_names)), cmap='viridis')
        
        for i, model in enumerate(model_names):
            ax.annotate(model, (inf_times[i], f1_scores[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Performance vs Efficiency\\n(bubble size = model size)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        save_path: str,
        include_raw_data: bool = False
    ) -> None:
        """Save evaluation results to JSON file."""
        
        # Prepare results for serialization
        serializable_results = {}
        
        for key, value in results.items():
            if key == 'detailed_results':
                serializable_results[key] = {}
                for model_name, model_results in value.items():
                    serializable_results[key][model_name] = self._make_serializable(
                        model_results, include_raw_data
                    )
            elif key == 'statistical_comparisons':
                serializable_results[key] = self._make_serializable(value, include_raw_data)
            else:
                serializable_results[key] = self._make_serializable(value, include_raw_data)
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {save_path}")
    
    def _make_serializable(self, obj: Any, include_raw_data: bool = False) -> Any:
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist() if include_raw_data else f"Array shape: {obj.shape}"
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v, include_raw_data) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item, include_raw_data) for item in obj]
        else:
            return obj


class SentimentBenchmark:
    """
    Comprehensive benchmarking suite for sentiment analysis models.
    
    Provides standardized evaluation protocols for reproducible research.
    """
    
    def __init__(
        self,
        output_dir: str = "./benchmark_results",
        device: Optional[torch.device] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = SentimentEvaluator(device=self.device)
        
        self.benchmark_results = []
        
    def run_benchmark(
        self,
        models: Dict[str, nn.Module],
        datasets: Dict[str, DataLoader],
        num_runs: int = 5,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across models and datasets.
        
        Args:
            models: Dictionary of model name -> model instance
            datasets: Dictionary of dataset name -> DataLoader
            num_runs: Number of evaluation runs per model-dataset combination
            save_results: Whether to save results to disk
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Starting benchmark with {len(models)} models and {len(datasets)} datasets...")
        
        benchmark_results = {}
        
        for dataset_name, data_loader in datasets.items():
            logger.info(f"Evaluating on dataset: {dataset_name}")
            
            # Compare models on this dataset
            comparison_results = self.evaluator.compare_models(
                models, data_loader, num_runs=num_runs
            )
            
            benchmark_results[dataset_name] = comparison_results
            
            # Save individual dataset results
            if save_results:
                dataset_path = self.output_dir / f"{dataset_name}_results.json"
                self.evaluator.save_evaluation_results(
                    comparison_results, str(dataset_path), include_raw_data=False
                )
        
        # Generate cross-dataset analysis
        cross_dataset_analysis = self._analyze_cross_dataset_performance(benchmark_results)
        
        # Create final benchmark report
        final_results = {
            'dataset_results': benchmark_results,
            'cross_dataset_analysis': cross_dataset_analysis,
            'benchmark_metadata': {
                'num_models': len(models),
                'num_datasets': len(datasets),
                'num_runs': num_runs,
                'device': str(self.device),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        if save_results:
            # Save comprehensive results
            final_path = self.output_dir / "benchmark_results.json"
            self.evaluator.save_evaluation_results(
                final_results, str(final_path), include_raw_data=False
            )
            
            # Generate benchmark report
            self._generate_benchmark_report(final_results)
        
        return final_results
    
    def _analyze_cross_dataset_performance(
        self,
        benchmark_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze model performance across different datasets."""
        
        model_names = set()
        for dataset_results in benchmark_results.values():
            model_names.update(dataset_results['detailed_results'].keys())
        
        model_names = list(model_names)
        dataset_names = list(benchmark_results.keys())
        
        # Calculate average performance across datasets
        cross_dataset_performance = {}
        
        for model in model_names:
            model_performance = {
                'accuracy': [],
                'f1_score': [],
                'inference_time': [],
                'model_size': None
            }
            
            for dataset in dataset_names:
                if model in benchmark_results[dataset]['detailed_results']:
                    results = benchmark_results[dataset]['detailed_results'][model]
                    
                    model_performance['accuracy'].append(
                        results['overall_metrics']['accuracy']['mean']
                    )
                    model_performance['f1_score'].append(
                        results['overall_metrics']['f1_score']['mean']
                    )
                    model_performance['inference_time'].append(
                        results['performance_metrics']['avg_inference_time']['mean']
                    )
                    
                    if model_performance['model_size'] is None:
                        model_performance['model_size'] = results['model_size']
            
            # Calculate aggregated metrics
            cross_dataset_performance[model] = {
                'avg_accuracy': np.mean(model_performance['accuracy']),
                'std_accuracy': np.std(model_performance['accuracy']),
                'avg_f1_score': np.mean(model_performance['f1_score']),
                'std_f1_score': np.std(model_performance['f1_score']),
                'avg_inference_time': np.mean(model_performance['inference_time']),
                'std_inference_time': np.std(model_performance['inference_time']),
                'model_size': model_performance['model_size'],
                'consistency_score': (np.std(model_performance['accuracy']) + 
                                    np.std(model_performance['f1_score'])) / 2
            }
        
        # Rank models by cross-dataset performance
        overall_ranking = sorted(
            cross_dataset_performance.items(),
            key=lambda x: (x[1]['avg_accuracy'] + x[1]['avg_f1_score']) / 2,
            reverse=True
        )
        
        # Find most consistent model
        most_consistent = min(
            cross_dataset_performance.items(),
            key=lambda x: x[1]['consistency_score']
        )[0]
        
        return {
            'model_performance': cross_dataset_performance,
            'overall_ranking': overall_ranking,
            'most_consistent_model': most_consistent,
            'dataset_coverage': {
                model: len([d for d in dataset_names 
                          if model in benchmark_results[d]['detailed_results']])
                for model in model_names
            }
        }
    
    def _generate_benchmark_report(self, results: Dict[str, Any]) -> None:
        """Generate a comprehensive benchmark report."""
        
        report_path = self.output_dir / "benchmark_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Sentiment Analysis Model Benchmark Report\\n\\n")
            f.write(f"Generated on: {results['benchmark_metadata']['timestamp']}\\n\\n")
            
            # Benchmark overview
            f.write("## Benchmark Overview\\n\\n")
            metadata = results['benchmark_metadata']
            f.write(f"- **Models Evaluated**: {metadata['num_models']}\\n")
            f.write(f"- **Datasets Used**: {metadata['num_datasets']}\\n")
            f.write(f"- **Evaluation Runs**: {metadata['num_runs']}\\n")
            f.write(f"- **Device**: {metadata['device']}\\n\\n")
            
            # Overall rankings
            f.write("## Overall Model Rankings\\n\\n")
            cross_analysis = results['cross_dataset_analysis']
            f.write("Ranked by average accuracy and F1 score across all datasets:\\n\\n")
            
            for i, (model, _) in enumerate(cross_analysis['overall_ranking'], 1):
                perf = cross_analysis['model_performance'][model]
                f.write(f"{i}. **{model}**\\n")
                f.write(f"   - Avg Accuracy: {perf['avg_accuracy']:.4f} ± {perf['std_accuracy']:.4f}\\n")
                f.write(f"   - Avg F1 Score: {perf['avg_f1_score']:.4f} ± {perf['std_f1_score']:.4f}\\n")
                f.write(f"   - Avg Inference Time: {perf['avg_inference_time']*1000:.2f}ms\\n")
                f.write(f"   - Model Size: {perf['model_size']/1e6:.1f}M parameters\\n\\n")
            
            # Dataset-specific results
            f.write("## Dataset-Specific Results\\n\\n")
            
            for dataset_name, dataset_results in results['dataset_results'].items():
                f.write(f"### {dataset_name}\\n\\n")
                
                rankings = dataset_results['rankings']
                f.write("**Top 3 Models by Accuracy:**\\n")
                for i, (model, score) in enumerate(rankings['accuracy'][:3], 1):
                    f.write(f"{i}. {model}: {score:.4f}\\n")
                f.write("\\n")
                
                f.write("**Fastest Models:**\\n")
                for i, (model, time) in enumerate(rankings['inference_speed'][:3], 1):
                    f.write(f"{i}. {model}: {time*1000:.2f}ms\\n")
                f.write("\\n")
            
            # Key findings
            f.write("## Key Findings\\n\\n")
            f.write(f"- **Best Overall Model**: {cross_analysis['overall_ranking'][0][0]}\\n")
            f.write(f"- **Most Consistent Model**: {cross_analysis['most_consistent_model']}\\n\\n")
            
            # Recommendations
            f.write("## Recommendations\\n\\n")
            best_model = cross_analysis['overall_ranking'][0][0]
            consistent_model = cross_analysis['most_consistent_model']
            
            f.write(f"- For **general use**: {best_model} provides the best overall performance\\n")
            f.write(f"- For **consistent performance**: {consistent_model} shows most stable results across datasets\\n")
            f.write(f"- For detailed analysis, see individual dataset results and statistical comparisons\\n\\n")
        
        logger.info(f"Benchmark report saved to {report_path}")


def create_synthetic_benchmark_data(
    num_samples: int = 1000,
    max_length: int = 128,
    vocab_size: int = 10000,
    num_classes: int = 3,
    batch_size: int = 32
) -> Tuple[DataLoader, Dict[str, int]]:
    """
    Create synthetic data for benchmarking when real datasets are not available.
    
    Args:
        num_samples: Number of samples to generate
        max_length: Maximum sequence length
        vocab_size: Vocabulary size
        num_classes: Number of classes
        batch_size: Batch size for DataLoader
        
    Returns:
        Tuple of (DataLoader, label_map)
    """
    from torch.utils.data import TensorDataset
    
    # Generate random sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, max_length))
    
    # Generate attention masks (random padding)
    attention_mask = torch.ones_like(input_ids)
    for i in range(num_samples):
        # Random sequence length
        seq_len = torch.randint(max_length // 2, max_length + 1, (1,)).item()
        if seq_len < max_length:
            attention_mask[i, seq_len:] = 0
            input_ids[i, seq_len:] = 0  # Pad token
    
    # Generate balanced labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    # Custom collate function to match expected format
    def collate_fn(batch):
        input_ids, attention_mask, labels = zip(*batch)
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels)
        }
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    label_map = {f'class_{i}': i for i in range(num_classes)}
    
    logger.info(f"Created synthetic benchmark dataset with {num_samples} samples")
    
    return dataloader, label_map
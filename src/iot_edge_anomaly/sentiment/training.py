"""
Training utilities for sentiment analysis models.

Includes training loops, evaluation metrics, model optimization,
and comparative analysis functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import logging
from pathlib import Path
import json
import copy
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SentimentTrainer:
    """
    Training manager for sentiment analysis models.
    
    Handles training loops, validation, early stopping, and model checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        save_dir: str = "./checkpoints",
        experiment_name: str = "sentiment_experiment",
        patience: int = 5,
        min_delta: float = 0.001
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Setup loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Setup directories and logging
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.model_save_path = self.save_dir / f"{experiment_name}_best.pth"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
        
        logger.info(f"Initialized trainer for {experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Log batch statistics
            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_predictions
        
        return epoch_loss, epoch_acc
    
    def validate(self, data_loader: Optional[DataLoader] = None) -> Tuple[float, float, Dict[str, Any]]:
        """Validate the model."""
        if data_loader is None:
            data_loader = self.val_loader
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Store results
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(torch.softmax(logits, dim=-1).cpu().numpy())
        
        epoch_loss = total_loss / len(data_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        
        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Calculate per-class metrics
        per_class_metrics = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        
        # Calculate AUC for multiclass (one-vs-rest)
        try:
            auc_score = roc_auc_score(
                all_labels, all_logits, 
                multi_class='ovr', average='weighted'
            )
        except ValueError:
            auc_score = 0.0
        
        metrics = {
            'accuracy': epoch_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'per_class_precision': per_class_metrics[0],
            'per_class_recall': per_class_metrics[1],
            'per_class_f1': per_class_metrics[2],
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
            'predictions': all_predictions,
            'labels': all_labels,
            'logits': all_logits
        }
        
        return epoch_loss, epoch_acc, metrics
    
    def train(self, num_epochs: int, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Early stopping check
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_model()
                
                if verbose:
                    logger.info(f"New best model saved at epoch {epoch+1}")
            else:
                self.epochs_without_improvement += 1
            
            # Print epoch results
            epoch_time = time.time() - start_time
            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model for final evaluation
        self.load_model()
        
        logger.info("Training completed!")
        return self.history
    
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            data_loader: DataLoader to evaluate on (uses test_loader if None)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if data_loader is None:
            data_loader = self.test_loader
        
        if data_loader is None:
            raise ValueError("No test data loader available for evaluation")
        
        logger.info("Starting model evaluation...")
        
        # Get validation metrics
        test_loss, test_acc, test_metrics = self.validate(data_loader)
        
        # Create classification report
        class_names = ['negative', 'neutral', 'positive']  # Adjust as needed
        classification_rep = classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=class_names[:len(np.unique(test_metrics['labels']))],
            output_dict=True
        )
        
        evaluation_results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_auc': test_metrics['auc'],
            'confusion_matrix': test_metrics['confusion_matrix'],
            'classification_report': classification_rep,
            'per_class_metrics': {
                'precision': test_metrics['per_class_precision'],
                'recall': test_metrics['per_class_recall'],
                'f1': test_metrics['per_class_f1']
            }
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        logger.info(f"  Test F1: {test_metrics['f1']:.4f}")
        logger.info(f"  Test AUC: {test_metrics['auc']:.4f}")
        
        return evaluation_results
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save model checkpoint."""
        if path is None:
            path = self.model_save_path
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'model_config': {
                'model_class': self.model.__class__.__name__,
                'model_params': getattr(self.model, 'config', {})
            }
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Optional[str] = None) -> None:
        """Load model checkpoint."""
        if path is None:
            path = self.model_save_path
        
        if not Path(path).exists():
            logger.warning(f"Checkpoint file {path} not found")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Model loaded from {path}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', color='orange')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(self.history['val_acc'], label='Validation Accuracy', color='orange')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['learning_rate'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss difference plot
        loss_diff = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        axes[1, 1].plot(loss_diff, color='red')
        axes[1, 1].set_title('Validation - Training Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def __del__(self):
        """Close tensorboard writer."""
        if hasattr(self, 'writer'):
            self.writer.close()


class ModelComparator:
    """
    Comparative analysis tool for different sentiment analysis models.
    
    Implements statistical significance testing and performance benchmarking.
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        num_runs: int = 3
    ):
        self.models = models
        self.test_loader = test_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_runs = num_runs
        
        self.results = defaultdict(list)
        self.detailed_results = {}
    
    def run_comparison(self) -> Dict[str, Any]:
        """
        Run comprehensive comparison across all models.
        
        Returns:
            Dictionary containing comparative analysis results
        """
        logger.info(f"Starting model comparison with {len(self.models)} models...")
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            model_results = []
            for run in range(self.num_runs):
                logger.info(f"  Run {run + 1}/{self.num_runs}")
                
                # Create trainer for evaluation
                trainer = SentimentTrainer(
                    model=copy.deepcopy(model),
                    train_loader=self.test_loader,  # Dummy
                    val_loader=self.test_loader,    # Dummy
                    test_loader=self.test_loader,
                    device=self.device,
                    experiment_name=f"{model_name}_run_{run}"
                )
                
                # Evaluate model
                results = trainer.evaluate()
                model_results.append(results)
            
            # Store results
            self.results[model_name] = model_results
            self.detailed_results[model_name] = self._aggregate_results(model_results)
        
        # Perform statistical analysis
        comparison_results = self._perform_statistical_analysis()
        
        return comparison_results
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple runs."""
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
        
        aggregated = {}
        for metric in metrics:
            values = [result[metric] for result in results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        return aggregated
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        from scipy import stats
        
        model_names = list(self.models.keys())
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
        
        statistical_results = {}
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                statistical_results[comparison_key] = {}
                
                for metric in metrics:
                    values1 = self.detailed_results[model1][metric]['values']
                    values2 = self.detailed_results[model2][metric]['values']
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        ((len(values1) - 1) * np.var(values1) + (len(values2) - 1) * np.var(values2)) /
                        (len(values1) + len(values2) - 2)
                    )
                    cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
                    
                    statistical_results[comparison_key][metric] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': cohens_d,
                        'mean_difference': np.mean(values1) - np.mean(values2)
                    }
        
        return {
            'detailed_results': self.detailed_results,
            'statistical_comparisons': statistical_results,
            'summary': self._create_summary()
        }
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create summary of comparison results."""
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
        
        summary = {}
        for metric in metrics:
            metric_summary = {}
            for model_name in self.models.keys():
                metric_summary[model_name] = self.detailed_results[model_name][metric]['mean']
            
            # Rank models by metric
            ranked_models = sorted(metric_summary.items(), key=lambda x: x[1], reverse=True)
            summary[metric] = {
                'scores': metric_summary,
                'ranking': ranked_models,
                'best_model': ranked_models[0][0]
            }
        
        return summary
    
    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """Create visualization of model comparison."""
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
        model_names = list(self.models.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for box plot
            data = []
            labels = []
            for model_name in model_names:
                data.append(self.detailed_results[model_name][metric]['values'])
                labels.append(model_name)
            
            # Create box plot
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            
            # Rotate labels if needed
            if len(max(labels, key=len)) > 10:
                ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, save_path: str) -> None:
        """Save comparison results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.detailed_results.items():
            serializable_results[model_name] = {}
            for metric, values in results.items():
                serializable_results[model_name][metric] = {
                    'mean': float(values['mean']),
                    'std': float(values['std']),
                    'min': float(values['min']),
                    'max': float(values['max']),
                    'values': [float(v) for v in values['values']]
                }
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Comparison results saved to {save_path}")


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer for training.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer parameters
        
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9, **kwargs)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'cosine',
    num_epochs: int = 100,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler ('cosine', 'step', 'plateau', 'exponential')
        num_epochs: Total number of training epochs
        **kwargs: Additional scheduler parameters
        
    Returns:
        Scheduler instance or None
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, **kwargs)
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1, **kwargs)
    elif scheduler_name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, **kwargs)
    elif scheduler_name == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, **kwargs)
    elif scheduler_name == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def hyperparameter_search(
    model_class: type,
    train_loader: DataLoader,
    val_loader: DataLoader,
    param_grid: Dict[str, List[Any]],
    num_epochs: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Perform hyperparameter search.
    
    Args:
        model_class: Model class to instantiate
        train_loader: Training data loader
        val_loader: Validation data loader
        param_grid: Dictionary of parameter lists to search
        num_epochs: Number of epochs for each trial
        device: Device to use for training
        
    Returns:
        Dictionary containing best parameters and results
    """
    from itertools import product
    
    logger.info("Starting hyperparameter search...")
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    best_score = 0.0
    best_params = {}
    all_results = []
    
    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        logger.info(f"Trial {i+1}/{len(param_combinations)}: {params}")
        
        try:
            # Create model with current parameters
            model = model_class(**params)
            
            # Create trainer
            trainer = SentimentTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                experiment_name=f"hyperparam_trial_{i}",
                patience=3  # Early stopping for efficiency
            )
            
            # Train model
            trainer.train(num_epochs, verbose=False)
            
            # Evaluate
            _, val_acc, _ = trainer.validate()
            
            # Store results
            result = {
                'params': params,
                'val_accuracy': val_acc,
                'trial': i
            }
            all_results.append(result)
            
            # Update best
            if val_acc > best_score:
                best_score = val_acc
                best_params = params
                logger.info(f"New best score: {best_score:.4f}")
        
        except Exception as e:
            logger.error(f"Trial {i} failed: {e}")
            continue
    
    logger.info(f"Hyperparameter search completed. Best score: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results
    }
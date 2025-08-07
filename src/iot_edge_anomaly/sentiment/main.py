"""
Main entry point for sentiment analysis system.

Provides command-line interface for training, evaluation, and benchmarking
of sentiment analysis models with comparative research capabilities.
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import DataLoader

from .models import create_sentiment_model
from .data_processing import (
    SentimentVocabulary, create_data_loaders, 
    load_standard_datasets
)
from .training import (
    SentimentTrainer, ModelComparator,
    create_optimizer, create_scheduler
)
from .evaluation import (
    SentimentEvaluator, SentimentBenchmark,
    create_synthetic_benchmark_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_datasets() -> Dict[str, DataLoader]:
    """Create sample datasets for demonstration."""
    logger.info("Creating synthetic datasets for demonstration...")
    
    # Create different synthetic datasets to simulate different domains
    datasets = {}
    
    # Movie reviews dataset
    movie_loader, _ = create_synthetic_benchmark_data(
        num_samples=1000,
        max_length=128,
        vocab_size=10000,
        num_classes=3,
        batch_size=32
    )
    datasets['movie_reviews'] = movie_loader
    
    # Product reviews dataset (slightly different characteristics)
    product_loader, _ = create_synthetic_benchmark_data(
        num_samples=800,
        max_length=96,
        vocab_size=8000,
        num_classes=3,
        batch_size=32
    )
    datasets['product_reviews'] = product_loader
    
    # Social media posts (shorter sequences)
    social_loader, _ = create_synthetic_benchmark_data(
        num_samples=1200,
        max_length=64,
        vocab_size=12000,
        num_classes=3,
        batch_size=32
    )
    datasets['social_media'] = social_loader
    
    return datasets


def create_model_suite(vocab_size: int = 10000) -> Dict[str, torch.nn.Module]:
    """Create a suite of sentiment analysis models for comparison."""
    logger.info("Creating model suite for comparative analysis...")
    
    models = {}
    
    # LSTM baseline
    models['LSTM_Baseline'] = create_sentiment_model(
        'lstm',
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.2
    )
    
    # BiLSTM with attention
    models['BiLSTM_Attention'] = create_sentiment_model(
        'bilstm_attention',
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.2
    )
    
    # LSTM-CNN hybrid
    models['LSTM_CNN_Hybrid'] = create_sentiment_model(
        'lstm_cnn',
        vocab_size=vocab_size,
        embedding_dim=128,
        lstm_hidden_dim=64,
        lstm_layers=1,
        cnn_filters=100,
        filter_sizes=[2, 3, 4],
        num_classes=3,
        dropout=0.2
    )
    
    # Lightweight transformer
    models['Lightweight_Transformer'] = create_sentiment_model(
        'transformer',
        vocab_size=vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        max_seq_len=128,
        num_classes=3,
        dropout=0.1
    )
    
    logger.info(f"Created {len(models)} models for comparison")
    return models


def train_single_model(args) -> None:
    """Train a single sentiment analysis model."""
    logger.info("Training single model...")
    
    # Create synthetic data for training
    logger.info("Creating synthetic training data...")
    texts = [f"This is sample text {i} for sentiment analysis" for i in range(1000)]
    labels = [i % 3 for i in range(1000)]  # Balanced labels
    
    # Create vocabulary
    vocab = SentimentVocabulary(min_freq=1, max_vocab_size=args.vocab_size)
    vocab.build_from_texts(texts)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        texts, labels, vocab,
        batch_size=args.batch_size,
        max_length=args.max_length,
        augment_data=args.augment_data
    )
    
    # Create model
    model = create_sentiment_model(
        args.model_type,
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model, args.optimizer, args.learning_rate, args.weight_decay
    )
    scheduler = create_scheduler(
        optimizer, args.scheduler, args.num_epochs
    )
    
    # Create trainer
    trainer = SentimentTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device(args.device),
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        patience=args.patience
    )
    
    # Train model
    history = trainer.train(args.num_epochs, verbose=True)
    
    # Evaluate model
    results = trainer.evaluate()
    
    # Print results
    logger.info("Training completed!")
    logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"Test F1 Score: {results['test_f1']:.4f}")
    
    # Save results
    if args.save_results:
        results_path = Path(args.save_dir) / f"{args.experiment_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")
    
    # Plot training history
    if args.plot_training:
        plot_path = Path(args.save_dir) / f"{args.experiment_name}_training_plot.png"
        trainer.plot_training_history(str(plot_path))


def compare_models(args) -> None:
    """Compare multiple sentiment analysis models."""
    logger.info("Starting model comparison...")
    
    # Create model suite
    models = create_model_suite(args.vocab_size)
    
    # Create test data
    test_loader, _ = create_synthetic_benchmark_data(
        num_samples=args.test_samples,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size
    )
    
    # Create evaluator
    evaluator = SentimentEvaluator(
        device=torch.device(args.device),
        num_classes=args.num_classes
    )
    
    # Run comparison
    comparison_results = evaluator.compare_models(
        models, test_loader, num_runs=args.num_runs
    )
    
    # Print summary
    summary = comparison_results['summary']
    logger.info("\\n=== MODEL COMPARISON RESULTS ===")
    logger.info(f"Best overall model: {summary['best_models']['overall']}")
    logger.info(f"Most consistent model: {summary['most_consistent_model']}")
    
    # Print key findings
    logger.info("\\nKey Findings:")
    for finding in summary['key_findings']:
        logger.info(f"  - {finding}")
    
    # Print recommendations
    logger.info("\\nRecommendations:")
    for rec in summary['recommendations']:
        logger.info(f"  - {rec}")
    
    # Save results
    if args.save_results:
        results_path = Path(args.save_dir) / "model_comparison_results.json"
        evaluator.save_evaluation_results(comparison_results, str(results_path))
    
    # Create visualization
    if args.plot_comparison:
        plot_path = Path(args.save_dir) / "model_comparison_plot.png"
        evaluator.plot_comparison_results(comparison_results, str(plot_path))


def run_benchmark(args) -> None:
    """Run comprehensive benchmark across models and datasets."""
    logger.info("Starting comprehensive benchmark...")
    
    # Create model suite
    models = create_model_suite(args.vocab_size)
    
    # Create datasets
    datasets = create_sample_datasets()
    
    # Create benchmark
    benchmark = SentimentBenchmark(
        output_dir=args.save_dir,
        device=torch.device(args.device)
    )
    
    # Run benchmark
    benchmark_results = benchmark.run_benchmark(
        models, datasets, num_runs=args.num_runs, save_results=args.save_results
    )
    
    # Print summary
    cross_analysis = benchmark_results['cross_dataset_analysis']
    logger.info("\\n=== BENCHMARK RESULTS ===")
    logger.info(f"Evaluated {len(models)} models on {len(datasets)} datasets")
    logger.info(f"Best overall model: {cross_analysis['overall_ranking'][0][0]}")
    logger.info(f"Most consistent model: {cross_analysis['most_consistent_model']}")
    
    logger.info("\\nOverall Rankings:")
    for i, (model, _) in enumerate(cross_analysis['overall_ranking'], 1):
        perf = cross_analysis['model_performance'][model]
        logger.info(f"  {i}. {model}: Acc={perf['avg_accuracy']:.4f}, F1={perf['avg_f1_score']:.4f}")


def evaluate_single_model(args) -> None:
    """Evaluate a single pre-trained model."""
    logger.info("Evaluating single model...")
    
    # Load model (placeholder - in practice would load from checkpoint)
    model = create_sentiment_model(
        args.model_type,
        vocab_size=args.vocab_size,
        num_classes=args.num_classes
    )
    
    # Create test data
    test_loader, _ = create_synthetic_benchmark_data(
        num_samples=args.test_samples,
        vocab_size=args.vocab_size,
        num_classes=args.num_classes
    )
    
    # Create evaluator
    evaluator = SentimentEvaluator(
        device=torch.device(args.device),
        num_classes=args.num_classes
    )
    
    # Evaluate
    results = evaluator.evaluate_model(
        model, test_loader, args.model_name,
        return_predictions=args.return_predictions,
        compute_feature_importance=args.compute_feature_importance
    )
    
    # Print results
    metrics = results['overall_metrics']
    logger.info(f"\\nEvaluation Results for {args.model_name}:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  AUC Score: {metrics['auc_score']:.4f}")
    
    perf_metrics = results['performance_metrics']
    logger.info(f"  Inference Time: {perf_metrics['avg_inference_time']*1000:.2f}ms")
    logger.info(f"  Throughput: {perf_metrics['throughput']:.1f} samples/sec")
    logger.info(f"  Model Size: {perf_metrics['model_size']/1e6:.1f}M parameters")
    logger.info(f"  Memory Usage: {perf_metrics['memory_usage_mb']:.1f}MB")
    
    # Save results
    if args.save_results:
        results_path = Path(args.save_dir) / f"{args.model_name}_evaluation.json"
        evaluator.save_evaluation_results(
            {'single_model_results': results}, str(results_path)
        )


def main():
    """Main entry point for sentiment analysis system."""
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis Research Framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--device', default='auto', 
                              help='Device to use (cuda/cpu/auto)')
    common_parser.add_argument('--save-dir', default='./results',
                              help='Directory to save results')
    common_parser.add_argument('--save-results', action='store_true',
                              help='Save results to disk')
    common_parser.add_argument('--vocab-size', type=int, default=10000,
                              help='Vocabulary size')
    common_parser.add_argument('--num-classes', type=int, default=3,
                              help='Number of sentiment classes')
    common_parser.add_argument('--max-length', type=int, default=128,
                              help='Maximum sequence length')
    common_parser.add_argument('--batch-size', type=int, default=32,
                              help='Batch size')
    
    # Train command
    train_parser = subparsers.add_parser('train', parents=[common_parser],
                                        help='Train a single model')
    train_parser.add_argument('--model-type', default='lstm',
                             choices=['lstm', 'bilstm_attention', 'lstm_cnn', 'transformer'],
                             help='Type of model to train')
    train_parser.add_argument('--experiment-name', default='sentiment_experiment',
                             help='Name for the experiment')
    train_parser.add_argument('--num-epochs', type=int, default=10,
                             help='Number of training epochs')
    train_parser.add_argument('--learning-rate', type=float, default=1e-3,
                             help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-4,
                             help='Weight decay')
    train_parser.add_argument('--optimizer', default='adamw',
                             choices=['adam', 'adamw', 'sgd'],
                             help='Optimizer type')
    train_parser.add_argument('--scheduler', default='cosine',
                             choices=['cosine', 'step', 'plateau', 'none'],
                             help='Learning rate scheduler')
    train_parser.add_argument('--embedding-dim', type=int, default=128,
                             help='Embedding dimension')
    train_parser.add_argument('--hidden-dim', type=int, default=64,
                             help='Hidden dimension')
    train_parser.add_argument('--num-layers', type=int, default=2,
                             help='Number of layers')
    train_parser.add_argument('--dropout', type=float, default=0.2,
                             help='Dropout rate')
    train_parser.add_argument('--patience', type=int, default=5,
                             help='Early stopping patience')
    train_parser.add_argument('--augment-data', action='store_true',
                             help='Apply data augmentation')
    train_parser.add_argument('--plot-training', action='store_true',
                             help='Plot training history')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', parents=[common_parser],
                                          help='Compare multiple models')
    compare_parser.add_argument('--num-runs', type=int, default=3,
                               help='Number of evaluation runs per model')
    compare_parser.add_argument('--test-samples', type=int, default=1000,
                               help='Number of test samples')
    compare_parser.add_argument('--plot-comparison', action='store_true',
                               help='Create comparison plots')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', parents=[common_parser],
                                           help='Run comprehensive benchmark')
    benchmark_parser.add_argument('--num-runs', type=int, default=5,
                                 help='Number of evaluation runs per model-dataset combination')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', parents=[common_parser],
                                       help='Evaluate a single model')
    eval_parser.add_argument('--model-type', default='lstm',
                            choices=['lstm', 'bilstm_attention', 'lstm_cnn', 'transformer'],
                            help='Type of model to evaluate')
    eval_parser.add_argument('--model-name', default='TestModel',
                            help='Name for the model')
    eval_parser.add_argument('--test-samples', type=int, default=1000,
                            help='Number of test samples')
    eval_parser.add_argument('--return-predictions', action='store_true',
                            help='Return predictions in results')
    eval_parser.add_argument('--compute-feature-importance', action='store_true',
                            help='Compute feature importance')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', parents=[common_parser],
                                       help='Run demonstration of all features')
    demo_parser.add_argument('--quick', action='store_true',
                            help='Run quick demo with fewer iterations')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {args.device}")
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Execute command
    if args.command == 'train':
        train_single_model(args)
    elif args.command == 'compare':
        compare_models(args)
    elif args.command == 'benchmark':
        run_benchmark(args)
    elif args.command == 'evaluate':
        evaluate_single_model(args)
    elif args.command == 'demo':
        run_demo(args)
    else:
        parser.print_help()


def run_demo(args) -> None:
    """Run demonstration of sentiment analysis capabilities."""
    logger.info("Running sentiment analysis demonstration...")
    
    # Set quick parameters if requested
    if args.quick:
        args.num_runs = 2
        args.test_samples = 500
        args.vocab_size = 5000
    
    logger.info("\\n=== DEMONSTRATION OVERVIEW ===")
    logger.info("This demo showcases the sentiment analysis research framework with:")
    logger.info("1. Multiple model architectures (LSTM, BiLSTM+Attention, LSTM-CNN, Transformer)")
    logger.info("2. Comprehensive evaluation metrics")
    logger.info("3. Statistical significance testing")
    logger.info("4. Performance benchmarking")
    logger.info("5. Cross-dataset analysis")
    
    # Run model comparison
    logger.info("\\n=== STEP 1: MODEL COMPARISON ===")
    compare_models(args)
    
    # Run benchmark
    logger.info("\\n=== STEP 2: CROSS-DATASET BENCHMARK ===")
    run_benchmark(args)
    
    logger.info("\\n=== DEMONSTRATION COMPLETE ===")
    logger.info(f"Results saved to: {args.save_dir}")
    logger.info("This framework provides:")
    logger.info("  ✓ Research-grade model comparison")
    logger.info("  ✓ Statistical significance testing")
    logger.info("  ✓ Comprehensive performance metrics")
    logger.info("  ✓ Publication-ready visualizations")
    logger.info("  ✓ Reproducible experimental protocols")


if __name__ == "__main__":
    main()
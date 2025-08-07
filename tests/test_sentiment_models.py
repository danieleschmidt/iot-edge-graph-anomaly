"""
Comprehensive test suite for sentiment analysis models.

Tests model architectures, training, evaluation, and comparative analysis
with statistical validation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Import sentiment analysis modules
from src.iot_edge_anomaly.sentiment.models import (
    SentimentLSTM, BiLSTMAttentionSentiment, LSTMCNNHybridSentiment,
    TransformerSentiment, SentimentEnsemble, create_sentiment_model
)
from src.iot_edge_anomaly.sentiment.data_processing import (
    SentimentVocabulary, SentimentDataset, SentimentDataAugmentation,
    create_data_loaders
)
from src.iot_edge_anomaly.sentiment.training import (
    SentimentTrainer, ModelComparator, create_optimizer, create_scheduler
)
from src.iot_edge_anomaly.sentiment.evaluation import (
    SentimentEvaluator, SentimentBenchmark, create_synthetic_benchmark_data
)


@pytest.fixture
def sample_vocab():
    """Create a sample vocabulary for testing."""
    vocab = SentimentVocabulary(min_freq=1, max_vocab_size=1000)
    texts = [
        "this movie is great and amazing",
        "terrible film with bad acting",
        "okay movie not bad not good",
        "love this fantastic wonderful film",
        "hate this awful terrible movie"
    ]
    vocab.build_from_texts(texts)
    return vocab


@pytest.fixture
def sample_data_loader():
    """Create a sample data loader for testing."""
    batch_size = 4
    seq_len = 16
    vocab_size = 1000
    num_samples = 20
    
    # Create synthetic data
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 3, (num_samples,))
    
    # Set some padding
    for i in range(num_samples):
        pad_len = torch.randint(0, seq_len // 2, (1,)).item()
        if pad_len > 0:
            attention_mask[i, -pad_len:] = 0
            input_ids[i, -pad_len:] = 0
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    def collate_fn(batch):
        input_ids, attention_mask, labels = zip(*batch)
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels)
        }
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


class TestSentimentModels:
    """Test sentiment analysis model architectures."""
    
    def test_sentiment_lstm_creation(self):
        """Test LSTM sentiment model creation."""
        model = SentimentLSTM(
            vocab_size=1000,
            embedding_dim=64,
            hidden_dim=32,
            num_layers=2,
            num_classes=3,
            dropout=0.1
        )
        
        assert isinstance(model, nn.Module)
        assert model.hidden_dim == 32
        assert model.num_layers == 2
        assert model.num_classes == 3
    
    def test_sentiment_lstm_forward(self, sample_data_loader):
        """Test LSTM forward pass."""
        model = SentimentLSTM(vocab_size=1000, embedding_dim=64, hidden_dim=32)
        model.eval()
        
        with torch.no_grad():
            for batch in sample_data_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                logits = model(input_ids, attention_mask)
                
                assert logits.shape == (input_ids.size(0), 3)  # batch_size x num_classes
                assert not torch.isnan(logits).any()
                assert not torch.isinf(logits).any()
                break
    
    def test_bilstm_attention_creation(self):
        """Test BiLSTM with attention model creation."""
        model = BiLSTMAttentionSentiment(
            vocab_size=1000,
            embedding_dim=64,
            hidden_dim=32,
            num_layers=2,
            num_classes=3
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'bilstm')
        assert hasattr(model, 'attention')
    
    def test_bilstm_attention_forward(self, sample_data_loader):
        """Test BiLSTM with attention forward pass."""
        model = BiLSTMAttentionSentiment(vocab_size=1000, embedding_dim=64, hidden_dim=32)
        model.eval()
        
        with torch.no_grad():
            for batch in sample_data_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                logits = model(input_ids, attention_mask)
                
                assert logits.shape == (input_ids.size(0), 3)
                assert not torch.isnan(logits).any()
                break
    
    def test_lstm_cnn_hybrid_creation(self):
        """Test LSTM-CNN hybrid model creation."""
        model = LSTMCNNHybridSentiment(
            vocab_size=1000,
            embedding_dim=64,
            lstm_hidden_dim=32,
            cnn_filters=50,
            filter_sizes=[2, 3, 4],
            num_classes=3
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'convs')
        assert len(model.convs) == 3  # Three filter sizes
    
    def test_lstm_cnn_hybrid_forward(self, sample_data_loader):
        """Test LSTM-CNN hybrid forward pass."""
        model = LSTMCNNHybridSentiment(
            vocab_size=1000,
            embedding_dim=64,
            lstm_hidden_dim=32,
            cnn_filters=50,
            filter_sizes=[2, 3]
        )
        model.eval()
        
        with torch.no_grad():
            for batch in sample_data_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                logits = model(input_ids, attention_mask)
                
                assert logits.shape == (input_ids.size(0), 3)
                assert not torch.isnan(logits).any()
                break
    
    def test_transformer_sentiment_creation(self):
        """Test transformer model creation."""
        model = TransformerSentiment(
            vocab_size=1000,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            max_seq_len=128,
            num_classes=3
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'position_embedding')
    
    def test_transformer_sentiment_forward(self, sample_data_loader):
        """Test transformer forward pass."""
        model = TransformerSentiment(
            vocab_size=1000,
            embedding_dim=64,
            num_heads=4,
            num_layers=1,
            ff_dim=128
        )
        model.eval()
        
        with torch.no_grad():
            for batch in sample_data_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                logits = model(input_ids, attention_mask)
                
                assert logits.shape == (input_ids.size(0), 3)
                assert not torch.isnan(logits).any()
                break
    
    def test_sentiment_ensemble_creation(self):
        """Test ensemble model creation."""
        base_models = [
            SentimentLSTM(vocab_size=1000, embedding_dim=64, hidden_dim=32),
            BiLSTMAttentionSentiment(vocab_size=1000, embedding_dim=64, hidden_dim=32)
        ]
        
        ensemble = SentimentEnsemble(base_models, weights=[0.6, 0.4])
        
        assert isinstance(ensemble, nn.Module)
        assert len(ensemble.models) == 2
        assert ensemble.weights == [0.6, 0.4]
    
    def test_sentiment_ensemble_forward(self, sample_data_loader):
        """Test ensemble forward pass."""
        base_models = [
            SentimentLSTM(vocab_size=1000, embedding_dim=64, hidden_dim=32),
            SentimentLSTM(vocab_size=1000, embedding_dim=64, hidden_dim=32)
        ]
        
        ensemble = SentimentEnsemble(base_models)
        ensemble.eval()
        
        with torch.no_grad():
            for batch in sample_data_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                logits = ensemble(input_ids, attention_mask)
                
                assert logits.shape == (input_ids.size(0), 3)
                assert not torch.isnan(logits).any()
                break
    
    def test_create_sentiment_model_factory(self):
        """Test model factory function."""
        vocab_size = 1000
        
        # Test all model types
        model_types = ['lstm', 'bilstm_attention', 'lstm_cnn', 'transformer']
        
        for model_type in model_types:
            model = create_sentiment_model(model_type, vocab_size=vocab_size)
            assert isinstance(model, nn.Module)
    
    def test_create_sentiment_model_invalid_type(self):
        """Test model factory with invalid model type."""
        with pytest.raises(ValueError):
            create_sentiment_model('invalid_model', vocab_size=1000)


class TestSentimentDataProcessing:
    """Test sentiment data processing utilities."""
    
    def test_sentiment_vocabulary_creation(self):
        """Test vocabulary creation."""
        vocab = SentimentVocabulary(min_freq=2, max_vocab_size=1000)
        
        assert vocab.pad_token == "<PAD>"
        assert vocab.unk_token == "<UNK>"
        assert vocab.pad_id == 0
        assert vocab.unk_id == 1
    
    def test_vocabulary_build_from_texts(self, sample_vocab):
        """Test vocabulary building from texts."""
        assert len(sample_vocab) > 4  # At least special tokens
        assert 'movie' in sample_vocab.token_to_id
        assert 'great' in sample_vocab.token_to_id
    
    def test_vocabulary_tokenization(self, sample_vocab):
        """Test text tokenization."""
        text = "This is a great movie!"
        tokens = sample_vocab.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert 'great' in tokens
        assert 'movie' in tokens
    
    def test_vocabulary_encoding(self, sample_vocab):
        """Test text encoding to token IDs."""
        text = "great movie"
        encoded = sample_vocab.encode(text, max_length=10)
        
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert encoded['input_ids'].shape == (10,)
        assert encoded['attention_mask'].shape == (10,)
    
    def test_vocabulary_decoding(self, sample_vocab):
        """Test token ID decoding to text."""
        text = "great movie"
        encoded = sample_vocab.encode(text, max_length=10)
        decoded = sample_vocab.decode(encoded['input_ids'].tolist())
        
        assert isinstance(decoded, str)
        assert 'great' in decoded
        assert 'movie' in decoded
    
    def test_vocabulary_save_load(self, sample_vocab):
        """Test vocabulary save/load functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            vocab_path = f.name
        
        try:
            # Save vocabulary
            sample_vocab.save(vocab_path)
            assert Path(vocab_path).exists()
            
            # Load vocabulary
            loaded_vocab = SentimentVocabulary.load(vocab_path)
            
            assert len(loaded_vocab) == len(sample_vocab)
            assert loaded_vocab.token_to_id == sample_vocab.token_to_id
        finally:
            Path(vocab_path).unlink(missing_ok=True)
    
    def test_sentiment_dataset_creation(self, sample_vocab):
        """Test sentiment dataset creation."""
        texts = ["great movie", "terrible film", "okay story"]
        labels = [2, 0, 1]  # positive, negative, neutral
        
        dataset = SentimentDataset(texts, labels, sample_vocab, max_length=10)
        
        assert len(dataset) == 3
        
        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample
        assert sample['input_ids'].shape == (10,)
    
    def test_data_augmentation(self):
        """Test data augmentation techniques."""
        augmenter = SentimentDataAugmentation(augmentation_prob=1.0)  # Always augment
        
        text = "This is a great movie"
        augmented = augmenter.augment(text, num_aug=2)
        
        assert isinstance(augmented, list)
        assert len(augmented) >= 1  # At least original text
        assert text in augmented  # Original text should be included
    
    def test_create_data_loaders(self, sample_vocab):
        """Test data loader creation."""
        texts = ["great movie"] * 50 + ["terrible film"] * 50
        labels = [2] * 50 + [0] * 50
        
        train_loader, val_loader, test_loader = create_data_loaders(
            texts, labels, sample_vocab,
            batch_size=8, test_size=0.2, val_size=0.2
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Check that loaders produce valid batches
        for batch in train_loader:
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'labels' in batch
            break


class TestSentimentTraining:
    """Test sentiment model training utilities."""
    
    def test_sentiment_trainer_creation(self, sample_data_loader):
        """Test sentiment trainer creation."""
        model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
        
        trainer = SentimentTrainer(
            model=model,
            train_loader=sample_data_loader,
            val_loader=sample_data_loader,
            test_loader=sample_data_loader,
            experiment_name="test_experiment"
        )
        
        assert trainer.model is not None
        assert trainer.train_loader is not None
        assert trainer.experiment_name == "test_experiment"
    
    def test_trainer_train_epoch(self, sample_data_loader):
        """Test single training epoch."""
        model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
        
        trainer = SentimentTrainer(
            model=model,
            train_loader=sample_data_loader,
            val_loader=sample_data_loader,
            experiment_name="test_epoch"
        )
        
        train_loss, train_acc = trainer.train_epoch()
        
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss > 0
        assert 0 <= train_acc <= 1
    
    def test_trainer_validation(self, sample_data_loader):
        """Test model validation."""
        model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
        
        trainer = SentimentTrainer(
            model=model,
            train_loader=sample_data_loader,
            val_loader=sample_data_loader,
            experiment_name="test_validation"
        )
        
        val_loss, val_acc, metrics = trainer.validate()
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
    
    def test_trainer_full_training(self, sample_data_loader):
        """Test full training process (short)."""
        model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
        
        trainer = SentimentTrainer(
            model=model,
            train_loader=sample_data_loader,
            val_loader=sample_data_loader,
            test_loader=sample_data_loader,
            experiment_name="test_full_training",
            patience=2
        )
        
        history = trainer.train(num_epochs=2, verbose=False)
        
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) <= 2
    
    def test_trainer_evaluation(self, sample_data_loader):
        """Test model evaluation."""
        model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
        
        trainer = SentimentTrainer(
            model=model,
            train_loader=sample_data_loader,
            val_loader=sample_data_loader,
            test_loader=sample_data_loader,
            experiment_name="test_evaluation"
        )
        
        results = trainer.evaluate()
        
        assert isinstance(results, dict)
        assert 'test_accuracy' in results
        assert 'test_f1' in results
        assert 'confusion_matrix' in results
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
        
        # Test different optimizers
        optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
        
        for opt_name in optimizers:
            optimizer = create_optimizer(model, opt_name, learning_rate=1e-3)
            assert optimizer is not None
    
    def test_create_scheduler(self):
        """Test scheduler creation."""
        model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
        optimizer = create_optimizer(model, 'adam')
        
        # Test different schedulers
        schedulers = ['cosine', 'step', 'plateau', 'exponential', 'none']
        
        for sched_name in schedulers:
            scheduler = create_scheduler(optimizer, sched_name, num_epochs=10)
            if sched_name == 'none':
                assert scheduler is None
            else:
                assert scheduler is not None


class TestSentimentEvaluation:
    """Test sentiment evaluation and benchmarking utilities."""
    
    def test_sentiment_evaluator_creation(self):
        """Test evaluator creation."""
        evaluator = SentimentEvaluator(num_classes=3)
        
        assert evaluator.num_classes == 3
        assert len(evaluator.class_names) == 3
    
    def test_evaluator_single_model_evaluation(self, sample_data_loader):
        """Test single model evaluation."""
        model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
        evaluator = SentimentEvaluator(num_classes=3)
        
        results = evaluator.evaluate_model(
            model, sample_data_loader, "TestModel"
        )
        
        assert isinstance(results, dict)
        assert 'overall_metrics' in results
        assert 'performance_metrics' in results
        assert 'confusion_matrix' in results
        
        # Check metrics
        metrics = results['overall_metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_score' in metrics
    
    def test_evaluator_model_comparison(self, sample_data_loader):
        """Test model comparison."""
        models = {
            'LSTM': SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16),
            'BiLSTM': BiLSTMAttentionSentiment(vocab_size=1000, embedding_dim=32, hidden_dim=16)
        }
        
        evaluator = SentimentEvaluator(num_classes=3)
        
        comparison_results = evaluator.compare_models(
            models, sample_data_loader, num_runs=2
        )
        
        assert isinstance(comparison_results, dict)
        assert 'detailed_results' in comparison_results
        assert 'statistical_comparisons' in comparison_results
        assert 'rankings' in comparison_results
        assert 'summary' in comparison_results
        
        # Check that both models are in results
        assert 'LSTM' in comparison_results['detailed_results']
        assert 'BiLSTM' in comparison_results['detailed_results']
    
    def test_create_synthetic_benchmark_data(self):
        """Test synthetic data creation for benchmarking."""
        data_loader, label_map = create_synthetic_benchmark_data(
            num_samples=100,
            max_length=32,
            vocab_size=1000,
            num_classes=3,
            batch_size=16
        )
        
        assert isinstance(data_loader, DataLoader)
        assert isinstance(label_map, dict)
        assert len(label_map) == 3
        
        # Check data loader produces valid batches
        for batch in data_loader:
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'labels' in batch
            break
    
    def test_sentiment_benchmark_creation(self):
        """Test benchmark creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = SentimentBenchmark(output_dir=temp_dir)
            
            assert benchmark.output_dir == Path(temp_dir)
            assert isinstance(benchmark.evaluator, SentimentEvaluator)


class TestSentimentIntegration:
    """Test integration between sentiment analysis components."""
    
    def test_end_to_end_training_evaluation(self, sample_vocab):
        """Test complete training and evaluation pipeline."""
        # Create synthetic data
        texts = ["great movie"] * 20 + ["terrible film"] * 20 + ["okay story"] * 20
        labels = [2] * 20 + [0] * 20 + [1] * 20
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            texts, labels, sample_vocab,
            batch_size=8, test_size=0.3, val_size=0.2
        )
        
        # Create model
        model = SentimentLSTM(
            vocab_size=len(sample_vocab),
            embedding_dim=32,
            hidden_dim=16,
            num_classes=3
        )
        
        # Train model (short training)
        trainer = SentimentTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            experiment_name="integration_test",
            patience=2
        )
        
        history = trainer.train(num_epochs=2, verbose=False)
        results = trainer.evaluate()
        
        # Verify results
        assert len(history['train_loss']) <= 2
        assert 'test_accuracy' in results
        assert 0 <= results['test_accuracy'] <= 1
    
    def test_model_comparison_integration(self, sample_vocab):
        """Test model comparison integration."""
        # Create minimal data
        texts = ["good"] * 10 + ["bad"] * 10
        labels = [1] * 10 + [0] * 10
        
        # Create data loader
        data_loader = create_data_loaders(
            texts, labels, sample_vocab, batch_size=4, test_size=0.0, val_size=0.0
        )[0]  # Just use train_loader as test set
        
        # Create models
        models = {
            'LSTM': SentimentLSTM(vocab_size=len(sample_vocab), embedding_dim=16, hidden_dim=8),
            'BiLSTM': BiLSTMAttentionSentiment(vocab_size=len(sample_vocab), embedding_dim=16, hidden_dim=8)
        }
        
        # Compare models
        evaluator = SentimentEvaluator(num_classes=2)
        results = evaluator.compare_models(models, data_loader, num_runs=2)
        
        assert 'LSTM' in results['detailed_results']
        assert 'BiLSTM' in results['detailed_results']
        assert 'LSTM_vs_BiLSTM' in results['statistical_comparisons']
    
    def test_benchmark_integration(self, sample_vocab):
        """Test benchmark integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal datasets
            datasets = {}
            for dataset_name in ['dataset1', 'dataset2']:
                texts = ["good"] * 8 + ["bad"] * 8
                labels = [1] * 8 + [0] * 8
                
                data_loader = create_data_loaders(
                    texts, labels, sample_vocab, batch_size=4, test_size=0.0, val_size=0.0
                )[0]
                datasets[dataset_name] = data_loader
            
            # Create models
            models = {
                'LSTM': SentimentLSTM(vocab_size=len(sample_vocab), embedding_dim=16, hidden_dim=8)
            }
            
            # Run benchmark
            benchmark = SentimentBenchmark(output_dir=temp_dir)
            results = benchmark.run_benchmark(
                models, datasets, num_runs=1, save_results=False
            )
            
            assert 'dataset_results' in results
            assert 'cross_dataset_analysis' in results
            assert 'dataset1' in results['dataset_results']
            assert 'dataset2' in results['dataset_results']


class TestSentimentEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_vocabulary(self):
        """Test vocabulary with no training texts."""
        vocab = SentimentVocabulary()
        vocab.build_from_texts([])
        
        # Should have only special tokens
        assert len(vocab) == 4  # PAD, UNK, CLS, SEP
    
    def test_model_with_zero_hidden_dim(self):
        """Test model creation with invalid parameters."""
        with pytest.raises(Exception):
            model = SentimentLSTM(
                vocab_size=1000,
                hidden_dim=0,  # Invalid
                embedding_dim=32
            )
    
    def test_empty_data_loader(self):
        """Test evaluation with empty data."""
        model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
        
        # Create empty data loader
        dataset = TensorDataset(
            torch.empty(0, 10, dtype=torch.long),
            torch.empty(0, 10, dtype=torch.long),
            torch.empty(0, dtype=torch.long)
        )
        
        def collate_fn(batch):
            if not batch:
                return {
                    'input_ids': torch.empty(0, 10, dtype=torch.long),
                    'attention_mask': torch.empty(0, 10, dtype=torch.long),
                    'labels': torch.empty(0, dtype=torch.long)
                }
            input_ids, attention_mask, labels = zip(*batch)
            return {
                'input_ids': torch.stack(input_ids),
                'attention_mask': torch.stack(attention_mask),
                'labels': torch.stack(labels)
            }
        
        empty_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        
        evaluator = SentimentEvaluator()
        
        # This should handle empty data gracefully
        results = evaluator.evaluate_model(model, empty_loader)
        
        # Results should contain default/empty values
        assert 'overall_metrics' in results
    
    def test_single_class_data(self, sample_vocab):
        """Test with data containing only one class."""
        texts = ["good movie"] * 10
        labels = [1] * 10  # Only positive class
        
        data_loader = create_data_loaders(
            texts, labels, sample_vocab, batch_size=4, test_size=0.0, val_size=0.0
        )[0]
        
        model = SentimentLSTM(vocab_size=len(sample_vocab), embedding_dim=16, hidden_dim=8)
        evaluator = SentimentEvaluator(num_classes=3)
        
        # Should handle single-class data without crashing
        results = evaluator.evaluate_model(model, data_loader)
        assert 'overall_metrics' in results
    
    def test_model_save_load_integration(self, sample_data_loader):
        """Test model save/load integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
            
            trainer = SentimentTrainer(
                model=model,
                train_loader=sample_data_loader,
                val_loader=sample_data_loader,
                test_loader=sample_data_loader,
                save_dir=temp_dir,
                experiment_name="save_load_test"
            )
            
            # Train briefly
            trainer.train(num_epochs=1, verbose=False)
            
            # Save model
            save_path = Path(temp_dir) / "test_model.pth"
            trainer.save_model(str(save_path))
            assert save_path.exists()
            
            # Create new trainer and load model
            new_model = SentimentLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=16)
            new_trainer = SentimentTrainer(
                model=new_model,
                train_loader=sample_data_loader,
                val_loader=sample_data_loader,
                test_loader=sample_data_loader,
                save_dir=temp_dir,
                experiment_name="save_load_test"
            )
            
            new_trainer.load_model(str(save_path))
            
            # Models should produce similar outputs
            model.eval()
            new_model.eval()
            
            with torch.no_grad():
                for batch in sample_data_loader:
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    
                    original_output = model(input_ids, attention_mask)
                    loaded_output = new_model(input_ids, attention_mask)
                    
                    # Should be very close (allowing for floating point differences)
                    assert torch.allclose(original_output, loaded_output, atol=1e-5)
                    break


# Performance and stress tests
class TestSentimentPerformance:
    """Test performance characteristics of sentiment analysis components."""
    
    @pytest.mark.slow
    def test_large_vocabulary_performance(self):
        """Test performance with large vocabulary."""
        large_vocab_size = 50000
        
        model = SentimentLSTM(
            vocab_size=large_vocab_size,
            embedding_dim=128,
            hidden_dim=64
        )
        
        # Test forward pass
        batch_size = 32
        seq_len = 128
        input_ids = torch.randint(0, large_vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            logits = model(input_ids, attention_mask)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                
                # Should complete within reasonable time (< 1000ms)
                assert elapsed_time < 1000
            
            assert logits.shape == (batch_size, 3)
    
    @pytest.mark.slow
    def test_batch_size_scaling(self):
        """Test performance with different batch sizes."""
        model = SentimentLSTM(vocab_size=10000, embedding_dim=64, hidden_dim=32)
        model.eval()
        
        seq_len = 64
        batch_sizes = [1, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            input_ids = torch.randint(0, 10000, (batch_size, seq_len))
            attention_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                assert logits.shape == (batch_size, 3)
                assert not torch.isnan(logits).any()
    
    def test_memory_efficiency(self):
        """Test memory efficiency of models."""
        models = {
            'LSTM': SentimentLSTM(vocab_size=10000, embedding_dim=64, hidden_dim=32),
            'BiLSTM': BiLSTMAttentionSentiment(vocab_size=10000, embedding_dim=64, hidden_dim=32),
            'Transformer': TransformerSentiment(vocab_size=10000, embedding_dim=64, num_heads=4, num_layers=2)
        }
        
        for name, model in models.items():
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            # Reasonable parameter counts
            assert param_count < 10_000_000  # Less than 10M parameters
            
            # Memory footprint should be reasonable
            model_size_mb = param_count * 4 / (1024 ** 2)  # Assuming float32
            assert model_size_mb < 100  # Less than 100MB


# Test fixtures for complex scenarios
@pytest.fixture
def multilingual_vocab():
    """Create vocabulary with multilingual content."""
    vocab = SentimentVocabulary(min_freq=1, max_vocab_size=2000)
    texts = [
        "This movie is great",  # English
        "Esta película es genial",  # Spanish
        "Ce film est génial",  # French
        "Dieser Film ist großartig",  # German
        "Этот фильм великолепен"  # Russian (Cyrillic)
    ]
    vocab.build_from_texts(texts)
    return vocab


class TestSentimentMultilingual:
    """Test multilingual sentiment analysis capabilities."""
    
    def test_multilingual_vocabulary(self, multilingual_vocab):
        """Test vocabulary with multilingual content."""
        assert len(multilingual_vocab) > 10
        
        # Test encoding different languages
        languages = [
            "great movie",
            "gran película",
            "excellent film",
            "großartiger Film"
        ]
        
        for text in languages:
            encoded = multilingual_vocab.encode(text, max_length=20)
            assert encoded['input_ids'].shape == (20,)
            assert encoded['attention_mask'].sum() > 0  # Some tokens should be non-padding
    
    def test_multilingual_model_training(self, multilingual_vocab):
        """Test model training with multilingual data."""
        texts = [
            "great movie", "excellent film", "wonderful story",
            "terrible movie", "bad film", "awful story",
            "okay movie", "average film", "decent story"
        ]
        labels = [2, 2, 2, 0, 0, 0, 1, 1, 1]
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            texts, labels, multilingual_vocab,
            batch_size=4, test_size=0.3, val_size=0.2
        )
        
        # Create and train model
        model = SentimentLSTM(
            vocab_size=len(multilingual_vocab),
            embedding_dim=32,
            hidden_dim=16,
            num_classes=3
        )
        
        trainer = SentimentTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            experiment_name="multilingual_test",
            patience=2
        )
        
        history = trainer.train(num_epochs=2, verbose=False)
        results = trainer.evaluate()
        
        # Should complete without errors
        assert len(history['train_loss']) <= 2
        assert 0 <= results['test_accuracy'] <= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
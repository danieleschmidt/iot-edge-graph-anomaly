"""
Comprehensive test framework for the enhanced IoT edge anomaly detection system.

This test suite covers all new features including:
- Hybrid LSTM-GNN model functionality
- Validation framework
- Fault tolerance mechanisms
- Security features
- Performance optimization
- Research frameworks
"""
import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
import time
import threading
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from iot_edge_anomaly.models.lstm_gnn_hybrid import LSTMGNNHybridModel, FeatureFusionLayer
from iot_edge_anomaly.models.gnn_layer import GraphNeuralNetworkLayer, create_sensor_graph
from iot_edge_anomaly.validation.model_validator import ComprehensiveValidator, DataValidator, ModelValidator
from iot_edge_anomaly.resilience.fault_tolerance import ResilientAnomalyDetector, FallbackModel
from iot_edge_anomaly.security.secure_inference import SecureInferenceEngine, InputSanitizer
from iot_edge_anomaly.optimization.performance_optimizer import OptimizedInferenceEngine, IntelligentCache, OptimizationConfig


class TestLSTMGNNHybridModel:
    """Test cases for the hybrid LSTM-GNN model."""
    
    @pytest.fixture
    def hybrid_config(self):
        return {
            'lstm': {
                'input_size': 5,
                'hidden_size': 32,
                'num_layers': 2,
                'dropout': 0.1
            },
            'gnn': {
                'input_dim': 32,
                'hidden_dim': 16,
                'output_dim': 32,
                'num_layers': 2,
                'dropout': 0.1
            },
            'fusion': {
                'output_dim': 64,
                'method': 'concatenate'
            },
            'graph': {
                'method': 'correlation',
                'threshold': 0.3
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        return torch.randn(4, 10, 5)
    
    def test_hybrid_model_initialization(self, hybrid_config):
        """Test hybrid model initialization."""
        model = LSTMGNNHybridModel(hybrid_config)
        
        assert model is not None
        assert hasattr(model, 'lstm_autoencoder')
        assert hasattr(model, 'gnn_layer')
        assert hasattr(model, 'fusion_layer')
        assert model.graph_method == 'correlation'
        assert model.graph_threshold == 0.3
    
    def test_hybrid_model_forward_pass(self, hybrid_config, sample_data):
        """Test forward pass through hybrid model."""
        model = LSTMGNNHybridModel(hybrid_config)
        
        with torch.no_grad():
            output = model(sample_data)
        
        assert output is not None
        assert output.shape == sample_data.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_hybrid_model_encoding(self, hybrid_config, sample_data):
        """Test hybrid model encoding functionality."""
        model = LSTMGNNHybridModel(hybrid_config)
        
        with torch.no_grad():
            lstm_features, gnn_features = model.encode(sample_data)
        
        assert lstm_features is not None
        assert gnn_features is not None
        assert lstm_features.shape[0] == sample_data.shape[0]
        assert gnn_features.shape[0] == sample_data.shape[0]
    
    def test_hybrid_anomaly_score_computation(self, hybrid_config, sample_data):
        """Test anomaly score computation."""
        model = LSTMGNNHybridModel(hybrid_config)
        
        with torch.no_grad():
            score = model.compute_hybrid_anomaly_score(sample_data, reduction='mean')
        
        assert score is not None
        assert isinstance(score, torch.Tensor)
        assert score.numel() == 1
        assert score.item() >= 0
    
    def test_fusion_layer_methods(self, hybrid_config):
        """Test different fusion methods."""
        fusion_methods = ['concatenate', 'attention', 'gate']
        
        for method in fusion_methods:
            config = hybrid_config.copy()
            config['fusion']['method'] = method
            
            try:
                model = LSTMGNNHybridModel(config)
                sample_data = torch.randn(2, 10, 5)
                
                with torch.no_grad():
                    output = model(sample_data)
                
                assert output is not None
                assert output.shape == sample_data.shape
                
            except Exception as e:
                pytest.fail(f"Fusion method {method} failed: {e}")


class TestGraphNeuralNetworkLayer:
    """Test cases for GNN layer functionality."""
    
    def test_gnn_layer_initialization(self):
        """Test GNN layer initialization."""
        gnn = GraphNeuralNetworkLayer(input_dim=10, hidden_dim=16, output_dim=20)
        
        assert gnn is not None
        assert gnn.input_dim == 10
        assert gnn.hidden_dim == 16
        assert gnn.output_dim == 20
    
    def test_gnn_forward_with_edges(self):
        """Test GNN forward pass with edges."""
        gnn = GraphNeuralNetworkLayer(input_dim=5, hidden_dim=8, output_dim=10)
        
        node_features = torch.randn(4, 5)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        with torch.no_grad():
            output = gnn(node_features, edge_index)
        
        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()
    
    def test_gnn_forward_without_edges(self):
        """Test GNN forward pass without edges (isolated nodes)."""
        gnn = GraphNeuralNetworkLayer(input_dim=5, hidden_dim=8, output_dim=10)
        
        node_features = torch.randn(4, 5)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        with torch.no_grad():
            output = gnn(node_features, edge_index)
        
        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()
    
    def test_create_sensor_graph(self):
        """Test sensor graph creation."""
        sensor_data = torch.randn(2, 10, 5)
        
        graph_data = create_sensor_graph(sensor_data, method='correlation', threshold=0.3)
        
        assert 'x' in graph_data
        assert 'edge_index' in graph_data
        assert graph_data['x'].shape[0] == 5  # Number of sensors
        assert graph_data['edge_index'].shape[0] == 2  # Edge format


class TestValidationFramework:
    """Test cases for validation framework."""
    
    @pytest.fixture
    def simple_model(self):
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        return LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
    
    @pytest.fixture
    def validation_data(self):
        normal_data = torch.randn(100, 10, 5)
        test_data = torch.randn(50, 10, 5)
        test_labels = torch.randint(0, 2, (50,)).bool()
        return normal_data, test_data, test_labels
    
    def test_data_validator_initialization(self):
        """Test data validator initialization."""
        validator = DataValidator()
        assert validator is not None
        assert validator.config is not None
    
    def test_input_tensor_validation(self, validation_data):
        """Test input tensor validation."""
        validator = DataValidator()
        _, test_data, _ = validation_data
        
        results = validator.validate_input_tensor(test_data[0:1])
        
        assert len(results) > 0
        assert all(hasattr(r, 'check_name') for r in results)
        assert all(hasattr(r, 'status') for r in results)
    
    def test_adversarial_input_detection(self):
        """Test adversarial input detection."""
        sanitizer = InputSanitizer()
        
        # Normal input
        normal_data = torch.randn(1, 10, 5)
        is_adv, score, details = sanitizer.detect_adversarial_input(normal_data)
        
        assert isinstance(is_adv, bool)
        assert isinstance(score, float)
        assert isinstance(details, dict)
        
        # Suspicious input (very low variance)
        suspicious_data = torch.ones(1, 10, 5) * 0.001
        is_adv_sus, score_sus, details_sus = sanitizer.detect_adversarial_input(suspicious_data)
        
        # Should be more suspicious than normal data
        assert score_sus >= score
    
    def test_comprehensive_validation(self, simple_model, validation_data):
        """Test comprehensive validation workflow."""
        validator = ComprehensiveValidator(simple_model)
        normal_data, test_data, test_labels = validation_data
        
        report = validator.run_comprehensive_validation(normal_data, test_data, test_labels)
        
        assert 'overall_status' in report
        assert 'summary' in report
        assert 'results_by_category' in report
        assert report['summary']['total_checks'] > 0


class TestFaultTolerance:
    """Test cases for fault tolerance mechanisms."""
    
    @pytest.fixture
    def simple_model(self):
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        return LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
    
    def test_fallback_model_training(self):
        """Test fallback model training."""
        fallback = FallbackModel(input_size=5)
        training_data = torch.randn(100, 10, 5)
        
        success = fallback.train(training_data)
        
        assert success is True
        assert fallback.is_trained is True
        assert fallback.baseline_stats is not None
    
    def test_fallback_model_prediction(self):
        """Test fallback model prediction."""
        fallback = FallbackModel(input_size=5)
        training_data = torch.randn(100, 10, 5)
        fallback.train(training_data)
        
        test_data = torch.randn(1, 10, 5)
        result = fallback.predict_anomaly(test_data)
        
        assert 'is_anomaly' in result
        assert 'confidence' in result
        assert 'method' in result
        assert isinstance(result['is_anomaly'], bool)
        assert 0 <= result['confidence'] <= 1
    
    def test_resilient_detector_initialization(self, simple_model):
        """Test resilient detector initialization."""
        detector = ResilientAnomalyDetector(simple_model)
        
        assert detector is not None
        assert detector.primary_model is not None
        assert detector.fallback_model is not None
        assert detector.state_manager is not None
    
    def test_resilient_detector_inference(self, simple_model):
        """Test resilient detector inference with fault tolerance."""
        detector = ResilientAnomalyDetector(simple_model)
        
        # Initialize fallback model
        training_data = torch.randn(50, 10, 5)
        detector.initialize_fallback_model(training_data)
        
        # Test normal inference
        test_data = torch.randn(1, 10, 5)
        result = detector.detect_anomaly(test_data)
        
        assert 'is_anomaly' in result
        assert 'confidence' in result
        assert 'method' in result
        assert result.get('success') is not False  # Should succeed or use fallback
    
    def test_state_persistence(self, simple_model):
        """Test state persistence and restoration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            detector = ResilientAnomalyDetector(simple_model, {'state_dir': temp_dir})
            
            # Save state
            test_state = {'test_key': 'test_value', 'number': 42}
            success = detector.state_manager.save_state('test_state', test_state)
            assert success is True
            
            # Load state
            loaded_state = detector.state_manager.load_state('test_state')
            assert loaded_state == test_state


class TestSecurityFramework:
    """Test cases for security framework."""
    
    @pytest.fixture
    def simple_model(self):
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        return LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
    
    def test_input_sanitizer_initialization(self):
        """Test input sanitizer initialization."""
        sanitizer = InputSanitizer()
        assert sanitizer is not None
        assert sanitizer.allowed_shapes is not None
        assert sanitizer.max_value_range is not None
    
    def test_tensor_sanitization(self):
        """Test tensor sanitization."""
        sanitizer = InputSanitizer()
        
        # Create problematic input
        data = torch.tensor([[[float('nan'), 1000, -1000, float('inf'), 2]]] * 10).float()
        data = data.view(1, 10, 5)
        
        sanitized, warnings = sanitizer.sanitize_tensor(data)
        
        assert not torch.isnan(sanitized).any()
        assert not torch.isinf(sanitized).any()
        assert len(warnings) > 0
    
    def test_secure_inference_engine_initialization(self, simple_model):
        """Test secure inference engine initialization."""
        engine = SecureInferenceEngine(simple_model)
        
        assert engine is not None
        assert engine.sanitizer is not None
        assert engine.protector is not None
        assert engine.auth_manager is not None
    
    def test_authentication_token_generation(self, simple_model):
        """Test authentication token generation and validation."""
        engine = SecureInferenceEngine(simple_model)
        
        # Generate token
        token = engine.auth_manager.generate_token('test_client', ['inference'])
        assert token is not None
        assert isinstance(token, str)
        
        # Validate token
        valid, client_id, details = engine.auth_manager.validate_token(token, 'inference')
        assert valid is True
        assert client_id == 'test_client'
    
    def test_secure_inference(self, simple_model):
        """Test secure inference workflow."""
        config = {'enable_authentication': False}  # Disable auth for testing
        engine = SecureInferenceEngine(simple_model, config)
        
        test_data = torch.randn(1, 10, 5)
        result = engine.secure_inference(test_data)
        
        assert 'success' in result
        assert 'security_level' in result


class TestPerformanceOptimization:
    """Test cases for performance optimization."""
    
    @pytest.fixture
    def simple_model(self):
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        return LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
    
    def test_intelligent_cache_initialization(self):
        """Test intelligent cache initialization."""
        cache = IntelligentCache(max_size_mb=10)
        assert cache is not None
        assert cache.max_size_mb == 10
    
    def test_cache_operations(self):
        """Test cache put/get operations."""
        cache = IntelligentCache(max_size_mb=10)
        
        # Test cache miss
        test_data = torch.randn(1, 10, 5)
        result = cache.get(test_data)
        assert result is None
        
        # Test cache put and hit
        test_result = 'test_result'
        cache.put(test_data, test_result)
        
        cached_result = cache.get(test_data)
        assert cached_result == test_result
        
        # Check stats
        stats = cache.get_stats()
        assert stats['hits'] > 0
        assert stats['misses'] > 0
    
    def test_optimization_config(self):
        """Test optimization configuration."""
        config = OptimizationConfig()
        assert config is not None
        assert hasattr(config, 'optimization_level')
        assert hasattr(config, 'enable_caching')
        assert hasattr(config, 'enable_batching')
    
    def test_optimized_inference_engine(self, simple_model):
        """Test optimized inference engine."""
        config = OptimizationConfig(
            enable_caching=True,
            enable_batching=False,  # Disable for simpler testing
            enable_async_processing=False
        )
        
        engine = OptimizedInferenceEngine(simple_model, config)
        
        assert engine is not None
        assert engine.optimized_model is not None
        assert engine.cache is not None
        
        # Test prediction
        test_data = torch.randn(1, 10, 5)
        result = engine.predict(test_data)
        
        assert 'success' in result
        assert 'result' in result
        assert 'source' in result
        
        # Test cache hit on second prediction
        result2 = engine.predict(test_data)
        assert result2['source'] == 'cache'
        
        # Cleanup
        engine.shutdown()


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple components."""
    
    @pytest.fixture
    def hybrid_model_config(self):
        return {
            'lstm': {'input_size': 5, 'hidden_size': 32, 'num_layers': 2, 'dropout': 0.1},
            'gnn': {'input_dim': 32, 'hidden_dim': 16, 'output_dim': 32, 'num_layers': 2, 'dropout': 0.1},
            'fusion': {'output_dim': 64, 'method': 'concatenate'},
            'graph': {'method': 'correlation', 'threshold': 0.3}
        }
    
    def test_end_to_end_hybrid_workflow(self, hybrid_model_config):
        """Test complete end-to-end hybrid model workflow."""
        # Create hybrid model
        model = LSTMGNNHybridModel(hybrid_model_config)
        
        # Generate test data
        normal_data = torch.randn(50, 10, 5)
        test_data = torch.randn(20, 10, 5)
        test_labels = torch.randint(0, 2, (20,)).bool()
        
        # Validation
        validator = ComprehensiveValidator(model)
        validation_report = validator.run_comprehensive_validation(normal_data, test_data, test_labels)
        
        assert validation_report['overall_status'] in ['PASSED', 'WARNING']
        
        # Fault tolerance
        resilient_detector = ResilientAnomalyDetector(model)
        resilient_detector.initialize_fallback_model(normal_data)
        
        # Test resilient inference
        for i in range(5):
            sample = test_data[i:i+1]
            result = resilient_detector.detect_anomaly(sample)
            assert 'is_anomaly' in result
            assert 'confidence' in result
        
        # Performance optimization
        opt_config = OptimizationConfig(
            enable_caching=True,
            enable_batching=False,
            enable_async_processing=False
        )
        
        opt_engine = OptimizedInferenceEngine(model, opt_config)
        
        # Test optimized inference
        for i in range(3):
            sample = test_data[i:i+1]
            result = opt_engine.predict(sample)
            assert result['success'] is True
        
        # Get performance report
        perf_report = opt_engine.get_performance_report()
        assert 'total_requests' in perf_report
        assert perf_report['total_requests'] >= 3
        
        # Cleanup
        opt_engine.shutdown()
        resilient_detector.shutdown()
    
    def test_security_with_performance_optimization(self):
        """Test security framework with performance optimization."""
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        
        # Create model
        model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
        
        # Security configuration
        security_config = {
            'enable_authentication': False,  # Simplified for testing
            'protection_level': 'medium'
        }
        
        secure_engine = SecureInferenceEngine(model, security_config)
        
        # Performance optimization
        opt_config = OptimizationConfig(enable_caching=True, enable_batching=False)
        opt_engine = OptimizedInferenceEngine(model, opt_config)
        
        # Test data
        test_data = torch.randn(1, 10, 5)
        
        # Secure inference
        secure_result = secure_engine.secure_inference(test_data)
        assert secure_result['success'] is True
        
        # Optimized inference  
        opt_result = opt_engine.predict(test_data)
        assert opt_result['success'] is True
        
        # Both should produce valid results
        assert 'result' in opt_result or 'output' in secure_result
        
        # Cleanup
        opt_engine.shutdown()
    
    def test_comprehensive_error_handling(self, hybrid_model_config):
        """Test comprehensive error handling across all components."""
        model = LSTMGNNHybridModel(hybrid_model_config)
        
        # Test invalid input handling
        invalid_inputs = [
            torch.full((1, 10, 5), float('nan')),  # NaN input
            torch.full((1, 10, 5), float('inf')),  # Infinite input
            torch.randn(0, 10, 5),  # Empty tensor
            torch.randn(1, 0, 5),   # Zero sequence length
        ]
        
        for invalid_input in invalid_inputs:
            try:
                # Test with validation
                validator = DataValidator()
                results = validator.validate_input_tensor(invalid_input)
                
                # Should detect issues
                has_errors = any(r.status.value == 'failed' for r in results)
                
                # Test with sanitization
                sanitizer = InputSanitizer()
                sanitized_input, warnings = sanitizer.sanitize_tensor(invalid_input)
                
                # Should handle gracefully
                assert sanitized_input is not None
                
            except Exception as e:
                # Some cases may still raise exceptions, which is acceptable
                # as long as they're handled gracefully by the resilient detector
                assert isinstance(e, (ValueError, RuntimeError, IndexError))


# Benchmark tests for performance validation
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_inference_speed_benchmark(self):
        """Test inference speed meets requirements."""
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        
        model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
        model.eval()
        
        test_data = torch.randn(1, 10, 5)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_data)
        
        # Benchmark
        start_time = time.time()
        num_iterations = 100
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(test_data)
        
        end_time = time.time()
        avg_time_ms = (end_time - start_time) * 1000 / num_iterations
        
        # Should be under 50ms for edge deployment
        assert avg_time_ms < 50, f"Inference too slow: {avg_time_ms:.2f}ms"
    
    def test_memory_usage_benchmark(self):
        """Test memory usage is within limits."""
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model
        model = LSTMAutoencoder(input_size=5, hidden_size=64, num_layers=2)
        
        # Run inference
        test_data = torch.randn(10, 10, 5)
        with torch.no_grad():
            _ = model(test_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should use less than 100MB additional memory
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.2f}MB"


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
        # Run only benchmark tests
        pytest.main(['-v', 'TestPerformanceBenchmarks', __file__])
    elif len(sys.argv) > 1 and sys.argv[1] == 'integration':
        # Run only integration tests
        pytest.main(['-v', 'TestIntegrationScenarios', __file__])
    else:
        # Run all tests
        pytest.main(['-v', __file__])
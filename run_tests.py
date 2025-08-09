#!/usr/bin/env python3
"""
Simple test runner for the comprehensive framework without pytest dependency.
"""
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_basic_tests():
    """Run basic tests to verify the framework works."""
    print("🧪 Running Comprehensive Framework Tests")
    print("=" * 50)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Hybrid Model Initialization
    try:
        from iot_edge_anomaly.models.lstm_gnn_hybrid import LSTMGNNHybridModel
        
        config = {
            'lstm': {'input_size': 5, 'hidden_size': 32, 'num_layers': 2, 'dropout': 0.1},
            'gnn': {'input_dim': 32, 'hidden_dim': 16, 'output_dim': 32, 'num_layers': 2, 'dropout': 0.1},
            'fusion': {'output_dim': 64, 'method': 'concatenate'},
            'graph': {'method': 'correlation', 'threshold': 0.3}
        }
        
        model = LSTMGNNHybridModel(config)
        assert model is not None
        assert hasattr(model, 'lstm_autoencoder')
        assert hasattr(model, 'gnn_layer')
        print("✅ Test 1 PASSED: Hybrid Model Initialization")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: Hybrid Model Initialization - {e}")
        tests_failed += 1
    
    # Test 2: GNN Layer Forward Pass
    try:
        import torch
        from iot_edge_anomaly.models.gnn_layer import GraphNeuralNetworkLayer
        
        gnn = GraphNeuralNetworkLayer(input_dim=5, hidden_dim=8, output_dim=10)
        node_features = torch.randn(4, 5)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        with torch.no_grad():
            output = gnn(node_features, edge_index)
        
        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()
        print("✅ Test 2 PASSED: GNN Layer Forward Pass")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: GNN Layer Forward Pass - {e}")
        tests_failed += 1
    
    # Test 3: Validation Framework
    try:
        from iot_edge_anomaly.validation.model_validator import DataValidator
        
        validator = DataValidator()
        test_data = torch.randn(1, 10, 5)
        results = validator.validate_input_tensor(test_data)
        
        assert len(results) > 0
        assert all(hasattr(r, 'check_name') for r in results)
        print("✅ Test 3 PASSED: Validation Framework")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: Validation Framework - {e}")
        tests_failed += 1
    
    # Test 4: Fault Tolerance
    try:
        from iot_edge_anomaly.resilience.fault_tolerance import FallbackModel
        
        fallback = FallbackModel(input_size=5)
        training_data = torch.randn(50, 10, 5)
        success = fallback.train(training_data)
        
        assert success is True
        assert fallback.is_trained is True
        print("✅ Test 4 PASSED: Fault Tolerance")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 4 FAILED: Fault Tolerance - {e}")
        tests_failed += 1
    
    # Test 5: Security Framework
    try:
        from iot_edge_anomaly.security.secure_inference import InputSanitizer
        
        sanitizer = InputSanitizer()
        # Create problematic input
        data = torch.tensor([[[float('nan'), 1000, -1000, 2, 3]]] * 10).float()
        data = data.view(1, 10, 5)
        
        sanitized, warnings = sanitizer.sanitize_tensor(data)
        
        assert not torch.isnan(sanitized).any()
        assert len(warnings) > 0
        print("✅ Test 5 PASSED: Security Framework")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 5 FAILED: Security Framework - {e}")
        tests_failed += 1
    
    # Test 6: Performance Optimization
    try:
        from iot_edge_anomaly.optimization.performance_optimizer import IntelligentCache
        
        cache = IntelligentCache(max_size_mb=10)
        test_data = torch.randn(1, 10, 5)
        
        # Test cache miss
        result = cache.get(test_data)
        assert result is None
        
        # Test cache put and hit
        cache.put(test_data, 'test_result')
        cached_result = cache.get(test_data)
        assert cached_result == 'test_result'
        
        print("✅ Test 6 PASSED: Performance Optimization")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 6 FAILED: Performance Optimization - {e}")
        tests_failed += 1
    
    # Test 7: Integration Test
    try:
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        from iot_edge_anomaly.validation.model_validator import ComprehensiveValidator
        
        # Create simple model
        model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
        validator = ComprehensiveValidator(model)
        
        # Generate test data
        normal_data = torch.randn(20, 10, 5)
        test_data = torch.randn(10, 10, 5)
        test_labels = torch.randint(0, 2, (10,)).bool()
        
        # Run validation
        report = validator.run_comprehensive_validation(normal_data, test_data, test_labels)
        
        assert 'overall_status' in report
        assert 'summary' in report
        assert report['summary']['total_checks'] > 0
        
        print("✅ Test 7 PASSED: Integration Test")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 7 FAILED: Integration Test - {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 TEST SUMMARY")
    print(f"✅ Tests Passed: {tests_passed}")
    print(f"❌ Tests Failed: {tests_failed}")
    print(f"📈 Success Rate: {tests_passed/(tests_passed+tests_failed)*100:.1f}%")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Framework is working correctly.")
        return True
    else:
        print(f"\n⚠️  {tests_failed} tests failed. Please check the implementation.")
        return False

def run_performance_benchmark():
    """Run performance benchmarks."""
    print("\n🚀 Running Performance Benchmarks")
    print("=" * 50)
    
    try:
        import torch
        import time
        import psutil
        import os
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        
        # Inference Speed Test
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
        
        print(f"⚡ Average Inference Time: {avg_time_ms:.2f}ms")
        
        if avg_time_ms < 50:
            print("✅ Inference speed meets requirements (<50ms)")
        else:
            print("⚠️  Inference speed may be too slow for edge deployment")
        
        # Memory Usage Test
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"💾 Current Memory Usage: {current_memory:.1f}MB")
        
        # Model Size Test
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        print(f"🧠 Model Size: {model_size:.2f}MB")
        
        if model_size < 100:
            print("✅ Model size meets requirements (<100MB)")
        else:
            print("⚠️  Model size may be too large for edge deployment")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 IoT Edge Anomaly Detection - Comprehensive Test Suite")
    print("=" * 70)
    
    # Run basic functionality tests
    basic_success = run_basic_tests()
    
    # Run performance benchmarks
    perf_success = run_performance_benchmark()
    
    # Overall result
    print("\n" + "=" * 70)
    if basic_success and perf_success:
        print("🎉 ALL TESTS AND BENCHMARKS PASSED!")
        print("✅ The enhanced IoT edge anomaly detection framework is ready!")
        exit(0)
    else:
        print("❌ Some tests failed. Please review the implementation.")
        exit(1)
#!/usr/bin/env python3
"""
Comprehensive test suite for Terragon Autonomous SDLC IoT Anomaly Detection System.

This script runs all quality gates and validation tests to ensure the system
meets production-ready standards across all three generations:
1. Generation 1: Basic Functionality
2. Generation 2: Robust and Reliable
3. Generation 3: Optimized and Scalable
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess
import json

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Test results storage
test_results = {
    'generation_1': {'tests': [], 'passed': 0, 'failed': 0, 'status': 'unknown'},
    'generation_2': {'tests': [], 'passed': 0, 'failed': 0, 'status': 'unknown'}, 
    'generation_3': {'tests': [], 'passed': 0, 'failed': 0, 'status': 'unknown'},
    'overall': {'total_tests': 0, 'total_passed': 0, 'total_failed': 0, 'status': 'unknown'}
}


def run_test(test_name: str, test_func, generation: str) -> bool:
    """Run individual test and record results."""
    print(f"🧪 Running {test_name}...")
    
    try:
        start_time = time.time()
        test_func()
        end_time = time.time()
        
        test_result = {
            'name': test_name,
            'status': 'PASSED',
            'duration': end_time - start_time,
            'error': None
        }
        
        test_results[generation]['tests'].append(test_result)
        test_results[generation]['passed'] += 1
        print(f"  ✅ {test_name} PASSED ({test_result['duration']:.3f}s)")
        return True
        
    except Exception as e:
        test_result = {
            'name': test_name,
            'status': 'FAILED',
            'duration': time.time() - start_time if 'start_time' in locals() else 0,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        
        test_results[generation]['tests'].append(test_result)
        test_results[generation]['failed'] += 1
        print(f"  ❌ {test_name} FAILED: {e}")
        return False


def test_basic_imports():
    """Test that all essential modules can be imported."""
    import torch
    import numpy as np
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    from iot_edge_anomaly.performance_optimizer import performance_monitor
    print("    All basic imports successful")


def test_model_creation_and_inference():
    """Test basic model creation and inference."""
    import torch
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    
    # Create model
    model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
    model.eval()
    
    # Test inference
    test_input = torch.randn(2, 20, 5)
    with torch.no_grad():
        reconstruction = model(test_input)
        error = model.compute_reconstruction_error(test_input)
    
    assert reconstruction.shape == test_input.shape, f"Shape mismatch: {reconstruction.shape} vs {test_input.shape}"
    assert error.numel() == 1, f"Error should be scalar, got shape: {error.shape}"
    assert not torch.isnan(error).any(), "Reconstruction error contains NaN"
    assert not torch.isinf(error).any(), "Reconstruction error contains Inf"
    
    print(f"    Model inference successful (error: {error.item():.4f})")


def test_performance_optimization():
    """Test performance optimization features."""
    import torch
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    from iot_edge_anomaly.performance_optimizer import performance_monitor
    
    # Create and optimize model
    model = LSTMAutoencoder(input_size=5, hidden_size=16, num_layers=1)  # Smaller for testing
    optimized_model = performance_monitor.optimize_model_for_inference(model)
    
    # Test optimized inference
    test_input = torch.randn(1, 20, 5)
    start_time = time.time()
    with torch.no_grad():
        error = optimized_model.compute_reconstruction_error(test_input)
    inference_time = time.time() - start_time
    
    # Get performance report
    report = performance_monitor.get_performance_report()
    
    assert not torch.isnan(error).any(), "Optimized model output contains NaN"
    assert inference_time < 1.0, f"Inference too slow: {inference_time:.3f}s"
    assert 'system' in report, "Performance report missing system info"
    
    print(f"    Optimized inference successful ({inference_time*1000:.1f}ms)")


def test_configuration_loading():
    """Test configuration loading and validation."""
    import yaml
    from iot_edge_anomaly.config_validator import validate_config, apply_config_defaults
    
    # Test default config loading
    config_path = Path(__file__).parent / 'config' / 'default.yaml'
    assert config_path.exists(), f"Default config not found: {config_path}"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply defaults and validate
    config = apply_config_defaults(config)
    issues = validate_config(config)
    
    errors = [issue for issue in issues if issue.severity.value == 'error']
    assert len(errors) == 0, f"Configuration validation errors: {[e.message for e in errors]}"
    
    print(f"    Configuration validation passed ({len(issues)} total issues)")


def test_security_features():
    """Test security framework components."""
    import torch
    from iot_edge_anomaly.security.secure_inference import SecureInferenceEngine, SecurityLevel
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    
    # Create security engine
    model = LSTMAutoencoder(input_size=5, hidden_size=16, num_layers=1)
    security_config = {
        'enable_authentication': False,  # Disable for testing
        'protection_level': 'medium'  # Use string instead of enum for testing
    }
    
    secure_engine = SecureInferenceEngine(model, security_config)
    
    # Test secure inference
    test_input = torch.randn(1, 20, 5)
    result = secure_engine.secure_inference(test_input, token=None, client_id="test_client")
    
    assert result['success'] == True, f"Secure inference failed: {result.get('error')}"
    assert 'security_level' in result, "Security metadata missing"
    
    # Test security report
    security_report = secure_engine.get_security_status()
    assert 'protection_level' in security_report, "Security report incomplete"
    
    print("    Security framework functional")


def test_metrics_collection():
    """Test advanced metrics collection."""
    import asyncio
    from iot_edge_anomaly.monitoring.advanced_metrics import AdvancedMetricsCollector
    
    # Create metrics collector
    collector = AdvancedMetricsCollector(
        enable_model_explanations=False,  # Simplified for testing
        enable_uncertainty_tracking=True
    )
    
    # Test async metrics recording
    async def test_metrics():
        await collector.record_prediction_async(
            reconstruction_error=0.5,
            uncertainty=0.1,
            anomaly_level='normal',
            model_contributions={'model1': 0.8, 'model2': 0.2},
            processing_time=0.05
        )
        
        metrics = await collector.export_advanced_metrics()
        return metrics
    
    # Run async test
    metrics = asyncio.run(test_metrics())
    
    assert 'timestamp' in metrics, "Metrics missing timestamp"
    assert 'performance_statistics' in metrics, "Performance statistics missing"
    
    print("    Advanced metrics collection functional")


def test_model_validation():
    """Test model validation framework."""
    import torch
    from iot_edge_anomaly.validation.model_validator import ComprehensiveValidator
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    
    # Create model and validator
    model = LSTMAutoencoder(input_size=5, hidden_size=16, num_layers=1)
    validator = ComprehensiveValidator(model, {})
    
    # Create test data
    train_data = torch.randn(10, 20, 5)
    test_data = torch.randn(5, 20, 5)
    test_labels = torch.zeros(5)  # All normal samples for testing
    
    # Test basic validation
    data_validator = validator.data_validator
    validation_results = data_validator.validate_input_tensor(test_data)
    
    assert len(validation_results) > 0, "Should have validation results"
    passed_results = [r for r in validation_results if r.status.value == 'passed']
    assert len(passed_results) > 0, "Should have some passed validations"
    
    print(f"    Model validation functional ({len(passed_results)} validations passed)")


def test_ensemble_system():
    """Test advanced ensemble system."""
    import torch
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    from iot_edge_anomaly.models.simple_ensemble import create_advanced_hybrid_system
    
    # Create models for ensemble
    models = {
        'lstm1': LSTMAutoencoder(input_size=5, hidden_size=16, num_layers=1),
        'lstm2': LSTMAutoencoder(input_size=5, hidden_size=16, num_layers=1)
    }
    
    # Create ensemble system
    config = {
        'models': models,
        'ensemble_method': 'simple_average',
        'uncertainty_quantification': True
    }
    
    ensemble = create_advanced_hybrid_system(config)
    
    # Test ensemble prediction
    test_input = torch.randn(1, 20, 5)
    result = ensemble.predict(
        test_input, 
        return_explanations=False,
        return_uncertainty=True
    )
    
    assert 'reconstruction_error' in result, "Ensemble result missing reconstruction error"
    assert 'uncertainty' in result, "Ensemble result missing uncertainty"
    assert result['ensemble_method'] == 'simple_average', "Ensemble method mismatch"
    
    print(f"    Ensemble system functional (error: {result['reconstruction_error']:.4f})")


def run_pytest_tests():
    """Run the existing pytest test suite."""
    try:
        # Run pytest with coverage
        cmd = ['python', '-m', 'pytest', 'tests/', '--tb=short', '-v', '--maxfail=5']
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent / 'src')
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
            env=env,
            cwd=str(Path(__file__).parent)
        )
        
        if result.returncode == 0:
            print("    Pytest suite passed")
        else:
            print(f"    Pytest suite failed (exit code: {result.returncode})")
            print(f"    STDOUT: {result.stdout[-500:]}")  # Last 500 chars
            print(f"    STDERR: {result.stderr[-500:]}")
            raise Exception(f"Pytest failed with exit code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("    Pytest suite timed out")
        raise Exception("Pytest suite timed out after 5 minutes")


def main():
    """Run comprehensive test suite."""
    print("🚀 Starting Terragon Autonomous SDLC Comprehensive Test Suite")
    print("=" * 70)
    
    start_time = time.time()
    
    # Generation 1 Tests: Basic Functionality
    print("\n📋 GENERATION 1: BASIC FUNCTIONALITY TESTS")
    print("-" * 50)
    
    gen1_tests = [
        ("Basic Imports", test_basic_imports),
        ("Model Creation and Inference", test_model_creation_and_inference),
        ("Configuration Loading", test_configuration_loading),
    ]
    
    for test_name, test_func in gen1_tests:
        run_test(test_name, test_func, 'generation_1')
    
    # Generation 2 Tests: Robust and Reliable
    print("\n🛡️  GENERATION 2: ROBUSTNESS AND RELIABILITY TESTS")
    print("-" * 50)
    
    gen2_tests = [
        ("Security Framework", test_security_features),
        ("Advanced Metrics Collection", test_metrics_collection),
        ("Model Validation Framework", test_model_validation),
    ]
    
    for test_name, test_func in gen2_tests:
        run_test(test_name, test_func, 'generation_2')
    
    # Generation 3 Tests: Optimized and Scalable
    print("\n⚡ GENERATION 3: OPTIMIZATION AND SCALABILITY TESTS")
    print("-" * 50)
    
    gen3_tests = [
        ("Performance Optimization", test_performance_optimization),
        ("Advanced Ensemble System", test_ensemble_system),
    ]
    
    for test_name, test_func in gen3_tests:
        run_test(test_name, test_func, 'generation_3')
    
    # Legacy Test Suite
    print("\n🧪 LEGACY PYTEST SUITE")
    print("-" * 50)
    
    run_test("Existing Pytest Suite", run_pytest_tests, 'generation_1')
    
    # Calculate final results
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for generation in ['generation_1', 'generation_2', 'generation_3']:
        gen_passed = test_results[generation]['passed']
        gen_failed = test_results[generation]['failed']
        gen_total = gen_passed + gen_failed
        
        total_tests += gen_total
        total_passed += gen_passed
        total_failed += gen_failed
        
        # Determine generation status
        if gen_failed == 0:
            test_results[generation]['status'] = 'PASSED'
        elif gen_passed > gen_failed:
            test_results[generation]['status'] = 'MOSTLY_PASSED'
        else:
            test_results[generation]['status'] = 'FAILED'
    
    # Overall status
    test_results['overall']['total_tests'] = total_tests
    test_results['overall']['total_passed'] = total_passed
    test_results['overall']['total_failed'] = total_failed
    
    if total_failed == 0:
        test_results['overall']['status'] = 'PASSED'
    elif total_passed > total_failed:
        test_results['overall']['status'] = 'MOSTLY_PASSED'
    else:
        test_results['overall']['status'] = 'FAILED'
    
    # Print final results
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("📊 FINAL TEST RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 OVERALL SUMMARY:")
    print(f"   Total Tests:  {total_tests}")
    print(f"   Passed:       {total_passed} ✅")
    print(f"   Failed:       {total_failed} ❌")
    print(f"   Success Rate: {total_passed/total_tests*100:.1f}%")
    print(f"   Duration:     {total_duration:.1f}s")
    print(f"   Status:       {test_results['overall']['status']}")
    
    print(f"\n📋 BY GENERATION:")
    for gen_name, gen_key in [
        ("Generation 1 (Basic)", "generation_1"),
        ("Generation 2 (Robust)", "generation_2"), 
        ("Generation 3 (Optimized)", "generation_3")
    ]:
        gen_data = test_results[gen_key]
        total_gen = gen_data['passed'] + gen_data['failed']
        if total_gen > 0:
            print(f"   {gen_name:25} {gen_data['passed']:2d}/{total_gen:2d} ({gen_data['status']})")
    
    # Save detailed results
    output_file = Path(__file__).parent / 'test_results.json'
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n📄 Detailed results saved to: {output_file}")
    
    # Final status
    if test_results['overall']['status'] == 'PASSED':
        print("\n🎉 ALL QUALITY GATES PASSED! System is production-ready.")
        return 0
    elif test_results['overall']['status'] == 'MOSTLY_PASSED':
        print("\n⚠️  MOST QUALITY GATES PASSED. Review failed tests before production.")
        return 1
    else:
        print("\n❌ QUALITY GATES FAILED. Significant issues need to be addressed.")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
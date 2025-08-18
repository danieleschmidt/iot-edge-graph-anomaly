# Comprehensive Testing Framework Documentation

## Overview

The IoT Edge Anomaly Detection system uses a comprehensive testing framework built on pytest with extensive fixtures, markers, and testing utilities to ensure code quality, performance, and reliability.

## Testing Philosophy

Our testing strategy follows the **Test Pyramid** approach with enhanced security and performance testing:
- **Unit Tests (60%)**: Fast, isolated tests for individual components
- **Integration Tests (25%)**: Tests for component interactions
- **End-to-End Tests (10%)**: Full system workflow validation
- **Security Tests (3%)**: Vulnerability and attack vector testing
- **Performance Tests (2%)**: Latency, throughput, and resource usage testing

## Test Structure

```
tests/
├── unit/           # Isolated unit tests
├── integration/    # Component interaction tests  
├── e2e/           # End-to-end workflow tests
├── performance/   # Performance and benchmark tests
├── fixtures/      # Test data and mocks
└── conftest.py    # Shared pytest configuration
```

## Running Tests

### Quick Commands (via package.json)
```bash
npm test                    # Run all tests
npm run test:unit          # Unit tests only
npm run test:integration   # Integration tests only
npm run test:e2e          # End-to-end tests only
npm run test:watch        # Watch mode for development
npm run test:coverage     # Generate coverage report
```

### Traditional Commands (via pytest)
```bash
pytest                     # Run all tests
pytest tests/unit         # Unit tests only
pytest -m integration     # Integration tests only
pytest -m e2e             # End-to-end tests only
pytest -m "not slow"      # Skip slow tests
pytest --cov=src         # Coverage report
```

### Make Commands
```bash
make test                 # Run all tests
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-e2e           # End-to-end tests only
make test-coverage      # Coverage report
```

## Test Categories

### Unit Tests (`tests/unit/`)
Test individual components in isolation:
- Model components (LSTM, GNN layers)
- Data processing functions
- Utility functions
- Configuration handlers

**Characteristics**:
- Fast execution (<1ms per test)
- No external dependencies
- Mock all I/O operations
- High code coverage target (>90%)

**Example**:
```python
def test_lstm_autoencoder_forward_pass():
    model = LSTMAutoencoder(input_size=10, hidden_size=64)
    x = torch.randn(32, 10, 10)  # batch, seq, features
    output = model(x)
    assert output.shape == x.shape
```

### Integration Tests (`tests/integration/`)
Test component interactions:
- Model training pipeline
- Data loading and preprocessing
- Metrics collection and export
- Health check endpoints

**Characteristics**:
- Medium execution time (<10s per test)
- May use test databases/files
- Test real component interactions
- Focus on interfaces and data flow

**Example**:
```python
def test_anomaly_detection_pipeline():
    loader = SWaTLoader(test_data_path)
    model = LSTMAutoencoder.load(test_model_path)
    detector = AnomalyDetector(model, threshold=0.5)
    
    data = loader.get_batch(32)
    anomalies = detector.detect(data)
    assert len(anomalies) == 32
```

### End-to-End Tests (`tests/e2e/`)
Test complete user workflows:
- Full training and inference pipeline
- Docker container deployment
- API endpoint functionality
- Monitoring and alerting

**Characteristics**:
- Slow execution (<60s per test)
- Use real data and configurations
- Test complete user scenarios
- Validate business requirements

**Example**:
```python
def test_container_anomaly_detection():
    # Start container with test configuration
    container = start_container(test_config)
    
    # Send test data
    response = send_sensor_data(container, test_data)
    
    # Verify anomaly detection
    assert response.status_code == 200
    assert response.json()['anomalies_detected'] > 0
```

### Performance Tests (`tests/performance/`)
Validate performance requirements:
- Inference latency benchmarks
- Memory usage profiling
- Throughput testing
- Resource constraint validation

**Characteristics**:
- Measure actual performance metrics
- Compare against SLA requirements
- Generate benchmark reports
- Run on CI/CD for regression detection

## Test Data Management

### Test Fixtures (`tests/fixtures/`)
Centralized test data management:
- Sample sensor data
- Pre-trained model weights
- Configuration files
- Mock response data

**Organization**:
```
fixtures/
├── data/
│   ├── sample_swat.csv
│   ├── normal_data.json
│   └── anomaly_data.json
├── models/
│   ├── test_lstm.pth
│   └── test_config.yaml
└── responses/
    ├── health_check.json
    └── metrics_export.json
```

### Data Generation
```python
# conftest.py
@pytest.fixture
def sample_sensor_data():
    return generate_synthetic_sensor_data(
        num_sensors=10,
        sequence_length=100,
        anomaly_rate=0.05
    )
```

## Mocking Strategy

### External Dependencies
Mock all external services and dependencies:
- File system operations
- Network requests
- Database connections
- System resources

### Mocking Patterns
```python
@pytest.fixture
def mock_model_loader(mocker):
    mock = mocker.patch('iot_edge_anomaly.models.load_model')
    mock.return_value = MockLSTMModel()
    return mock

def test_with_mocked_model(mock_model_loader):
    # Test uses mocked model instead of real one
    pass
```

## Performance Testing

### Latency Requirements
- **Inference Latency**: <10ms per sample
- **Model Loading**: <5 seconds
- **Startup Time**: <30 seconds

### Memory Requirements
- **Peak Memory**: <100MB
- **Model Size**: <50MB
- **Working Set**: <75MB

### Benchmark Tests
```python
@pytest.mark.slow
def test_inference_latency():
    model = load_test_model()
    data = generate_test_batch(1000)
    
    start_time = time.time()
    results = model.predict(data)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / len(data)
    assert avg_latency < 0.01  # <10ms requirement
```

## Coverage Requirements

### Coverage Targets
- **Overall Coverage**: >90%
- **Unit Test Coverage**: >95%
- **Integration Coverage**: >85%
- **Critical Path Coverage**: 100%

### Coverage Exclusions
- `__init__.py` files
- Configuration files
- CLI entry points (unless critical)
- Deprecated code paths

### Generating Reports
```bash
# HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Terminal coverage report
pytest --cov=src --cov-report=term-missing

# XML coverage for CI/CD
pytest --cov=src --cov-report=xml
```

## Continuous Integration

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest-unit
      name: Run unit tests
      entry: pytest tests/unit -v
      language: system
      pass_filenames: false
```

### CI/CD Pipeline
```bash
# Run in CI/CD pipeline
make test-unit              # Fast feedback
make test-integration       # Medium feedback  
make test-e2e              # Full validation
make test-performance      # SLA validation
```

## Test Best Practices

### Writing Good Tests
1. **Test One Thing**: Each test should verify a single behavior
2. **Clear Names**: Test names should describe what is being tested
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Independent**: Tests should not depend on each other
5. **Deterministic**: Tests should produce consistent results

### Test Naming Convention
```python
def test_[component]_[scenario]_[expected_result]():
    """
    Test that [component] [behavior] when [condition].
    """
    pass

# Examples:
def test_lstm_autoencoder_raises_error_with_invalid_input():
def test_anomaly_detector_returns_true_above_threshold():
def test_data_loader_handles_missing_files_gracefully():
```

### Test Organization
```python
class TestLSTMAutoencoder:
    """Test suite for LSTM Autoencoder component."""
    
    def test_initialization(self):
        """Test model initializes with correct parameters."""
        pass
    
    def test_forward_pass(self):
        """Test forward pass produces expected output shape."""
        pass
    
    def test_backward_pass(self):
        """Test backward pass updates parameters correctly."""
        pass
```

## Debugging Tests

### Common Issues
1. **Flaky Tests**: Use fixed random seeds, mock time-dependent operations
2. **Slow Tests**: Profile and optimize, consider using smaller test data
3. **Memory Leaks**: Clean up resources in teardown methods
4. **Environment Dependencies**: Use containers or virtual environments

### Debugging Commands
```bash
# Run single test with verbose output
pytest tests/unit/test_lstm.py::test_forward_pass -v -s

# Run tests with pdb debugger
pytest --pdb tests/unit/test_lstm.py

# Run tests with coverage and open browser
pytest --cov=src --cov-report=html && open htmlcov/index.html
```

## Documentation Testing

### Docstring Tests  
```python
def anomaly_threshold(reconstruction_error: float) -> bool:
    """
    Determine if reconstruction error indicates anomaly.
    
    Args:
        reconstruction_error: MSE between input and reconstruction
    
    Returns:
        True if anomaly detected, False otherwise
        
    Examples:
        >>> anomaly_threshold(0.1)
        False
        >>> anomaly_threshold(0.8)
        True
    """
    return reconstruction_error > 0.5
```

### Running Docstring Tests
```bash
python -m doctest src/iot_edge_anomaly/models.py -v
```

---

**Testing Guide Version**: 1.0  
**Last Updated**: 2025-08-02  
**Maintainer**: IoT Edge Anomaly Detection Team
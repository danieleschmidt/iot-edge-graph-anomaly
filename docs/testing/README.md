# Testing Framework - IoT Edge Graph Anomaly Detection

This document provides comprehensive guidance for testing the IoT Edge Graph Anomaly Detection system.

## 📋 Testing Strategy

### Testing Pyramid

```
    /\
   /  \
  / E2E \          - End-to-End Tests (few, slow, high confidence)
 /______\
/        \
| Integration |    - Integration Tests (some, medium speed)
|____________|
|            |
|    Unit     |    - Unit Tests (many, fast, focused)
|______________|
```

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual components in isolation
   - Fast execution (<1ms per test)
   - High coverage of business logic
   - Mock external dependencies

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - Medium execution time (1-100ms per test)
   - Test real integrations with controlled environments
   - Limited external service dependencies

3. **End-to-End Tests** (`tests/e2e/`)
   - Test complete user workflows
   - Slow execution (100ms-10s per test)
   - Test realistic scenarios
   - May use external services

4. **Performance Tests** (`tests/performance/`)
   - Benchmark inference latency
   - Memory usage validation
   - Throughput testing
   - Resource constraint validation

## 🏗️ Test Structure

### Directory Organization

```
tests/
├── __init__.py
├── conftest.py                    # Shared pytest configuration
├── fixtures/                     # Test data and fixtures
│   ├── __init__.py
│   └── test_data.py              # Data fixtures and generators
├── mocks/                        # Mock objects and services
│   ├── __init__.py
│   └── mock_services.py          # Mock implementations
├── unit/                         # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data_processing.py
│   └── test_monitoring.py
├── integration/                  # Integration tests
│   ├── __init__.py
│   ├── test_model_integration.py
│   ├── test_monitoring_integration.py
│   └── test_deployment_integration.py
├── e2e/                         # End-to-end tests
│   ├── __init__.py
│   ├── test_deployment_workflow.py
│   ├── test_monitoring_workflow.py
│   └── test_failure_recovery.py
├── performance/                 # Performance tests
│   ├── __init__.py
│   └── test_benchmarks.py
└── data/                       # Test datasets (not committed)
    ├── sample_data.csv
    └── test_models/
```

### Test Naming Conventions

- **File names**: `test_<component>.py`
- **Class names**: `Test<ComponentName>`
- **Method names**: `test_<behavior>_<condition>`

Examples:
```python
# Good naming
def test_detect_anomaly_returns_true_when_threshold_exceeded():
def test_model_loading_raises_error_when_file_not_found():
def test_metrics_export_succeeds_with_valid_config():

# Avoid
def test_anomaly():
def test_model():
def test_export():
```

## 🔧 Test Configuration

### pytest Configuration

The project uses pytest with the following configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "6.0"
addopts = [
    "-ra",                        # Show test summary
    "-q",                         # Quiet output
    "--strict-markers",           # Enforce marker registration
    "--strict-config",            # Strict configuration
    "--cov=src",                  # Coverage for src directory
    "--cov-report=term-missing",  # Show missing lines
    "--cov-report=html",          # HTML coverage report
    "--cov-report=xml",           # XML coverage report for CI
    "--cov-fail-under=80",        # Minimum 80% coverage
]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests", 
    "e2e: marks tests as end-to-end tests",
]
```

### Test Markers

Use markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_model_initialization():
    """Unit test for model initialization."""
    pass

@pytest.mark.integration
def test_model_data_pipeline():
    """Integration test for model and data pipeline."""
    pass

@pytest.mark.e2e
@pytest.mark.slow
def test_complete_deployment():
    """End-to-end test for complete deployment workflow."""
    pass

@pytest.mark.performance
def test_inference_latency():
    """Performance test for inference latency."""
    pass
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m e2e                    # End-to-end tests only
pytest -m "not slow"             # Exclude slow tests

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test files
pytest tests/unit/test_models.py
pytest tests/integration/ -v

# Run tests matching pattern
pytest -k "test_anomaly"

# Run tests in parallel
pytest -n auto                   # Requires pytest-xdist
```

## 🎯 Test Data Management

### Fixtures

Use fixtures for reusable test data:

```python
import pytest
import numpy as np

@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 10)

@pytest.fixture(scope="module")
def trained_model():
    """Load trained model (expensive operation, cache per module)."""
    # Mock or load actual model
    return MockModel()

@pytest.fixture(scope="session") 
def test_database():
    """Set up test database (once per test session)."""
    # Setup and teardown database
    db = create_test_database()
    yield db
    db.cleanup()
```

### Mock Data Generation

Use the `MockSensorDataGenerator` for streaming scenarios:

```python
from tests.fixtures.test_data import MockSensorDataGenerator

def test_streaming_processing():
    generator = MockSensorDataGenerator(num_sensors=10, anomaly_rate=0.05)
    
    for _ in range(100):
        sample = generator.generate_sample()
        # Process sample
        assert len(sample) == 10
```

### Test Data Isolation

- Each test should be independent
- Use fresh data for each test
- Clean up resources after tests
- Don't rely on test execution order

## 🛠️ Mocking Strategy

### Mock External Dependencies

```python
from unittest.mock import Mock, patch
from tests.mocks.mock_services import MockOTLPExporter

def test_metrics_export():
    # Use custom mock
    exporter = MockOTLPExporter()
    
    # Use unittest.mock
    with patch('prometheus_client.push_to_gateway') as mock_push:
        # Test code here
        mock_push.assert_called_once()
```

### Mock Guidelines

1. **Mock at the boundary**: Mock external services, not internal logic
2. **Verify interactions**: Assert that mocks are called correctly
3. **Use realistic responses**: Mock responses should match real service behavior
4. **Keep mocks simple**: Don't over-complicate mock implementations

## ⚡ Performance Testing

### Benchmarking

```python
import time
import pytest

@pytest.mark.performance
def test_inference_latency_benchmark():
    """Benchmark model inference latency."""
    model = load_test_model()
    test_data = generate_test_batch(batch_size=1)
    
    # Warm up
    for _ in range(10):
        model.predict(test_data)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        model.predict(test_data)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / 100 * 1000  # ms
    assert avg_latency < 10.0, f"Inference too slow: {avg_latency:.2f}ms"
```

### Resource Usage Testing

```python
import psutil
import pytest

@pytest.mark.performance
def test_memory_usage_under_load():
    """Test memory usage under continuous load."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run continuous processing
    for _ in range(1000):
        # Simulate processing
        pass
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 10, f"Memory leak detected: {memory_increase:.2f}MB"
```

## 🔍 Test Coverage

### Coverage Goals

- **Overall Coverage**: Minimum 80%
- **Critical Paths**: 95%+ coverage for anomaly detection logic
- **Edge Cases**: 90%+ coverage for error handling
- **Integration Points**: 85%+ coverage for component interfaces

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Generate terminal coverage report
pytest --cov=src --cov-report=term-missing

# Export coverage for CI
pytest --cov=src --cov-report=xml
```

### Coverage Configuration

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*.py", 
    "*/__init__.py",
    "*/conftest.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
]
```

## 🔄 Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Pull requests
- Push to main branch
- Release tags

### Test Matrix

Tests run across:
- Python versions: 3.8, 3.9, 3.10, 3.11
- Operating systems: Ubuntu, macOS, Windows
- PyTorch versions: Latest stable

### CI Test Commands

```yaml
# .github/workflows/tests.yml
- name: Run unit tests
  run: pytest tests/unit/ --cov=src --cov-report=xml

- name: Run integration tests  
  run: pytest tests/integration/ -v

- name: Run performance tests
  run: pytest tests/performance/ -m "not slow"
```

## 🐛 Debugging Tests

### Test Debugging Tips

1. **Use descriptive assertions**:
   ```python
   # Good
   assert actual == expected, f"Expected {expected}, got {actual}"
   
   # Better
   assert len(results) == 5, f"Expected 5 results, got {len(results)}: {results}"
   ```

2. **Use pytest's detailed output**:
   ```bash
   pytest -v -s tests/test_specific.py::test_function
   ```

3. **Debug with pdb**:
   ```python
   import pdb; pdb.set_trace()
   ```

4. **Use pytest fixtures for debugging**:
   ```python
   @pytest.fixture(autouse=True)
   def debug_info(request):
       print(f"Running {request.node.name}")
   ```

### Common Testing Patterns

```python
# Test exception handling
with pytest.raises(ValueError, match="Invalid input"):
    process_invalid_data()

# Test warnings
with pytest.warns(UserWarning):
    deprecated_function()

# Parametrized testing
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4), 
    (3, 6),
])
def test_double(input, expected):
    assert double(input) == expected

# Async testing
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

## 📊 Test Metrics

### Key Metrics to Track

1. **Test Coverage**: Overall and per-component coverage
2. **Test Execution Time**: Monitor for test suite performance
3. **Flaky Tests**: Tests that intermittently fail
4. **Test Maintenance**: Time spent maintaining tests vs. production code

### Reporting

- Generate daily coverage reports
- Track test execution time trends
- Monitor test failure rates
- Review test quality metrics monthly

## 🎓 Best Practices

### Do's

✅ **Write tests first** (TDD approach when possible)  
✅ **Keep tests simple** and focused on one behavior  
✅ **Use descriptive test names** that explain the scenario  
✅ **Mock external dependencies** to ensure test isolation  
✅ **Test edge cases** and error conditions  
✅ **Maintain test data** separately from test logic  
✅ **Use appropriate test markers** for categorization  

### Don'ts

❌ **Don't test implementation details**, test behavior  
❌ **Don't use sleep()** in tests, use proper synchronization  
❌ **Don't share state** between tests  
❌ **Don't ignore flaky tests**, fix them immediately  
❌ **Don't skip testing error paths**  
❌ **Don't use production data** in tests  
❌ **Don't write overly complex test setups**  

---

**Testing Documentation Version**: 1.0  
**Last Updated**: 2025-01-27  
**Next Review**: 2025-04-27  
**Maintainer**: Engineering Team
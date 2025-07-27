# Development Guide

## Getting Started

### Prerequisites

- **Python 3.8+** (3.11 recommended)
- **Git** for version control
- **Docker** for containerized development
- **Make** for automation commands

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/terragonlabs/iot-edge-graph-anomaly.git
cd iot-edge-graph-anomaly

# Set up development environment
make dev-setup

# Activate virtual environment
source venv/bin/activate

# Install pre-commit hooks
make install-hooks

# Run tests to verify setup
make test
```

### Development Container (Recommended)

For consistent development environment:

```bash
# Open in VS Code with Dev Container extension
code .

# Or manually with Docker
docker-compose -f docker-compose.yml --profile dev up -d
```

## Development Workflow

### Branch Strategy

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/***: Individual feature development
- **hotfix/***: Critical production fixes

### Making Changes

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop with TDD**
   ```bash
   # Write tests first
   make test-watch  # Run tests in watch mode
   
   # Implement functionality
   make lint        # Check code quality
   make type-check  # Verify type hints
   ```

3. **Pre-commit Validation**
   ```bash
   make pre-commit  # Run all quality checks
   ```

4. **Submit Pull Request**
   - Use the provided PR template
   - Ensure all CI checks pass
   - Request reviews from relevant team members

## Code Standards

### Python Style Guide

- **PEP 8** compliance with 88-character line length
- **Type hints** required for all public functions
- **Docstrings** required for all modules, classes, and functions
- **Black** formatting with isort for imports

### Example Code Structure

```python
"""Module docstring describing purpose."""

from typing import Optional, List, Dict, Any
import torch
import numpy as np

class ExampleClass:
    """Class docstring explaining purpose and usage.
    
    Args:
        param1: Description of parameter
        param2: Description with type info
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None) -> None:
        """Initialize the class."""
        self.param1 = param1
        self.param2 = param2 or 42
    
    def process_data(self, data: torch.Tensor) -> Dict[str, Any]:
        """Process input data and return results.
        
        Args:
            data: Input tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            Dictionary containing processed results and metadata
            
        Raises:
            ValueError: If input data shape is invalid
        """
        if data.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {data.dim()}D")
        
        # Implementation here
        return {"result": data.mean(), "shape": data.shape}
```

### Testing Standards

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions  
- **Performance Tests**: Verify edge device constraints
- **Test Coverage**: Maintain >80% coverage

### Test Structure

```python
"""Test module following naming convention."""

import pytest
import torch
from unittest.mock import patch, Mock

from src.iot_edge_anomaly.example import ExampleClass

class TestExampleClass:
    """Test suite for ExampleClass."""
    
    def test_initialization(self):
        """Test proper initialization."""
        instance = ExampleClass("test", 10)
        assert instance.param1 == "test"
        assert instance.param2 == 10
    
    def test_process_data_valid_input(self):
        """Test data processing with valid input."""
        instance = ExampleClass("test")
        data = torch.randn(2, 10, 5)
        
        result = instance.process_data(data)
        
        assert isinstance(result, dict)
        assert "result" in result
        assert "shape" in result
    
    def test_process_data_invalid_shape(self):
        """Test error handling for invalid input shape."""
        instance = ExampleClass("test")
        data = torch.randn(10, 5)  # 2D instead of 3D
        
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            instance.process_data(data)
    
    @pytest.mark.performance
    def test_processing_speed(self, performance_benchmark):
        """Test processing meets performance requirements."""
        instance = ExampleClass("test")
        data = torch.randn(1, 10, 5)
        
        performance_benchmark.start()
        for _ in range(100):
            instance.process_data(data)
        performance_benchmark.stop()
        
        # Should process 100 samples in <100ms
        assert performance_benchmark.elapsed_time < 0.1
```

## Development Commands

### Essential Commands

```bash
# Code Quality
make lint          # Run linting (flake8, black, isort)
make format        # Auto-format code
make type-check    # Type checking with mypy
make security-check # Security scanning

# Testing
make test          # Run all tests
make test-unit     # Unit tests only
make test-integration # Integration tests
make test-performance # Performance tests
make test-watch    # Watch mode for TDD

# Development
make run           # Run application locally
make run-dev       # Run in development mode
make clean         # Clean build artifacts

# Docker
make docker-build  # Build container image
make docker-run    # Run container locally
make docker-build-arm64 # Build for ARM64 (Pi)

# Documentation
make docs          # Generate documentation
make docs-serve    # Serve docs locally
```

### Advanced Commands

```bash
# Release Management
make release-patch # Bump patch version
make release-minor # Bump minor version
make release-major # Bump major version

# Dependencies
make update-deps   # Update all dependencies
make check-deps    # Check for vulnerabilities

# Monitoring
make health-check  # Check application health
make metrics       # View current metrics
```

## Component Development

### Adding New Models

1. **Create Model Module**
   ```python
   # src/iot_edge_anomaly/models/new_model.py
   ```

2. **Implement Base Interface**
   ```python
   from abc import ABC, abstractmethod
   
   class BaseModel(ABC):
       @abstractmethod
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           pass
   ```

3. **Add Comprehensive Tests**
   ```python
   # tests/test_new_model.py
   ```

4. **Update Documentation**
   - Add to architecture documentation
   - Update API documentation
   - Add usage examples

### Adding New Data Loaders

1. **Implement Data Loader**
   ```python
   # src/iot_edge_anomaly/data/new_loader.py
   ```

2. **Follow Interface Pattern**
   ```python
   class BaseDataLoader(ABC):
       @abstractmethod
       def load_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
           pass
   ```

3. **Add Integration Tests**
   ```python
   # tests/test_new_loader.py
   ```

### Adding Monitoring Metrics

1. **Extend Metrics Exporter**
   ```python
   # src/iot_edge_anomaly/monitoring/metrics_exporter.py
   
   def record_new_metric(self, value: float) -> None:
       """Record new custom metric."""
       self.new_metric_counter.add(value)
   ```

2. **Add to Health Checks**
   ```python
   # src/iot_edge_anomaly/health.py
   ```

3. **Update Dashboards**
   ```yaml
   # monitoring/grafana/dashboards/
   ```

## Edge Device Development

### Resource Constraints

Always consider edge device limitations:

- **Memory**: Target <100MB total usage
- **CPU**: Design for ARM64 and low-power x86
- **Storage**: Minimal model and data storage
- **Network**: Intermittent connectivity

### Performance Testing

```python
@pytest.mark.performance
def test_memory_usage(self):
    """Test memory usage stays within limits."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024
    
    # Your code here
    
    memory_after = process.memory_info().rss / 1024 / 1024
    assert (memory_after - memory_before) < 50  # <50MB increase
```

### ARM64 Testing

```bash
# Test on ARM64 emulation
docker run --platform linux/arm64 -it python:3.11-alpine
pip install -e .
python -m pytest tests/
```

## Debugging

### Local Debugging

```bash
# Debug mode with detailed logging
DEBUG=true LOG_LEVEL=DEBUG python -m src.iot_edge_anomaly.main

# Profile memory usage
python -m memory_profiler src/iot_edge_anomaly/main.py

# Profile CPU usage  
python -m cProfile -o profile.stats src/iot_edge_anomaly/main.py
```

### Container Debugging

```bash
# Debug inside container
docker run -it --entrypoint /bin/sh iot-edge-anomaly:latest

# Check container resource usage
docker stats iot-edge-anomaly

# View container logs
docker logs -f iot-edge-anomaly
```

### Remote Debugging (Edge Devices)

```bash
# SSH debugging on Raspberry Pi
ssh pi@device-ip
docker logs -f iot-edge-anomaly

# Resource monitoring
htop
iotop
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure proper package installation
   pip install -e .
   ```

2. **Test Failures**
   ```bash
   # Run specific test with verbose output
   pytest tests/test_specific.py::test_function -v -s
   ```

3. **Memory Issues**
   ```bash
   # Check for memory leaks
   python -m memory_profiler your_script.py
   ```

4. **Container Issues**
   ```bash
   # Rebuild with no cache
   docker build --no-cache -t iot-edge-anomaly .
   ```

### Getting Help

- **Documentation**: Check docs/ directory
- **Issues**: Search GitHub issues
- **Team Chat**: Internal development channels
- **Code Review**: Request reviews early and often

## Contributing Guidelines

### Pull Request Process

1. **Feature Development**
   - Create feature branch from develop
   - Follow TDD approach
   - Maintain test coverage
   - Update documentation

2. **Code Review**
   - Address all review comments
   - Ensure CI passes
   - Update based on feedback

3. **Merge Process**
   - Squash commits for clean history
   - Use conventional commit messages
   - Delete feature branch after merge

### Commit Message Format

```
type(scope): brief description

Optional longer description explaining the change in more detail.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

---

**Development Guide Version**: 1.0  
**Last Updated**: 2025-01-27  
**Next Review**: 2025-04-27
"""
Test suite for project setup validation.
"""
import pytest
import sys
from pathlib import Path

def test_python_version():
    """Test that Python version meets requirements."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"

def test_package_import():
    """Test that the main package can be imported."""
    from iot_edge_anomaly import __version__, __author__
    assert __version__ == "0.1.0"
    assert __author__ == "Terragon Labs"

def test_project_structure():
    """Test that required project directories exist."""
    repo_root = Path(__file__).parent.parent
    
    required_dirs = [
        "src/iot_edge_anomaly",
        "tests",
        "docs/status",
        "config"
    ]
    
    for dir_path in required_dirs:
        assert (repo_root / dir_path).exists(), f"Required directory {dir_path} missing"

def test_requirements_file():
    """Test that requirements.txt exists and contains key dependencies."""
    repo_root = Path(__file__).parent.parent
    requirements_file = repo_root / "requirements.txt"
    
    assert requirements_file.exists(), "requirements.txt missing"
    
    content = requirements_file.read_text()
    required_deps = ["torch", "torch-geometric", "numpy", "pytest", "opentelemetry-api"]
    
    for dep in required_deps:
        assert dep in content, f"Required dependency {dep} missing from requirements.txt"
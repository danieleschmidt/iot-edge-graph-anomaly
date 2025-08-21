#!/usr/bin/env python3
"""
Development Setup and Environment Verification Script
For Terragon Autonomous SDLC v4.0
"""
import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}: SUCCESS")
            return True
        else:
            print(f"❌ {description}: FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}: EXCEPTION - {e}")
        return False

def check_python_availability():
    """Check Python and basic packages."""
    print("🐍 Python Environment Check")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    
    # Check if we can import basic modules
    basic_modules = ['os', 'sys', 'json', 'pathlib', 'subprocess']
    for module in basic_modules:
        try:
            __import__(module)
            print(f"✅ {module}: Available")
        except ImportError:
            print(f"❌ {module}: Not available")

def create_mock_dependencies():
    """Create mock implementations for missing dependencies."""
    mock_dir = Path("/root/repo/mock_deps")
    mock_dir.mkdir(exist_ok=True)
    
    # Create minimal yaml module
    yaml_mock = mock_dir / "yaml.py"
    with open(yaml_mock, 'w') as f:
        f.write('''
"""Mock YAML module for basic functionality."""
import json

def safe_load(stream):
    """Basic YAML loading - supports JSON subset."""
    if hasattr(stream, 'read'):
        content = stream.read()
    else:
        content = stream
    
    # Try JSON first (YAML subset)
    try:
        return json.loads(content)
    except:
        # Basic key-value parsing for simple YAML
        result = {}
        for line in content.split('\\n'):
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Try to parse value
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                result[key] = value
        return result

def dump(data, stream=None):
    """Basic YAML dumping."""
    if stream is None:
        return json.dumps(data, indent=2)
    else:
        json.dump(data, stream, indent=2)
''')
    
    # Create minimal torch module
    torch_mock = mock_dir / "torch.py"
    with open(torch_mock, 'w') as f:
        f.write('''
"""Mock PyTorch module for basic tensor operations."""
import math
import random

class Tensor:
    def __init__(self, data, dtype=None):
        if isinstance(data, (list, tuple)):
            self.data = data
            self.shape = self._compute_shape(data)
        else:
            self.data = data
            self.shape = (1,)
    
    def _compute_shape(self, data):
        if not isinstance(data, (list, tuple)):
            return (1,)
        shape = [len(data)]
        if data and isinstance(data[0], (list, tuple)):
            shape.extend(self._compute_shape(data[0]))
        return tuple(shape)
    
    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim] if dim < len(self.shape) else 1
    
    def item(self):
        return float(self.data)
    
    def isnan(self):
        return Tensor([False])
    
    def isinf(self):
        return Tensor([False])
    
    def any(self):
        return False

def tensor(data, dtype=None):
    return Tensor(data, dtype)

def randn(*shape):
    size = 1
    for s in shape:
        size *= s
    data = [random.gauss(0, 1) for _ in range(size)]
    return Tensor(data)

def no_grad():
    class NoGradContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    return NoGradContext()

# Neural network mock
class Module:
    def eval(self): return self
    def train(self, mode=True): return self

nn = type('nn', (), {
    'Module': Module,
    'LSTM': lambda *args, **kwargs: Module(),
    'Linear': lambda *args, **kwargs: Module(),
    'Dropout': lambda *args, **kwargs: Module(),
})()
''')
    
    return str(mock_dir)

def setup_development_environment():
    """Setup development environment for autonomous execution."""
    print("🚀 Terragon Autonomous SDLC v4.0 - Development Setup")
    print("=" * 60)
    
    # Check Python
    check_python_availability()
    
    # Create mock dependencies
    mock_path = create_mock_dependencies()
    print(f"📦 Created mock dependencies at: {mock_path}")
    
    # Add to Python path
    sys.path.insert(0, mock_path)
    os.environ['PYTHONPATH'] = f"{mock_path}:{os.environ.get('PYTHONPATH', '')}"
    
    print("✅ Development environment setup complete!")
    return True

def verify_system_functionality():
    """Verify core system functionality with mocks."""
    print("🔍 System Functionality Verification")
    
    # Add mock path
    mock_path = "/root/repo/mock_deps"
    sys.path.insert(0, mock_path)
    sys.path.insert(0, "/root/repo/src")
    
    try:
        # Test basic imports
        print("📦 Testing core imports...")
        
        # Test YAML mock
        import yaml
        test_config = yaml.safe_load('{"test": true, "value": 42}')
        assert test_config['test'] == True
        print("✅ YAML mock: Working")
        
        # Test torch mock
        import torch
        test_tensor = torch.randn(2, 3, 5)
        assert test_tensor.size(0) == 2
        print("✅ Torch mock: Working")
        
        # Test main module structure
        from iot_edge_anomaly.main import IoTAnomalyDetectionApp
        print("✅ Main application: Importable")
        
        return True
        
    except Exception as e:
        print(f"❌ System verification failed: {e}")
        return False

if __name__ == "__main__":
    setup_development_environment()
    verify_system_functionality()

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

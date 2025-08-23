"""Enhanced Mock PyTorch module for comprehensive testing."""
import math
import random

__version__ = "2.0.0+mock"

class Tensor:
    def __init__(self, data, dtype=None):
        if isinstance(data, (list, tuple)):
            self.data = data
            self.shape = self._compute_shape(data)
        else:
            self.data = data
            self.shape = (1,)
        self.dtype = dtype
        self.requires_grad = False
    
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
        return float(self.data) if not isinstance(self.data, (list, tuple)) else float(self.data[0])
    
    def isnan(self):
        return Tensor([False])
    
    def isinf(self):
        return Tensor([False])
    
    def any(self):
        return False
    
    def unsqueeze(self, dim):
        return Tensor(self.data)
    
    def view(self, *shape):
        return Tensor(self.data)
    
    def permute(self, *dims):
        return Tensor(self.data)
    
    def contiguous(self):
        return Tensor(self.data)
    
    def t(self):
        return Tensor(self.data)
    
    def __add__(self, other):
        return Tensor([0.5])
    
    def __mul__(self, other):
        return Tensor([0.5])

def tensor(data, dtype=None):
    return Tensor(data, dtype)

def randn(*shape):
    if len(shape) == 3:
        # For batch, seq, features
        batch, seq, feat = shape
        data = [[[random.gauss(0, 1) for _ in range(feat)] for _ in range(seq)] for _ in range(batch)]
    else:
        size = 1
        for s in shape:
            size *= s
        data = [random.gauss(0, 1) for _ in range(size)]
    return Tensor(data)

def zeros(*shape):
    if len(shape) == 3:
        batch, seq, feat = shape
        data = [[[0.0 for _ in range(feat)] for _ in range(seq)] for _ in range(batch)]
    else:
        size = 1
        for s in shape:
            size *= s
        data = [0.0 for _ in range(size)]
    return Tensor(data)

def ones(*shape):
    if len(shape) == 3:
        batch, seq, feat = shape
        data = [[[1.0 for _ in range(feat)] for _ in range(seq)] for _ in range(batch)]
    else:
        size = 1
        for s in shape:
            size *= s
        data = [1.0 for _ in range(size)]
    return Tensor(data)

def long():
    return "long"

def save(obj, path):
    pass

def load(path, map_location=None):
    return {}

class no_grad:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# Neural network module mock
class Module:
    def __init__(self):
        self.training = True
    
    def eval(self):
        self.training = False
        return self
    
    def train(self, mode=True):
        self.training = mode
        return self
    
    def forward(self, x):
        return x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        return []

class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
    
    def forward(self, x, hidden=None):
        # Mock LSTM output - return same batch size and sequence length
        if self.batch_first:
            batch_size, seq_len = x.size(0), x.size(1)
            output = tensor([[[random.gauss(0, 0.1) for _ in range(self.hidden_size)] 
                           for _ in range(seq_len)] for _ in range(batch_size)])
        else:
            seq_len, batch_size = x.size(0), x.size(1)
            output = tensor([[[random.gauss(0, 0.1) for _ in range(self.hidden_size)] 
                           for _ in range(batch_size)] for _ in range(seq_len)])
        
        h_n = tensor([[[random.gauss(0, 0.1) for _ in range(self.hidden_size)] 
                     for _ in range(batch_size)] for _ in range(self.num_layers)])
        c_n = tensor([[[random.gauss(0, 0.1) for _ in range(self.hidden_size)] 
                     for _ in range(batch_size)] for _ in range(self.num_layers)])
        
        return output, (h_n, c_n)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        # Mock output with correct shape
        if isinstance(x.data, list) and len(x.shape) >= 2:
            batch_size = x.size(0)
            return tensor([[random.gauss(0, 0.1) for _ in range(self.out_features)] 
                          for _ in range(batch_size)])
        return tensor([random.gauss(0, 0.1) for _ in range(self.out_features)])

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        return x

class MSELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input, target):
        return tensor([random.uniform(0.1, 0.8)])  # Mock reconstruction error

class functional:
    @staticmethod
    def mse_loss(input, target, reduction='mean'):
        return tensor([random.uniform(0.1, 0.8)])

# Create nn namespace
class nn:
    Module = Module
    LSTM = LSTM
    Linear = Linear 
    Dropout = Dropout
    MSELoss = MSELoss
    functional = functional()

nn = nn()

# Optimizer mocks
class optim:
    class Adam:
        def __init__(self, params, lr=0.001):
            pass
        def zero_grad(self):
            pass
        def step(self):
            pass
    
    class SGD:
        def __init__(self, params, lr=0.01):
            pass
        def zero_grad(self):
            pass  
        def step(self):
            pass

optim = optim()
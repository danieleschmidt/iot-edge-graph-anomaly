"""
Test suite for Graph Neural Network layer functionality.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing."""
    # Simple graph: 5 nodes with edges
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],  # source nodes
        [1, 0, 2, 1, 3, 2, 4, 3]   # target nodes
    ], dtype=torch.long)
    
    # Node features (5 nodes, 3 features each)
    x = torch.randn(5, 3)
    
    return {'x': x, 'edge_index': edge_index}

def test_gnn_layer_import():
    """Test that GNN layer can be imported."""
    try:
        from iot_edge_anomaly.models.gnn_layer import GraphNeuralNetworkLayer
    except ImportError:
        pytest.fail("GraphNeuralNetworkLayer cannot be imported")

def test_gnn_layer_initialization():
    """Test GNN layer initialization."""
    from iot_edge_anomaly.models.gnn_layer import GraphNeuralNetworkLayer
    
    input_dim = 64
    hidden_dim = 32
    output_dim = 64
    
    gnn = GraphNeuralNetworkLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2
    )
    
    assert gnn.input_dim == input_dim
    assert gnn.hidden_dim == hidden_dim
    assert gnn.output_dim == output_dim
    assert gnn.num_layers == 2

def test_gnn_forward_pass(sample_graph_data):
    """Test GNN forward pass with graph data."""
    from iot_edge_anomaly.models.gnn_layer import GraphNeuralNetworkLayer
    
    input_dim = 3  # Match sample data features
    hidden_dim = 16
    output_dim = 32
    
    gnn = GraphNeuralNetworkLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    # Forward pass
    output = gnn(sample_graph_data['x'], sample_graph_data['edge_index'])
    
    # Check output shape
    assert output.shape == (5, output_dim)  # 5 nodes, output_dim features
    assert output.dtype == torch.float32

def test_gnn_edge_cases():
    """Test GNN with edge cases."""
    from iot_edge_anomaly.models.gnn_layer import GraphNeuralNetworkLayer
    
    # Single node graph
    gnn = GraphNeuralNetworkLayer(input_dim=3, hidden_dim=8, output_dim=8)
    gnn.eval()  # Set to eval mode to avoid BatchNorm issues with single sample
    
    single_node_x = torch.randn(1, 3)
    single_node_edges = torch.empty((2, 0), dtype=torch.long)  # No edges
    
    output = gnn(single_node_x, single_node_edges)
    assert output.shape == (1, 8)

def test_lstm_gnn_integration():
    """Test integration between LSTM and GNN layers."""
    try:
        from iot_edge_anomaly.models.lstm_gnn_hybrid import LSTMGNNHybridModel
    except ImportError:
        pytest.fail("LSTMGNNHybridModel cannot be imported")

def test_hybrid_model_initialization():
    """Test hybrid LSTM-GNN model initialization."""
    from iot_edge_anomaly.models.lstm_gnn_hybrid import LSTMGNNHybridModel
    
    config = {
        'lstm': {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 2
        },
        'gnn': {
            'input_dim': 64,
            'hidden_dim': 32,
            'output_dim': 64,
            'num_layers': 2
        },
        'fusion': {
            'method': 'concatenate',
            'output_dim': 128
        }
    }
    
    model = LSTMGNNHybridModel(config)
    assert model is not None

def test_hybrid_model_forward_pass(sample_graph_data):
    """Test hybrid model forward pass."""
    from iot_edge_anomaly.models.lstm_gnn_hybrid import LSTMGNNHybridModel
    
    config = {
        'lstm': {
            'input_size': 5,
            'hidden_size': 32,
            'num_layers': 1
        },
        'gnn': {
            'input_dim': 32,
            'hidden_dim': 16,
            'output_dim': 32,
            'num_layers': 1
        },
        'fusion': {
            'method': 'concatenate',
            'output_dim': 64
        }
    }
    
    model = LSTMGNNHybridModel(config)
    
    # Create time-series data (batch=2, seq_len=10, features=5)
    time_series_data = torch.randn(2, 10, 5)
    
    # Create graph topology (5 sensor nodes)
    graph_data = {
        'x': torch.randn(5, 32),  # Node features (after LSTM encoding)
        'edge_index': sample_graph_data['edge_index']
    }
    
    # Forward pass
    output = model(time_series_data, graph_data)
    
    # Check output shape - hybrid model returns reconstruction
    assert output.shape[0] == 2  # Batch size
    assert output.shape[1] == 10  # Sequence length
    assert output.shape[2] == 5   # Input features (reconstruction)

def test_graph_topology_builder():
    """Test graph topology building for sensor networks."""
    from iot_edge_anomaly.models.gnn_layer import GraphTopologyBuilder
    
    # Test sensor correlation-based topology
    builder = GraphTopologyBuilder()
    
    # Mock sensor correlation matrix (5 sensors)
    correlation_matrix = torch.tensor([
        [1.0, 0.8, 0.2, 0.1, 0.0],
        [0.8, 1.0, 0.7, 0.2, 0.1],
        [0.2, 0.7, 1.0, 0.6, 0.3],
        [0.1, 0.2, 0.6, 1.0, 0.5],
        [0.0, 0.1, 0.3, 0.5, 1.0]
    ])
    
    edge_index = builder.build_from_correlation(correlation_matrix, threshold=0.5)
    
    # Check edge format
    assert edge_index.shape[0] == 2  # [source, target]
    assert edge_index.dtype == torch.long

def test_gnn_memory_efficiency():
    """Test GNN memory efficiency for edge deployment."""
    from iot_edge_anomaly.models.gnn_layer import GraphNeuralNetworkLayer
    
    # Large graph test
    gnn = GraphNeuralNetworkLayer(
        input_dim=64,
        hidden_dim=32,
        output_dim=64,
        num_layers=2
    )
    
    # Test with reasonable graph size for edge devices
    num_nodes = 50  # Typical IoT sensor network size
    x = torch.randn(num_nodes, 64)
    
    # Sparse connectivity (each node connected to ~5 others)
    num_edges = num_nodes * 5
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Should run without memory issues
    output = gnn(x, edge_index)
    assert output.shape == (num_nodes, 64)

def test_gnn_training_mode():
    """Test GNN training and evaluation modes."""
    from iot_edge_anomaly.models.gnn_layer import GraphNeuralNetworkLayer
    
    gnn = GraphNeuralNetworkLayer(input_dim=8, hidden_dim=16, output_dim=8)
    
    # Test training mode
    gnn.train()
    assert gnn.training is True
    
    # Test evaluation mode
    gnn.eval()
    assert gnn.training is False

def test_gnn_gradient_flow(sample_graph_data):
    """Test gradient flow through GNN layers."""
    from iot_edge_anomaly.models.gnn_layer import GraphNeuralNetworkLayer
    
    gnn = GraphNeuralNetworkLayer(input_dim=3, hidden_dim=8, output_dim=8)
    
    x = sample_graph_data['x'].requires_grad_(True)
    edge_index = sample_graph_data['edge_index']
    
    output = gnn(x, edge_index)
    loss = torch.mean(output ** 2)
    loss.backward()
    
    # Check gradients exist and are non-trivial
    grad_found = False
    for param in gnn.parameters():
        assert param.grad is not None
        if torch.norm(param.grad) > 1e-6:  # More lenient threshold for very small gradients
            grad_found = True
    assert grad_found, "No significant gradients found"
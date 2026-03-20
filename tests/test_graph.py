"""Tests for SensorGraph."""

import numpy as np
import pytest

from iot_anomaly.graph import SensorGraph


N_NODES = 5
FEAT_DIM = 4


@pytest.fixture
def graph():
    g = SensorGraph(n_nodes=N_NODES)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    return g


def test_sensor_graph_adjacency(graph):
    adj = graph.adjacency
    assert adj.shape == (N_NODES, N_NODES)
    # Edge (0,1) and (1,0) should be set
    assert adj[0, 1] == 1.0
    assert adj[1, 0] == 1.0
    # No self-loops in raw adjacency
    assert adj[0, 0] == 0.0


def test_sensor_graph_weighted_edge():
    g = SensorGraph(n_nodes=3)
    g.add_edge(0, 2, weight=0.5)
    assert g.adjacency[0, 2] == 0.5
    assert g.adjacency[2, 0] == 0.5


def test_sensor_graph_normalize(graph):
    adj_norm = graph.normalize()
    assert adj_norm.shape == (N_NODES, N_NODES)
    # Row sums should be 1 (approximately)
    row_sums = adj_norm.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(N_NODES), atol=1e-5)


def test_sensor_graph_conv(graph):
    features = np.random.randn(N_NODES, FEAT_DIM).astype(np.float32)
    out = graph.graph_conv(features)
    assert out.shape == (N_NODES, FEAT_DIM), f"Expected ({N_NODES}, {FEAT_DIM}), got {out.shape}"


def test_sensor_graph_conv_custom_adj(graph):
    features = np.random.randn(N_NODES, FEAT_DIM).astype(np.float32)
    adj = np.eye(N_NODES, dtype=np.float32)
    out = graph.graph_conv(features, adj)
    # With identity adjacency, output equals input
    np.testing.assert_allclose(out, features, atol=1e-6)


def test_sensor_graph_neighbor_smoothing(graph):
    features = np.random.randn(N_NODES, FEAT_DIM).astype(np.float32)
    smoothed = graph.neighbor_smoothing(features)
    assert smoothed.shape == features.shape
    # After smoothing, values should differ from raw (because of neighbor aggregation)
    assert not np.allclose(smoothed, features)

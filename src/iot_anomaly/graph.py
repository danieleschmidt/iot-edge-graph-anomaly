"""Sensor graph with adjacency matrix and graph convolution."""

import numpy as np


class SensorGraph:
    """Graph representation of sensor relationships.
    
    Args:
        n_nodes: Number of sensor nodes in the graph.
    """

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    def add_edge(self, i: int, j: int, weight: float = 1.0):
        """Add an undirected edge between sensors i and j.
        
        Args:
            i: Source node index.
            j: Target node index.
            weight: Edge weight.
        """
        self.adjacency[i, j] = weight
        self.adjacency[j, i] = weight

    def normalize(self) -> np.ndarray:
        """Compute row-normalized adjacency matrix (including self-loops).
        
        Returns:
            Row-normalized adjacency matrix of shape (n_nodes, n_nodes).
        """
        adj = self.adjacency + np.eye(self.n_nodes, dtype=np.float32)
        row_sums = adj.sum(axis=1, keepdims=True)
        # Avoid division by zero for isolated nodes
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return adj / row_sums

    def graph_conv(self, features: np.ndarray, adj: np.ndarray = None) -> np.ndarray:
        """Simple graph convolution: adj @ features.
        
        Args:
            features: Node features of shape (n_nodes, feat_dim) or (n_nodes,).
            adj: Adjacency matrix to use. If None, uses normalized adjacency.
        
        Returns:
            Convolved features, same shape as input.
        """
        if adj is None:
            adj = self.normalize()
        return adj @ features

    def neighbor_smoothing(self, features: np.ndarray) -> np.ndarray:
        """Smooth features by aggregating from neighbors using normalized adjacency.
        
        Args:
            features: Node features of shape (n_nodes, feat_dim) or (n_nodes,).
        
        Returns:
            Smoothed features, same shape as input.
        """
        adj_norm = self.normalize()
        return self.graph_conv(features, adj_norm)

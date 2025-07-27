"""Integration tests for the full anomaly detection pipeline."""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import pandas as pd
import numpy as np

from iot_edge_anomaly.models.lstm_gnn_hybrid import LSTMGNNHybrid
from iot_edge_anomaly.data.swat_loader import SWaTLoader
from iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter


@pytest.mark.integration
class TestFullPipeline:
    """Test the complete anomaly detection pipeline."""

    def test_end_to_end_training_and_inference(
        self,
        sample_sensor_data: pd.DataFrame,
        sample_graph_topology: Dict[str, Any],
        temp_dir: Path,
        sample_config: Dict[str, Any]
    ) -> None:
        """Test end-to-end training and inference pipeline."""
        # Setup data
        data_file = temp_dir / "sensor_data.csv"
        sample_sensor_data.to_csv(data_file, index=False)
        
        # Initialize data loader
        loader = SWaTLoader(
            data_path=str(data_file),
            graph_topology=sample_graph_topology,
            window_size=sample_config["data"]["window_size"],
            batch_size=sample_config["data"]["batch_size"]
        )
        
        # Prepare data
        train_loader, val_loader = loader.get_data_loaders()
        
        # Initialize model
        model = LSTMGNNHybrid(
            input_size=len(sample_sensor_data.columns),
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=sample_config["model"]["learning_rate"]
        )
        
        # Training loop (simplified)
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data, edge_index in train_loader:
            if num_batches >= 5:  # Limit for testing
                break
                
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = model(batch_data, edge_index)
            
            # Calculate reconstruction loss
            loss = torch.nn.functional.mse_loss(reconstructed, batch_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        assert avg_loss > 0, "Training loss should be positive"
        assert avg_loss < 10, "Training loss should be reasonable"
        
        # Inference test
        model.eval()
        with torch.no_grad():
            for batch_data, edge_index in val_loader:
                # Forward pass
                reconstructed = model(batch_data, edge_index)
                
                # Calculate anomaly scores
                reconstruction_error = torch.nn.functional.mse_loss(
                    reconstructed, batch_data, reduction='none'
                )
                anomaly_scores = torch.mean(reconstruction_error, dim=(1, 2))
                
                # Assertions
                assert reconstructed.shape == batch_data.shape
                assert len(anomaly_scores) == batch_data.shape[0]
                assert torch.all(anomaly_scores >= 0)
                break

    def test_model_save_and_load(
        self,
        sample_sensor_data: pd.DataFrame,
        sample_graph_topology: Dict[str, Any],
        temp_dir: Path,
        sample_config: Dict[str, Any]
    ) -> None:
        """Test model saving and loading functionality."""
        # Initialize model
        model = LSTMGNNHybrid(
            input_size=len(sample_sensor_data.columns),
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        
        # Save model
        model_path = temp_dir / "test_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Load model
        loaded_model = LSTMGNNHybrid(
            input_size=len(sample_sensor_data.columns),
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        loaded_model.load_state_dict(torch.load(model_path))
        
        # Test that models produce same output
        test_input = torch.randn(1, len(sample_sensor_data.columns), 50)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            output1 = model(test_input, edge_index)
            output2 = loaded_model(test_input, edge_index)
            
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_data_preprocessing_pipeline(
        self,
        sample_sensor_data: pd.DataFrame,
        sample_graph_topology: Dict[str, Any],
        temp_dir: Path
    ) -> None:
        """Test data preprocessing pipeline."""
        # Save data to file
        data_file = temp_dir / "sensor_data.csv"
        sample_sensor_data.to_csv(data_file, index=False)
        
        # Initialize loader
        loader = SWaTLoader(
            data_path=str(data_file),
            graph_topology=sample_graph_topology,
            window_size=50,
            batch_size=32
        )
        
        # Test data loading
        train_loader, val_loader = loader.get_data_loaders()
        
        # Test data shapes and types
        for batch_data, edge_index in train_loader:
            # Check data shapes
            assert len(batch_data.shape) == 3  # (batch, sensors, time)
            assert batch_data.shape[1] == len(sample_sensor_data.columns)
            assert batch_data.shape[2] == 50  # window_size
            
            # Check edge index
            assert edge_index.dtype == torch.long
            assert edge_index.shape[0] == 2  # source and target nodes
            
            # Check data normalization (should be roughly zero-centered)
            data_mean = torch.mean(batch_data)
            assert abs(data_mean.item()) < 1.0, "Data should be approximately normalized"
            break

    @pytest.mark.slow
    def test_performance_benchmarks(
        self,
        sample_sensor_data: pd.DataFrame,
        sample_graph_topology: Dict[str, Any],
        sample_config: Dict[str, Any]
    ) -> None:
        """Test performance benchmarks for inference time."""
        import time
        
        # Initialize model
        model = LSTMGNNHybrid(
            input_size=len(sample_sensor_data.columns),
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        
        model.eval()
        
        # Prepare test data
        test_input = torch.randn(1, len(sample_sensor_data.columns), 50)
        edge_index = torch.tensor(
            sample_graph_topology["edges"], dtype=torch.long
        ).t().contiguous()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input, edge_index)
        
        # Benchmark inference time
        inference_times = []
        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                _ = model(test_input, edge_index)
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        p95_inference_time = np.percentile(inference_times, 95) * 1000
        
        # Performance assertions (adjust based on requirements)
        assert avg_inference_time < 100, f"Average inference time {avg_inference_time:.2f}ms exceeds 100ms"
        assert p95_inference_time < 200, f"P95 inference time {p95_inference_time:.2f}ms exceeds 200ms"

    def test_metrics_integration(
        self,
        sample_sensor_data: pd.DataFrame,
        sample_graph_topology: Dict[str, Any],
        sample_config: Dict[str, Any]
    ) -> None:
        """Test integration with metrics exporter."""
        # Initialize metrics exporter (in test mode)
        metrics_exporter = MetricsExporter(
            service_name="test-anomaly-detector",
            service_version="0.1.0-test"
        )
        
        # Initialize model
        model = LSTMGNNHybrid(
            input_size=len(sample_sensor_data.columns),
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        
        model.eval()
        
        # Simulate inference with metrics collection
        test_input = torch.randn(1, len(sample_sensor_data.columns), 50)
        edge_index = torch.tensor(
            sample_graph_topology["edges"], dtype=torch.long
        ).t().contiguous()
        
        with torch.no_grad():
            # Record inference start
            start_time = time.time()
            
            # Run inference
            output = model(test_input, edge_index)
            
            # Record inference end
            inference_time = time.time() - start_time
            
            # Calculate anomaly score
            reconstruction_error = torch.nn.functional.mse_loss(
                output, test_input, reduction='mean'
            )
            
            # Export metrics (should not raise exceptions)
            metrics_exporter.record_inference_time(inference_time)
            metrics_exporter.record_anomaly_score(reconstruction_error.item())
            metrics_exporter.increment_inference_counter()
        
        # Test that metrics were recorded (basic check)
        assert True  # If we reach here, no exceptions were thrown
"""Sample data fixtures for testing."""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple


def generate_swat_like_data(
    n_samples: int = 1000,
    n_sensors: int = 51,
    anomaly_ratio: float = 0.05,
    random_seed: int = 42
) -> pd.DataFrame:
    """Generate SWaT-like dataset for testing."""
    np.random.seed(random_seed)
    
    # Sensor names similar to SWaT dataset
    sensor_names = [
        # Physical sensors
        "FIT101", "LIT101", "MV101", "P101", "P102",
        "AIT201", "AIT202", "AIT203", "FIT201", "MV201",
        "P201", "P202", "P203", "P204", "P205", "P206",
        "DPIT301", "FIT301", "LIT301", "MV301", "MV302",
        "MV303", "MV304", "P301", "P302",
        "AIT401", "AIT402", "FIT401", "LIT401", "P401",
        "P402", "P403", "P404", "UV401",
        "AIT501", "AIT502", "AIT503", "AIT504", "FIT501",
        "FIT502", "FIT503", "FIT504", "P501", "P502",
        "PIT501", "PIT502", "PIT503",
        "FIT601", "P601", "P602", "P603"
    ]
    
    data = {}
    
    for i, sensor_name in enumerate(sensor_names):
        # Generate base pattern (different for each sensor type)
        if "FIT" in sensor_name:  # Flow sensors
            base_pattern = 2.5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, n_samples))
        elif "LIT" in sensor_name:  # Level sensors
            base_pattern = 800 + 100 * np.sin(np.linspace(0, 2*np.pi, n_samples))
        elif "PIT" in sensor_name or "DPIT" in sensor_name:  # Pressure sensors
            base_pattern = 2.0 + 0.3 * np.sin(np.linspace(0, 3*np.pi, n_samples))
        elif "AIT" in sensor_name:  # Analyzer sensors
            base_pattern = 7.0 + 0.5 * np.sin(np.linspace(0, 6*np.pi, n_samples))
        elif "MV" in sensor_name:  # Valve sensors (binary-like)
            base_pattern = np.random.choice([0, 1], n_samples, p=[0.7, 0.3]).astype(float)
        elif "P" in sensor_name:  # Pump sensors (binary)
            base_pattern = np.random.choice([0, 1], n_samples, p=[0.6, 0.4]).astype(float)
        elif "UV" in sensor_name:  # UV sensors
            base_pattern = 100 + 20 * np.sin(np.linspace(0, 5*np.pi, n_samples))
        else:
            base_pattern = np.random.normal(1.0, 0.1, n_samples)
        
        # Add normal noise
        noise = np.random.normal(0, 0.05, n_samples)
        sensor_data = base_pattern + noise
        
        # Inject anomalies
        n_anomalies = int(n_samples * anomaly_ratio)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            if "MV" in sensor_name or "P" in sensor_name:
                # Binary sensors: flip value
                sensor_data[idx] = 1 - sensor_data[idx]
            else:
                # Continuous sensors: add significant deviation
                anomaly_magnitude = np.random.uniform(2, 5)
                anomaly_sign = np.random.choice([-1, 1])
                sensor_data[idx] += anomaly_sign * anomaly_magnitude * np.std(sensor_data)
        
        data[sensor_name] = sensor_data
    
    # Add attack labels
    attack_labels = np.zeros(n_samples)
    attack_labels[anomaly_indices] = 1
    data["Attack"] = attack_labels
    
    return pd.DataFrame(data)


def generate_graph_topology(n_sensors: int = 51) -> Dict[str, Any]:
    """Generate realistic sensor network topology."""
    # SWaT-like process structure
    processes = {
        "P1": list(range(0, 5)),      # Raw water intake
        "P2": list(range(5, 16)),     # Chemical dosing
        "P3": list(range(16, 25)),    # Ultra-filtration
        "P4": list(range(25, 33)),    # Dechlorination
        "P5": list(range(33, 47)),    # Reverse osmosis
        "P6": list(range(47, 51))     # Clean water tank
    }
    
    # Generate edges within processes (fully connected within each process)
    edges = []
    for process_nodes in processes.values():
        for i in process_nodes:
            for j in process_nodes:
                if i != j:
                    edges.append([i, j])
    
    # Add inter-process connections (sequential flow)
    process_list = list(processes.keys())
    for i in range(len(process_list) - 1):
        current_process = processes[process_list[i]]
        next_process = processes[process_list[i + 1]]
        
        # Connect last node of current process to first node of next process
        edges.append([current_process[-1], next_process[0]])
    
    # Add some additional cross-process connections for realism
    edges.extend([
        [4, 16],   # P1 to P3 (bypass)
        [15, 25],  # P2 to P4 (control)
        [32, 40],  # P4 to P5 (feedback)
    ])
    
    return {
        "nodes": list(range(n_sensors)),
        "edges": edges,
        "processes": processes
    }


def save_sample_data(output_dir: Path) -> None:
    """Save sample data files for testing."""
    output_dir.mkdir(exist_ok=True)
    
    # Generate and save sensor data
    sensor_data = generate_swat_like_data()
    sensor_data.to_csv(output_dir / "sample_sensor_data.csv", index=False)
    
    # Generate and save graph topology
    graph_topology = generate_graph_topology()
    with open(output_dir / "sample_graph_topology.json", "w") as f:
        json.dump(graph_topology, f, indent=2)
    
    # Generate configuration file
    config = {
        "model": {
            "lstm_hidden_size": 64,
            "lstm_num_layers": 2,
            "gnn_hidden_size": 32,
            "dropout": 0.1,
            "learning_rate": 0.001
        },
        "data": {
            "window_size": 50,
            "batch_size": 32,
            "normalize": True,
            "train_split": 0.7,
            "val_split": 0.15
        },
        "training": {
            "epochs": 100,
            "patience": 10,
            "threshold": 0.5,
            "min_delta": 0.001
        },
        "monitoring": {
            "metrics_enabled": True,
            "export_interval": 30,
            "log_level": "INFO"
        }
    }
    
    with open(output_dir / "sample_config.json", "w") as f:
        json.dump(config, f, indent=2)


def create_test_batch(
    batch_size: int = 32,
    n_sensors: int = 10,
    window_size: int = 50,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a test batch of sensor data and edge indices."""
    # Generate sensor data
    sensor_data = torch.randn(batch_size, n_sensors, window_size, device=device)
    
    # Generate simple ring topology
    edges = []
    for i in range(n_sensors):
        edges.append([i, (i + 1) % n_sensors])
        edges.append([(i + 1) % n_sensors, i])  # Make it undirected
    
    edge_index = torch.tensor(edges, device=device).t().contiguous()
    
    return sensor_data, edge_index


def create_anomaly_scenarios() -> List[Dict[str, Any]]:
    """Create different anomaly scenarios for testing."""
    scenarios = [
        {
            "name": "sensor_failure",
            "description": "Single sensor fails and outputs constant value",
            "affected_sensors": [5],
            "duration": 100,
            "pattern": "constant"
        },
        {
            "name": "coordinated_attack",
            "description": "Multiple sensors show synchronized anomalous behavior",
            "affected_sensors": [2, 3, 7, 8],
            "duration": 50,
            "pattern": "synchronized_spike"
        },
        {
            "name": "gradual_drift",
            "description": "Sensor readings gradually drift from normal range",
            "affected_sensors": [1],
            "duration": 200,
            "pattern": "linear_drift"
        },
        {
            "name": "intermittent_fault",
            "description": "Sensor shows periodic anomalous readings",
            "affected_sensors": [9],
            "duration": 150,
            "pattern": "periodic_spikes"
        },
        {
            "name": "process_disruption",
            "description": "Multiple related sensors affected by process change",
            "affected_sensors": [0, 1, 2],
            "duration": 75,
            "pattern": "step_change"
        }
    ]
    
    return scenarios


if __name__ == "__main__":
    # Generate sample data when run directly
    from pathlib import Path
    output_dir = Path("tests/fixtures/data")
    save_sample_data(output_dir)
    print(f"Sample data saved to {output_dir}")
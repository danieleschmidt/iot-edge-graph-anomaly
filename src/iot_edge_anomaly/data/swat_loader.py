"""
SWaT Dataset Loader for IoT Edge Anomaly Detection.

This module provides functionality to load, preprocess, and manage the
Secure Water Treatment (SWaT) dataset for training and evaluation of
the LSTM-GNN anomaly detection model.

The SWaT dataset contains sensor readings from a water treatment testbed
with both normal operation data and attack scenarios.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SWaTDataset(Dataset):
    """
    PyTorch Dataset wrapper for SWaT time-series sequences.
    
    Args:
        sequences: Time-series sequences of shape (n_samples, seq_len, n_features)
        labels: Labels for each sequence (0=normal, 1=anomaly)
    """
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class SWaTDataLoader:
    """
    SWaT Dataset Loader and Preprocessor.
    
    Handles loading, preprocessing, and batching of the SWaT dataset
    for training LSTM-based anomaly detection models.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - data_path: Path to SWaT CSV file
            - sequence_length: Length of time-series sequences
            - step_size: Step size for sliding window (default: 1)  
            - test_split: Fraction of data for testing (default: 0.2)
            - validation_split: Fraction of remaining data for validation (default: 0.1)
            - normalize: Whether to normalize features (default: True)
            - normalization_method: 'standard' or 'minmax' (default: 'standard')
            - batch_size: Batch size for DataLoader (default: 32)
            - shuffle: Whether to shuffle training data (default: True)
            - chunk_size: Process data in chunks for memory efficiency (optional)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = config.get('data_path')
        self.sequence_length = config.get('sequence_length', 20)
        self.step_size = config.get('step_size', 1)
        self.test_split = config.get('test_split', 0.2)
        self.validation_split = config.get('validation_split', 0.1)
        self.normalize = config.get('normalize', True)
        self.normalization_method = config.get('normalization_method', 'standard')
        self.batch_size = config.get('batch_size', 32)
        self.shuffle = config.get('shuffle', True)
        self.chunk_size = config.get('chunk_size', None)
        
        # Initialize scalers
        self.feature_scaler = None
        if self.normalization_method == 'standard':
            self.feature_scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            self.feature_scaler = MinMaxScaler()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        
        logger.info(f"Initialized SWaTDataLoader with sequence_length={self.sequence_length}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load SWaT dataset from CSV file.
        
        Returns:
            pd.DataFrame: Raw SWaT dataset
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        if not self.data_path:
            raise ValueError("data_path must be specified in config")
        
        data_file = Path(self.data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"SWaT data file not found: {self.data_path}")
        
        logger.info(f"Loading SWaT data from {self.data_path}")
        
        try:
            # Load CSV data
            data = pd.read_csv(self.data_path)
            
            # Basic validation
            if data.empty:
                raise ValueError("Loaded data is empty")
            
            if 'label' not in data.columns and data.columns[-1] != 'label':
                logger.warning("No 'label' column found, assuming last column is labels")
                data.columns = list(data.columns[:-1]) + ['label']
            
            logger.info(f"Loaded {len(data)} samples with {len(data.columns)-1} features")
            
            # Convert label column to binary (0/1)
            if 'label' in data.columns:
                # Handle different label formats (Normal/Attack, 0/1, etc.)
                unique_labels = data['label'].unique()
                if len(unique_labels) == 2:
                    # Binary mapping
                    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                    if isinstance(unique_labels[0], str):
                        # String labels - map Normal to 0, anything else to 1
                        label_map = {'Normal': 0, 'Attack': 1}
                        for label in unique_labels:
                            if label.lower() in ['normal', 'norm']:
                                label_map[label] = 0
                            else:
                                label_map[label] = 1
                    data['label'] = data['label'].map(label_map).fillna(1)
                else:
                    # Multi-class to binary: 0 for normal, 1 for any anomaly
                    data['label'] = (data['label'] != 0).astype(int)
            
            self.raw_data = data
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to load SWaT data: {str(e)}")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the SWaT dataset.
        
        Args:
            data: Raw SWaT dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        logger.info("Preprocessing SWaT data")
        
        # Separate features and labels
        if 'label' in data.columns:
            features = data.drop(['label'], axis=1)
            labels = data['label']
        else:
            features = data.iloc[:, :-1]
            labels = data.iloc[:, -1]
        
        # Handle missing values
        if features.isnull().any().any():
            logger.warning("Missing values detected, forward filling")
            features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize features if requested
        if self.normalize and self.feature_scaler is not None:
            logger.info(f"Normalizing features using {self.normalization_method} scaler")
            features_normalized = self.feature_scaler.fit_transform(features)
            features = pd.DataFrame(
                features_normalized, 
                columns=features.columns, 
                index=features.index
            )
        
        # Recombine features and labels
        processed_data = features.copy()
        processed_data['label'] = labels
        
        self.processed_data = processed_data
        logger.info(f"Preprocessing complete. Shape: {processed_data.shape}")
        
        return processed_data
    
    def create_sequences(
        self, 
        data: pd.DataFrame, 
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series sequences from tabular data.
        
        Args:
            data: Preprocessed dataset
            sequence_length: Override default sequence length
            
        Returns:
            Tuple of (sequences, labels) as numpy arrays
        """
        seq_len = sequence_length or self.sequence_length
        
        # Separate features and labels
        if 'label' in data.columns:
            features = data.drop(['label'], axis=1).values
            labels = data['label'].values
        else:
            features = data.iloc[:, :-1].values
            labels = data.iloc[:, -1].values
        
        sequences = []
        sequence_labels = []
        
        logger.info(f"Creating sequences with length {seq_len}, step size {self.step_size}")
        
        # Create sliding window sequences
        for i in range(0, len(features) - seq_len + 1, self.step_size):
            # Extract sequence
            seq = features[i:i + seq_len]
            
            # Label is based on the last timestep in the sequence
            # or majority vote for the sequence
            seq_label = labels[i + seq_len - 1]  # Last timestep approach
            
            sequences.append(seq)
            sequence_labels.append(seq_label)
        
        sequences = np.array(sequences, dtype=np.float32)
        sequence_labels = np.array(sequence_labels, dtype=np.float32)
        
        logger.info(f"Created {len(sequences)} sequences of shape {sequences.shape}")
        
        return sequences, sequence_labels
    
    def split_data(
        self, 
        sequences: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Split sequences into train, validation, and test sets.
        
        Args:
            sequences: Time-series sequences
            labels: Corresponding labels
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        # First split: separate test set
        train_val_seq, test_seq, train_val_labels, test_labels = train_test_split(
            sequences, labels, 
            test_size=self.test_split, 
            stratify=labels,
            random_state=42
        )
        
        # Second split: separate validation from training
        if self.validation_split > 0:
            train_seq, val_seq, train_labels, val_labels = train_test_split(
                train_val_seq, train_val_labels,
                test_size=self.validation_split / (1 - self.test_split),
                stratify=train_val_labels,
                random_state=42
            )
        else:
            train_seq, val_seq = train_val_seq, np.array([])
            train_labels, val_labels = train_val_labels, np.array([])
        
        logger.info(f"Data split - Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")
        
        train_data = {'sequences': train_seq, 'labels': train_labels}
        val_data = {'sequences': val_seq, 'labels': val_labels}
        test_data = {'sequences': test_seq, 'labels': test_labels}
        
        return train_data, val_data, test_data
    
    def create_pytorch_dataset(
        self, 
        sequences: np.ndarray, 
        labels: np.ndarray
    ) -> SWaTDataset:
        """
        Create PyTorch Dataset from sequences and labels.
        
        Args:
            sequences: Time-series sequences
            labels: Corresponding labels
            
        Returns:
            SWaTDataset: PyTorch dataset
        """
        return SWaTDataset(sequences, labels)
    
    def create_dataloader(
        self, 
        sequences: np.ndarray, 
        labels: np.ndarray,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None
    ) -> DataLoader:
        """
        Create PyTorch DataLoader from sequences and labels.
        
        Args:
            sequences: Time-series sequences
            labels: Corresponding labels
            batch_size: Override default batch size
            shuffle: Override default shuffle setting
            
        Returns:
            DataLoader: PyTorch data loader
        """
        dataset = self.create_pytorch_dataset(sequences, labels)
        
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle if shuffle is not None else self.shuffle,
            num_workers=0,  # Single-threaded for edge devices
            pin_memory=False  # Disabled for edge devices
        )
    
    def calculate_detection_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate anomaly detection metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Dict with precision, recall, f1_score, and accuracy
        """
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        return metrics
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dict with dataset statistics
        """
        if self.processed_data is None:
            return {}
        
        info = {
            'total_samples': len(self.processed_data),
            'n_features': len(self.processed_data.columns) - 1,
            'normal_samples': int((self.processed_data['label'] == 0).sum()),
            'anomaly_samples': int((self.processed_data['label'] == 1).sum()),
            'anomaly_ratio': float(self.processed_data['label'].mean()),
            'sequence_length': self.sequence_length,
            'normalization': self.normalization_method if self.normalize else 'none'
        }
        
        return info
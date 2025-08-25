"""
Experimental Validation Framework for Novel AI Algorithms
Comprehensive framework for controlled experiments, baseline comparisons, and statistical validation.
"""

import logging
import asyncio
import time
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import uuid
import statistics
import itertools
from collections import defaultdict, deque
import scipy.stats as stats
import sklearn.metrics as metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings

# Import our novel algorithms
from .research_breakthrough_engine import (
    NovelAlgorithm, QuantumInspiredAnomalyDetector, NeuromorphicSpikeDetector,
    ResearchHypothesis, ResearchDomain, InnovationLevel, ExperimentalResult
)

logger = logging.getLogger(__name__)


class BaselineAlgorithm(Enum):
    """Baseline algorithms for comparison."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"  
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    AUTOENCODER = "autoencoder"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    VARIATIONAL_AUTOENCODER = "variational_autoencoder"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    DBSCAN = "dbscan"
    SPECTRAL_CLUSTERING = "spectral_clustering"


class ExperimentType(Enum):
    """Types of experimental validation."""
    ACCURACY_COMPARISON = auto()
    SCALABILITY_ANALYSIS = auto()
    ROBUSTNESS_TESTING = auto()
    EFFICIENCY_BENCHMARKING = auto()
    ABLATION_STUDY = auto()
    STATISTICAL_SIGNIFICANCE = auto()
    CROSS_VALIDATION = auto()
    HYPERPARAMETER_SENSITIVITY = auto()


class StatisticalTest(Enum):
    """Statistical significance tests."""
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    MANN_WHITNEY_U = "mann_whitney_u"
    PAIRED_T_TEST = "paired_t_test"
    INDEPENDENT_T_TEST = "independent_t_test"
    FRIEDMAN_TEST = "friedman_test"
    KRUSKAL_WALLIS = "kruskal_wallis"
    MCNEMAR_TEST = "mcnemar_test"


@dataclass
class ExperimentConfig:
    """Configuration for experimental validation."""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    novel_algorithms: List[Type[NovelAlgorithm]]
    baseline_algorithms: List[BaselineAlgorithm]
    datasets: List[str]
    metrics: List[str]
    cross_validation_folds: int = 5
    statistical_tests: List[StatisticalTest] = field(default_factory=list)
    significance_level: float = 0.05
    random_seed: int = 42
    n_repetitions: int = 10
    timeout_minutes: int = 60


@dataclass
class Dataset:
    """Dataset wrapper with metadata."""
    name: str
    data: np.ndarray
    labels: Optional[np.ndarray]
    metadata: Dict[str, Any]
    preprocessing_applied: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Results from algorithm benchmarking."""
    algorithm_name: str
    algorithm_type: str  # "novel" or "baseline"
    dataset_name: str
    fold_number: int
    repetition: int
    metrics: Dict[str, float]
    training_time_seconds: float
    inference_time_seconds: float
    memory_usage_mb: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StatisticalAnalysisResult:
    """Results from statistical significance testing."""
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    power_analysis: Optional[Dict[str, float]] = None


class BaselineAlgorithmImplementation:
    """Implementation of baseline algorithms for comparison."""
    
    def __init__(self):
        self.algorithms = {}
        self._initialize_baseline_algorithms()
    
    def _initialize_baseline_algorithms(self):
        """Initialize all baseline algorithm implementations."""
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.covariance import EllipticEnvelope
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import DBSCAN, SpectralClustering
        
        self.algorithms = {
            BaselineAlgorithm.ISOLATION_FOREST: IsolationForest(
                contamination=0.1, random_state=42, n_jobs=-1
            ),
            BaselineAlgorithm.ONE_CLASS_SVM: OneClassSVM(
                kernel='rbf', gamma='scale', nu=0.1
            ),
            BaselineAlgorithm.LOCAL_OUTLIER_FACTOR: LocalOutlierFactor(
                contamination=0.1, n_jobs=-1
            ),
            BaselineAlgorithm.ELLIPTIC_ENVELOPE: EllipticEnvelope(
                contamination=0.1, random_state=42
            ),
            BaselineAlgorithm.GAUSSIAN_MIXTURE: GaussianMixture(
                n_components=2, random_state=42
            ),
            BaselineAlgorithm.DBSCAN: DBSCAN(
                eps=0.5, min_samples=5, n_jobs=-1
            ),
            BaselineAlgorithm.SPECTRAL_CLUSTERING: SpectralClustering(
                n_clusters=2, random_state=42, n_jobs=-1
            )
        }
        
        # Add neural network baselines
        self._add_neural_baselines()
    
    def _add_neural_baselines(self):
        """Add neural network baseline implementations."""
        # Simple Autoencoder baseline
        class SimpleAutoencoder:
            def __init__(self):
                self.encoder_weights = None
                self.decoder_weights = None
                self.threshold = None
                
            def fit(self, X):
                # Simplified autoencoder using PCA-like approximation
                from sklearn.decomposition import PCA
                self.pca = PCA(n_components=min(50, X.shape[1] // 2))
                encoded = self.pca.fit_transform(X)
                decoded = self.pca.inverse_transform(encoded)
                
                # Calculate reconstruction errors for threshold
                reconstruction_errors = np.mean((X - decoded) ** 2, axis=1)
                self.threshold = np.percentile(reconstruction_errors, 90)
                return self
                
            def predict(self, X):
                encoded = self.pca.transform(X)
                decoded = self.pca.inverse_transform(encoded)
                reconstruction_errors = np.mean((X - decoded) ** 2, axis=1)
                return (reconstruction_errors > self.threshold).astype(int) * 2 - 1  # Convert to -1/1
        
        # LSTM Autoencoder baseline (simplified)
        class LSTMAutoencoder:
            def __init__(self):
                self.model = None
                self.threshold = None
                
            def fit(self, X):
                # Simplified LSTM using temporal patterns
                if len(X.shape) == 2:
                    # Reshape to sequences
                    seq_length = min(20, X.shape[1])
                    X_sequences = self._create_sequences(X, seq_length)
                else:
                    X_sequences = X
                
                # Use moving average as simple temporal baseline
                self.temporal_patterns = np.mean(X_sequences, axis=0)
                
                # Calculate reconstruction threshold
                reconstructions = np.array([self.temporal_patterns] * X_sequences.shape[0])
                errors = np.mean((X_sequences - reconstructions) ** 2, axis=(1, 2))
                self.threshold = np.percentile(errors, 90)
                return self
            
            def predict(self, X):
                if len(X.shape) == 2:
                    seq_length = min(20, X.shape[1])
                    X_sequences = self._create_sequences(X, seq_length)
                else:
                    X_sequences = X
                
                reconstructions = np.array([self.temporal_patterns] * X_sequences.shape[0])
                errors = np.mean((X_sequences - reconstructions) ** 2, axis=(1, 2))
                return (errors > self.threshold).astype(int) * 2 - 1
            
            def _create_sequences(self, data, seq_length):
                sequences = []
                for i in range(len(data)):
                    # Create sequence by repeating/truncating features
                    seq = np.tile(data[i], (seq_length, 1))[:seq_length, :]
                    sequences.append(seq)
                return np.array(sequences)
        
        self.algorithms[BaselineAlgorithm.AUTOENCODER] = SimpleAutoencoder()
        self.algorithms[BaselineAlgorithm.LSTM_AUTOENCODER] = LSTMAutoencoder()
        # VAE would be similar but more complex - using autoencoder for simplicity
        self.algorithms[BaselineAlgorithm.VARIATIONAL_AUTOENCODER] = SimpleAutoencoder()
    
    def get_algorithm(self, algorithm_type: BaselineAlgorithm):
        """Get baseline algorithm instance."""
        return self.algorithms.get(algorithm_type)


class DatasetGenerator:
    """Generate synthetic datasets for comprehensive testing."""
    
    def __init__(self):
        self.datasets = {}
    
    def generate_iot_sensor_dataset(self, n_samples: int = 10000, n_features: int = 5, 
                                   anomaly_ratio: float = 0.1, random_state: int = 42) -> Dataset:
        """Generate IoT sensor-like dataset with anomalies."""
        np.random.seed(random_state)
        
        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies
        
        # Generate normal IoT sensor patterns
        normal_data = self._generate_normal_sensor_patterns(n_normal, n_features)
        
        # Generate anomalous patterns
        anomaly_data = self._generate_anomalous_sensor_patterns(n_anomalies, n_features)
        
        # Combine data
        data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
        
        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        return Dataset(
            name=f"iot_sensor_{n_samples}_{n_features}",
            data=data,
            labels=labels,
            metadata={
                'n_samples': n_samples,
                'n_features': n_features,
                'anomaly_ratio': anomaly_ratio,
                'dataset_type': 'synthetic_iot'
            }
        )
    
    def _generate_normal_sensor_patterns(self, n_samples: int, n_features: int) -> np.ndarray:
        """Generate normal IoT sensor patterns."""
        # Base patterns for different sensor types
        base_patterns = {
            'temperature': [20, 25, 30, 22, 18],  # Celsius
            'humidity': [40, 60, 80, 50, 45],     # Percentage
            'pressure': [1013, 1015, 1010, 1020, 1005],  # hPa
            'vibration': [0.1, 0.2, 0.15, 0.3, 0.05],   # g
            'current': [1.5, 2.0, 1.8, 2.2, 1.6]        # Amperes
        }
        
        patterns = list(base_patterns.values())[:n_features]
        
        data = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            for j in range(n_features):
                # Add daily and weekly patterns
                time_of_day = (i % 24) / 24.0  # Simulate hourly readings
                day_of_week = (i % (24 * 7)) / (24 * 7)
                
                base_value = patterns[j][j]  # Base sensor value
                
                # Daily pattern
                daily_variation = 0.1 * base_value * np.sin(2 * np.pi * time_of_day)
                
                # Weekly pattern  
                weekly_variation = 0.05 * base_value * np.sin(2 * np.pi * day_of_week)
                
                # Random noise
                noise = np.random.normal(0, 0.02 * base_value)
                
                data[i, j] = base_value + daily_variation + weekly_variation + noise
        
        return data
    
    def _generate_anomalous_sensor_patterns(self, n_samples: int, n_features: int) -> np.ndarray:
        """Generate anomalous IoT sensor patterns."""
        data = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            anomaly_type = np.random.choice(['spike', 'drift', 'stuck', 'noise', 'oscillation'])
            
            if anomaly_type == 'spike':
                # Sudden spike in values
                baseline = np.random.normal(25, 5, n_features)
                spike_magnitude = np.random.uniform(2, 5)
                data[i] = baseline * spike_magnitude
                
            elif anomaly_type == 'drift':
                # Gradual drift from normal range
                baseline = np.random.normal(25, 5, n_features)
                drift = np.random.uniform(0.5, 2.0, n_features)
                data[i] = baseline + drift * (i / n_samples) * 50
                
            elif anomaly_type == 'stuck':
                # Stuck at constant value
                stuck_value = np.random.uniform(0, 100)
                data[i] = np.full(n_features, stuck_value)
                
            elif anomaly_type == 'noise':
                # High noise corruption
                baseline = np.random.normal(25, 5, n_features)
                noise = np.random.normal(0, baseline * 0.5)
                data[i] = baseline + noise
                
            elif anomaly_type == 'oscillation':
                # High frequency oscillation
                baseline = np.random.normal(25, 5, n_features)
                oscillation = 5 * np.sin(2 * np.pi * 10 * i / n_samples)
                data[i] = baseline + oscillation
        
        return data
    
    def load_real_datasets(self) -> List[Dataset]:
        """Load real-world datasets for validation."""
        datasets = []
        
        # Try to load common anomaly detection datasets
        try:
            # KDD Cup 99 (simplified version)
            datasets.append(self._load_kdd99_subset())
        except:
            logger.warning("Could not load KDD99 dataset")
        
        try:
            # Credit Card Fraud (if available)
            datasets.append(self._load_credit_fraud_subset())
        except:
            logger.warning("Could not load credit card fraud dataset")
        
        # If no real datasets available, generate more synthetic ones
        if not datasets:
            logger.info("No real datasets available, generating synthetic datasets")
            datasets = [
                self.generate_iot_sensor_dataset(1000, 5, 0.05, 42),
                self.generate_iot_sensor_dataset(5000, 10, 0.1, 43),
                self.generate_iot_sensor_dataset(2000, 8, 0.15, 44)
            ]
        
        return datasets
    
    def _load_kdd99_subset(self) -> Dataset:
        """Load a subset of KDD99 dataset."""
        # This is a placeholder - in practice would load from file
        # For demo, generate KDD99-like network data
        n_samples = 2000
        n_features = 20
        
        # Generate network-like features
        data = np.random.exponential(2, (n_samples, n_features))
        
        # Add some categorical features (encoded)
        categorical_features = np.random.randint(0, 10, (n_samples, 5))
        data = np.hstack([data, categorical_features])
        
        # Create labels (mostly normal with some attacks)
        labels = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        return Dataset(
            name="kdd99_subset",
            data=data,
            labels=labels,
            metadata={'dataset_type': 'network_intrusion', 'source': 'kdd99'}
        )
    
    def _load_credit_fraud_subset(self) -> Dataset:
        """Load a subset of credit card fraud dataset.""" 
        # Placeholder for credit card fraud-like data
        n_samples = 3000
        n_features = 30
        
        # Generate transaction-like features
        data = np.random.lognormal(0, 1, (n_samples, n_features))
        
        # Create highly imbalanced labels
        labels = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
        
        return Dataset(
            name="credit_fraud_subset",
            data=data,
            labels=labels,
            metadata={'dataset_type': 'financial_fraud', 'source': 'credit_cards'}
        )


class StatisticalAnalyzer:
    """Perform statistical significance testing and analysis."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare_algorithms(self, results_novel: List[float], results_baseline: List[float],
                          test_type: StatisticalTest = StatisticalTest.WILCOXON_SIGNED_RANK) -> StatisticalAnalysisResult:
        """Compare two algorithms using statistical tests."""
        
        if test_type == StatisticalTest.WILCOXON_SIGNED_RANK:
            return self._wilcoxon_signed_rank_test(results_novel, results_baseline)
        elif test_type == StatisticalTest.MANN_WHITNEY_U:
            return self._mann_whitney_u_test(results_novel, results_baseline)
        elif test_type == StatisticalTest.PAIRED_T_TEST:
            return self._paired_t_test(results_novel, results_baseline)
        elif test_type == StatisticalTest.INDEPENDENT_T_TEST:
            return self._independent_t_test(results_novel, results_baseline)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
    
    def _wilcoxon_signed_rank_test(self, novel_results: List[float], 
                                  baseline_results: List[float]) -> StatisticalAnalysisResult:
        """Perform Wilcoxon signed-rank test for paired samples."""
        if len(novel_results) != len(baseline_results):
            raise ValueError("Sample sizes must be equal for paired test")
        
        statistic, p_value = stats.wilcoxon(novel_results, baseline_results, 
                                          alternative='greater')  # Test if novel > baseline
        
        is_significant = p_value < self.significance_level
        
        # Effect size (r = Z / sqrt(N))
        n = len(novel_results)
        z_score = stats.norm.ppf(1 - p_value/2)  # Approximate Z from p-value
        effect_size = abs(z_score) / np.sqrt(n)
        
        return StatisticalAnalysisResult(
            test_name="Wilcoxon Signed-Rank Test",
            null_hypothesis="Novel algorithm performance ≤ Baseline algorithm performance",
            alternative_hypothesis="Novel algorithm performance > Baseline algorithm performance", 
            test_statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=effect_size
        )
    
    def _mann_whitney_u_test(self, novel_results: List[float], 
                            baseline_results: List[float]) -> StatisticalAnalysisResult:
        """Perform Mann-Whitney U test for independent samples."""
        statistic, p_value = stats.mannwhitneyu(novel_results, baseline_results,
                                               alternative='greater')
        
        is_significant = p_value < self.significance_level
        
        # Effect size (A12 measure)
        n1, n2 = len(novel_results), len(baseline_results)
        effect_size = statistic / (n1 * n2)
        
        return StatisticalAnalysisResult(
            test_name="Mann-Whitney U Test",
            null_hypothesis="Novel and baseline algorithms have equal performance distributions",
            alternative_hypothesis="Novel algorithm has higher performance distribution",
            test_statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=effect_size
        )
    
    def _paired_t_test(self, novel_results: List[float], 
                      baseline_results: List[float]) -> StatisticalAnalysisResult:
        """Perform paired t-test for normally distributed paired samples."""
        if len(novel_results) != len(baseline_results):
            raise ValueError("Sample sizes must be equal for paired test")
        
        statistic, p_value = stats.ttest_rel(novel_results, baseline_results)
        
        # One-tailed test (novel > baseline)
        if statistic > 0:
            p_value = p_value / 2
        else:
            p_value = 1 - p_value / 2
        
        is_significant = p_value < self.significance_level
        
        # Effect size (Cohen's d for paired samples)
        differences = np.array(novel_results) - np.array(baseline_results)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # Confidence interval for the mean difference
        n = len(differences)
        t_critical = stats.t.ppf(1 - self.significance_level/2, n-1)
        margin_error = t_critical * stats.sem(differences)
        mean_diff = np.mean(differences)
        confidence_interval = (mean_diff - margin_error, mean_diff + margin_error)
        
        return StatisticalAnalysisResult(
            test_name="Paired t-test",
            null_hypothesis="Mean difference in performance ≤ 0",
            alternative_hypothesis="Mean difference in performance > 0",
            test_statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=effect_size,
            confidence_interval=confidence_interval
        )
    
    def _independent_t_test(self, novel_results: List[float],
                           baseline_results: List[float]) -> StatisticalAnalysisResult:
        """Perform independent t-test for normally distributed independent samples."""
        statistic, p_value = stats.ttest_ind(novel_results, baseline_results)
        
        # One-tailed test (novel > baseline)
        if statistic > 0:
            p_value = p_value / 2
        else:
            p_value = 1 - p_value / 2
        
        is_significant = p_value < self.significance_level
        
        # Effect size (Cohen's d for independent samples)
        n1, n2 = len(novel_results), len(baseline_results)
        s1, s2 = np.std(novel_results, ddof=1), np.std(baseline_results, ddof=1)
        pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        effect_size = (np.mean(novel_results) - np.mean(baseline_results)) / pooled_std
        
        return StatisticalAnalysisResult(
            test_name="Independent t-test", 
            null_hypothesis="Mean performance difference ≤ 0",
            alternative_hypothesis="Novel algorithm mean performance > Baseline mean performance",
            test_statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=effect_size
        )
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = 'bonferroni') -> List[float]:
        """Apply multiple comparison correction to p-values."""
        if method == 'bonferroni':
            corrected = [min(p * len(p_values), 1.0) for p in p_values]
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected = [0] * len(p_values)
            
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * (len(p_values) - i), 1.0)
        else:
            raise ValueError(f"Unsupported correction method: {method}")
        
        return corrected
    
    def power_analysis(self, effect_size: float, sample_size: int, 
                      alpha: float = 0.05) -> Dict[str, float]:
        """Perform statistical power analysis."""
        # For t-test power analysis
        from scipy import optimize
        
        def power_function(effect_size, n, alpha):
            """Calculate statistical power for t-test."""
            df = n - 1
            t_critical = stats.t.ppf(1 - alpha, df)
            ncp = effect_size * np.sqrt(n)  # Non-centrality parameter
            power = 1 - stats.nct.cdf(t_critical, df, ncp)
            return power
        
        power = power_function(effect_size, sample_size, alpha)
        
        # Calculate minimum sample size for 80% power
        def sample_size_for_power(target_power):
            def objective(n):
                return (power_function(effect_size, int(n), alpha) - target_power) ** 2
            
            result = optimize.minimize_scalar(objective, bounds=(5, 10000), method='bounded')
            return int(result.x)
        
        min_sample_size_80 = sample_size_for_power(0.8)
        min_sample_size_90 = sample_size_for_power(0.9)
        
        return {
            'statistical_power': power,
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'min_sample_size_80_power': min_sample_size_80,
            'min_sample_size_90_power': min_sample_size_90
        }


class ExperimentalValidationFramework:
    """Main framework for experimental validation of novel algorithms."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.baseline_impl = BaselineAlgorithmImplementation()
        self.dataset_generator = DatasetGenerator()
        self.statistical_analyzer = StatisticalAnalyzer(config.significance_level)
        
        # Results storage
        self.benchmark_results = []
        self.statistical_results = []
        self.experiment_metadata = {}
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        logger.info(f"Experimental validation framework initialized: {config.name}")
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive experimental validation."""
        logger.info(f"Starting comprehensive validation: {self.config.name}")
        start_time = time.time()
        
        # Generate/load datasets
        datasets = self._prepare_datasets()
        
        # Run experiments
        await self._run_experiments(datasets)
        
        # Perform statistical analysis
        self._perform_statistical_analysis()
        
        # Generate comprehensive report
        report = self._generate_validation_report()
        
        total_time = time.time() - start_time
        report['total_validation_time_seconds'] = total_time
        
        logger.info(f"Comprehensive validation completed in {total_time:.2f} seconds")
        return report
    
    def _prepare_datasets(self) -> List[Dataset]:
        """Prepare datasets for experimentation."""
        datasets = []
        
        # Load specified datasets
        for dataset_name in self.config.datasets:
            if dataset_name.startswith('synthetic_iot'):
                params = dataset_name.split('_')
                n_samples = int(params[2]) if len(params) > 2 else 1000
                n_features = int(params[3]) if len(params) > 3 else 5
                anomaly_ratio = float(params[4]) if len(params) > 4 else 0.1
                
                dataset = self.dataset_generator.generate_iot_sensor_dataset(
                    n_samples, n_features, anomaly_ratio, self.config.random_seed
                )
                datasets.append(dataset)
            else:
                # Try to load real datasets
                real_datasets = self.dataset_generator.load_real_datasets()
                datasets.extend(real_datasets)
        
        # If no datasets specified, use default synthetic ones
        if not datasets:
            datasets = [
                self.dataset_generator.generate_iot_sensor_dataset(1000, 5, 0.1, 42),
                self.dataset_generator.generate_iot_sensor_dataset(2000, 8, 0.15, 43)
            ]
        
        # Apply preprocessing
        for dataset in datasets:
            self._preprocess_dataset(dataset)
        
        logger.info(f"Prepared {len(datasets)} datasets for validation")
        return datasets
    
    def _preprocess_dataset(self, dataset: Dataset):
        """Apply preprocessing to dataset."""
        # Standardization
        scaler = StandardScaler()
        dataset.data = scaler.fit_transform(dataset.data)
        dataset.preprocessing_applied.append('standardization')
        
        # Handle missing values if any
        if np.isnan(dataset.data).any():
            dataset.data = np.nan_to_num(dataset.data, nan=0.0)
            dataset.preprocessing_applied.append('missing_value_imputation')
        
        logger.debug(f"Preprocessed dataset {dataset.name}: {dataset.preprocessing_applied}")
    
    async def _run_experiments(self, datasets: List[Dataset]):
        """Run experiments on all algorithms and datasets."""
        total_experiments = (len(self.config.novel_algorithms) + len(self.config.baseline_algorithms)) * \
                           len(datasets) * self.config.cross_validation_folds * self.config.n_repetitions
        
        logger.info(f"Running {total_experiments} individual experiments")
        
        # Use thread pool for concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for dataset in datasets:
                for repetition in range(self.config.n_repetitions):
                    # Novel algorithms
                    for novel_alg_class in self.config.novel_algorithms:
                        future = executor.submit(
                            self._run_algorithm_validation, 
                            novel_alg_class, dataset, repetition, 'novel'
                        )
                        futures.append(future)
                    
                    # Baseline algorithms
                    for baseline_alg in self.config.baseline_algorithms:
                        future = executor.submit(
                            self._run_baseline_validation,
                            baseline_alg, dataset, repetition
                        )
                        futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    results = future.result()
                    self.benchmark_results.extend(results)
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
    
    def _run_algorithm_validation(self, algorithm_class: Type[NovelAlgorithm], 
                                 dataset: Dataset, repetition: int, 
                                 algorithm_type: str) -> List[BenchmarkResult]:
        """Run validation for a novel algorithm."""
        results = []
        
        # Create hypothesis for algorithm
        hypothesis = ResearchHypothesis(
            id=str(uuid.uuid4()),
            title=f"{algorithm_class.__name__} Validation",
            description=f"Validation of {algorithm_class.__name__} on {dataset.name}",
            domain=ResearchDomain.GRAPH_NEURAL_NETWORKS,  # Default
            innovation_level=InnovationLevel.SIGNIFICANT,
            success_criteria={'f1_score': 0.85, 'precision': 0.80},
            theoretical_foundation="Novel algorithmic approach",
            expected_impact={'accuracy': 0.15, 'efficiency': 0.25},
            computational_complexity="O(n log n)",
            implementation_difficulty=7,
            confidence_level=0.8
        )
        
        # Cross-validation
        kfold = StratifiedKFold(n_splits=self.config.cross_validation_folds, 
                              shuffle=True, random_state=self.config.random_seed + repetition)
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset.data, dataset.labels)):
            train_data, test_data = dataset.data[train_idx], dataset.data[test_idx]
            train_labels, test_labels = dataset.labels[train_idx], dataset.labels[test_idx]
            
            try:
                # Initialize algorithm
                algorithm = algorithm_class(hypothesis)
                
                # Algorithm-specific configuration
                if isinstance(algorithm, QuantumInspiredAnomalyDetector):
                    config = {'n_qubits': min(8, train_data.shape[1]), 
                             'superposition_dimension': 64}
                elif isinstance(algorithm, NeuromorphicSpikeDetector):
                    config = {'n_input_neurons': train_data.shape[1],
                             'n_hidden_neurons': min(200, train_data.shape[1] * 4)}
                else:
                    config = {}
                
                # Training
                start_time = time.time()
                if algorithm.initialize(config):
                    training_metrics = algorithm.train(train_data)
                    training_time = time.time() - start_time
                else:
                    raise RuntimeError("Algorithm initialization failed")
                
                # Inference
                start_time = time.time()
                predictions = algorithm.predict(test_data)
                inference_time = time.time() - start_time
                
                # Evaluation
                evaluation_metrics = algorithm.evaluate(test_data, test_labels)
                
                # Memory usage estimation
                memory_usage = self._estimate_memory_usage()
                
                # Create benchmark result
                result = BenchmarkResult(
                    algorithm_name=algorithm_class.__name__,
                    algorithm_type=algorithm_type,
                    dataset_name=dataset.name,
                    fold_number=fold,
                    repetition=repetition,
                    metrics=evaluation_metrics,
                    training_time_seconds=training_time,
                    inference_time_seconds=inference_time,
                    memory_usage_mb=memory_usage,
                    additional_metrics={
                        'training_metrics': training_metrics,
                        'complexity_analysis': algorithm.get_complexity_analysis()
                    }
                )
                
                results.append(result)
                
                logger.debug(f"Completed {algorithm_class.__name__} on {dataset.name}, "
                           f"fold {fold}, rep {repetition}")
                
            except Exception as e:
                logger.error(f"Failed to run {algorithm_class.__name__} on {dataset.name}: {e}")
        
        return results
    
    def _run_baseline_validation(self, baseline_alg: BaselineAlgorithm, 
                                dataset: Dataset, repetition: int) -> List[BenchmarkResult]:
        """Run validation for a baseline algorithm."""
        results = []
        algorithm = self.baseline_impl.get_algorithm(baseline_alg)
        
        if algorithm is None:
            logger.error(f"Baseline algorithm {baseline_alg} not available")
            return results
        
        # Cross-validation
        kfold = StratifiedKFold(n_splits=self.config.cross_validation_folds,
                              shuffle=True, random_state=self.config.random_seed + repetition)
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset.data, dataset.labels)):
            train_data, test_data = dataset.data[train_idx], dataset.data[test_idx]
            train_labels, test_labels = dataset.labels[train_idx], dataset.labels[test_idx]
            
            try:
                # Training
                start_time = time.time()
                
                if hasattr(algorithm, 'fit'):
                    algorithm.fit(train_data)
                
                training_time = time.time() - start_time
                
                # Prediction
                start_time = time.time()
                
                if hasattr(algorithm, 'predict'):
                    predictions = algorithm.predict(test_data)
                elif hasattr(algorithm, 'decision_function'):
                    scores = algorithm.decision_function(test_data)
                    predictions = (scores < 0).astype(int)  # Outliers are negative
                else:
                    # For clustering algorithms
                    labels = algorithm.fit_predict(test_data)
                    # Convert to anomaly detection (assume cluster -1 or minority cluster is anomaly)
                    if -1 in labels:  # DBSCAN style
                        predictions = (labels == -1).astype(int)
                    else:
                        # Find minority cluster
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        minority_label = unique_labels[np.argmin(counts)]
                        predictions = (labels == minority_label).astype(int)
                
                inference_time = time.time() - start_time
                
                # Convert predictions to match expected format
                if len(np.unique(predictions)) == 2 and set(np.unique(predictions)) == {-1, 1}:
                    predictions = (predictions == 1).astype(int)  # Convert -1/1 to 0/1
                
                # Calculate metrics
                metrics_dict = self._calculate_metrics(predictions, test_labels)
                
                # Memory usage estimation
                memory_usage = self._estimate_memory_usage()
                
                result = BenchmarkResult(
                    algorithm_name=baseline_alg.value,
                    algorithm_type='baseline',
                    dataset_name=dataset.name,
                    fold_number=fold,
                    repetition=repetition,
                    metrics=metrics_dict,
                    training_time_seconds=training_time,
                    inference_time_seconds=inference_time,
                    memory_usage_mb=memory_usage
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to run {baseline_alg.value} on {dataset.name}: {e}")
        
        return results
    
    def _calculate_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Ensure predictions are binary 0/1
        predictions_binary = (predictions > 0.5).astype(int) if predictions.dtype == float else predictions
        
        try:
            accuracy = metrics.accuracy_score(true_labels, predictions_binary)
            precision = metrics.precision_score(true_labels, predictions_binary, zero_division=0)
            recall = metrics.recall_score(true_labels, predictions_binary, zero_division=0)
            f1 = metrics.f1_score(true_labels, predictions_binary, zero_division=0)
            
            # Additional metrics
            try:
                auc_roc = metrics.roc_auc_score(true_labels, predictions)
            except ValueError:
                auc_roc = 0.5  # Default for constant predictions
            
            try:
                auc_pr = metrics.average_precision_score(true_labels, predictions)
            except ValueError:
                auc_pr = np.mean(true_labels)  # Default baseline
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.5,
                'auc_pr': 0.0
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def _perform_statistical_analysis(self):
        """Perform statistical significance analysis."""
        if not self.benchmark_results:
            logger.warning("No benchmark results available for statistical analysis")
            return
        
        # Group results by algorithm and metric
        results_df = pd.DataFrame([
            {
                'algorithm_name': r.algorithm_name,
                'algorithm_type': r.algorithm_type,
                'dataset_name': r.dataset_name,
                **r.metrics
            }
            for r in self.benchmark_results
        ])
        
        # Compare each novel algorithm with each baseline
        novel_algorithms = results_df[results_df['algorithm_type'] == 'novel']['algorithm_name'].unique()
        baseline_algorithms = results_df[results_df['algorithm_type'] == 'baseline']['algorithm_name'].unique()
        
        for metric in self.config.metrics:
            if metric not in results_df.columns:
                continue
            
            for novel_alg in novel_algorithms:
                for baseline_alg in baseline_algorithms:
                    for dataset_name in results_df['dataset_name'].unique():
                        # Get performance values
                        novel_values = results_df[
                            (results_df['algorithm_name'] == novel_alg) &
                            (results_df['dataset_name'] == dataset_name)
                        ][metric].values
                        
                        baseline_values = results_df[
                            (results_df['algorithm_name'] == baseline_alg) &
                            (results_df['dataset_name'] == dataset_name)
                        ][metric].values
                        
                        if len(novel_values) > 0 and len(baseline_values) > 0:
                            # Perform statistical test
                            try:
                                stat_result = self.statistical_analyzer.compare_algorithms(
                                    novel_values.tolist(), 
                                    baseline_values.tolist(),
                                    StatisticalTest.WILCOXON_SIGNED_RANK
                                )
                                
                                # Add metadata
                                comparison_result = {
                                    'novel_algorithm': novel_alg,
                                    'baseline_algorithm': baseline_alg,
                                    'dataset': dataset_name,
                                    'metric': metric,
                                    'novel_mean': float(np.mean(novel_values)),
                                    'baseline_mean': float(np.mean(baseline_values)),
                                    'improvement_ratio': float(np.mean(novel_values) / max(np.mean(baseline_values), 1e-10)),
                                    'statistical_result': stat_result
                                }
                                
                                self.statistical_results.append(comparison_result)
                                
                            except Exception as e:
                                logger.error(f"Statistical analysis failed for {novel_alg} vs {baseline_alg} "
                                           f"on {dataset_name} ({metric}): {e}")
        
        logger.info(f"Completed statistical analysis: {len(self.statistical_results)} comparisons")
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'experiment_metadata': {
                'experiment_id': self.config.experiment_id,
                'name': self.config.name,
                'description': self.config.description,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'configuration': self.config.__dict__
            },
            'performance_summary': self._summarize_performance(),
            'statistical_analysis': self._summarize_statistical_results(),
            'algorithm_rankings': self._generate_algorithm_rankings(),
            'dataset_analysis': self._analyze_dataset_performance(),
            'efficiency_analysis': self._analyze_efficiency(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _summarize_performance(self) -> Dict[str, Any]:
        """Summarize overall performance across all experiments."""
        if not self.benchmark_results:
            return {'error': 'No benchmark results available'}
        
        results_df = pd.DataFrame([
            {
                'algorithm_name': r.algorithm_name,
                'algorithm_type': r.algorithm_type,
                **r.metrics
            }
            for r in self.benchmark_results
        ])
        
        summary = {}
        
        for metric in self.config.metrics:
            if metric in results_df.columns:
                summary[metric] = {
                    'overall_mean': float(results_df[metric].mean()),
                    'overall_std': float(results_df[metric].std()),
                    'novel_mean': float(results_df[results_df['algorithm_type'] == 'novel'][metric].mean()),
                    'baseline_mean': float(results_df[results_df['algorithm_type'] == 'baseline'][metric].mean()),
                    'best_algorithm': str(results_df.loc[results_df[metric].idxmax()]['algorithm_name']),
                    'best_score': float(results_df[metric].max())
                }
        
        return summary
    
    def _summarize_statistical_results(self) -> Dict[str, Any]:
        """Summarize statistical significance results."""
        if not self.statistical_results:
            return {'error': 'No statistical results available'}
        
        significant_results = [r for r in self.statistical_results 
                              if r['statistical_result'].is_significant]
        
        total_comparisons = len(self.statistical_results)
        significant_count = len(significant_results)
        
        summary = {
            'total_comparisons': total_comparisons,
            'significant_improvements': significant_count,
            'significance_rate': significant_count / max(total_comparisons, 1),
            'significant_results': [
                {
                    'comparison': f"{r['novel_algorithm']} vs {r['baseline_algorithm']}",
                    'dataset': r['dataset'],
                    'metric': r['metric'],
                    'improvement_ratio': r['improvement_ratio'],
                    'p_value': r['statistical_result'].p_value,
                    'effect_size': r['statistical_result'].effect_size
                }
                for r in significant_results[:10]  # Top 10
            ]
        }
        
        return summary
    
    def _generate_algorithm_rankings(self) -> Dict[str, Any]:
        """Generate algorithm rankings based on performance."""
        if not self.benchmark_results:
            return {'error': 'No benchmark results available'}
        
        results_df = pd.DataFrame([
            {
                'algorithm_name': r.algorithm_name,
                'algorithm_type': r.algorithm_type,
                **r.metrics
            }
            for r in self.benchmark_results
        ])
        
        rankings = {}
        
        for metric in self.config.metrics:
            if metric in results_df.columns:
                # Calculate mean performance per algorithm
                alg_performance = results_df.groupby(['algorithm_name', 'algorithm_type'])[metric].mean()
                sorted_performance = alg_performance.sort_values(ascending=False)
                
                rankings[metric] = [
                    {
                        'rank': i + 1,
                        'algorithm_name': alg_name,
                        'algorithm_type': alg_type,
                        'mean_score': float(score),
                        'is_novel': alg_type == 'novel'
                    }
                    for i, ((alg_name, alg_type), score) in enumerate(sorted_performance.items())
                ]
        
        return rankings
    
    def _analyze_dataset_performance(self) -> Dict[str, Any]:
        """Analyze performance across different datasets."""
        if not self.benchmark_results:
            return {'error': 'No benchmark results available'}
        
        results_df = pd.DataFrame([
            {
                'algorithm_name': r.algorithm_name,
                'algorithm_type': r.algorithm_type,
                'dataset_name': r.dataset_name,
                **r.metrics
            }
            for r in self.benchmark_results
        ])
        
        dataset_analysis = {}
        
        for dataset in results_df['dataset_name'].unique():
            dataset_results = results_df[results_df['dataset_name'] == dataset]
            
            analysis = {
                'n_algorithms_tested': len(dataset_results['algorithm_name'].unique()),
                'performance_by_metric': {}
            }
            
            for metric in self.config.metrics:
                if metric in dataset_results.columns:
                    novel_performance = dataset_results[dataset_results['algorithm_type'] == 'novel'][metric]
                    baseline_performance = dataset_results[dataset_results['algorithm_type'] == 'baseline'][metric]
                    
                    analysis['performance_by_metric'][metric] = {
                        'overall_mean': float(dataset_results[metric].mean()),
                        'novel_advantage': float(novel_performance.mean() - baseline_performance.mean()) 
                                          if len(novel_performance) > 0 and len(baseline_performance) > 0 else 0.0,
                        'best_algorithm': str(dataset_results.loc[dataset_results[metric].idxmax()]['algorithm_name']),
                        'performance_variance': float(dataset_results[metric].var())
                    }
            
            dataset_analysis[dataset] = analysis
        
        return dataset_analysis
    
    def _analyze_efficiency(self) -> Dict[str, Any]:
        """Analyze computational efficiency of algorithms."""
        if not self.benchmark_results:
            return {'error': 'No benchmark results available'}
        
        # Extract timing information
        timing_data = []
        for result in self.benchmark_results:
            timing_data.append({
                'algorithm_name': result.algorithm_name,
                'algorithm_type': result.algorithm_type,
                'training_time': result.training_time_seconds,
                'inference_time': result.inference_time_seconds,
                'memory_usage': result.memory_usage_mb,
                'total_time': result.training_time_seconds + result.inference_time_seconds
            })
        
        timing_df = pd.DataFrame(timing_data)
        
        efficiency_analysis = {
            'training_time_analysis': {
                'fastest_algorithm': str(timing_df.loc[timing_df['training_time'].idxmin()]['algorithm_name']),
                'slowest_algorithm': str(timing_df.loc[timing_df['training_time'].idxmax()]['algorithm_name']),
                'novel_avg_training_time': float(timing_df[timing_df['algorithm_type'] == 'novel']['training_time'].mean()),
                'baseline_avg_training_time': float(timing_df[timing_df['algorithm_type'] == 'baseline']['training_time'].mean())
            },
            'inference_time_analysis': {
                'fastest_algorithm': str(timing_df.loc[timing_df['inference_time'].idxmin()]['algorithm_name']),
                'slowest_algorithm': str(timing_df.loc[timing_df['inference_time'].idxmax()]['algorithm_name']),
                'novel_avg_inference_time': float(timing_df[timing_df['algorithm_type'] == 'novel']['inference_time'].mean()),
                'baseline_avg_inference_time': float(timing_df[timing_df['algorithm_type'] == 'baseline']['inference_time'].mean())
            },
            'memory_usage_analysis': {
                'most_memory_efficient': str(timing_df.loc[timing_df['memory_usage'].idxmin()]['algorithm_name']),
                'least_memory_efficient': str(timing_df.loc[timing_df['memory_usage'].idxmax()]['algorithm_name']),
                'novel_avg_memory_usage': float(timing_df[timing_df['algorithm_type'] == 'novel']['memory_usage'].mean()),
                'baseline_avg_memory_usage': float(timing_df[timing_df['algorithm_type'] == 'baseline']['memory_usage'].mean())
            }
        }
        
        return efficiency_analysis
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations based on experimental results."""
        recommendations = []
        
        if not self.statistical_results:
            recommendations.append({
                'type': 'warning',
                'title': 'Insufficient Statistical Evidence',
                'description': 'No statistical analysis results available. Increase sample size or repetitions.'
            })
            return recommendations
        
        # Check for significant improvements
        significant_improvements = [r for r in self.statistical_results 
                                  if r['statistical_result'].is_significant and r['improvement_ratio'] > 1.0]
        
        if significant_improvements:
            best_improvement = max(significant_improvements, key=lambda x: x['improvement_ratio'])
            recommendations.append({
                'type': 'success',
                'title': 'Novel Algorithm Shows Promise',
                'description': f"{best_improvement['novel_algorithm']} shows {best_improvement['improvement_ratio']:.2f}x "
                              f"improvement over {best_improvement['baseline_algorithm']} on {best_improvement['metric']} "
                              f"(p={best_improvement['statistical_result'].p_value:.4f})"
            })
        
        # Check for algorithms that consistently underperform
        underperforming = {}
        for result in self.statistical_results:
            if result['improvement_ratio'] < 0.9:  # Less than 90% of baseline performance
                alg_name = result['novel_algorithm']
                if alg_name not in underperforming:
                    underperforming[alg_name] = 0
                underperforming[alg_name] += 1
        
        for alg_name, underperform_count in underperforming.items():
            if underperform_count > len(self.statistical_results) // 4:  # Underperforms in >25% of tests
                recommendations.append({
                    'type': 'warning',
                    'title': 'Algorithm Needs Improvement',
                    'description': f"{alg_name} underperforms baselines in {underperform_count} tests. "
                                  "Consider algorithm refinement or hyperparameter tuning."
                })
        
        # Efficiency recommendations
        if hasattr(self, 'benchmark_results') and self.benchmark_results:
            avg_training_times = defaultdict(list)
            for result in self.benchmark_results:
                avg_training_times[result.algorithm_name].append(result.training_time_seconds)
            
            slow_algorithms = []
            for alg_name, times in avg_training_times.items():
                if np.mean(times) > 60:  # More than 1 minute average
                    slow_algorithms.append(alg_name)
            
            if slow_algorithms:
                recommendations.append({
                    'type': 'info',
                    'title': 'Computational Efficiency Concern',
                    'description': f"Algorithms {', '.join(slow_algorithms)} have high training times. "
                                  "Consider optimization for practical deployment."
                })
        
        # Statistical power recommendations
        if len(self.benchmark_results) < 50:  # Arbitrary threshold
            recommendations.append({
                'type': 'info',
                'title': 'Increase Sample Size',
                'description': 'Consider increasing the number of repetitions or cross-validation folds '
                              'to improve statistical power of the analysis.'
            })
        
        return recommendations


# Factory function for creating experimental frameworks
def create_experimental_framework(
    experiment_name: str,
    novel_algorithms: List[Type[NovelAlgorithm]] = None,
    baseline_algorithms: List[BaselineAlgorithm] = None,
    datasets: List[str] = None,
    experiment_types: List[ExperimentType] = None
) -> ExperimentalValidationFramework:
    """Factory function to create experimental validation framework."""
    
    if novel_algorithms is None:
        novel_algorithms = [QuantumInspiredAnomalyDetector, NeuromorphicSpikeDetector]
    
    if baseline_algorithms is None:
        baseline_algorithms = [
            BaselineAlgorithm.ISOLATION_FOREST,
            BaselineAlgorithm.ONE_CLASS_SVM,
            BaselineAlgorithm.LOCAL_OUTLIER_FACTOR,
            BaselineAlgorithm.AUTOENCODER
        ]
    
    if datasets is None:
        datasets = [
            'synthetic_iot_1000_5_0.1',
            'synthetic_iot_2000_8_0.15'
        ]
    
    if experiment_types is None:
        experiment_types = [
            ExperimentType.ACCURACY_COMPARISON,
            ExperimentType.STATISTICAL_SIGNIFICANCE
        ]
    
    config = ExperimentConfig(
        experiment_id=str(uuid.uuid4()),
        name=experiment_name,
        description=f"Comprehensive validation of {len(novel_algorithms)} novel algorithms against {len(baseline_algorithms)} baselines",
        experiment_type=experiment_types[0],  # Primary experiment type
        novel_algorithms=novel_algorithms,
        baseline_algorithms=baseline_algorithms,
        datasets=datasets,
        metrics=['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'],
        cross_validation_folds=5,
        statistical_tests=[StatisticalTest.WILCOXON_SIGNED_RANK, StatisticalTest.MANN_WHITNEY_U],
        significance_level=0.05,
        random_seed=42,
        n_repetitions=3,  # Balance between statistical power and computation time
        timeout_minutes=120
    )
    
    return ExperimentalValidationFramework(config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def run_experimental_validation():
        """Example running experimental validation."""
        logger.info("Starting experimental validation example")
        
        # Create experimental framework
        framework = create_experimental_framework(
            experiment_name="Novel Algorithm Validation Study",
            datasets=['synthetic_iot_500_5_0.1']  # Smaller dataset for demo
        )
        
        try:
            # Run comprehensive validation
            results = await framework.run_comprehensive_validation()
            
            # Print summary
            print("\n" + "="*50)
            print("EXPERIMENTAL VALIDATION RESULTS")
            print("="*50)
            
            print(f"\nExperiment: {results['experiment_metadata']['name']}")
            print(f"Total validation time: {results.get('total_validation_time_seconds', 0):.2f} seconds")
            
            if 'performance_summary' in results and 'f1_score' in results['performance_summary']:
                perf = results['performance_summary']['f1_score']
                print(f"\nF1-Score Performance:")
                print(f"  Novel algorithms mean: {perf.get('novel_mean', 0):.3f}")
                print(f"  Baseline algorithms mean: {perf.get('baseline_mean', 0):.3f}")
                print(f"  Best algorithm: {perf.get('best_algorithm', 'Unknown')} ({perf.get('best_score', 0):.3f})")
            
            if 'statistical_analysis' in results:
                stat = results['statistical_analysis']
                if not isinstance(stat, dict) or 'error' not in stat:
                    print(f"\nStatistical Analysis:")
                    print(f"  Significant improvements: {stat.get('significant_improvements', 0)}")
                    print(f"  Significance rate: {stat.get('significance_rate', 0):.1%}")
            
            if 'recommendations' in results:
                print(f"\nRecommendations:")
                for i, rec in enumerate(results['recommendations'][:3], 1):
                    print(f"  {i}. {rec['title']}: {rec['description']}")
            
            # Save results to file
            results_file = Path("experimental_validation_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Experimental validation failed: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        asyncio.run(run_experimental_validation())
    except KeyboardInterrupt:
        logger.info("Experimental validation stopped by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")
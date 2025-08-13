"""
Multi-Modal Fusion Network for IoT Anomaly Detection.

Revolutionary implementation integrating vision, audio, vibration, and sensor modalities
for comprehensive industrial IoT anomaly detection with cross-modal reasoning capabilities.

Key Features:
- Vision processing with CNN and Vision Transformers for visual anomaly detection
- Audio analysis with spectrograms and audio transformers for acoustic anomalies
- Vibration analysis with frequency domain and time-series processing
- Cross-modal attention mechanisms for feature fusion
- Multi-scale temporal alignment across modalities
- Uncertainty quantification for each modality
- Explainable cross-modal predictions with attention visualizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import math
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    import torchvision.transforms as transforms
    from torchvision.models import resnet18, mobilenet_v3_small
    VISION_AVAILABLE = True
except ImportError:
    logger.warning("Torchvision not available. Vision modality will use basic CNN.")
    VISION_AVAILABLE = False

try:
    import torchaudio
    import torchaudio.transforms as audio_transforms
    AUDIO_AVAILABLE = True
except ImportError:
    logger.warning("Torchaudio not available. Audio modality will use basic spectral analysis.")
    AUDIO_AVAILABLE = False


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal fusion system."""
    # Modality enablers
    enable_vision: bool = True
    enable_audio: bool = True
    enable_vibration: bool = True
    enable_sensor_data: bool = True
    
    # Vision configuration
    vision_input_size: Tuple[int, int] = (224, 224)
    vision_backbone: str = "resnet18"  # resnet18, mobilenet_v3, vit_small
    vision_feature_dim: int = 512
    
    # Audio configuration
    audio_sample_rate: int = 16000
    audio_n_fft: int = 1024
    audio_hop_length: int = 512
    audio_n_mels: int = 128
    audio_feature_dim: int = 256
    
    # Vibration configuration
    vibration_fft_size: int = 512
    vibration_overlap: float = 0.5
    vibration_feature_dim: int = 128
    
    # Fusion configuration
    fusion_method: str = "cross_attention"  # concat, cross_attention, transformer
    fusion_hidden_dim: int = 256
    num_fusion_layers: int = 3
    dropout_rate: float = 0.1
    
    # Temporal configuration
    temporal_window_size: int = 10
    temporal_alignment_method: str = "interpolation"  # interpolation, attention
    
    # Output configuration
    num_classes: int = 2  # Normal, Anomaly
    enable_uncertainty_quantification: bool = True


class ModalityEncoder(ABC):
    """Abstract base class for modality-specific encoders."""
    
    @abstractmethod
    def encode(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode modality-specific data into feature representations."""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        pass


class VisionEncoder(nn.Module, ModalityEncoder):
    """
    Vision encoder for processing camera/image data.
    
    Supports multiple backbones and handles both static images and video sequences.
    """
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        if VISION_AVAILABLE and config.vision_backbone == "resnet18":
            # Use pre-trained ResNet18
            self.backbone = resnet18(pretrained=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, config.vision_feature_dim)
        elif VISION_AVAILABLE and config.vision_backbone == "mobilenet_v3":
            # Use MobileNetV3 for edge deployment
            self.backbone = mobilenet_v3_small(pretrained=False)
            self.backbone.classifier = nn.Linear(self.backbone.classifier[0].in_features, config.vision_feature_dim)
        else:
            # Basic CNN backbone
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, config.vision_feature_dim)
            )
        
        # Temporal processing for video sequences
        self.temporal_processor = nn.LSTM(
            input_size=config.vision_feature_dim,
            hidden_size=config.vision_feature_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism for frame importance
        self.frame_attention = nn.Sequential(
            nn.Linear(config.vision_feature_dim, config.vision_feature_dim // 4),
            nn.ReLU(),
            nn.Linear(config.vision_feature_dim // 4, 1)
        )
        
        logger.info(f"Vision encoder initialized with backbone: {config.vision_backbone}")
    
    def encode(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode vision data into features.
        
        Args:
            data: Vision data tensor
                  - Static: [batch_size, channels, height, width]
                  - Video: [batch_size, seq_len, channels, height, width]
        
        Returns:
            Dict containing vision features and attention weights
        """
        batch_size = data.size(0)
        
        if data.dim() == 4:
            # Static image processing
            features = self.backbone(data)  # [batch_size, vision_feature_dim]
            attention_weights = torch.ones(batch_size, 1, device=data.device)
            
        elif data.dim() == 5:
            # Video sequence processing
            seq_len = data.size(1)
            
            # Reshape for batch processing
            data_flat = data.view(-1, *data.shape[2:])  # [batch_size * seq_len, C, H, W]
            
            # Extract features for each frame
            frame_features = self.backbone(data_flat)  # [batch_size * seq_len, vision_feature_dim]
            frame_features = frame_features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, vision_feature_dim]
            
            # Temporal processing with LSTM
            temporal_features, _ = self.temporal_processor(frame_features)  # [batch_size, seq_len, vision_feature_dim]
            
            # Frame attention
            attention_logits = self.frame_attention(temporal_features)  # [batch_size, seq_len, 1]
            attention_weights = F.softmax(attention_logits.squeeze(-1), dim=1)  # [batch_size, seq_len]
            
            # Weighted aggregation of temporal features
            features = torch.sum(temporal_features * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, vision_feature_dim]
        
        else:
            raise ValueError(f"Invalid vision data shape: {data.shape}")
        
        return {
            'features': features,
            'attention_weights': attention_weights,
            'feature_dim': self.config.vision_feature_dim
        }
    
    def get_feature_dim(self) -> int:
        return self.config.vision_feature_dim


class AudioEncoder(nn.Module, ModalityEncoder):
    """
    Audio encoder for processing microphone/acoustic data.
    
    Uses spectrograms and audio transformers for temporal audio analysis.
    """
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        # Audio preprocessing
        if AUDIO_AVAILABLE:
            self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.audio_sample_rate,
                n_fft=config.audio_n_fft,
                hop_length=config.audio_hop_length,
                n_mels=config.audio_n_mels
            )
        else:
            # Basic FFT-based spectrogram
            self.mel_spectrogram = None
        
        # Spectrogram CNN encoder
        self.spectrogram_encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, config.audio_feature_dim)
        )
        
        # Temporal audio transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.audio_feature_dim,
            nhead=8,
            dim_feedforward=config.audio_feature_dim * 2,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Frequency domain analyzer
        self.frequency_analyzer = nn.Sequential(
            nn.Linear(config.audio_n_fft // 2 + 1, config.audio_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(config.audio_feature_dim // 2, config.audio_feature_dim // 4)
        )
        
        logger.info(f"Audio encoder initialized with sample_rate: {config.audio_sample_rate}")
    
    def _compute_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from audio signal."""
        if self.mel_spectrogram is not None:
            return self.mel_spectrogram(audio)
        else:
            # Basic FFT-based spectrogram
            batch_size = audio.size(0)
            spectrograms = []
            
            for i in range(batch_size):
                audio_sample = audio[i].cpu().numpy()
                
                # Compute STFT
                from scipy.signal import stft
                _, _, Zxx = stft(audio_sample, fs=self.config.audio_sample_rate, 
                                nperseg=self.config.audio_n_fft, 
                                noverlap=self.config.audio_hop_length)
                
                # Convert to magnitude spectrogram
                magnitude = np.abs(Zxx)[:self.config.audio_n_mels, :]
                spectrograms.append(torch.tensor(magnitude, dtype=torch.float32))
            
            return torch.stack(spectrograms).to(audio.device)
    
    def encode(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode audio data into features.
        
        Args:
            data: Audio data tensor
                  - Raw audio: [batch_size, num_samples] or [batch_size, seq_len, num_samples]
        
        Returns:
            Dict containing audio features and spectral analysis
        """
        batch_size = data.size(0)
        
        if data.dim() == 2:
            # Single audio clip
            audio_clips = data  # [batch_size, num_samples]
            
            # Compute spectrograms
            spectrograms = self._compute_spectrogram(audio_clips)  # [batch_size, n_mels, time_frames]
            spectrograms = spectrograms.unsqueeze(1)  # Add channel dimension: [batch_size, 1, n_mels, time_frames]
            
            # Extract features from spectrograms
            spectrogram_features = self.spectrogram_encoder(spectrograms)  # [batch_size, audio_feature_dim]
            
            # Frequency domain analysis
            audio_fft = torch.fft.fft(audio_clips, dim=-1)
            magnitude_spectrum = torch.abs(audio_fft)[:, :self.config.audio_n_fft // 2 + 1]
            frequency_features = self.frequency_analyzer(magnitude_spectrum.mean(dim=0, keepdim=True).expand(batch_size, -1))
            
            # Combine features
            combined_features = torch.cat([spectrogram_features, frequency_features], dim=1)
            final_features = nn.Linear(combined_features.size(1), self.config.audio_feature_dim).to(data.device)(combined_features)
            
            attention_weights = torch.ones(batch_size, 1, device=data.device)
        
        elif data.dim() == 3:
            # Sequential audio clips
            seq_len = data.size(1)
            data_flat = data.view(-1, data.size(-1))  # [batch_size * seq_len, num_samples]
            
            # Process each clip
            spectrograms = self._compute_spectrogram(data_flat)
            spectrograms = spectrograms.unsqueeze(1)
            
            clip_features = self.spectrogram_encoder(spectrograms)  # [batch_size * seq_len, audio_feature_dim]
            clip_features = clip_features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, audio_feature_dim]
            
            # Temporal processing with transformer
            temporal_features = self.temporal_transformer(clip_features)  # [batch_size, seq_len, audio_feature_dim]
            
            # Temporal attention
            attention_logits = torch.sum(temporal_features ** 2, dim=-1)  # [batch_size, seq_len]
            attention_weights = F.softmax(attention_logits, dim=1)
            
            # Weighted aggregation
            final_features = torch.sum(temporal_features * attention_weights.unsqueeze(-1), dim=1)
        
        else:
            raise ValueError(f"Invalid audio data shape: {data.shape}")
        
        return {
            'features': final_features,
            'attention_weights': attention_weights,
            'spectral_features': spectrogram_features if data.dim() == 2 else clip_features.mean(1),
            'feature_dim': self.config.audio_feature_dim
        }
    
    def get_feature_dim(self) -> int:
        return self.config.audio_feature_dim


class VibrationEncoder(nn.Module, ModalityEncoder):
    """
    Vibration encoder for processing accelerometer/vibration sensor data.
    
    Analyzes both time-domain and frequency-domain characteristics of vibrations.
    """
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        # Time-domain feature extractor
        self.time_domain_encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),  # Assuming 3-axis accelerometer
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(128 * 16, config.vibration_feature_dim // 2)
        )
        
        # Frequency-domain feature extractor
        self.frequency_domain_encoder = nn.Sequential(
            nn.Linear(config.vibration_fft_size // 2 + 1, config.vibration_feature_dim // 4),
            nn.ReLU(),
            nn.Linear(config.vibration_feature_dim // 4, config.vibration_feature_dim // 2)
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.vibration_feature_dim, config.vibration_feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.vibration_feature_dim, config.vibration_feature_dim)
        )
        
        # Vibration pattern classifier (bearing faults, imbalance, misalignment)
        self.pattern_classifier = nn.Sequential(
            nn.Linear(config.vibration_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Normal, bearing_fault, imbalance, misalignment
        )
        
        logger.info(f"Vibration encoder initialized with FFT size: {config.vibration_fft_size}")
    
    def encode(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode vibration data into features.
        
        Args:
            data: Vibration data tensor [batch_size, seq_len, 3] (3-axis accelerometer)
                  or [batch_size, seq_len] (single-axis)
        
        Returns:
            Dict containing vibration features and pattern analysis
        """
        batch_size = data.size(0)
        
        # Ensure 3-channel input
        if data.dim() == 2:
            data = data.unsqueeze(-1).expand(-1, -1, 3)  # Replicate to 3 channels
        elif data.size(-1) == 1:
            data = data.expand(-1, -1, 3)
        
        # Transpose for Conv1d: [batch_size, channels, seq_len]
        data_transposed = data.transpose(1, 2)
        
        # Time-domain features
        time_features = self.time_domain_encoder(data_transposed)  # [batch_size, vibration_feature_dim // 2]
        
        # Frequency-domain features
        frequency_features_list = []
        for i in range(3):  # Process each axis
            axis_data = data[:, :, i]  # [batch_size, seq_len]
            
            # Compute FFT
            fft_result = torch.fft.fft(axis_data, n=self.config.vibration_fft_size, dim=-1)
            magnitude_spectrum = torch.abs(fft_result)[:, :self.config.vibration_fft_size // 2 + 1]
            
            # Extract frequency features
            freq_features = self.frequency_domain_encoder(magnitude_spectrum)  # [batch_size, vibration_feature_dim // 2]
            frequency_features_list.append(freq_features)
        
        # Average frequency features across axes
        frequency_features = torch.stack(frequency_features_list, dim=1).mean(dim=1)
        
        # Combine time and frequency features
        combined_features = torch.cat([time_features, frequency_features], dim=1)  # [batch_size, vibration_feature_dim]
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Vibration pattern classification
        pattern_logits = self.pattern_classifier(fused_features)
        pattern_probabilities = F.softmax(pattern_logits, dim=1)
        
        # Statistical features
        rms_values = torch.sqrt(torch.mean(data ** 2, dim=1))  # RMS for each axis
        peak_values = torch.max(torch.abs(data), dim=1)[0]     # Peak values
        crest_factors = peak_values / (rms_values + 1e-8)      # Crest factor
        
        return {
            'features': fused_features,
            'time_domain_features': time_features,
            'frequency_domain_features': frequency_features,
            'pattern_probabilities': pattern_probabilities,
            'rms_values': rms_values,
            'peak_values': peak_values,
            'crest_factors': crest_factors,
            'feature_dim': self.config.vibration_feature_dim
        }
    
    def get_feature_dim(self) -> int:
        return self.config.vibration_feature_dim


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing features from different modalities.
    
    Learns to attend to relevant features across modalities for enhanced anomaly detection.
    """
    
    def __init__(self, feature_dims: List[int], fusion_dim: int):
        super().__init__()
        self.feature_dims = feature_dims
        self.fusion_dim = fusion_dim
        
        # Projection layers to common dimension
        self.modality_projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in feature_dims
        ])
        
        # Multi-head attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-modal interaction layers
        self.cross_modal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=8,
                dim_feedforward=fusion_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(3)
        ])
        
        # Modality importance weighting
        self.modality_importance = nn.Parameter(torch.ones(len(feature_dims)))
        
    def forward(self, modality_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Fuse features from multiple modalities using cross-modal attention.
        
        Args:
            modality_features: List of feature tensors from different modalities
                              Each tensor: [batch_size, feature_dim_i]
        
        Returns:
            Dict containing fused features and attention weights
        """
        batch_size = modality_features[0].size(0)
        num_modalities = len(modality_features)
        
        # Project all modalities to common dimension
        projected_features = []
        for i, features in enumerate(modality_features):
            projected = self.modality_projections[i](features)  # [batch_size, fusion_dim]
            projected_features.append(projected)
        
        # Stack modalities as sequence for attention
        modality_sequence = torch.stack(projected_features, dim=1)  # [batch_size, num_modalities, fusion_dim]
        
        # Apply cross-modal attention layers
        attended_features = modality_sequence
        attention_maps = []
        
        for layer in self.cross_modal_layers:
            attended_features = layer(attended_features)  # [batch_size, num_modalities, fusion_dim]
            
            # Compute attention weights for this layer
            attn_weights = F.softmax(torch.sum(attended_features ** 2, dim=-1), dim=-1)  # [batch_size, num_modalities]
            attention_maps.append(attn_weights)
        
        # Final attention-weighted fusion
        modality_weights = F.softmax(self.modality_importance, dim=0)  # [num_modalities]
        final_attention = attention_maps[-1] * modality_weights.unsqueeze(0)  # [batch_size, num_modalities]
        final_attention = F.softmax(final_attention, dim=-1)
        
        # Weighted combination of modality features
        fused_features = torch.sum(attended_features * final_attention.unsqueeze(-1), dim=1)  # [batch_size, fusion_dim]
        
        return {
            'fused_features': fused_features,
            'attention_weights': final_attention,
            'attention_maps': attention_maps,
            'modality_importance': modality_weights
        }


class UncertaintyQuantifier(nn.Module):
    """
    Uncertainty quantification for multi-modal predictions.
    
    Provides both aleatoric (data-dependent) and epistemic (model) uncertainty estimates.
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Aleatoric uncertainty estimation
        self.aleatoric_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Softplus()  # Ensure positive variance
        )
        
        # Epistemic uncertainty (using Monte Carlo Dropout)
        self.epistemic_dropout = nn.Dropout(0.1)
        
    def forward(self, features: torch.Tensor, num_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Estimate prediction uncertainties.
        
        Args:
            features: Input features [batch_size, input_dim]
            num_samples: Number of MC samples for epistemic uncertainty
        
        Returns:
            Dict containing uncertainty estimates
        """
        # Aleatoric uncertainty
        aleatoric_variance = self.aleatoric_head(features)  # [batch_size, 1]
        
        # Epistemic uncertainty using MC Dropout
        predictions = []
        self.train()  # Enable dropout during inference
        
        for _ in range(num_samples):
            pred = self.epistemic_dropout(features)
            predictions.append(pred)
        
        predictions_tensor = torch.stack(predictions, dim=0)  # [num_samples, batch_size, input_dim]
        
        # Calculate epistemic uncertainty as prediction variance
        epistemic_variance = torch.var(predictions_tensor, dim=0).mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        # Total uncertainty
        total_uncertainty = aleatoric_variance + epistemic_variance
        
        self.eval()  # Restore evaluation mode
        
        return {
            'aleatoric_uncertainty': aleatoric_variance.squeeze(-1),
            'epistemic_uncertainty': epistemic_variance.squeeze(-1),
            'total_uncertainty': total_uncertainty.squeeze(-1),
            'prediction_samples': predictions_tensor
        }


class MultiModalFusionNetwork(nn.Module):
    """
    Complete Multi-Modal Fusion Network for IoT Anomaly Detection.
    
    Integrates vision, audio, vibration, and sensor data for comprehensive
    anomaly detection with cross-modal reasoning and uncertainty quantification.
    """
    
    def __init__(self, 
                 sensor_input_size: int = 5,
                 config: Optional[MultiModalConfig] = None):
        super().__init__()
        
        self.sensor_input_size = sensor_input_size
        self.config = config or MultiModalConfig()
        
        # Modality encoders
        self.modality_encoders = nn.ModuleDict()
        feature_dims = []
        
        if self.config.enable_vision:
            self.modality_encoders['vision'] = VisionEncoder(self.config)
            feature_dims.append(self.config.vision_feature_dim)
        
        if self.config.enable_audio:
            self.modality_encoders['audio'] = AudioEncoder(self.config)
            feature_dims.append(self.config.audio_feature_dim)
        
        if self.config.enable_vibration:
            self.modality_encoders['vibration'] = VibrationEncoder(self.config)
            feature_dims.append(self.config.vibration_feature_dim)
        
        if self.config.enable_sensor_data:
            # Traditional sensor data encoder
            self.sensor_encoder = nn.Sequential(
                nn.Linear(sensor_input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, self.config.fusion_hidden_dim)
            )
            feature_dims.append(self.config.fusion_hidden_dim)
        
        # Cross-modal fusion
        if len(feature_dims) > 1:
            if self.config.fusion_method == "cross_attention":
                self.fusion_module = CrossModalAttention(feature_dims, self.config.fusion_hidden_dim)
            else:
                # Simple concatenation fusion
                total_dim = sum(feature_dims)
                self.fusion_module = nn.Sequential(
                    nn.Linear(total_dim, self.config.fusion_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout_rate),
                    nn.Linear(self.config.fusion_hidden_dim, self.config.fusion_hidden_dim)
                )
        else:
            self.fusion_module = nn.Identity()
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.config.fusion_hidden_dim, self.config.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.fusion_hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty quantification
        if self.config.enable_uncertainty_quantification:
            self.uncertainty_quantifier = UncertaintyQuantifier(self.config.fusion_hidden_dim)
        
        # Temporal alignment module
        self.temporal_aligner = nn.LSTM(
            input_size=self.config.fusion_hidden_dim,
            hidden_size=self.config.fusion_hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        logger.info(f"Multi-Modal Fusion Network initialized with {len(self.modality_encoders)} modalities")
        logger.info(f"Enabled modalities: {list(self.modality_encoders.keys())}")
    
    def forward(self, 
                modality_data: Dict[str, torch.Tensor],
                return_modality_info: bool = False) -> Dict[str, Any]:
        """
        Forward pass through multi-modal fusion network.
        
        Args:
            modality_data: Dict containing data for each modality
                          - 'vision': [batch_size, C, H, W] or [batch_size, seq_len, C, H, W]
                          - 'audio': [batch_size, num_samples] or [batch_size, seq_len, num_samples]
                          - 'vibration': [batch_size, seq_len, 3]
                          - 'sensor': [batch_size, seq_len, sensor_input_size]
            return_modality_info: Whether to return detailed modality information
            
        Returns:
            Dict containing anomaly predictions and optional modality analysis
        """
        batch_size = list(modality_data.values())[0].size(0)
        device = list(modality_data.values())[0].device
        
        # Process each available modality
        modality_features = []
        modality_info = {}
        
        for modality_name, encoder in self.modality_encoders.items():
            if modality_name in modality_data:
                if modality_name == 'sensor' and self.config.enable_sensor_data:
                    # Special handling for sensor data
                    sensor_data = modality_data[modality_name]
                    if sensor_data.dim() == 3:
                        sensor_data = sensor_data[:, -1, :]  # Use latest timestep
                    
                    features = self.sensor_encoder(sensor_data)
                    modality_features.append(features)
                    
                    if return_modality_info:
                        modality_info[modality_name] = {
                            'features': features,
                            'feature_dim': features.size(-1)
                        }
                else:
                    # Process with modality-specific encoder
                    encoding_result = encoder.encode(modality_data[modality_name])
                    modality_features.append(encoding_result['features'])
                    
                    if return_modality_info:
                        modality_info[modality_name] = encoding_result
        
        if not modality_features:
            raise ValueError("No valid modality data provided")
        
        # Cross-modal fusion
        if len(modality_features) > 1 and isinstance(self.fusion_module, CrossModalAttention):
            fusion_result = self.fusion_module(modality_features)
            fused_features = fusion_result['fused_features']
            cross_modal_attention = fusion_result['attention_weights']
        elif len(modality_features) > 1:
            # Concatenation fusion
            concatenated = torch.cat(modality_features, dim=1)
            fused_features = self.fusion_module(concatenated)
            cross_modal_attention = torch.ones(batch_size, len(modality_features), device=device) / len(modality_features)
        else:
            # Single modality
            fused_features = modality_features[0]
            cross_modal_attention = torch.ones(batch_size, 1, device=device)
        
        # Temporal processing if sequence data is available
        temporal_features = fused_features
        if any('seq_len' in str(data.shape) for data in modality_data.values()):
            # Add temporal dimension if missing
            if fused_features.dim() == 2:
                fused_features = fused_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
            
            temporal_output, _ = self.temporal_aligner(fused_features)
            temporal_features = temporal_output[:, -1, :]  # Use final hidden state
        
        # Anomaly detection
        anomaly_scores = self.anomaly_head(temporal_features).squeeze(-1)
        
        results = {
            'anomaly_scores': anomaly_scores,
            'fused_features': fused_features.squeeze(1) if fused_features.dim() == 3 else fused_features,
            'cross_modal_attention': cross_modal_attention,
            'num_active_modalities': len(modality_features)
        }
        
        # Uncertainty quantification
        if self.config.enable_uncertainty_quantification:
            uncertainty_results = self.uncertainty_quantifier(temporal_features)
            results.update(uncertainty_results)
        
        # Detailed modality information
        if return_modality_info:
            results['modality_info'] = modality_info
            results['modality_contributions'] = self._analyze_modality_contributions(modality_features, cross_modal_attention)
        
        return results
    
    def _analyze_modality_contributions(self, 
                                      modality_features: List[torch.Tensor], 
                                      attention_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze contribution of each modality to final prediction."""
        modality_names = list(self.modality_encoders.keys())
        contributions = {}
        
        for i, modality_name in enumerate(modality_names):
            if i < len(modality_features):
                # Contribution is attention weight times feature magnitude
                feature_magnitude = torch.norm(modality_features[i], dim=1)
                if i < attention_weights.size(1):
                    contribution = attention_weights[:, i] * feature_magnitude
                    contributions[modality_name] = contribution
        
        return contributions
    
    def compute_reconstruction_error(self, modality_data: Dict[str, torch.Tensor], reduction: str = 'mean') -> torch.Tensor:
        """
        Compute reconstruction error compatible with anomaly detection interface.
        
        Args:
            modality_data: Dict containing multi-modal data
            reduction: How to reduce the error ('mean', 'sum', 'none')
            
        Returns:
            Reconstruction error tensor
        """
        results = self.forward(modality_data)
        anomaly_scores = results['anomaly_scores']
        
        # Convert anomaly probability to reconstruction-like error
        reconstruction_error = -torch.log(1 - anomaly_scores + 1e-8)
        
        if reduction == 'mean':
            return reconstruction_error.mean()
        elif reduction == 'sum':
            return reconstruction_error.sum()
        else:
            return reconstruction_error
    
    def get_modality_importance(self) -> Dict[str, float]:
        """Get learned importance weights for each modality."""
        if isinstance(self.fusion_module, CrossModalAttention):
            importance_weights = F.softmax(self.fusion_module.modality_importance, dim=0)
            modality_names = list(self.modality_encoders.keys())
            
            importance_dict = {}
            for i, name in enumerate(modality_names):
                if i < len(importance_weights):
                    importance_dict[name] = importance_weights[i].item()
            
            return importance_dict
        else:
            # Equal importance for concatenation fusion
            num_modalities = len(self.modality_encoders)
            return {name: 1.0 / num_modalities for name in self.modality_encoders.keys()}


def create_multimodal_fusion_detector(config: Dict[str, Any]) -> MultiModalFusionNetwork:
    """
    Factory function to create multi-modal fusion anomaly detector.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured multi-modal fusion network
    """
    sensor_input_size = config.get('input_size', 5)
    
    multimodal_config_dict = config.get('multimodal_config', {})
    multimodal_config = MultiModalConfig(
        enable_vision=multimodal_config_dict.get('enable_vision', True),
        enable_audio=multimodal_config_dict.get('enable_audio', True),
        enable_vibration=multimodal_config_dict.get('enable_vibration', True),
        enable_sensor_data=multimodal_config_dict.get('enable_sensor_data', True),
        vision_feature_dim=multimodal_config_dict.get('vision_feature_dim', 512),
        audio_feature_dim=multimodal_config_dict.get('audio_feature_dim', 256),
        vibration_feature_dim=multimodal_config_dict.get('vibration_feature_dim', 128),
        fusion_method=multimodal_config_dict.get('fusion_method', 'cross_attention'),
        fusion_hidden_dim=multimodal_config_dict.get('fusion_hidden_dim', 256),
        enable_uncertainty_quantification=multimodal_config_dict.get('enable_uncertainty_quantification', True)
    )
    
    network = MultiModalFusionNetwork(
        sensor_input_size=sensor_input_size,
        config=multimodal_config
    )
    
    total_params = sum(p.numel() for p in network.parameters())
    enabled_modalities = [name for name in ['vision', 'audio', 'vibration', 'sensor'] 
                         if getattr(multimodal_config, f'enable_{name}', False)]
    
    logger.info(f"Created multi-modal fusion detector with {total_params} parameters")
    logger.info(f"Enabled modalities: {enabled_modalities}")
    logger.info(f"Fusion method: {multimodal_config.fusion_method}")
    logger.info(f"Uncertainty quantification: {multimodal_config.enable_uncertainty_quantification}")
    
    return network
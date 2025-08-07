"""
Data models and enumerations for sentiment analysis.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime


class ModelType(Enum):
    """Available sentiment analysis models."""
    VADER = "vader"
    TEXTBLOB = "textblob"
    TRANSFORMERS = "transformers"
    LSTM = "lstm"


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    text: str
    label: SentimentLabel
    confidence: float
    scores: Dict[str, float]
    model_type: ModelType
    processing_time: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "label": self.label.value,
            "confidence": self.confidence,
            "scores": self.scores,
            "model_type": self.model_type.value,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


@dataclass
class AnalysisConfig:
    """Configuration for sentiment analysis."""
    model_type: ModelType = ModelType.VADER
    confidence_threshold: float = 0.6
    batch_size: int = 32
    max_length: int = 512
    cache_results: bool = True
    enable_preprocessing: bool = True
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (
            0.0 <= self.confidence_threshold <= 1.0 and
            self.batch_size > 0 and
            self.max_length > 0
        )
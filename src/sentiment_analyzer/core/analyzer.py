"""
Core sentiment analysis engine with multiple model support.
"""
import time
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import re

from .models import SentimentResult, SentimentLabel, ModelType, AnalysisConfig

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove mentions and hashtags (optional - keep for context)
        # text = self.mention_pattern.sub(' ', text)
        # text = self.hashtag_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip and basic cleanup
        text = text.strip()
        
        return text


class VaderAnalyzer:
    """VADER sentiment analysis implementation."""
    
    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.available = True
        except ImportError:
            logger.warning("VADER not available. Install vaderSentiment package.")
            self.available = False
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        if not self.available:
            raise RuntimeError("VADER analyzer not available")
            
        scores = self.analyzer.polarity_scores(text)
        return scores


class TextBlobAnalyzer:
    """TextBlob sentiment analysis implementation."""
    
    def __init__(self):
        try:
            from textblob import TextBlob
            self.TextBlob = TextBlob
            self.available = True
        except ImportError:
            logger.warning("TextBlob not available. Install textblob package.")
            self.available = False
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob."""
        if not self.available:
            raise RuntimeError("TextBlob analyzer not available")
            
        blob = self.TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Convert to VADER-like format
        return {
            "positive": max(0, polarity),
            "negative": max(0, -polarity),
            "neutral": 1 - abs(polarity),
            "compound": polarity,
            "subjectivity": subjectivity
        }


class SentimentAnalyzer:
    """
    Main sentiment analysis engine with multiple model support.
    
    Supports VADER, TextBlob, and extensible architecture for additional models.
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.preprocessor = TextPreprocessor()
        
        # Initialize analyzers
        self.analyzers = {
            ModelType.VADER: VaderAnalyzer(),
            ModelType.TEXTBLOB: TextBlobAnalyzer(),
        }
        
        # Result cache
        self._cache = {} if self.config.cache_results else None
        
        logger.info(f"Initialized SentimentAnalyzer with model: {self.config.model_type}")
    
    def _get_cache_key(self, text: str, model_type: ModelType) -> str:
        """Generate cache key for text and model combination."""
        return f"{model_type.value}:{hash(text)}"
    
    def _classify_sentiment(self, scores: Dict[str, float], model_type: ModelType) -> tuple[SentimentLabel, float]:
        """Classify sentiment based on scores and model type."""
        if model_type == ModelType.VADER:
            compound = scores.get("compound", 0)
            if compound >= 0.05:
                return SentimentLabel.POSITIVE, scores.get("pos", 0)
            elif compound <= -0.05:
                return SentimentLabel.NEGATIVE, scores.get("neg", 0)
            else:
                return SentimentLabel.NEUTRAL, scores.get("neu", 0)
        
        elif model_type == ModelType.TEXTBLOB:
            compound = scores.get("compound", 0)
            if compound > 0.1:
                return SentimentLabel.POSITIVE, abs(compound)
            elif compound < -0.1:
                return SentimentLabel.NEGATIVE, abs(compound)
            else:
                return SentimentLabel.NEUTRAL, scores.get("neutral", 0)
        
        # Default classification
        return SentimentLabel.NEUTRAL, 0.5
    
    def analyze_text(self, text: str, model_type: Optional[ModelType] = None) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            model_type: Model to use (defaults to config model)
            
        Returns:
            SentimentResult with analysis results
        """
        start_time = time.time()
        model_type = model_type or self.config.model_type
        
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        if len(text) > self.config.max_length:
            text = text[:self.config.max_length]
            logger.warning(f"Text truncated to {self.config.max_length} characters")
        
        # Check cache
        cache_key = None
        if self._cache is not None:
            cache_key = self._get_cache_key(text, model_type)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached_result
        
        # Preprocess text
        if self.config.enable_preprocessing:
            processed_text = self.preprocessor.clean_text(text)
        else:
            processed_text = text
        
        # Get analyzer
        analyzer = self.analyzers.get(model_type)
        if not analyzer or not analyzer.available:
            raise RuntimeError(f"Analyzer {model_type.value} not available")
        
        try:
            # Perform analysis
            scores = analyzer.analyze(processed_text)
            label, confidence = self._classify_sentiment(scores, model_type)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = SentimentResult(
                text=text,
                label=label,
                confidence=confidence,
                scores=scores,
                model_type=model_type,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={
                    "processed_text": processed_text,
                    "text_length": len(text),
                    "preprocessed": self.config.enable_preprocessing
                }
            )
            
            # Cache result
            if cache_key and self._cache is not None:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for text: {text[:50]}... Error: {e}")
            raise RuntimeError(f"Sentiment analysis failed: {e}")
    
    def analyze_batch(self, texts: List[str], model_type: Optional[ModelType] = None) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            model_type: Model to use (defaults to config model)
            
        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []
        
        results = []
        for text in texts:
            try:
                result = self.analyze_text(text, model_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze text: {text[:50]}... Error: {e}")
                # Create error result
                error_result = SentimentResult(
                    text=text,
                    label=SentimentLabel.NEUTRAL,
                    confidence=0.0,
                    scores={},
                    model_type=model_type or self.config.model_type,
                    processing_time=0.0,
                    timestamp=datetime.now(),
                    metadata={"error": str(e)}
                )
                results.append(error_result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "available_models": [model.value for model in ModelType],
            "current_model": self.config.model_type.value,
            "model_status": {
                model.value: analyzer.available 
                for model, analyzer in self.analyzers.items()
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Analysis cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if self._cache is None:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cache_size": len(self._cache),
            "cache_hits": getattr(self, '_cache_hits', 0),
            "cache_misses": getattr(self, '_cache_misses', 0)
        }
"""
Sentiment Analyzer Pro - Advanced sentiment analysis with ML capabilities.

A production-ready sentiment analysis system with real-time processing,
multiple model support, and enterprise-grade monitoring.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"

from .core.analyzer import SentimentAnalyzer
from .core.models import ModelType, SentimentResult
from .api.server import create_app

__all__ = [
    "SentimentAnalyzer",
    "ModelType", 
    "SentimentResult",
    "create_app"
]
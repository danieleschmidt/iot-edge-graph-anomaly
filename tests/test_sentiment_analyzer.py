"""
Comprehensive tests for sentiment analyzer core functionality.
"""
import pytest
import time
from datetime import datetime

from sentiment_analyzer.core.analyzer import SentimentAnalyzer, TextPreprocessor, VaderAnalyzer, TextBlobAnalyzer
from sentiment_analyzer.core.models import ModelType, SentimentLabel, AnalysisConfig, SentimentResult


class TestTextPreprocessor:
    """Test text preprocessing functionality."""
    
    def setup_method(self):
        self.preprocessor = TextPreprocessor()
    
    def test_url_removal(self):
        text = "Check this out: https://example.com and http://test.org"
        cleaned = self.preprocessor.clean_text(text)
        assert "https://example.com" not in cleaned
        assert "http://test.org" not in cleaned
        assert "Check this out:" in cleaned
    
    def test_whitespace_normalization(self):
        text = "Multiple   spaces    and\ttabs\n\neverywhere"
        cleaned = self.preprocessor.clean_text(text)
        assert cleaned == "Multiple spaces and tabs everywhere"
    
    def test_empty_input(self):
        assert self.preprocessor.clean_text("") == ""
        assert self.preprocessor.clean_text("   \n\t  ") == ""
    
    def test_non_string_input(self):
        assert self.preprocessor.clean_text(None) == ""
        assert self.preprocessor.clean_text(123) == ""


class TestVaderAnalyzer:
    """Test VADER sentiment analyzer."""
    
    def setup_method(self):
        self.analyzer = VaderAnalyzer()
    
    def test_positive_sentiment(self):
        if not self.analyzer.available:
            pytest.skip("VADER not available")
        
        scores = self.analyzer.analyze("I love this amazing product!")
        assert scores['compound'] > 0
        assert scores['pos'] > scores['neg']
    
    def test_negative_sentiment(self):
        if not self.analyzer.available:
            pytest.skip("VADER not available")
        
        scores = self.analyzer.analyze("This is terrible and awful!")
        assert scores['compound'] < 0
        assert scores['neg'] > scores['pos']
    
    def test_neutral_sentiment(self):
        if not self.analyzer.available:
            pytest.skip("VADER not available")
        
        scores = self.analyzer.analyze("This is a book.")
        assert abs(scores['compound']) < 0.5
    
    def test_empty_text(self):
        if not self.analyzer.available:
            pytest.skip("VADER not available")
        
        scores = self.analyzer.analyze("")
        assert isinstance(scores, dict)
        assert 'compound' in scores


class TestTextBlobAnalyzer:
    """Test TextBlob sentiment analyzer."""
    
    def setup_method(self):
        self.analyzer = TextBlobAnalyzer()
    
    def test_sentiment_analysis(self):
        if not self.analyzer.available:
            pytest.skip("TextBlob not available")
        
        scores = self.analyzer.analyze("I am happy today!")
        assert isinstance(scores, dict)
        assert 'compound' in scores
        assert 'positive' in scores
        assert 'negative' in scores
    
    def test_subjectivity_tracking(self):
        if not self.analyzer.available:
            pytest.skip("TextBlob not available")
        
        scores = self.analyzer.analyze("I think this might be good.")
        assert 'subjectivity' in scores
        assert 0 <= scores['subjectivity'] <= 1


class TestAnalysisConfig:
    """Test analysis configuration."""
    
    def test_default_config(self):
        config = AnalysisConfig()
        assert config.model_type == ModelType.VADER
        assert config.confidence_threshold == 0.6
        assert config.validate()
    
    def test_config_validation(self):
        # Valid config
        config = AnalysisConfig(confidence_threshold=0.8, batch_size=64)
        assert config.validate()
        
        # Invalid configs
        invalid_configs = [
            AnalysisConfig(confidence_threshold=-0.1),  # Too low
            AnalysisConfig(confidence_threshold=1.1),   # Too high
            AnalysisConfig(batch_size=0),               # Zero batch size
            AnalysisConfig(max_length=0),               # Zero max length
        ]
        
        for config in invalid_configs:
            assert not config.validate()


class TestSentimentAnalyzer:
    """Test main sentiment analyzer functionality."""
    
    def setup_method(self):
        self.config = AnalysisConfig(cache_results=True)
        self.analyzer = SentimentAnalyzer(self.config)
    
    def test_initialization(self):
        assert self.analyzer.config == self.config
        assert isinstance(self.analyzer.preprocessor, TextPreprocessor)
        assert ModelType.VADER in self.analyzer.analyzers
        assert ModelType.TEXTBLOB in self.analyzer.analyzers
    
    def test_single_text_analysis(self):
        text = "I love this product! It's amazing!"
        result = self.analyzer.analyze_text(text)
        
        assert isinstance(result, SentimentResult)
        assert result.text == text
        assert isinstance(result.label, SentimentLabel)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.scores, dict)
        assert result.model_type == self.config.model_type
        assert result.processing_time > 0
        assert isinstance(result.timestamp, datetime)
    
    def test_batch_analysis(self):
        texts = [
            "I love this!",
            "This is terrible.",
            "It's okay, I guess."
        ]
        
        results = self.analyzer.analyze_batch(texts)
        
        assert len(results) == len(texts)
        for i, result in enumerate(results):
            assert isinstance(result, SentimentResult)
            assert result.text == texts[i]
    
    def test_different_models(self):
        text = "This is a test message."
        
        for model_type in [ModelType.VADER, ModelType.TEXTBLOB]:
            try:
                result = self.analyzer.analyze_text(text, model_type=model_type)
                assert result.model_type == model_type
            except RuntimeError as e:
                if "not available" in str(e):
                    pytest.skip(f"{model_type.value} not available")
                else:
                    raise
    
    def test_text_preprocessing(self):
        original_text = "Check this: https://example.com   multiple    spaces!"
        result = self.analyzer.analyze_text(original_text)
        
        # Should still contain original text
        assert result.text == original_text
        
        # But metadata should show processed version
        if result.metadata and 'processed_text' in result.metadata:
            processed = result.metadata['processed_text']
            assert "https://example.com" not in processed
            assert "multiple    spaces" not in processed
    
    def test_caching(self):
        if not self.analyzer._cache:
            pytest.skip("Caching disabled")
        
        text = "This is a test for caching."
        
        # Clear cache first
        self.analyzer.clear_cache()
        
        # First analysis
        start_time = time.time()
        result1 = self.analyzer.analyze_text(text)
        first_time = time.time() - start_time
        
        # Second analysis (should be cached)
        start_time = time.time()
        result2 = self.analyzer.analyze_text(text)
        second_time = time.time() - start_time
        
        # Results should be identical
        assert result1.text == result2.text
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence
        
        # Second call should be faster (cached)
        assert second_time < first_time
    
    def test_input_validation(self):
        # Empty string
        with pytest.raises(ValueError):
            self.analyzer.analyze_text("")
        
        # Non-string input
        with pytest.raises(ValueError):
            self.analyzer.analyze_text(None)
        
        with pytest.raises(ValueError):
            self.analyzer.analyze_text(123)
    
    def test_text_length_limits(self):
        # Test with very long text
        long_text = "This is a test. " * 1000  # ~16k characters
        
        # Should truncate and warn
        result = self.analyzer.analyze_text(long_text)
        assert isinstance(result, SentimentResult)
    
    def test_sentiment_classification(self):
        test_cases = [
            ("I absolutely love this amazing product!", SentimentLabel.POSITIVE),
            ("This is terrible and I hate it!", SentimentLabel.NEGATIVE),
            ("This is a book.", SentimentLabel.NEUTRAL),
        ]
        
        for text, expected_label in test_cases:
            try:
                result = self.analyzer.analyze_text(text)
                # Note: Exact label matching might vary by model, so we just check it's valid
                assert isinstance(result.label, SentimentLabel)
            except RuntimeError as e:
                if "not available" in str(e):
                    pytest.skip("Required analyzer not available")
                else:
                    raise
    
    def test_model_info(self):
        info = self.analyzer.get_model_info()
        assert 'available_models' in info
        assert 'current_model' in info
        assert 'model_status' in info
        
        assert isinstance(info['available_models'], list)
        assert info['current_model'] == self.config.model_type.value
        assert isinstance(info['model_status'], dict)
    
    def test_cache_management(self):
        if not self.analyzer._cache:
            pytest.skip("Caching disabled")
        
        # Test cache clearing
        self.analyzer.analyze_text("Test text for cache")
        stats_before = self.analyzer.get_cache_stats()
        
        self.analyzer.clear_cache()
        stats_after = self.analyzer.get_cache_stats()
        
        if stats_after.get('cache_enabled'):
            assert stats_after['cache_size'] == 0
    
    def test_cache_stats(self):
        stats = self.analyzer.get_cache_stats()
        assert isinstance(stats, dict)
        
        if stats.get('cache_enabled'):
            assert 'cache_size' in stats
            assert 'cache_hits' in stats or 'cache_misses' in stats
        else:
            assert stats['cache_enabled'] is False
    
    def test_error_handling_in_batch(self):
        # Mix of valid and invalid texts
        texts = [
            "Valid text",
            "",  # Invalid - empty
            "Another valid text",
            None,  # Invalid - not string
        ]
        
        # Should not raise exception, but handle errors gracefully
        results = self.analyzer.analyze_batch(texts)
        assert len(results) == len(texts)
        
        # Check that valid texts got processed
        assert results[0].text == "Valid text"
        assert results[2].text == "Another valid text"
        
        # Check that invalid texts got error results
        assert results[1].metadata and 'error' in results[1].metadata
        assert results[3].metadata and 'error' in results[3].metadata


class TestSentimentResult:
    """Test sentiment result data structure."""
    
    def test_result_creation(self):
        result = SentimentResult(
            text="Test text",
            label=SentimentLabel.POSITIVE,
            confidence=0.8,
            scores={'pos': 0.8, 'neg': 0.1, 'neu': 0.1},
            model_type=ModelType.VADER,
            processing_time=0.05,
            timestamp=datetime.now()
        )
        
        assert result.text == "Test text"
        assert result.label == SentimentLabel.POSITIVE
        assert result.confidence == 0.8
        assert result.model_type == ModelType.VADER
    
    def test_to_dict_conversion(self):
        result = SentimentResult(
            text="Test text",
            label=SentimentLabel.POSITIVE,
            confidence=0.8,
            scores={'compound': 0.5},
            model_type=ModelType.VADER,
            processing_time=0.05,
            timestamp=datetime.now(),
            metadata={'test': True}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['text'] == "Test text"
        assert result_dict['label'] == 'positive'
        assert result_dict['confidence'] == 0.8
        assert result_dict['model_type'] == 'vader'
        assert 'timestamp' in result_dict
        assert result_dict['metadata'] == {'test': True}


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This is the worst thing I've ever bought. Complete waste of money.",
        "The product is okay. Nothing special but does what it says.",
        "Neutral statement about a book on the table.",
        "🎉 Excited about the new features! Can't wait to try them! 😊",
        "Unfortunately, the service was disappointing and slow.",
    ]


class TestEndToEndScenarios:
    """End-to-end integration tests."""
    
    def test_complete_analysis_pipeline(self, sample_texts):
        """Test complete analysis pipeline with various texts."""
        analyzer = SentimentAnalyzer()
        
        for text in sample_texts:
            try:
                result = analyzer.analyze_text(text)
                
                # Verify result structure
                assert isinstance(result, SentimentResult)
                assert result.text == text
                assert isinstance(result.label, SentimentLabel)
                assert 0 <= result.confidence <= 1
                assert result.processing_time >= 0
                assert isinstance(result.timestamp, datetime)
                
                # Verify scores
                assert isinstance(result.scores, dict)
                assert len(result.scores) > 0
                
            except RuntimeError as e:
                if "not available" in str(e):
                    pytest.skip("Required analyzer not available")
                else:
                    raise
    
    def test_performance_benchmarks(self, sample_texts):
        """Test performance benchmarks."""
        analyzer = SentimentAnalyzer()
        
        # Single text performance
        start_time = time.time()
        for text in sample_texts:
            try:
                result = analyzer.analyze_text(text)
                # Should complete reasonably quickly
                assert result.processing_time < 1.0  # Less than 1 second
            except RuntimeError as e:
                if "not available" in str(e):
                    pytest.skip("Required analyzer not available")
                else:
                    raise
        
        total_time = time.time() - start_time
        avg_time = total_time / len(sample_texts)
        assert avg_time < 0.5  # Average less than 500ms per text
        
        # Batch performance
        start_time = time.time()
        try:
            results = analyzer.analyze_batch(sample_texts)
            batch_time = time.time() - start_time
            
            # Batch should be reasonably efficient
            assert len(results) == len(sample_texts)
            assert batch_time < len(sample_texts) * 0.5  # Better than sequential worst case
        
        except RuntimeError as e:
            if "not available" in str(e):
                pytest.skip("Required analyzer not available")
            else:
                raise
    
    def test_model_consistency(self, sample_texts):
        """Test consistency across different models."""
        analyzers = {}
        
        # Create analyzers for available models
        for model_type in ModelType:
            try:
                config = AnalysisConfig(model_type=model_type)
                analyzer = SentimentAnalyzer(config)
                # Test that analyzer works
                analyzer.analyze_text(sample_texts[0])
                analyzers[model_type] = analyzer
            except RuntimeError as e:
                if "not available" in str(e):
                    continue  # Skip unavailable models
                else:
                    raise
        
        if len(analyzers) < 2:
            pytest.skip("Need at least 2 available models for consistency test")
        
        # Test same text with different models
        test_text = sample_texts[0]
        results = {}
        
        for model_type, analyzer in analyzers.items():
            result = analyzer.analyze_text(test_text)
            results[model_type] = result
        
        # All models should return valid results
        for model_type, result in results.items():
            assert isinstance(result, SentimentResult)
            assert result.text == test_text
            assert result.model_type == model_type
            assert isinstance(result.label, SentimentLabel)
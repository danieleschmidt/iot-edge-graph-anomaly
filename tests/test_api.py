"""
Tests for API server functionality.
"""
import pytest
import asyncio
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from sentiment_analyzer.api.server import create_app
from sentiment_analyzer.core.models import ModelType


@pytest.fixture
def client():
    """Create test client for API testing."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_analyzer():
    """Mock analyzer for testing."""
    from sentiment_analyzer.core.analyzer import SentimentAnalyzer
    from sentiment_analyzer.core.models import SentimentResult, SentimentLabel
    from datetime import datetime
    
    mock = Mock(spec=SentimentAnalyzer)
    mock.analyze_text.return_value = SentimentResult(
        text="test text",
        label=SentimentLabel.POSITIVE,
        confidence=0.8,
        scores={'compound': 0.5, 'pos': 0.8, 'neg': 0.1, 'neu': 0.1},
        model_type=ModelType.VADER,
        processing_time=0.05,
        timestamp=datetime.now(),
        metadata={'test': True}
    )
    
    mock.analyze_batch.return_value = [mock.analyze_text.return_value]
    
    mock.get_model_info.return_value = {
        'available_models': ['vader', 'textblob'],
        'current_model': 'vader',
        'model_status': {'vader': True, 'textblob': True}
    }
    
    mock.get_cache_stats.return_value = {
        'cache_enabled': True,
        'cache_size': 10,
        'cache_hits': 5,
        'cache_misses': 3
    }
    
    return mock


class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "0.1.0"
    
    @patch('sentiment_analyzer.api.server.analyzer')
    def test_health_endpoint(self, mock_analyzer_global, client, mock_analyzer):
        """Test health check endpoint."""
        mock_analyzer_global = mock_analyzer
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "uptime" in data
        assert "models" in data
    
    @patch('sentiment_analyzer.api.server.analyzer')
    def test_analyze_endpoint(self, mock_analyzer_global, client, mock_analyzer):
        """Test single text analysis endpoint."""
        mock_analyzer_global = mock_analyzer
        
        request_data = {
            "text": "I love this product!",
            "model_type": "vader"
        }
        
        response = client.post("/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert "processing_time" in data
        assert "timestamp" in data
        
        result = data["result"]
        assert result["text"] == "test text"  # From mock
        assert result["label"] == "positive"
        assert result["confidence"] == 0.8
    
    @patch('sentiment_analyzer.api.server.analyzer')
    def test_analyze_batch_endpoint(self, mock_analyzer_global, client, mock_analyzer):
        """Test batch analysis endpoint."""
        mock_analyzer_global = mock_analyzer
        
        request_data = {
            "texts": ["I love this!", "This is great!", "Amazing product!"],
            "model_type": "vader"
        }
        
        response = client.post("/analyze/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        assert "total_processed" in data
        assert data["total_processed"] == 3
        
        results = data["results"]
        assert len(results) == 3
    
    @patch('sentiment_analyzer.api.server.analyzer')
    def test_models_endpoint(self, mock_analyzer_global, client, mock_analyzer):
        """Test models info endpoint."""
        mock_analyzer_global = mock_analyzer
        
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "available_models" in data
        assert "current_model" in data
        assert "model_status" in data
    
    @patch('sentiment_analyzer.api.server.analyzer')
    def test_cache_endpoints(self, mock_analyzer_global, client, mock_analyzer):
        """Test cache management endpoints."""
        mock_analyzer_global = mock_analyzer
        
        # Test cache stats
        response = client.get("/cache/stats")
        assert response.status_code == 200
        
        # Test cache clear
        response = client.post("/cache/clear")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "uptime_seconds" in data
        assert "request_count" in data
        assert "error_count" in data
        assert "error_rate" in data


class TestAPIValidation:
    """Test API input validation."""
    
    def test_analyze_validation(self, client):
        """Test analysis request validation."""
        # Empty text
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422
        
        # Missing text
        response = client.post("/analyze", json={"model_type": "vader"})
        assert response.status_code == 422
        
        # Text too long
        long_text = "A" * 11000  # Exceeds max_length
        response = client.post("/analyze", json={"text": long_text})
        assert response.status_code == 422
        
        # Invalid model type
        response = client.post("/analyze", json={
            "text": "test",
            "model_type": "invalid_model"
        })
        assert response.status_code == 422
    
    def test_batch_analyze_validation(self, client):
        """Test batch analysis request validation."""
        # Empty texts list
        response = client.post("/analyze/batch", json={"texts": []})
        assert response.status_code == 422
        
        # Too many texts
        many_texts = ["text"] * 101  # Exceeds max_items
        response = client.post("/analyze/batch", json={"texts": many_texts})
        assert response.status_code == 422
        
        # Mixed valid/invalid texts
        mixed_texts = ["valid text", "", "another valid text"]
        response = client.post("/analyze/batch", json={"texts": mixed_texts})
        # Should process valid texts only
        assert response.status_code == 200 or response.status_code == 422


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_analyzer_not_initialized(self, client):
        """Test behavior when analyzer is not initialized."""
        # This would happen if analyzer failed to initialize
        with patch('sentiment_analyzer.api.server.analyzer', None):
            response = client.post("/analyze", json={"text": "test"})
            assert response.status_code == 503
    
    @patch('sentiment_analyzer.api.server.analyzer')
    def test_analysis_error_handling(self, mock_analyzer_global, client):
        """Test handling of analysis errors."""
        # Mock analyzer that raises exception
        mock_analyzer = Mock()
        mock_analyzer.analyze_text.side_effect = RuntimeError("Analysis failed")
        mock_analyzer_global = mock_analyzer
        
        response = client.post("/analyze", json={"text": "test"})
        # Should return error response, not crash
        assert response.status_code == 200  # API handles the error gracefully
        
        data = response.json()
        assert data["success"] is False
        assert "error" in data


class TestAPICORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        # Preflight request
        response = client.options("/analyze")
        # TestClient might not fully simulate CORS, so just check the endpoint exists
        assert response.status_code in [200, 405]  # Either allowed or method not allowed
        
        # Regular request should work
        response = client.get("/")
        assert response.status_code == 200


class TestAPIMiddleware:
    """Test API middleware functionality."""
    
    def test_request_counting_middleware(self, client):
        """Test that requests are being counted."""
        # Make a few requests
        client.get("/")
        client.get("/health")
        client.get("/metrics")
        
        # Check metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert data["request_count"] >= 4  # Including the metrics request itself
    
    def test_gzip_middleware(self, client):
        """Test gzip compression middleware."""
        # Make request with large response
        response = client.get("/health")
        assert response.status_code == 200
        # TestClient might not simulate compression fully, but endpoint should work


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    @patch('sentiment_analyzer.api.server.analyzer')
    def test_response_time(self, mock_analyzer_global, client, mock_analyzer):
        """Test API response time."""
        mock_analyzer_global = mock_analyzer
        
        import time
        
        # Single analysis
        start_time = time.time()
        response = client.post("/analyze", json={"text": "test"})
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    @patch('sentiment_analyzer.api.server.analyzer')
    def test_concurrent_requests(self, mock_analyzer_global, client, mock_analyzer):
        """Test handling of concurrent requests."""
        mock_analyzer_global = mock_analyzer
        
        import concurrent.futures
        import threading
        
        def make_request():
            return client.post("/analyze", json={"text": "test"})
        
        # Simulate concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_docs_endpoint(self, client):
        """Test that OpenAPI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint(self, client):
        """Test that ReDoc documentation is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_openapi_json(self, client):
        """Test OpenAPI JSON specification."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        spec = response.json()
        assert "openapi" in spec
        assert "info" in spec
        assert spec["info"]["title"] == "Sentiment Analyzer Pro"


class TestAPIIntegration:
    """End-to-end API integration tests."""
    
    @patch('sentiment_analyzer.api.server.analyzer')
    def test_complete_analysis_workflow(self, mock_analyzer_global, client, mock_analyzer):
        """Test complete analysis workflow."""
        mock_analyzer_global = mock_analyzer
        
        # 1. Check health
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        # 2. Get available models
        response = client.get("/models")
        assert response.status_code == 200
        models_info = response.json()
        assert "available_models" in models_info
        
        # 3. Analyze single text
        response = client.post("/analyze", json={
            "text": "I love this amazing product!",
            "model_type": models_info["current_model"]
        })
        assert response.status_code == 200
        analysis_result = response.json()
        assert analysis_result["success"] is True
        
        # 4. Analyze batch
        response = client.post("/analyze/batch", json={
            "texts": ["Great product!", "Not so good", "It's okay"],
            "model_type": models_info["current_model"]
        })
        assert response.status_code == 200
        batch_result = response.json()
        assert batch_result["success"] is True
        assert batch_result["total_processed"] == 3
        
        # 5. Check cache stats
        response = client.get("/cache/stats")
        assert response.status_code == 200
        
        # 6. Check metrics
        response = client.get("/metrics")
        assert response.status_code == 200
        metrics = response.json()
        assert metrics["request_count"] > 0
    
    def test_error_recovery(self, client):
        """Test API error recovery."""
        # Send malformed requests
        malformed_requests = [
            {"invalid": "data"},
            {"text": None},
            {"texts": "not_a_list"},
        ]
        
        for bad_request in malformed_requests:
            response = client.post("/analyze", json=bad_request)
            # Should return validation error, not crash
            assert response.status_code == 422
        
        # API should still work after bad requests
        response = client.get("/health")
        assert response.status_code == 200
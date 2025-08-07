"""
FastAPI server for sentiment analysis API.
"""
import time
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from ..core.analyzer import SentimentAnalyzer
from ..core.models import ModelType, AnalysisConfig, SentimentResult

logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer: Optional[SentimentAnalyzer] = None


class AnalysisRequest(BaseModel):
    """Request model for sentiment analysis."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    model_type: Optional[ModelType] = Field(None, description="Model to use for analysis")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()


class BatchAnalysisRequest(BaseModel):
    """Request model for batch sentiment analysis."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Texts to analyze")
    model_type: Optional[ModelType] = Field(None, description="Model to use for analysis")
    
    @validator('texts')
    def validate_texts(cls, v):
        cleaned = []
        for text in v:
            if isinstance(text, str) and text.strip():
                cleaned.append(text.strip())
        if not cleaned:
            raise ValueError('At least one valid text required')
        return cleaned[:100]  # Limit batch size


class AnalysisResponse(BaseModel):
    """Response model for sentiment analysis."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: str


class BatchAnalysisResponse(BaseModel):
    """Response model for batch sentiment analysis."""
    success: bool
    results: List[Dict[str, Any]] = []
    error: Optional[str] = None
    processing_time: float
    total_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime: float
    models: Dict[str, Any]


# Application state
app_state = {
    "start_time": time.time(),
    "request_count": 0,
    "error_count": 0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global analyzer
    
    # Startup
    logger.info("Starting Sentiment Analyzer API...")
    
    # Initialize analyzer with default config
    config = AnalysisConfig(
        model_type=ModelType.VADER,
        confidence_threshold=0.6,
        batch_size=32,
        cache_results=True,
        enable_preprocessing=True
    )
    
    analyzer = SentimentAnalyzer(config)
    logger.info("Sentiment analyzer initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentiment Analyzer API...")


def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Sentiment Analyzer Pro",
        description="Advanced sentiment analysis API with multiple model support",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request counting middleware
    @app.middleware("http")
    async def count_requests(request, call_next):
        app_state["request_count"] += 1
        start_time = time.time()
        
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            app_state["error_count"] += 1
            logger.error(f"Request failed: {e}")
            raise
    
    # Dependency to get analyzer
    def get_analyzer() -> SentimentAnalyzer:
        if analyzer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Sentiment analyzer not initialized"
            )
        return analyzer
    
    # Routes
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "Sentiment Analyzer Pro API",
            "version": "0.1.0",
            "docs": "/docs"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check(sentiment_analyzer: SentimentAnalyzer = Depends(get_analyzer)):
        """Health check endpoint."""
        uptime = time.time() - app_state["start_time"]
        
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            uptime=uptime,
            models=sentiment_analyzer.get_model_info()
        )
    
    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze_sentiment(
        request: AnalysisRequest,
        sentiment_analyzer: SentimentAnalyzer = Depends(get_analyzer)
    ):
        """Analyze sentiment of a single text."""
        start_time = time.time()
        
        try:
            result = sentiment_analyzer.analyze_text(
                text=request.text,
                model_type=request.model_type
            )
            
            processing_time = time.time() - start_time
            
            return AnalysisResponse(
                success=True,
                result=result.to_dict(),
                processing_time=processing_time,
                timestamp=result.timestamp.isoformat()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Analysis failed: {e}")
            
            return AnalysisResponse(
                success=False,
                error=str(e),
                processing_time=processing_time,
                timestamp=time.time()
            )
    
    @app.post("/analyze/batch", response_model=BatchAnalysisResponse)
    async def analyze_batch_sentiment(
        request: BatchAnalysisRequest,
        sentiment_analyzer: SentimentAnalyzer = Depends(get_analyzer)
    ):
        """Analyze sentiment of multiple texts."""
        start_time = time.time()
        
        try:
            results = sentiment_analyzer.analyze_batch(
                texts=request.texts,
                model_type=request.model_type
            )
            
            processing_time = time.time() - start_time
            
            return BatchAnalysisResponse(
                success=True,
                results=[result.to_dict() for result in results],
                processing_time=processing_time,
                total_processed=len(results),
                timestamp=time.time()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Batch analysis failed: {e}")
            
            return BatchAnalysisResponse(
                success=False,
                error=str(e),
                processing_time=processing_time,
                total_processed=0,
                timestamp=time.time()
            )
    
    @app.get("/models", response_model=Dict[str, Any])
    async def get_models_info(sentiment_analyzer: SentimentAnalyzer = Depends(get_analyzer)):
        """Get information about available models."""
        return sentiment_analyzer.get_model_info()
    
    @app.post("/cache/clear")
    async def clear_cache(sentiment_analyzer: SentimentAnalyzer = Depends(get_analyzer)):
        """Clear the analysis cache."""
        sentiment_analyzer.clear_cache()
        return {"message": "Cache cleared successfully"}
    
    @app.get("/cache/stats")
    async def get_cache_stats(sentiment_analyzer: SentimentAnalyzer = Depends(get_analyzer)):
        """Get cache statistics."""
        return sentiment_analyzer.get_cache_stats()
    
    @app.get("/metrics")
    async def get_metrics():
        """Get application metrics."""
        uptime = time.time() - app_state["start_time"]
        
        return {
            "uptime_seconds": uptime,
            "request_count": app_state["request_count"],
            "error_count": app_state["error_count"],
            "error_rate": app_state["error_count"] / max(app_state["request_count"], 1)
        }
    
    return app


# CLI entry point
def main():
    """Main entry point for API server."""
    logging.basicConfig(level=logging.INFO)
    
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
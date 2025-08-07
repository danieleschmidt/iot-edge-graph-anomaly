# 🎯 Sentiment Analyzer Pro

**Advanced Sentiment Analysis with Production-Grade Features**

[![Build Status](https://img.shields.io/github/actions/workflow/status/terragon-labs/sentiment-analyzer-pro/ci.yml?branch=main)](https://github.com/terragon-labs/sentiment-analyzer-pro/actions)
[![Coverage](https://img.shields.io/codecov/c/github/terragon-labs/sentiment-analyzer-pro)](https://codecov.io/gh/terragon-labs/sentiment-analyzer-pro)
[![License](https://img.shields.io/github/license/terragon-labs/sentiment-analyzer-pro)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue)](https://python.org)

A production-ready sentiment analysis system with enterprise-grade security, multi-language support, and global compliance features. Built with modern async architecture for high-throughput processing.

## ✨ Key Features

### 🧠 Advanced Sentiment Analysis
- **Multiple Models**: VADER, TextBlob, with extensible architecture
- **Real-time Processing**: Sub-200ms response times
- **Batch Processing**: Efficient bulk analysis with async workers
- **Confidence Scoring**: Detailed sentiment confidence metrics

### 🚀 Production-Ready Architecture
- **High Performance**: Async processing with worker pools
- **Smart Caching**: LRU cache with adaptive optimization
- **Circuit Breakers**: Fault tolerance and graceful degradation
- **Health Monitoring**: Comprehensive health checks and metrics

### 🔒 Enterprise Security
- **Input Validation**: Multi-level security with XSS/injection protection
- **Rate Limiting**: Configurable per-client rate limits
- **Security Levels**: Basic, Strict, and Paranoid security modes
- **Audit Logging**: Complete security event tracking

### 🌍 Global-First Design
- **Multi-Language**: Support for 10 languages (EN, ES, FR, DE, JA, ZH, PT, IT, RU, AR)
- **Privacy Compliance**: GDPR, CCPA, PDPA, UK GDPR compliance
- **Data Sovereignty**: Regional data processing controls
- **Localized APIs**: Language-aware error messages and responses

### 📊 Monitoring & Observability
- **Prometheus Metrics**: Detailed performance and usage metrics
- **OpenTelemetry**: Distributed tracing support
- **Grafana Dashboards**: Pre-built visualization dashboards
- **Health Endpoints**: Service and component health monitoring

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install sentiment-analyzer-pro

# Or install from source
git clone https://github.com/terragon-labs/sentiment-analyzer-pro.git
cd sentiment-analyzer-pro
pip install -e .
```

### Basic Usage

```python
from sentiment_analyzer import SentimentAnalyzer
from sentiment_analyzer.core.models import ModelType

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Analyze single text
result = analyzer.analyze_text("I love this amazing product!")
print(f"Sentiment: {result.label} (confidence: {result.confidence:.3f})")

# Batch analysis
texts = ["Great service!", "Could be better", "Absolutely terrible"]
results = analyzer.analyze_batch(texts)
```

### API Server

```bash
# Start development server
sentiment-analyzer serve --host 0.0.0.0 --port 8000

# Or with gunicorn for production
gunicorn sentiment_analyzer.api.server:create_app --factory --bind 0.0.0.0:8000
```

### CLI Usage

```bash
# Analyze single text
sentiment-analyzer analyze "I'm feeling great today!"

# Analyze from file
sentiment-analyzer batch -i input.txt -o results.json

# Compare models
sentiment-analyzer demo "This is a test message"
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🛠️ Built With

- **FastAPI** - Modern async web framework
- **VADER Sentiment** - Rule-based sentiment analysis
- **TextBlob** - Natural language processing
- **Prometheus** - Metrics and monitoring
- **Redis** - Caching and session storage
- **Docker** - Containerization
- **Kubernetes** - Orchestration

## 🏢 Enterprise

For enterprise deployments, custom models, and professional support, contact [enterprise@terragon-labs.com](mailto:enterprise@terragon-labs.com).

---

**Made with ❤️ by [Terragon Labs](https://terragon-labs.com)**

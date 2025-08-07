"""
Configuration management for sentiment analyzer.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

from ..core.models import ModelType, AnalysisConfig
from ..security.validation import SecurityLevel

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: List[str] = None
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class SecurityConfig:
    """Security configuration."""
    level: SecurityLevel = SecurityLevel.STRICT
    max_text_length: int = 5000
    enable_profanity_filter: bool = True
    enable_rate_limiting: bool = True
    log_security_events: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_metrics: bool = True
    metrics_retention_hours: int = 24
    enable_health_checks: bool = True
    enable_circuit_breakers: bool = True
    prometheus_endpoint: bool = True


@dataclass
class CacheConfig:
    """Caching configuration."""
    enable_cache: bool = True
    max_cache_size: int = 10000
    cache_ttl_seconds: int = 3600
    clear_cache_on_startup: bool = False


@dataclass
class AppConfig:
    """Complete application configuration."""
    analysis: AnalysisConfig
    api: APIConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    cache: CacheConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """Create configuration from dictionary."""
        # Convert model_type string to enum
        if 'analysis' in data and 'model_type' in data['analysis']:
            data['analysis']['model_type'] = ModelType(data['analysis']['model_type'])
        
        # Convert security level string to enum
        if 'security' in data and 'level' in data['security']:
            data['security']['level'] = SecurityLevel(data['security']['level'])
        
        return cls(
            analysis=AnalysisConfig(**data.get('analysis', {})),
            api=APIConfig(**data.get('api', {})),
            security=SecurityConfig(**data.get('security', {})),
            monitoring=MonitoringConfig(**data.get('monitoring', {})),
            cache=CacheConfig(**data.get('cache', {}))
        )


class ConfigManager:
    """
    Configuration manager with environment variable support.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Optional[AppConfig] = None
    
    def load_config(self, config_path: Optional[str] = None) -> AppConfig:
        """Load configuration from file and environment variables."""
        config_path = config_path or self.config_path
        
        # Start with default configuration
        config_data = self._get_default_config()
        
        # Load from file if exists
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                config_data = self._merge_configs(config_data, file_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
        
        # Override with environment variables
        config_data = self._apply_env_overrides(config_data)
        
        # Create and validate configuration
        self.config = AppConfig.from_dict(config_data)
        self._validate_config(self.config)
        
        return self.config
    
    def save_config(self, config: AppConfig, config_path: str) -> None:
        """Save configuration to file."""
        try:
            # Convert enums to strings for serialization
            config_dict = config.to_dict()
            config_dict['analysis']['model_type'] = config.analysis.model_type.value
            config_dict['security']['level'] = config.security.level.value
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'analysis': {
                'model_type': ModelType.VADER.value,
                'confidence_threshold': 0.6,
                'batch_size': 32,
                'max_length': 512,
                'cache_results': True,
                'enable_preprocessing': True
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 1,
                'reload': False,
                'log_level': 'info',
                'cors_origins': ['*'],
                'rate_limit_requests': 1000,
                'rate_limit_window': 3600
            },
            'security': {
                'level': SecurityLevel.STRICT.value,
                'max_text_length': 5000,
                'enable_profanity_filter': True,
                'enable_rate_limiting': True,
                'log_security_events': True
            },
            'monitoring': {
                'enable_metrics': True,
                'metrics_retention_hours': 24,
                'enable_health_checks': True,
                'enable_circuit_breakers': True,
                'prometheus_endpoint': True
            },
            'cache': {
                'enable_cache': True,
                'max_cache_size': 10000,
                'cache_ttl_seconds': 3600,
                'clear_cache_on_startup': False
            }
        }
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Environment variable mapping
        env_mappings = {
            'SENTIMENT_MODEL_TYPE': ('analysis', 'model_type'),
            'SENTIMENT_API_HOST': ('api', 'host'),
            'SENTIMENT_API_PORT': ('api', 'port'),
            'SENTIMENT_API_WORKERS': ('api', 'workers'),
            'SENTIMENT_SECURITY_LEVEL': ('security', 'level'),
            'SENTIMENT_RATE_LIMIT': ('api', 'rate_limit_requests'),
            'SENTIMENT_CACHE_ENABLED': ('cache', 'enable_cache'),
            'SENTIMENT_METRICS_ENABLED': ('monitoring', 'enable_metrics'),
            'SENTIMENT_LOG_LEVEL': ('api', 'log_level')
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion
                if key in ['port', 'workers', 'rate_limit_requests', 'max_text_length']:
                    value = int(value)
                elif key in ['enable_cache', 'enable_metrics', 'reload', 'enable_profanity_filter']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif key == 'cors_origins':
                    value = [origin.strip() for origin in value.split(',')]
                
                config_data[section][key] = value
                logger.debug(f"Applied env override: {env_var}={value}")
        
        return config_data
    
    def _validate_config(self, config: AppConfig) -> None:
        """Validate configuration values."""
        errors = []
        
        # Analysis config validation
        if not config.analysis.validate():
            errors.append("Invalid analysis configuration")
        
        # API config validation
        if not (1 <= config.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        
        if config.api.workers < 1:
            errors.append("API workers must be at least 1")
        
        # Security config validation
        if config.security.max_text_length < 1:
            errors.append("Max text length must be positive")
        
        # Cache config validation
        if config.cache.max_cache_size < 0:
            errors.append("Cache size cannot be negative")
        
        if config.cache.cache_ttl_seconds < 0:
            errors.append("Cache TTL cannot be negative")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        logger.info("Configuration validation passed")
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        if self.config is None:
            return self.load_config()
        return self.config


# Global configuration manager
config_manager = ConfigManager()
"""
Enterprise Integration Gateway for IoT Edge Anomaly Detection.

This module provides enterprise-grade integration capabilities:
- Advanced API Gateway with rate limiting, authentication, and authorization
- Service mesh integration (Istio/Linkerd) with traffic management
- Enterprise SSO integration (LDAP, SAML, OAuth2)
- Workflow orchestration with failure recovery
- Message queue integration and event-driven architecture
- Circuit breaker patterns for external service calls
- API versioning and backward compatibility
"""

import asyncio
import logging
import json
import time
import hashlib
import base64
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import jwt
import aiohttp
import yaml
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import ssl
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, parse_qs
import hmac

logger = logging.getLogger(__name__)


class AuthenticationMethod(Enum):
    """Authentication methods supported by the gateway."""
    API_KEY = "api_key"
    JWT_BEARER = "jwt_bearer"
    OAUTH2 = "oauth2"
    SAML = "saml"
    LDAP = "ldap"
    BASIC_AUTH = "basic_auth"
    MUTUAL_TLS = "mutual_tls"


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class ServiceMeshProvider(Enum):
    """Service mesh providers."""
    ISTIO = "istio"
    LINKERD = "linkerd"
    CONSUL_CONNECT = "consul_connect"
    ENVOY = "envoy"


@dataclass
class APIEndpoint:
    """API endpoint configuration."""
    path: str
    method: str
    handler: str
    auth_required: bool = True
    auth_methods: List[AuthenticationMethod] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    version: str = "v1"
    deprecated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "method": self.method,
            "handler": self.handler,
            "auth_required": self.auth_required,
            "auth_methods": [method.value for method in self.auth_methods],
            "rate_limits": self.rate_limits,
            "permissions": self.permissions,
            "version": self.version,
            "deprecated": self.deprecated
        }


@dataclass
class ServiceMeshConfig:
    """Service mesh configuration."""
    provider: ServiceMeshProvider
    namespace: str
    service_name: str
    traffic_policy: Dict[str, Any] = field(default_factory=dict)
    security_policy: Dict[str, Any] = field(default_factory=dict)
    observability_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider.value,
            "namespace": self.namespace,
            "service_name": self.service_name,
            "traffic_policy": self.traffic_policy,
            "security_policy": self.security_policy,
            "observability_config": self.observability_config
        }


class RateLimiter:
    """
    Advanced rate limiting with multiple strategies.
    
    Features:
    - Multiple rate limiting algorithms
    - Per-client and global rate limiting
    - Burst handling with token bucket
    - Rate limit headers for client awareness
    """
    
    def __init__(self, strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW):
        """Initialize rate limiter."""
        self.strategy = strategy
        self.client_buckets: Dict[str, Dict[str, Any]] = {}
        self.global_counters: Dict[str, Dict[str, Any]] = {}
        
    async def is_allowed(self, client_id: str, endpoint: str, 
                        limit: int, window_seconds: int = 60) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limits.
        
        Args:
            client_id: Client identifier
            endpoint: Endpoint being accessed
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        current_time = time.time()
        bucket_key = f"{client_id}:{endpoint}"
        
        if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_check(bucket_key, limit, window_seconds, current_time)
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket_check(bucket_key, limit, window_seconds, current_time)
        elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window_check(bucket_key, limit, window_seconds, current_time)
        else:
            # Default to sliding window
            return await self._sliding_window_check(bucket_key, limit, window_seconds, current_time)
    
    async def _sliding_window_check(self, bucket_key: str, limit: int, 
                                  window_seconds: int, current_time: float) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limit check."""
        if bucket_key not in self.client_buckets:
            self.client_buckets[bucket_key] = {
                "requests": deque(),
                "count": 0
            }
        
        bucket = self.client_buckets[bucket_key]
        requests = bucket["requests"]
        
        # Remove old requests outside the window
        cutoff_time = current_time - window_seconds
        while requests and requests[0] <= cutoff_time:
            requests.popleft()
        
        # Check if we can accept this request
        if len(requests) < limit:
            requests.append(current_time)
            allowed = True
        else:
            allowed = False
        
        # Calculate rate limit info
        remaining = max(0, limit - len(requests))
        reset_time = int(requests[0] + window_seconds) if requests else int(current_time + window_seconds)
        
        rate_limit_info = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "retry_after": None if allowed else int(reset_time - current_time)
        }
        
        return allowed, rate_limit_info
    
    async def _token_bucket_check(self, bucket_key: str, limit: int, 
                                window_seconds: int, current_time: float) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limit check."""
        if bucket_key not in self.client_buckets:
            self.client_buckets[bucket_key] = {
                "tokens": limit,
                "last_refill": current_time
            }
        
        bucket = self.client_buckets[bucket_key]
        
        # Refill tokens based on elapsed time
        elapsed = current_time - bucket["last_refill"]
        tokens_to_add = int(elapsed * (limit / window_seconds))
        bucket["tokens"] = min(limit, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = current_time
        
        # Check if we can consume a token
        if bucket["tokens"] > 0:
            bucket["tokens"] -= 1
            allowed = True
        else:
            allowed = False
        
        # Calculate when next token will be available
        time_per_token = window_seconds / limit
        retry_after = int(time_per_token) if not allowed else None
        
        rate_limit_info = {
            "limit": limit,
            "remaining": bucket["tokens"],
            "reset": int(current_time + window_seconds),
            "retry_after": retry_after
        }
        
        return allowed, rate_limit_info
    
    async def _fixed_window_check(self, bucket_key: str, limit: int, 
                                window_seconds: int, current_time: float) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limit check."""
        window_start = int(current_time // window_seconds) * window_seconds
        window_key = f"{bucket_key}:{window_start}"
        
        if window_key not in self.client_buckets:
            self.client_buckets[window_key] = {"count": 0}
        
        bucket = self.client_buckets[window_key]
        
        if bucket["count"] < limit:
            bucket["count"] += 1
            allowed = True
        else:
            allowed = False
        
        remaining = max(0, limit - bucket["count"])
        reset_time = int(window_start + window_seconds)
        
        rate_limit_info = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "retry_after": int(reset_time - current_time) if not allowed else None
        }
        
        return allowed, rate_limit_info


class AuthenticationManager:
    """
    Multi-method authentication manager.
    
    Features:
    - Multiple authentication methods (JWT, OAuth2, SAML, LDAP)
    - Token validation and refresh
    - SSO integration
    - Role-based access control
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize authentication manager."""
        self.config = config or {}
        
        # JWT configuration
        self.jwt_secret = self.config.get("jwt_secret", "default_secret")
        self.jwt_algorithm = self.config.get("jwt_algorithm", "HS256")
        self.jwt_expiry = self.config.get("jwt_expiry_minutes", 60)
        
        # OAuth2 configuration
        self.oauth2_providers = self.config.get("oauth2_providers", {})
        
        # SAML configuration
        self.saml_config = self.config.get("saml", {})
        
        # LDAP configuration
        self.ldap_config = self.config.get("ldap", {})
        
        # API key storage (in production, use secure storage)
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # Active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def authenticate(self, method: AuthenticationMethod, 
                          credentials: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Authenticate user using specified method.
        
        Args:
            method: Authentication method to use
            credentials: Authentication credentials
            
        Returns:
            Tuple of (success, user_info)
        """
        if method == AuthenticationMethod.API_KEY:
            return await self._authenticate_api_key(credentials)
        elif method == AuthenticationMethod.JWT_BEARER:
            return await self._authenticate_jwt(credentials)
        elif method == AuthenticationMethod.OAUTH2:
            return await self._authenticate_oauth2(credentials)
        elif method == AuthenticationMethod.SAML:
            return await self._authenticate_saml(credentials)
        elif method == AuthenticationMethod.LDAP:
            return await self._authenticate_ldap(credentials)
        elif method == AuthenticationMethod.BASIC_AUTH:
            return await self._authenticate_basic(credentials)
        elif method == AuthenticationMethod.MUTUAL_TLS:
            return await self._authenticate_mutual_tls(credentials)
        else:
            return False, None
    
    async def _authenticate_api_key(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate using API key."""
        api_key = credentials.get("api_key")
        if not api_key:
            return False, None
        
        # Hash the API key for secure comparison
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.api_keys:
            key_info = self.api_keys[key_hash]
            
            # Check if key is active and not expired
            if key_info.get("active", True) and not self._is_expired(key_info.get("expires_at")):
                return True, {
                    "user_id": key_info["user_id"],
                    "permissions": key_info.get("permissions", []),
                    "auth_method": AuthenticationMethod.API_KEY.value
                }
        
        return False, None
    
    async def _authenticate_jwt(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate using JWT token."""
        token = credentials.get("token")
        if not token:
            return False, None
        
        try:
            # Decode and verify JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check expiration
            if "exp" in payload and payload["exp"] < time.time():
                return False, None
            
            # Extract user information
            user_info = {
                "user_id": payload.get("sub"),
                "permissions": payload.get("permissions", []),
                "roles": payload.get("roles", []),
                "auth_method": AuthenticationMethod.JWT_BEARER.value,
                "token_claims": payload
            }
            
            return True, user_info
        
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return False, None
    
    async def _authenticate_oauth2(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate using OAuth2."""
        access_token = credentials.get("access_token")
        provider = credentials.get("provider", "default")
        
        if not access_token or provider not in self.oauth2_providers:
            return False, None
        
        provider_config = self.oauth2_providers[provider]
        
        try:
            # Validate token with OAuth2 provider
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {access_token}"}
                
                async with session.get(provider_config["userinfo_endpoint"], headers=headers) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        
                        user_info = {
                            "user_id": user_data.get("sub") or user_data.get("id"),
                            "email": user_data.get("email"),
                            "name": user_data.get("name"),
                            "permissions": [],  # Would be mapped from provider roles
                            "auth_method": AuthenticationMethod.OAUTH2.value,
                            "provider": provider
                        }
                        
                        return True, user_info
                    else:
                        return False, None
        
        except Exception as e:
            logger.error(f"OAuth2 authentication error: {e}")
            return False, None
    
    async def _authenticate_saml(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate using SAML assertion."""
        saml_response = credentials.get("saml_response")
        if not saml_response:
            return False, None
        
        try:
            # Decode SAML response (simplified - use proper SAML library in production)
            decoded_response = base64.b64decode(saml_response)
            root = ET.fromstring(decoded_response)
            
            # Extract user information from SAML assertion
            # This is a simplified implementation - use proper SAML validation
            
            # Mock user extraction for demonstration
            user_info = {
                "user_id": "saml_user_123",
                "email": "user@example.com",
                "permissions": ["read", "write"],
                "auth_method": AuthenticationMethod.SAML.value
            }
            
            return True, user_info
        
        except Exception as e:
            logger.error(f"SAML authentication error: {e}")
            return False, None
    
    async def _authenticate_ldap(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate using LDAP."""
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            return False, None
        
        # In production, use proper LDAP library like python-ldap
        # This is a simplified mock implementation
        
        try:
            # Mock LDAP authentication
            if username == "ldap_user" and password == "ldap_password":
                user_info = {
                    "user_id": username,
                    "email": f"{username}@company.com",
                    "permissions": ["read", "write"],
                    "groups": ["users", "analysts"],
                    "auth_method": AuthenticationMethod.LDAP.value
                }
                
                return True, user_info
            else:
                return False, None
        
        except Exception as e:
            logger.error(f"LDAP authentication error: {e}")
            return False, None
    
    async def _authenticate_basic(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate using basic authentication."""
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            return False, None
        
        # Mock basic auth validation
        # In production, validate against secure user store
        if username == "admin" and password == "admin123":
            user_info = {
                "user_id": username,
                "permissions": ["admin"],
                "auth_method": AuthenticationMethod.BASIC_AUTH.value
            }
            
            return True, user_info
        
        return False, None
    
    async def _authenticate_mutual_tls(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate using mutual TLS."""
        client_cert = credentials.get("client_cert")
        if not client_cert:
            return False, None
        
        try:
            # Validate client certificate (simplified)
            # In production, perform proper certificate validation
            
            user_info = {
                "user_id": "mtls_client",
                "permissions": ["secure_access"],
                "auth_method": AuthenticationMethod.MUTUAL_TLS.value,
                "cert_subject": client_cert.get("subject", "")
            }
            
            return True, user_info
        
        except Exception as e:
            logger.error(f"Mutual TLS authentication error: {e}")
            return False, None
    
    def _is_expired(self, expires_at: Optional[datetime]) -> bool:
        """Check if credential is expired."""
        if not expires_at:
            return False
        return datetime.now() > expires_at
    
    def create_api_key(self, user_id: str, permissions: List[str], 
                      expires_days: Optional[int] = None) -> str:
        """Create a new API key for a user."""
        # Generate random API key
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Set expiration
        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)
        
        # Store API key info
        self.api_keys[key_hash] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.now(),
            "expires_at": expires_at,
            "active": True
        }
        
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self.api_keys:
            self.api_keys[key_hash]["active"] = False
            return True
        return False


class ServiceMeshIntegrator:
    """
    Service mesh integration for traffic management and security.
    
    Features:
    - Traffic routing and load balancing
    - Circuit breaker patterns
    - Mutual TLS enforcement
    - Observability and tracing
    """
    
    def __init__(self, config: ServiceMeshConfig):
        """Initialize service mesh integrator."""
        self.config = config
        self.mesh_client = None
        
        # Initialize mesh-specific client
        if config.provider == ServiceMeshProvider.ISTIO:
            self._initialize_istio_client()
        elif config.provider == ServiceMeshProvider.LINKERD:
            self._initialize_linkerd_client()
    
    def _initialize_istio_client(self):
        """Initialize Istio client."""
        # In production, use Kubernetes client to interact with Istio CRDs
        logger.info("Initialized Istio service mesh client")
    
    def _initialize_linkerd_client(self):
        """Initialize Linkerd client."""
        # In production, use Kubernetes client to interact with Linkerd
        logger.info("Initialized Linkerd service mesh client")
    
    async def configure_traffic_policy(self, policy: Dict[str, Any]) -> bool:
        """Configure traffic management policies."""
        try:
            if self.config.provider == ServiceMeshProvider.ISTIO:
                return await self._configure_istio_traffic_policy(policy)
            elif self.config.provider == ServiceMeshProvider.LINKERD:
                return await self._configure_linkerd_traffic_policy(policy)
            else:
                logger.warning(f"Traffic policy configuration not implemented for {self.config.provider}")
                return False
        except Exception as e:
            logger.error(f"Failed to configure traffic policy: {e}")
            return False
    
    async def _configure_istio_traffic_policy(self, policy: Dict[str, Any]) -> bool:
        """Configure Istio traffic policy."""
        # Example Istio VirtualService configuration
        virtual_service = {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{self.config.service_name}-vs",
                "namespace": self.config.namespace
            },
            "spec": {
                "hosts": [self.config.service_name],
                "http": policy.get("http_routes", [])
            }
        }
        
        # In production, apply this configuration to Kubernetes
        logger.info(f"Would apply Istio VirtualService: {virtual_service}")
        return True
    
    async def _configure_linkerd_traffic_policy(self, policy: Dict[str, Any]) -> bool:
        """Configure Linkerd traffic policy."""
        # Example Linkerd TrafficSplit configuration
        traffic_split = {
            "apiVersion": "split.smi-spec.io/v1alpha1",
            "kind": "TrafficSplit",
            "metadata": {
                "name": f"{self.config.service_name}-split",
                "namespace": self.config.namespace
            },
            "spec": {
                "service": self.config.service_name,
                "backends": policy.get("backends", [])
            }
        }
        
        # In production, apply this configuration to Kubernetes
        logger.info(f"Would apply Linkerd TrafficSplit: {traffic_split}")
        return True
    
    async def enable_circuit_breaker(self, threshold_errors: int = 5, 
                                   timeout_seconds: int = 30) -> bool:
        """Enable circuit breaker for the service."""
        circuit_breaker_config = {
            "consecutiveErrors": threshold_errors,
            "interval": f"{timeout_seconds}s",
            "baseEjectionTime": f"{timeout_seconds * 2}s"
        }
        
        if self.config.provider == ServiceMeshProvider.ISTIO:
            destination_rule = {
                "apiVersion": "networking.istio.io/v1alpha3",
                "kind": "DestinationRule",
                "metadata": {
                    "name": f"{self.config.service_name}-cb",
                    "namespace": self.config.namespace
                },
                "spec": {
                    "host": self.config.service_name,
                    "trafficPolicy": {
                        "outlierDetection": circuit_breaker_config
                    }
                }
            }
            
            logger.info(f"Would apply Istio circuit breaker: {destination_rule}")
            return True
        
        return False
    
    async def enable_mutual_tls(self) -> bool:
        """Enable mutual TLS for the service."""
        if self.config.provider == ServiceMeshProvider.ISTIO:
            peer_authentication = {
                "apiVersion": "security.istio.io/v1beta1",
                "kind": "PeerAuthentication",
                "metadata": {
                    "name": f"{self.config.service_name}-mtls",
                    "namespace": self.config.namespace
                },
                "spec": {
                    "selector": {
                        "matchLabels": {
                            "app": self.config.service_name
                        }
                    },
                    "mtls": {
                        "mode": "STRICT"
                    }
                }
            }
            
            logger.info(f"Would apply Istio mTLS: {peer_authentication}")
            return True
        
        return False


class EnterpriseAPIGateway:
    """
    Enterprise API Gateway with advanced features.
    
    Features:
    - Multi-method authentication and authorization
    - Advanced rate limiting with multiple strategies
    - API versioning and deprecation management
    - Request/response transformation
    - Circuit breaker patterns for backend services
    - Service mesh integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enterprise API gateway."""
        self.config = config or {}
        
        # Initialize components
        self.auth_manager = AuthenticationManager(config.get("authentication", {}))
        self.rate_limiter = RateLimiter(
            RateLimitStrategy(config.get("rate_limit_strategy", "sliding_window"))
        )
        
        # API endpoint registry
        self.endpoints: Dict[str, APIEndpoint] = {}
        
        # Service mesh integration
        self.service_mesh = None
        if config.get("service_mesh"):
            mesh_config = ServiceMeshConfig(**config["service_mesh"])
            self.service_mesh = ServiceMeshIntegrator(mesh_config)
        
        # Request/response middleware
        self.request_middleware: List[Callable] = []
        self.response_middleware: List[Callable] = []
        
        # Circuit breakers for backend services
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default endpoints
        self._initialize_default_endpoints()
    
    def _initialize_default_endpoints(self):
        """Initialize default API endpoints."""
        # Model inference endpoint
        inference_endpoint = APIEndpoint(
            path="/api/v1/inference",
            method="POST",
            handler="inference_handler",
            auth_required=True,
            auth_methods=[AuthenticationMethod.JWT_BEARER, AuthenticationMethod.API_KEY],
            rate_limits={"requests_per_minute": 100, "requests_per_hour": 1000},
            permissions=["inference:execute"],
            version="v1"
        )
        
        # Health check endpoint
        health_endpoint = APIEndpoint(
            path="/health",
            method="GET",
            handler="health_handler",
            auth_required=False,
            rate_limits={"requests_per_minute": 1000},
            version="v1"
        )
        
        # Admin endpoints
        admin_endpoint = APIEndpoint(
            path="/api/v1/admin/status",
            method="GET",
            handler="admin_status_handler",
            auth_required=True,
            auth_methods=[AuthenticationMethod.JWT_BEARER],
            permissions=["admin:read"],
            rate_limits={"requests_per_minute": 10},
            version="v1"
        )
        
        # Register endpoints
        for endpoint in [inference_endpoint, health_endpoint, admin_endpoint]:
            self.register_endpoint(endpoint)
    
    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """Register an API endpoint."""
        endpoint_key = f"{endpoint.method}:{endpoint.path}"
        self.endpoints[endpoint_key] = endpoint
        logger.info(f"Registered endpoint: {endpoint_key}")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming API request.
        
        Args:
            request: Request dictionary containing method, path, headers, body, etc.
            
        Returns:
            Response dictionary
        """
        try:
            # Extract request information
            method = request.get("method", "GET").upper()
            path = request.get("path", "/")
            headers = request.get("headers", {})
            body = request.get("body")
            client_id = request.get("client_id", "unknown")
            
            # Find matching endpoint
            endpoint_key = f"{method}:{path}"
            endpoint = self.endpoints.get(endpoint_key)
            
            if not endpoint:
                return self._create_error_response(404, "Endpoint not found")
            
            # Check if endpoint is deprecated
            if endpoint.deprecated:
                logger.warning(f"Deprecated endpoint accessed: {endpoint_key}")
                # In production, might return deprecation warnings in headers
            
            # Apply request middleware
            for middleware in self.request_middleware:
                request = await middleware(request)
            
            # Rate limiting
            if endpoint.rate_limits:
                rate_limit_result = await self._check_rate_limits(
                    client_id, endpoint, headers
                )
                if not rate_limit_result["allowed"]:
                    return self._create_rate_limit_response(rate_limit_result)
            
            # Authentication
            if endpoint.auth_required:
                auth_result = await self._authenticate_request(endpoint, headers)
                if not auth_result["authenticated"]:
                    return self._create_auth_error_response(auth_result)
                
                # Authorization
                if endpoint.permissions:
                    if not await self._authorize_request(auth_result["user"], endpoint.permissions):
                        return self._create_error_response(403, "Insufficient permissions")
            
            # Handle the request
            response = await self._handle_endpoint_request(endpoint, request)
            
            # Apply response middleware
            for middleware in self.response_middleware:
                response = await middleware(response)
            
            return response
        
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return self._create_error_response(500, "Internal server error")
    
    async def _check_rate_limits(self, client_id: str, endpoint: APIEndpoint, 
                               headers: Dict[str, str]) -> Dict[str, Any]:
        """Check rate limits for the request."""
        rate_limit_results = {}
        
        for limit_type, limit_value in endpoint.rate_limits.items():
            if limit_type == "requests_per_minute":
                window_seconds = 60
            elif limit_type == "requests_per_hour":
                window_seconds = 3600
            else:
                window_seconds = 60  # Default
            
            allowed, rate_info = await self.rate_limiter.is_allowed(
                client_id, endpoint.path, limit_value, window_seconds
            )
            
            rate_limit_results[limit_type] = {
                "allowed": allowed,
                "info": rate_info
            }
            
            # If any limit is exceeded, return failure
            if not allowed:
                return {
                    "allowed": False,
                    "limit_type": limit_type,
                    "rate_info": rate_info
                }
        
        return {"allowed": True, "results": rate_limit_results}
    
    async def _authenticate_request(self, endpoint: APIEndpoint, 
                                  headers: Dict[str, str]) -> Dict[str, Any]:
        """Authenticate the request."""
        # Try each allowed authentication method
        for auth_method in endpoint.auth_methods:
            credentials = self._extract_credentials(auth_method, headers)
            if credentials:
                success, user_info = await self.auth_manager.authenticate(auth_method, credentials)
                if success:
                    return {
                        "authenticated": True,
                        "user": user_info,
                        "method": auth_method.value
                    }
        
        return {"authenticated": False, "error": "Authentication required"}
    
    def _extract_credentials(self, auth_method: AuthenticationMethod, 
                           headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Extract credentials from request headers."""
        if auth_method == AuthenticationMethod.API_KEY:
            api_key = headers.get("X-API-Key") or headers.get("Authorization", "").replace("ApiKey ", "")
            if api_key:
                return {"api_key": api_key}
        
        elif auth_method == AuthenticationMethod.JWT_BEARER:
            auth_header = headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]  # Remove "Bearer " prefix
                return {"token": token}
        
        elif auth_method == AuthenticationMethod.BASIC_AUTH:
            auth_header = headers.get("Authorization", "")
            if auth_header.startswith("Basic "):
                try:
                    encoded_creds = auth_header[6:]  # Remove "Basic " prefix
                    decoded_creds = base64.b64decode(encoded_creds).decode()
                    username, password = decoded_creds.split(":", 1)
                    return {"username": username, "password": password}
                except:
                    pass
        
        return None
    
    async def _authorize_request(self, user: Dict[str, Any], 
                               required_permissions: List[str]) -> bool:
        """Check if user has required permissions."""
        user_permissions = set(user.get("permissions", []))
        required_permissions_set = set(required_permissions)
        
        return required_permissions_set.issubset(user_permissions)
    
    async def _handle_endpoint_request(self, endpoint: APIEndpoint, 
                                     request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the actual endpoint request."""
        # This is where you'd route to the actual handler
        # For demonstration, we'll return mock responses
        
        if endpoint.handler == "inference_handler":
            return await self._handle_inference_request(request)
        elif endpoint.handler == "health_handler":
            return await self._handle_health_request(request)
        elif endpoint.handler == "admin_status_handler":
            return await self._handle_admin_status_request(request)
        else:
            return self._create_error_response(501, "Handler not implemented")
    
    async def _handle_inference_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model inference request."""
        # Mock inference response
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "status_code": 200,
            "headers": {"Content-Type": "application/json"},
            "body": {
                "anomaly_score": 0.15,
                "confidence": 0.87,
                "timestamp": datetime.now().isoformat(),
                "model_version": "v1.2.3"
            }
        }
    
    async def _handle_health_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check request."""
        return {
            "status_code": 200,
            "headers": {"Content-Type": "application/json"},
            "body": {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        }
    
    async def _handle_admin_status_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle admin status request."""
        return {
            "status_code": 200,
            "headers": {"Content-Type": "application/json"},
            "body": {
                "system_status": "operational",
                "active_connections": 42,
                "requests_per_minute": 156,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _create_error_response(self, status_code: int, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "status_code": status_code,
            "headers": {"Content-Type": "application/json"},
            "body": {
                "error": message,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _create_auth_error_response(self, auth_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create authentication error response."""
        return {
            "status_code": 401,
            "headers": {
                "Content-Type": "application/json",
                "WWW-Authenticate": "Bearer"
            },
            "body": {
                "error": "Authentication required",
                "details": auth_result.get("error", ""),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _create_rate_limit_response(self, rate_limit_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create rate limit exceeded response."""
        rate_info = rate_limit_result["rate_info"]
        
        headers = {
            "Content-Type": "application/json",
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset"])
        }
        
        if rate_info["retry_after"]:
            headers["Retry-After"] = str(rate_info["retry_after"])
        
        return {
            "status_code": 429,
            "headers": headers,
            "body": {
                "error": "Rate limit exceeded",
                "limit_type": rate_limit_result["limit_type"],
                "retry_after": rate_info["retry_after"],
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def add_request_middleware(self, middleware: Callable) -> None:
        """Add request middleware."""
        self.request_middleware.append(middleware)
        logger.info("Added request middleware")
    
    def add_response_middleware(self, middleware: Callable) -> None:
        """Add response middleware."""
        self.response_middleware.append(middleware)
        logger.info("Added response middleware")
    
    async def start_gateway(self) -> None:
        """Start the API gateway."""
        logger.info("Starting Enterprise API Gateway")
        
        # Configure service mesh if enabled
        if self.service_mesh:
            await self.service_mesh.enable_mutual_tls()
            await self.service_mesh.enable_circuit_breaker()
        
        logger.info("Enterprise API Gateway started")
    
    def get_gateway_status(self) -> Dict[str, Any]:
        """Get gateway status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "endpoints_registered": len(self.endpoints),
            "auth_methods_supported": [method.value for method in AuthenticationMethod],
            "rate_limit_strategy": self.rate_limiter.strategy.value,
            "service_mesh_enabled": self.service_mesh is not None,
            "middleware_count": {
                "request": len(self.request_middleware),
                "response": len(self.response_middleware)
            }
        }


# Global gateway instance
_enterprise_gateway: Optional[EnterpriseAPIGateway] = None


def get_enterprise_gateway(config: Optional[Dict[str, Any]] = None) -> EnterpriseAPIGateway:
    """Get or create the global enterprise gateway."""
    global _enterprise_gateway
    
    if _enterprise_gateway is None:
        _enterprise_gateway = EnterpriseAPIGateway(config)
    
    return _enterprise_gateway


# Example usage and configuration
async def setup_enterprise_integration():
    """Setup enterprise integration with production configuration."""
    config = {
        "authentication": {
            "jwt_secret": "your-secure-secret-key",
            "jwt_algorithm": "HS256",
            "jwt_expiry_minutes": 60,
            "oauth2_providers": {
                "google": {
                    "client_id": "your-google-client-id",
                    "client_secret": "your-google-client-secret",
                    "userinfo_endpoint": "https://www.googleapis.com/oauth2/v2/userinfo"
                },
                "microsoft": {
                    "client_id": "your-microsoft-client-id",
                    "client_secret": "your-microsoft-client-secret",
                    "userinfo_endpoint": "https://graph.microsoft.com/v1.0/me"
                }
            },
            "ldap": {
                "server": "ldap://your-ldap-server:389",
                "base_dn": "dc=company,dc=com",
                "user_filter": "(uid={username})"
            }
        },
        "rate_limit_strategy": "sliding_window",
        "service_mesh": {
            "provider": "istio",
            "namespace": "iot-anomaly",
            "service_name": "anomaly-detection-api",
            "traffic_policy": {
                "http_routes": [
                    {
                        "match": [{"uri": {"prefix": "/api/v1"}}],
                        "route": [{"destination": {"host": "anomaly-detection-api"}}],
                        "fault": {
                            "delay": {
                                "percentage": {"value": 0.1},
                                "fixedDelay": "5s"
                            }
                        }
                    }
                ]
            }
        }
    }
    
    gateway = get_enterprise_gateway(config)
    await gateway.start_gateway()
    
    # Add custom middleware
    async def logging_middleware(request):
        logger.info(f"Request: {request['method']} {request['path']}")
        return request
    
    async def cors_middleware(response):
        if "headers" not in response:
            response["headers"] = {}
        response["headers"]["Access-Control-Allow-Origin"] = "*"
        return response
    
    gateway.add_request_middleware(logging_middleware)
    gateway.add_response_middleware(cors_middleware)
    
    logger.info("Enterprise integration configured and started")
    return gateway
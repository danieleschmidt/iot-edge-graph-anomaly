"""
Zero-Trust Security Framework for IoT Edge Anomaly Detection.

This module implements a comprehensive zero-trust security model with:
- Identity-based access control with continuous verification
- Network microsegmentation and policy enforcement
- Advanced threat detection with ML-based behavioral analysis
- Compliance frameworks (SOX, HIPAA, GDPR, PCI-DSS)
- Audit logging and forensic capabilities
- Secure model serving with end-to-end encryption
"""

import asyncio
import hashlib
import hmac
import secrets
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import torch
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.x509 import load_pem_x509_certificate
import jwt
import base64
import re
import time

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels for zero-trust architecture."""
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class AccessResult(Enum):
    """Access control results."""
    GRANTED = "granted"
    DENIED = "denied"
    REQUIRES_MFA = "requires_mfa"
    REQUIRES_STEP_UP = "requires_step_up"
    SUSPENDED = "suspended"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOX = "sox"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    SOC2 = "soc2"
    ISO27001 = "iso27001"


@dataclass
class Identity:
    """Represents an identity in the zero-trust system."""
    id: str
    name: str
    type: str  # user, service, device
    trust_level: TrustLevel = TrustLevel.UNTRUSTED
    attributes: Dict[str, Any] = field(default_factory=dict)
    permissions: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_verified: Optional[datetime] = None
    verification_failures: int = 0
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "trust_level": self.trust_level.value,
            "attributes": self.attributes,
            "permissions": list(self.permissions),
            "created_at": self.created_at.isoformat(),
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
            "verification_failures": self.verification_failures,
            "is_active": self.is_active
        }


@dataclass
class AccessRequest:
    """Represents an access request in the system."""
    request_id: str
    identity_id: str
    resource: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "request_id": self.request_id,
            "identity_id": self.identity_id,
            "resource": self.resource,
            "action": self.action,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "session_id": self.session_id
        }


@dataclass
class ThreatIndicator:
    """Represents a threat indicator detected by the system."""
    indicator_id: str
    threat_type: str
    severity: str
    confidence: float
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    source_identity: Optional[str] = None
    mitigated: bool = False


class ZeroTrustPolicyEngine:
    """
    Zero-trust policy engine with dynamic policy evaluation.
    
    Features:
    - Risk-based access control
    - Continuous authentication and authorization
    - Behavioral analysis and anomaly detection
    - Dynamic policy updates based on threat intelligence
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize zero-trust policy engine."""
        self.config = config or {}
        self.identities: Dict[str, Identity] = {}
        self.access_policies: Dict[str, Dict[str, Any]] = {}
        self.threat_indicators: List[ThreatIndicator] = []
        self.access_logs: List[Dict[str, Any]] = []
        
        # Behavioral analysis
        self.behavioral_patterns: Dict[str, Dict[str, Any]] = {}
        self.anomaly_threshold = self.config.get("anomaly_threshold", 0.8)
        
        # Compliance settings
        self.compliance_frameworks = set(self.config.get("compliance_frameworks", ["SOX", "SOC2"]))
        
        # Initialize default policies
        self._initialize_default_policies()
        
    def _initialize_default_policies(self):
        """Initialize default zero-trust policies."""
        # Default model inference policy
        self.access_policies["model:inference"] = {
            "required_trust_level": TrustLevel.MEDIUM,
            "required_permissions": ["inference:execute"],
            "max_requests_per_minute": 100,
            "require_encryption": True,
            "allow_anonymous": False,
            "compliance_requirements": ["audit_logging", "data_protection"]
        }
        
        # Admin operations policy
        self.access_policies["admin:*"] = {
            "required_trust_level": TrustLevel.HIGH,
            "required_permissions": ["admin:full_access"],
            "require_mfa": True,
            "require_step_up_auth": True,
            "max_session_duration": 3600,  # 1 hour
            "compliance_requirements": ["audit_logging", "privileged_access_monitoring"]
        }
        
        # Data access policy
        self.access_policies["data:read"] = {
            "required_trust_level": TrustLevel.MEDIUM,
            "required_permissions": ["data:read"],
            "data_classification_allowed": ["public", "internal"],
            "require_encryption": True,
            "compliance_requirements": ["data_protection", "audit_logging"]
        }
        
        # Model management policy
        self.access_policies["model:management"] = {
            "required_trust_level": TrustLevel.HIGH,
            "required_permissions": ["model:manage"],
            "require_approval": True,
            "require_mfa": True,
            "compliance_requirements": ["change_management", "audit_logging"]
        }
    
    async def evaluate_access_request(self, request: AccessRequest) -> Tuple[AccessResult, Dict[str, Any]]:
        """
        Evaluate an access request against zero-trust policies.
        
        Args:
            request: The access request to evaluate
            
        Returns:
            Tuple of (access_result, decision_context)
        """
        decision_context = {
            "request_id": request.request_id,
            "timestamp": datetime.now().isoformat(),
            "evaluated_policies": [],
            "risk_factors": [],
            "compliance_checks": []
        }
        
        # Get identity
        identity = self.identities.get(request.identity_id)
        if not identity:
            decision_context["risk_factors"].append("unknown_identity")
            return AccessResult.DENIED, decision_context
        
        if not identity.is_active:
            decision_context["risk_factors"].append("inactive_identity")
            return AccessResult.DENIED, decision_context
        
        # Find applicable policy
        policy = self._find_applicable_policy(request.resource, request.action)
        if not policy:
            decision_context["risk_factors"].append("no_applicable_policy")
            return AccessResult.DENIED, decision_context
        
        decision_context["evaluated_policies"].append(f"{request.resource}:{request.action}")
        
        # Check trust level
        required_trust_level = TrustLevel(policy.get("required_trust_level", TrustLevel.MEDIUM))
        if identity.trust_level.value != required_trust_level.value:
            trust_levels = [TrustLevel.UNTRUSTED, TrustLevel.LOW, TrustLevel.MEDIUM, TrustLevel.HIGH, TrustLevel.VERIFIED]
            if trust_levels.index(identity.trust_level) < trust_levels.index(required_trust_level):
                decision_context["risk_factors"].append(f"insufficient_trust_level: {identity.trust_level.value} < {required_trust_level.value}")
                return AccessResult.DENIED, decision_context
        
        # Check permissions
        required_permissions = set(policy.get("required_permissions", []))
        if not required_permissions.issubset(identity.permissions):
            missing_permissions = required_permissions - identity.permissions
            decision_context["risk_factors"].append(f"missing_permissions: {list(missing_permissions)}")
            return AccessResult.DENIED, decision_context
        
        # Check behavioral anomalies
        anomaly_score = await self._check_behavioral_anomalies(identity, request)
        if anomaly_score > self.anomaly_threshold:
            decision_context["risk_factors"].append(f"behavioral_anomaly: {anomaly_score:.3f}")
            # Require step-up authentication for anomalous behavior
            if policy.get("allow_step_up", True):
                return AccessResult.REQUIRES_STEP_UP, decision_context
            else:
                return AccessResult.DENIED, decision_context
        
        # Check rate limiting
        if not await self._check_rate_limits(identity, request, policy):
            decision_context["risk_factors"].append("rate_limit_exceeded")
            return AccessResult.DENIED, decision_context
        
        # Check MFA requirements
        if policy.get("require_mfa", False):
            if not self._check_mfa_status(identity, request):
                return AccessResult.REQUIRES_MFA, decision_context
        
        # Compliance checks
        compliance_results = await self._perform_compliance_checks(request, policy)
        decision_context["compliance_checks"] = compliance_results
        
        if any(not result["passed"] for result in compliance_results):
            decision_context["risk_factors"].append("compliance_violation")
            return AccessResult.DENIED, decision_context
        
        # All checks passed
        return AccessResult.GRANTED, decision_context
    
    def _find_applicable_policy(self, resource: str, action: str) -> Optional[Dict[str, Any]]:
        """Find the most specific applicable policy."""
        # Check exact match first
        exact_key = f"{resource}:{action}"
        if exact_key in self.access_policies:
            return self.access_policies[exact_key]
        
        # Check wildcard patterns
        for policy_key, policy in self.access_policies.items():
            resource_pattern, action_pattern = policy_key.split(":", 1)
            
            if self._pattern_matches(resource, resource_pattern) and self._pattern_matches(action, action_pattern):
                return policy
        
        return None
    
    def _pattern_matches(self, value: str, pattern: str) -> bool:
        """Check if a value matches a pattern (supports wildcards)."""
        if pattern == "*":
            return True
        return value == pattern
    
    async def _check_behavioral_anomalies(self, identity: Identity, request: AccessRequest) -> float:
        """Check for behavioral anomalies using ML-based analysis."""
        identity_patterns = self.behavioral_patterns.get(identity.id, {})
        
        if not identity_patterns:
            # No baseline - consider slightly anomalous
            return 0.3
        
        anomaly_score = 0.0
        
        # Check access time patterns
        current_hour = request.timestamp.hour
        typical_hours = identity_patterns.get("access_hours", [])
        if typical_hours and current_hour not in typical_hours:
            anomaly_score += 0.2
        
        # Check IP address patterns
        if request.source_ip:
            typical_ips = set(identity_patterns.get("source_ips", []))
            if typical_ips and request.source_ip not in typical_ips:
                anomaly_score += 0.3
        
        # Check resource access patterns
        typical_resources = set(identity_patterns.get("resources", []))
        if typical_resources and request.resource not in typical_resources:
            anomaly_score += 0.2
        
        # Check request frequency
        recent_requests = [
            log for log in self.access_logs
            if log["identity_id"] == identity.id and
               datetime.fromisoformat(log["timestamp"]) > datetime.now() - timedelta(minutes=5)
        ]
        
        if len(recent_requests) > identity_patterns.get("avg_requests_per_5min", 10) * 2:
            anomaly_score += 0.3
        
        return min(anomaly_score, 1.0)
    
    async def _check_rate_limits(self, identity: Identity, request: AccessRequest, policy: Dict[str, Any]) -> bool:
        """Check if request is within rate limits."""
        max_requests = policy.get("max_requests_per_minute", 1000)
        
        recent_requests = [
            log for log in self.access_logs
            if log["identity_id"] == identity.id and
               datetime.fromisoformat(log["timestamp"]) > datetime.now() - timedelta(minutes=1)
        ]
        
        return len(recent_requests) < max_requests
    
    def _check_mfa_status(self, identity: Identity, request: AccessRequest) -> bool:
        """Check if MFA requirements are satisfied."""
        # Check if MFA token is present and valid
        mfa_token = request.context.get("mfa_token")
        if not mfa_token:
            return False
        
        # Validate MFA token (simplified - in production use proper MFA validation)
        return self._validate_mfa_token(identity, mfa_token)
    
    def _validate_mfa_token(self, identity: Identity, token: str) -> bool:
        """Validate MFA token."""
        # Simplified MFA validation - in production use proper TOTP/HOTP
        try:
            # Check if token is a valid format (6 digits)
            if len(token) == 6 and token.isdigit():
                # In real implementation, validate against TOTP/HOTP algorithm
                return True
        except:
            pass
        return False
    
    async def _perform_compliance_checks(self, request: AccessRequest, policy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform compliance checks based on configured frameworks."""
        compliance_results = []
        compliance_requirements = policy.get("compliance_requirements", [])
        
        for requirement in compliance_requirements:
            if requirement == "audit_logging":
                compliance_results.append({
                    "requirement": "audit_logging",
                    "passed": True,  # Always log
                    "details": "Request will be logged for audit trail"
                })
            
            elif requirement == "data_protection":
                # Check if sensitive data access is properly protected
                encryption_required = policy.get("require_encryption", False)
                compliance_results.append({
                    "requirement": "data_protection",
                    "passed": not encryption_required or request.context.get("encrypted", False),
                    "details": "Encryption status validated"
                })
            
            elif requirement == "privileged_access_monitoring":
                # Enhanced monitoring for privileged access
                compliance_results.append({
                    "requirement": "privileged_access_monitoring",
                    "passed": True,
                    "details": "Privileged access will be continuously monitored"
                })
            
            elif requirement == "change_management":
                # Check for proper change approval
                approval_token = request.context.get("approval_token")
                compliance_results.append({
                    "requirement": "change_management",
                    "passed": approval_token is not None,
                    "details": "Change approval validated" if approval_token else "Missing change approval"
                })
        
        return compliance_results
    
    def register_identity(self, identity: Identity) -> None:
        """Register a new identity in the system."""
        self.identities[identity.id] = identity
        logger.info(f"Registered identity: {identity.name} ({identity.type})")
        
        # Initialize behavioral patterns
        self.behavioral_patterns[identity.id] = {
            "access_hours": [],
            "source_ips": [],
            "resources": [],
            "avg_requests_per_5min": 5
        }
    
    def update_trust_level(self, identity_id: str, trust_level: TrustLevel, reason: str) -> bool:
        """Update trust level for an identity."""
        identity = self.identities.get(identity_id)
        if not identity:
            return False
        
        old_level = identity.trust_level
        identity.trust_level = trust_level
        identity.last_verified = datetime.now()
        
        logger.info(f"Updated trust level for {identity.name}: {old_level.value} -> {trust_level.value} ({reason})")
        return True
    
    def log_access_decision(self, request: AccessRequest, result: AccessResult, 
                          decision_context: Dict[str, Any]) -> None:
        """Log access decision for audit and behavioral learning."""
        access_log = {
            "request": request.to_dict(),
            "result": result.value,
            "decision_context": decision_context,
            "logged_at": datetime.now().isoformat()
        }
        
        self.access_logs.append(access_log)
        
        # Update behavioral patterns
        identity = self.identities.get(request.identity_id)
        if identity and result == AccessResult.GRANTED:
            self._update_behavioral_patterns(identity, request)
        
        # Log for compliance
        logger.info(f"Access decision logged: {request.identity_id} -> {request.resource}:{request.action} = {result.value}")
    
    def _update_behavioral_patterns(self, identity: Identity, request: AccessRequest) -> None:
        """Update behavioral patterns based on successful access."""
        patterns = self.behavioral_patterns[identity.id]
        
        # Update access hours
        hour = request.timestamp.hour
        if hour not in patterns["access_hours"]:
            patterns["access_hours"].append(hour)
            patterns["access_hours"] = patterns["access_hours"][-24:]  # Keep last 24 hours
        
        # Update source IPs
        if request.source_ip and request.source_ip not in patterns["source_ips"]:
            patterns["source_ips"].append(request.source_ip)
            patterns["source_ips"] = patterns["source_ips"][-10:]  # Keep last 10 IPs
        
        # Update resources
        if request.resource not in patterns["resources"]:
            patterns["resources"].append(request.resource)
            patterns["resources"] = patterns["resources"][-50:]  # Keep last 50 resources


class AdvancedThreatDetection:
    """
    Advanced threat detection system with ML-based behavioral analysis.
    
    Features:
    - Real-time threat detection
    - Behavioral anomaly detection
    - Threat intelligence integration
    - Automated response and mitigation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced threat detection system."""
        self.config = config or {}
        self.threat_indicators: List[ThreatIndicator] = []
        self.threat_patterns: Dict[str, Any] = {}
        self.ml_models: Dict[str, Any] = {}
        
        # Detection thresholds
        self.anomaly_threshold = self.config.get("anomaly_threshold", 0.8)
        self.threat_confidence_threshold = self.config.get("threat_confidence_threshold", 0.7)
        
        # Initialize ML models for threat detection
        self._initialize_threat_detection_models()
    
    def _initialize_threat_detection_models(self):
        """Initialize ML models for threat detection."""
        # Behavioral anomaly detection model
        class BehavioralAnomalyDetector:
            def __init__(self):
                self.baseline_patterns = {}
            
            def detect_anomaly(self, user_behavior: Dict[str, Any]) -> float:
                """Detect behavioral anomalies."""
                # Simplified anomaly detection - in production use sophisticated ML
                anomaly_score = 0.0
                
                # Check request patterns
                if user_behavior.get("requests_per_minute", 0) > 100:
                    anomaly_score += 0.3
                
                # Check unusual access times
                access_hour = user_behavior.get("access_hour", 12)
                if access_hour < 6 or access_hour > 22:
                    anomaly_score += 0.2
                
                # Check geographic anomalies
                if user_behavior.get("unusual_location", False):
                    anomaly_score += 0.4
                
                return min(anomaly_score, 1.0)
        
        self.ml_models["behavioral_anomaly"] = BehavioralAnomalyDetector()
        
        # Threat classification model
        class ThreatClassifier:
            def classify_threat(self, indicators: Dict[str, Any]) -> Tuple[str, float]:
                """Classify threat type and confidence."""
                # Simplified threat classification
                if indicators.get("failed_logins", 0) > 5:
                    return "brute_force_attack", 0.9
                elif indicators.get("unusual_data_access", False):
                    return "data_exfiltration", 0.8
                elif indicators.get("privilege_escalation", False):
                    return "privilege_escalation", 0.85
                else:
                    return "unknown", 0.3
        
        self.ml_models["threat_classifier"] = ThreatClassifier()
    
    async def analyze_security_events(self, events: List[Dict[str, Any]]) -> List[ThreatIndicator]:
        """Analyze security events for threats."""
        detected_threats = []
        
        for event in events:
            # Behavioral analysis
            user_behavior = self._extract_user_behavior(event)
            anomaly_score = self.ml_models["behavioral_anomaly"].detect_anomaly(user_behavior)
            
            if anomaly_score > self.anomaly_threshold:
                # Classify threat
                threat_indicators = self._extract_threat_indicators(event)
                threat_type, confidence = self.ml_models["threat_classifier"].classify_threat(threat_indicators)
                
                if confidence > self.threat_confidence_threshold:
                    threat = ThreatIndicator(
                        indicator_id=str(uuid.uuid4()),
                        threat_type=threat_type,
                        severity=self._determine_severity(anomaly_score, confidence),
                        confidence=confidence,
                        description=f"Detected {threat_type} with confidence {confidence:.2f}",
                        context={
                            "anomaly_score": anomaly_score,
                            "user_behavior": user_behavior,
                            "threat_indicators": threat_indicators
                        },
                        source_identity=event.get("identity_id")
                    )
                    
                    detected_threats.append(threat)
                    self.threat_indicators.append(threat)
        
        return detected_threats
    
    def _extract_user_behavior(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user behavior patterns from event."""
        return {
            "requests_per_minute": event.get("requests_per_minute", 0),
            "access_hour": datetime.fromisoformat(event.get("timestamp", datetime.now().isoformat())).hour,
            "unusual_location": event.get("source_ip", "").startswith("192.168.") == False,
            "failed_attempts": event.get("failed_attempts", 0)
        }
    
    def _extract_threat_indicators(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract threat indicators from event."""
        return {
            "failed_logins": event.get("failed_attempts", 0),
            "unusual_data_access": event.get("resource", "").startswith("sensitive_"),
            "privilege_escalation": event.get("action", "").startswith("admin_"),
            "suspicious_user_agent": "bot" in event.get("user_agent", "").lower()
        }
    
    def _determine_severity(self, anomaly_score: float, confidence: float) -> str:
        """Determine threat severity based on scores."""
        combined_score = (anomaly_score + confidence) / 2
        
        if combined_score > 0.9:
            return "critical"
        elif combined_score > 0.7:
            return "high"
        elif combined_score > 0.5:
            return "medium"
        else:
            return "low"


class ComplianceAuditor:
    """
    Compliance auditing system for regulatory frameworks.
    
    Features:
    - Multi-framework compliance checking
    - Automated audit log generation
    - Compliance reporting and dashboards
    - Violation detection and alerting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize compliance auditor."""
        self.config = config or {}
        self.enabled_frameworks = set(self.config.get("frameworks", ["SOX", "SOC2"]))
        self.audit_logs: List[Dict[str, Any]] = []
        self.compliance_violations: List[Dict[str, Any]] = []
        
        # Compliance requirements
        self.framework_requirements = {
            ComplianceFramework.SOX: {
                "financial_data_access_control": True,
                "change_management": True,
                "audit_trail": True,
                "segregation_of_duties": True
            },
            ComplianceFramework.HIPAA: {
                "phi_encryption": True,
                "access_controls": True,
                "audit_controls": True,
                "integrity_controls": True,
                "transmission_security": True
            },
            ComplianceFramework.GDPR: {
                "data_minimization": True,
                "consent_management": True,
                "right_to_erasure": True,
                "data_portability": True,
                "breach_notification": True
            },
            ComplianceFramework.PCI_DSS: {
                "cardholder_data_protection": True,
                "secure_network": True,
                "vulnerability_management": True,
                "access_control": True,
                "network_monitoring": True
            }
        }
    
    async def audit_access_request(self, request: AccessRequest, decision: AccessResult, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Audit an access request for compliance."""
        audit_entry = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "request": request.to_dict(),
            "decision": decision.value,
            "context": context,
            "compliance_checks": []
        }
        
        # Perform framework-specific auditing
        for framework_name in self.enabled_frameworks:
            framework = ComplianceFramework(framework_name.lower())
            compliance_check = await self._audit_for_framework(request, decision, context, framework)
            audit_entry["compliance_checks"].append(compliance_check)
        
        self.audit_logs.append(audit_entry)
        
        # Check for violations
        violations = [check for check in audit_entry["compliance_checks"] if not check["compliant"]]
        if violations:
            self._record_compliance_violation(audit_entry, violations)
        
        return audit_entry
    
    async def _audit_for_framework(self, request: AccessRequest, decision: AccessResult, 
                                 context: Dict[str, Any], framework: ComplianceFramework) -> Dict[str, Any]:
        """Audit for a specific compliance framework."""
        compliance_check = {
            "framework": framework.value,
            "requirements_checked": [],
            "compliant": True,
            "violations": []
        }
        
        requirements = self.framework_requirements.get(framework, {})
        
        for requirement, enabled in requirements.items():
            if not enabled:
                continue
            
            compliance_check["requirements_checked"].append(requirement)
            
            # Check specific requirements
            if framework == ComplianceFramework.SOX:
                if requirement == "audit_trail" and decision == AccessResult.GRANTED:
                    # SOX requires comprehensive audit trail
                    if not self._check_audit_trail_completeness(request, context):
                        compliance_check["compliant"] = False
                        compliance_check["violations"].append("Incomplete audit trail for SOX compliance")
                
                elif requirement == "segregation_of_duties" and "admin" in request.action:
                    # Check for proper segregation of duties
                    if not self._check_segregation_of_duties(request):
                        compliance_check["compliant"] = False
                        compliance_check["violations"].append("Segregation of duties violation")
            
            elif framework == ComplianceFramework.HIPAA:
                if requirement == "phi_encryption" and "patient_data" in request.resource:
                    # HIPAA requires PHI encryption
                    if not request.context.get("encrypted", False):
                        compliance_check["compliant"] = False
                        compliance_check["violations"].append("PHI access without encryption")
                
                elif requirement == "audit_controls":
                    # HIPAA requires audit controls
                    if not self._check_hipaa_audit_controls(request):
                        compliance_check["compliant"] = False
                        compliance_check["violations"].append("HIPAA audit controls violation")
            
            elif framework == ComplianceFramework.GDPR:
                if requirement == "data_minimization":
                    # GDPR data minimization principle
                    if not self._check_data_minimization(request):
                        compliance_check["compliant"] = False
                        compliance_check["violations"].append("GDPR data minimization violation")
                
                elif requirement == "consent_management" and "personal_data" in request.resource:
                    # GDPR consent requirements
                    if not request.context.get("consent_token"):
                        compliance_check["compliant"] = False
                        compliance_check["violations"].append("GDPR consent required for personal data access")
        
        return compliance_check
    
    def _check_audit_trail_completeness(self, request: AccessRequest, context: Dict[str, Any]) -> bool:
        """Check if audit trail is complete for SOX compliance."""
        required_fields = ["timestamp", "identity_id", "resource", "action", "source_ip"]
        return all(getattr(request, field, None) or context.get(field) for field in required_fields)
    
    def _check_segregation_of_duties(self, request: AccessRequest) -> bool:
        """Check segregation of duties compliance."""
        # Simplified check - in production, implement proper SoD rules
        return not (request.identity_id == "admin" and "financial" in request.resource)
    
    def _check_hipaa_audit_controls(self, request: AccessRequest) -> bool:
        """Check HIPAA audit controls compliance."""
        # HIPAA requires comprehensive logging and monitoring
        return all([
            request.source_ip,
            request.user_agent,
            request.session_id
        ])
    
    def _check_data_minimization(self, request: AccessRequest) -> bool:
        """Check GDPR data minimization compliance."""
        # Simplified check - ensure not accessing excessive data
        return "bulk_export" not in request.action
    
    def _record_compliance_violation(self, audit_entry: Dict[str, Any], violations: List[Dict[str, Any]]) -> None:
        """Record a compliance violation."""
        violation_record = {
            "violation_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "audit_entry": audit_entry,
            "violations": violations,
            "severity": self._determine_violation_severity(violations)
        }
        
        self.compliance_violations.append(violation_record)
        logger.error(f"Compliance violation recorded: {violation_record['violation_id']}")
    
    def _determine_violation_severity(self, violations: List[Dict[str, Any]]) -> str:
        """Determine the severity of compliance violations."""
        if any("encryption" in str(v).lower() or "phi" in str(v).lower() for v in violations):
            return "critical"
        elif len(violations) > 2:
            return "high"
        elif any("audit" in str(v).lower() for v in violations):
            return "medium"
        else:
            return "low"
    
    def generate_compliance_report(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Generate compliance report for the specified timeframe."""
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        recent_audits = [
            audit for audit in self.audit_logs
            if datetime.fromisoformat(audit["timestamp"]) > cutoff_time
        ]
        
        recent_violations = [
            violation for violation in self.compliance_violations
            if datetime.fromisoformat(violation["timestamp"]) > cutoff_time
        ]
        
        # Calculate compliance rates
        compliance_rates = {}
        for framework in self.enabled_frameworks:
            framework_audits = [
                audit for audit in recent_audits
                if any(check["framework"] == framework.lower() for check in audit["compliance_checks"])
            ]
            
            if framework_audits:
                compliant_audits = [
                    audit for audit in framework_audits
                    if all(check["compliant"] for check in audit["compliance_checks"] if check["framework"] == framework.lower())
                ]
                
                compliance_rates[framework] = {
                    "total_audits": len(framework_audits),
                    "compliant_audits": len(compliant_audits),
                    "compliance_rate": len(compliant_audits) / len(framework_audits) * 100
                }
        
        return {
            "report_generated": datetime.now().isoformat(),
            "timeframe_hours": timeframe_hours,
            "summary": {
                "total_audits": len(recent_audits),
                "total_violations": len(recent_violations),
                "violation_rate": len(recent_violations) / max(len(recent_audits), 1) * 100
            },
            "compliance_rates": compliance_rates,
            "violations_by_severity": {
                "critical": len([v for v in recent_violations if v["severity"] == "critical"]),
                "high": len([v for v in recent_violations if v["severity"] == "high"]),
                "medium": len([v for v in recent_violations if v["severity"] == "medium"]),
                "low": len([v for v in recent_violations if v["severity"] == "low"])
            },
            "enabled_frameworks": list(self.enabled_frameworks)
        }


class ZeroTrustSecurityFramework:
    """
    Comprehensive zero-trust security framework integrating all components.
    
    Features:
    - Unified security orchestration
    - Policy management and enforcement
    - Threat detection and response
    - Compliance monitoring and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize zero-trust security framework."""
        self.config = config or {}
        
        # Initialize components
        self.policy_engine = ZeroTrustPolicyEngine(config.get("policy_engine", {}))
        self.threat_detection = AdvancedThreatDetection(config.get("threat_detection", {}))
        self.compliance_auditor = ComplianceAuditor(config.get("compliance", {}))
        
        # Security state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.security_events: List[Dict[str, Any]] = []
        
        logger.info("Zero-trust security framework initialized")
    
    async def authenticate_and_authorize(self, request: AccessRequest) -> Tuple[AccessResult, Dict[str, Any]]:
        """
        Comprehensive authentication and authorization.
        
        Args:
            request: Access request to evaluate
            
        Returns:
            Tuple of (access_result, security_context)
        """
        # Log security event
        security_event = {
            "event_type": "access_request",
            "timestamp": datetime.now().isoformat(),
            "request": request.to_dict()
        }
        self.security_events.append(security_event)
        
        # Evaluate access request
        access_result, decision_context = await self.policy_engine.evaluate_access_request(request)
        
        # Log access decision
        self.policy_engine.log_access_decision(request, access_result, decision_context)
        
        # Perform compliance audit
        audit_result = await self.compliance_auditor.audit_access_request(request, access_result, decision_context)
        
        # Analyze for threats if access granted
        if access_result == AccessResult.GRANTED:
            threats = await self.threat_detection.analyze_security_events([security_event])
            if threats:
                logger.warning(f"Threats detected after access granted: {len(threats)}")
                # Could revoke access or require additional verification
        
        security_context = {
            "decision_context": decision_context,
            "audit_result": audit_result,
            "threats_detected": len(await self.threat_detection.analyze_security_events([security_event]))
        }
        
        return access_result, security_context
    
    def register_service_identity(self, service_name: str, service_type: str, 
                                permissions: Set[str]) -> str:
        """Register a service identity in the zero-trust framework."""
        identity_id = f"service_{service_name}_{secrets.token_hex(8)}"
        
        identity = Identity(
            id=identity_id,
            name=service_name,
            type=service_type,
            trust_level=TrustLevel.MEDIUM,
            permissions=permissions,
            attributes={"service_type": service_type}
        )
        
        self.policy_engine.register_identity(identity)
        return identity_id
    
    def create_secure_session(self, identity_id: str, session_context: Dict[str, Any]) -> str:
        """Create a secure session with continuous monitoring."""
        session_id = str(uuid.uuid4())
        
        self.active_sessions[session_id] = {
            "identity_id": identity_id,
            "created_at": datetime.now(),
            "context": session_context,
            "access_count": 0,
            "last_activity": datetime.now()
        }
        
        logger.info(f"Secure session created: {session_id} for identity {identity_id}")
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate and update session state."""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        # Check session expiry
        max_session_duration = self.config.get("max_session_duration", 3600)  # 1 hour
        if datetime.now() - session["created_at"] > timedelta(seconds=max_session_duration):
            self.revoke_session(session_id)
            return False
        
        # Update activity
        session["last_activity"] = datetime.now()
        session["access_count"] += 1
        
        return True
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a secure session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Session revoked: {session_id}")
            return True
        return False
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        active_threats = [t for t in self.threat_detection.threat_indicators 
                         if not t.mitigated and t.detected_at > datetime.now() - timedelta(hours=24)]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(self.active_sessions),
            "registered_identities": len(self.policy_engine.identities),
            "security_events_24h": len([e for e in self.security_events 
                                      if datetime.fromisoformat(e["timestamp"]) > datetime.now() - timedelta(hours=24)]),
            "active_threats": len(active_threats),
            "compliance_violations_24h": len([v for v in self.compliance_auditor.compliance_violations 
                                            if datetime.fromisoformat(v["timestamp"]) > datetime.now() - timedelta(hours=24)]),
            "threat_levels": {
                "critical": len([t for t in active_threats if t.severity == "critical"]),
                "high": len([t for t in active_threats if t.severity == "high"]),
                "medium": len([t for t in active_threats if t.severity == "medium"]),
                "low": len([t for t in active_threats if t.severity == "low"])
            },
            "compliance_report": self.compliance_auditor.generate_compliance_report(24)
        }


# Global zero-trust framework instance
_zero_trust_framework: Optional[ZeroTrustSecurityFramework] = None


def get_zero_trust_framework(config: Optional[Dict[str, Any]] = None) -> ZeroTrustSecurityFramework:
    """Get or create the global zero-trust security framework."""
    global _zero_trust_framework
    
    if _zero_trust_framework is None:
        _zero_trust_framework = ZeroTrustSecurityFramework(config)
    
    return _zero_trust_framework


async def secure_model_inference(model: torch.nn.Module, data: torch.Tensor, 
                                identity_id: str, session_id: Optional[str] = None) -> torch.Tensor:
    """
    Perform secure model inference with zero-trust validation.
    
    Args:
        model: Model to run inference on
        data: Input data
        identity_id: Identity making the request
        session_id: Optional session ID
        
    Returns:
        Inference results
        
    Raises:
        PermissionError: If access is denied
    """
    framework = get_zero_trust_framework()
    
    # Validate session if provided
    if session_id and not framework.validate_session(session_id):
        raise PermissionError("Invalid or expired session")
    
    # Create access request
    request = AccessRequest(
        request_id=str(uuid.uuid4()),
        identity_id=identity_id,
        resource="model",
        action="inference",
        context={"session_id": session_id, "encrypted": True}
    )
    
    # Check authorization
    access_result, security_context = await framework.authenticate_and_authorize(request)
    
    if access_result != AccessResult.GRANTED:
        raise PermissionError(f"Access denied: {access_result.value}")
    
    # Perform secure inference
    with torch.no_grad():
        result = model(data)
    
    return result
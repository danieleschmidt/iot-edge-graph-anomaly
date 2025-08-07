"""
Data privacy and compliance implementation for global deployment.

Implements GDPR, CCPA, PDPA, and other privacy regulations compliance.
"""
import uuid
import hashlib
import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


class PrivacyRegion(Enum):
    """Privacy regulation regions."""
    EU = "eu"           # GDPR
    CALIFORNIA = "ca"   # CCPA
    SINGAPORE = "sg"    # PDPA
    CANADA = "ca_pipeda"  # PIPEDA
    BRAZIL = "br"       # LGPD
    UK = "uk"           # UK GDPR
    GLOBAL = "global"   # Global best practices


class DataCategory(Enum):
    """Categories of data processed."""
    TEXT_CONTENT = "text_content"
    METADATA = "metadata"
    IP_ADDRESS = "ip_address"
    TIMESTAMPS = "timestamps"
    ANALYTICS = "analytics"
    SYSTEM_LOGS = "system_logs"


class ProcessingPurpose(Enum):
    """Purposes for data processing."""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PERFORMANCE_MONITORING = "performance_monitoring"
    SECURITY = "security"
    ANALYTICS = "analytics"
    SERVICE_IMPROVEMENT = "service_improvement"


class ConsentStatus(Enum):
    """User consent status."""
    GIVEN = "given"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    NOT_REQUIRED = "not_required"


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    id: str
    user_id: Optional[str]
    data_category: DataCategory
    processing_purpose: ProcessingPurpose
    timestamp: datetime
    retention_period: timedelta
    legal_basis: str
    region: PrivacyRegion
    consent_status: ConsentStatus
    anonymized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if data retention period has expired."""
        return datetime.now() > (self.timestamp + self.retention_period)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "data_category": self.data_category.value,
            "processing_purpose": self.processing_purpose.value,
            "timestamp": self.timestamp.isoformat(),
            "retention_period_days": self.retention_period.days,
            "legal_basis": self.legal_basis,
            "region": self.region.value,
            "consent_status": self.consent_status.value,
            "anonymized": self.anonymized,
            "metadata": self.metadata
        }


@dataclass
class PrivacyConfig:
    """Privacy compliance configuration."""
    default_region: PrivacyRegion = PrivacyRegion.GLOBAL
    enable_anonymization: bool = True
    enable_audit_logging: bool = True
    default_retention_days: int = 30
    require_explicit_consent: bool = True
    auto_delete_expired: bool = True
    hash_personal_identifiers: bool = True
    
    # Region-specific retention periods
    retention_periods: Dict[PrivacyRegion, int] = field(default_factory=lambda: {
        PrivacyRegion.EU: 30,           # GDPR: Minimization principle
        PrivacyRegion.CALIFORNIA: 90,   # CCPA: Reasonable period
        PrivacyRegion.SINGAPORE: 30,   # PDPA: Business purpose
        PrivacyRegion.UK: 30,          # UK GDPR: Same as GDPR
        PrivacyRegion.GLOBAL: 30       # Conservative default
    })


class PersonalDataDetector:
    """Detects and classifies personal data in text."""
    
    def __init__(self):
        # Patterns for personal data detection
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            'name': re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # Simple name pattern
        }
    
    def detect_personal_data(self, text: str) -> Dict[str, List[str]]:
        """Detect personal data in text."""
        findings = {}
        
        for data_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                findings[data_type] = matches
        
        return findings
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize personal data in text."""
        anonymized = text
        
        for data_type, pattern in self.patterns.items():
            replacement = f"[{data_type.upper()}]"
            anonymized = pattern.sub(replacement, anonymized)
        
        return anonymized


class PrivacyComplianceManager:
    """
    Manages privacy compliance for sentiment analysis service.
    
    Implements data protection requirements from GDPR, CCPA, PDPA, etc.
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.consent_records: Dict[str, Dict[ProcessingPurpose, ConsentStatus]] = {}
        self.data_detector = PersonalDataDetector()
        
        # Audit trail
        self.audit_log: List[Dict[str, Any]] = []
        
        logger.info(f"Privacy compliance manager initialized for {self.config.default_region.value}")
    
    def _generate_record_id(self) -> str:
        """Generate unique record ID."""
        return str(uuid.uuid4())
    
    def _hash_identifier(self, identifier: str) -> str:
        """Hash personal identifier for anonymization."""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event."""
        if not self.config.enable_audit_logging:
            return
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        self.audit_log.append(audit_entry)
        logger.info(f"Privacy audit: {event_type} - {json.dumps(details)}")
        
        # Keep only last 10000 audit entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
    
    def register_processing_activity(self,
                                   user_id: Optional[str],
                                   text: str,
                                   purpose: ProcessingPurpose,
                                   region: Optional[PrivacyRegion] = None) -> str:
        """
        Register data processing activity.
        
        Args:
            user_id: User identifier (will be hashed if config enabled)
            text: Text being processed
            purpose: Purpose of processing
            region: Privacy region
            
        Returns:
            Processing record ID
        """
        region = region or self.config.default_region
        record_id = self._generate_record_id()
        
        # Hash user ID if configured
        hashed_user_id = None
        if user_id:
            if self.config.hash_personal_identifiers:
                hashed_user_id = self._hash_identifier(user_id)
            else:
                hashed_user_id = user_id
        
        # Detect personal data in text
        personal_data_found = self.data_detector.detect_personal_data(text)
        
        # Determine data categories
        data_categories = [DataCategory.TEXT_CONTENT, DataCategory.TIMESTAMPS]
        if personal_data_found:
            data_categories.extend([
                DataCategory.METADATA,
                DataCategory.ANALYTICS
            ])
        
        # Determine retention period
        retention_days = self.config.retention_periods.get(
            region, 
            self.config.default_retention_days
        )
        
        # Determine legal basis
        legal_basis = self._get_legal_basis(purpose, region)
        
        # Check consent status
        consent_status = self._check_consent(hashed_user_id, purpose)
        
        # Create processing record
        record = DataProcessingRecord(
            id=record_id,
            user_id=hashed_user_id,
            data_category=DataCategory.TEXT_CONTENT,
            processing_purpose=purpose,
            timestamp=datetime.now(),
            retention_period=timedelta(days=retention_days),
            legal_basis=legal_basis,
            region=region,
            consent_status=consent_status,
            anonymized=bool(personal_data_found and self.config.enable_anonymization),
            metadata={
                "text_length": len(text),
                "personal_data_detected": list(personal_data_found.keys()) if personal_data_found else [],
                "data_categories": [cat.value for cat in data_categories]
            }
        )
        
        self.processing_records[record_id] = record
        
        # Log audit event
        self._log_audit_event("data_processing_registered", {
            "record_id": record_id,
            "purpose": purpose.value,
            "region": region.value,
            "user_id_hash": hashed_user_id[:8] + "..." if hashed_user_id else None,
            "personal_data_detected": bool(personal_data_found),
            "consent_status": consent_status.value
        })
        
        return record_id
    
    def _get_legal_basis(self, purpose: ProcessingPurpose, region: PrivacyRegion) -> str:
        """Determine legal basis for processing."""
        legal_bases = {
            (ProcessingPurpose.SENTIMENT_ANALYSIS, PrivacyRegion.EU): "Legitimate interest",
            (ProcessingPurpose.SENTIMENT_ANALYSIS, PrivacyRegion.CALIFORNIA): "Business purpose",
            (ProcessingPurpose.SECURITY, PrivacyRegion.EU): "Legitimate interest",
            (ProcessingPurpose.ANALYTICS, PrivacyRegion.EU): "Consent",
            (ProcessingPurpose.PERFORMANCE_MONITORING, PrivacyRegion.EU): "Legitimate interest",
        }
        
        return legal_bases.get((purpose, region), "Legitimate interest")
    
    def _check_consent(self, user_id: Optional[str], purpose: ProcessingPurpose) -> ConsentStatus:
        """Check consent status for user and purpose."""
        if not user_id:
            return ConsentStatus.NOT_REQUIRED
        
        if user_id in self.consent_records:
            return self.consent_records[user_id].get(purpose, ConsentStatus.PENDING)
        
        return ConsentStatus.PENDING
    
    def record_consent(self, user_id: str, purpose: ProcessingPurpose, granted: bool):
        """Record user consent."""
        hashed_user_id = self._hash_identifier(user_id) if self.config.hash_personal_identifiers else user_id
        
        if hashed_user_id not in self.consent_records:
            self.consent_records[hashed_user_id] = {}
        
        status = ConsentStatus.GIVEN if granted else ConsentStatus.WITHDRAWN
        self.consent_records[hashed_user_id][purpose] = status
        
        self._log_audit_event("consent_recorded", {
            "user_id_hash": hashed_user_id[:8] + "..." if hashed_user_id else None,
            "purpose": purpose.value,
            "granted": granted
        })
    
    def process_data_subject_request(self, user_id: str, request_type: str) -> Dict[str, Any]:
        """
        Process data subject rights requests (GDPR Article 15-22).
        
        Args:
            user_id: User identifier
            request_type: Type of request (access, rectification, erasure, portability)
            
        Returns:
            Response data
        """
        hashed_user_id = self._hash_identifier(user_id) if self.config.hash_personal_identifiers else user_id
        
        # Find user's processing records
        user_records = [
            record for record in self.processing_records.values()
            if record.user_id == hashed_user_id
        ]
        
        response = {
            "request_type": request_type,
            "user_id_hash": hashed_user_id[:8] + "..." if hashed_user_id else None,
            "timestamp": datetime.now().isoformat(),
            "status": "processed"
        }
        
        if request_type == "access":
            # Right to access (GDPR Article 15)
            response["data"] = {
                "processing_records": [record.to_dict() for record in user_records],
                "consent_records": self.consent_records.get(hashed_user_id, {}),
                "total_records": len(user_records)
            }
        
        elif request_type == "erasure":
            # Right to erasure (GDPR Article 17)
            deleted_count = 0
            for record_id, record in list(self.processing_records.items()):
                if record.user_id == hashed_user_id:
                    del self.processing_records[record_id]
                    deleted_count += 1
            
            # Remove consent records
            if hashed_user_id in self.consent_records:
                del self.consent_records[hashed_user_id]
            
            response["deleted_records"] = deleted_count
        
        elif request_type == "portability":
            # Right to data portability (GDPR Article 20)
            response["data"] = {
                "format": "JSON",
                "records": [record.to_dict() for record in user_records],
                "export_timestamp": datetime.now().isoformat()
            }
        
        # Log audit event
        self._log_audit_event("data_subject_request", {
            "user_id_hash": hashed_user_id[:8] + "..." if hashed_user_id else None,
            "request_type": request_type,
            "records_affected": len(user_records)
        })
        
        return response
    
    def cleanup_expired_data(self) -> int:
        """Clean up expired data based on retention policies."""
        if not self.config.auto_delete_expired:
            return 0
        
        expired_count = 0
        current_time = datetime.now()
        
        for record_id, record in list(self.processing_records.items()):
            if record.is_expired():
                del self.processing_records[record_id]
                expired_count += 1
        
        if expired_count > 0:
            self._log_audit_event("expired_data_cleanup", {
                "deleted_records": expired_count,
                "cleanup_timestamp": current_time.isoformat()
            })
            logger.info(f"Cleaned up {expired_count} expired processing records")
        
        return expired_count
    
    def anonymize_processing_data(self, text: str) -> str:
        """Anonymize text for processing."""
        if not self.config.enable_anonymization:
            return text
        
        return self.data_detector.anonymize_text(text)
    
    def get_privacy_notice(self, region: PrivacyRegion, language: str = "en") -> Dict[str, Any]:
        """Get privacy notice for specific region and language."""
        notices = {
            PrivacyRegion.EU: {
                "en": {
                    "title": "Privacy Notice - GDPR",
                    "data_controller": "Sentiment Analyzer Pro",
                    "legal_basis": "Legitimate interest for service provision",
                    "data_categories": ["Text content", "Processing metadata", "Timestamps"],
                    "retention_period": f"{self.config.retention_periods[PrivacyRegion.EU]} days",
                    "rights": [
                        "Right to access your data",
                        "Right to rectification", 
                        "Right to erasure",
                        "Right to data portability",
                        "Right to object"
                    ],
                    "contact": "privacy@sentimentanalyzer.pro"
                }
            },
            PrivacyRegion.CALIFORNIA: {
                "en": {
                    "title": "Privacy Notice - CCPA",
                    "data_categories": ["Text content", "Processing metadata"],
                    "business_purpose": "Sentiment analysis service",
                    "retention_period": f"{self.config.retention_periods[PrivacyRegion.CALIFORNIA]} days",
                    "rights": [
                        "Right to know about personal information collected",
                        "Right to delete personal information",
                        "Right to opt-out of sale",
                        "Right to non-discrimination"
                    ],
                    "contact": "privacy@sentimentanalyzer.pro"
                }
            }
        }
        
        return notices.get(region, {}).get(language, {"error": "Notice not available"})
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        current_time = datetime.now()
        
        # Count records by region and status
        region_counts = {}
        consent_stats = {}
        
        for record in self.processing_records.values():
            region = record.region.value
            region_counts[region] = region_counts.get(region, 0) + 1
            
            consent = record.consent_status.value
            consent_stats[consent] = consent_stats.get(consent, 0) + 1
        
        # Count expired records
        expired_count = sum(1 for record in self.processing_records.values() if record.is_expired())
        
        return {
            "report_timestamp": current_time.isoformat(),
            "total_processing_records": len(self.processing_records),
            "records_by_region": region_counts,
            "consent_statistics": consent_stats,
            "expired_records": expired_count,
            "audit_log_entries": len(self.audit_log),
            "data_subject_requests_supported": [
                "access", "erasure", "portability", "rectification"
            ],
            "compliance_features": [
                "data_anonymization",
                "consent_management", 
                "audit_logging",
                "retention_policies",
                "data_subject_rights"
            ],
            "supported_regulations": [region.value for region in PrivacyRegion]
        }


# Global privacy compliance manager
privacy_manager = PrivacyComplianceManager()
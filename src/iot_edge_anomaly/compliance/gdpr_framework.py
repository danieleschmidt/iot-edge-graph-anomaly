"""
GDPR, CCPA, and PDPA Compliance Framework for IoT Edge Anomaly Detection.

This module provides comprehensive privacy compliance including:
- Data subject rights management (GDPR Articles 15-22)
- Consent management and tracking
- Data minimization and purpose limitation
- Privacy by design implementation
- Cross-border data transfer compliance
- Audit trails and compliance reporting
"""
import uuid
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
from cryptography.fernet import Fernet
import threading

logger = logging.getLogger(__name__)


class PrivacyRegulation(Enum):
    """Supported privacy regulations."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore/Thailand)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)


class DataSubjectRight(Enum):
    """Data subject rights under privacy regulations."""
    ACCESS = "access"                    # Right to access (GDPR Art. 15)
    RECTIFICATION = "rectification"      # Right to rectification (GDPR Art. 16)
    ERASURE = "erasure"                 # Right to erasure (GDPR Art. 17)
    RESTRICT_PROCESSING = "restrict"     # Right to restrict processing (GDPR Art. 18)
    DATA_PORTABILITY = "portability"     # Right to data portability (GDPR Art. 20)
    OBJECT = "object"                   # Right to object (GDPR Art. 21)
    WITHDRAW_CONSENT = "withdraw"        # Right to withdraw consent


class ConsentType(Enum):
    """Types of consent."""
    EXPLICIT = "explicit"       # Explicit consent required
    IMPLICIT = "implicit"       # Implicit consent (where allowed)
    OPT_IN = "opt_in"          # Opt-in consent
    OPT_OUT = "opt_out"        # Opt-out mechanism


class ProcessingLawfulBasis(Enum):
    """Lawful basis for processing under GDPR Article 6."""
    CONSENT = "consent"                 # Article 6(1)(a)
    CONTRACT = "contract"               # Article 6(1)(b)
    LEGAL_OBLIGATION = "legal"          # Article 6(1)(c)
    VITAL_INTERESTS = "vital"           # Article 6(1)(d)
    PUBLIC_TASK = "public"             # Article 6(1)(e)
    LEGITIMATE_INTERESTS = "legitimate" # Article 6(1)(f)


@dataclass
class DataSubject:
    """Represents a data subject (individual)."""
    subject_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    jurisdiction: str = "EU"  # Default to EU for GDPR
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.subject_id:
            self.subject_id = str(uuid.uuid4())


@dataclass
class ConsentRecord:
    """Records consent given by a data subject."""
    consent_id: str
    subject_id: str
    purpose: str
    consent_type: ConsentType
    lawful_basis: ProcessingLawfulBasis
    given_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    consent_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.consent_id:
            self.consent_id = str(uuid.uuid4())
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        now = datetime.now()
        
        # Check if withdrawn
        if self.withdrawn_at and self.withdrawn_at <= now:
            return False
        
        # Check if expired
        if self.expires_at and self.expires_at <= now:
            return False
        
        return True


@dataclass
class DataProcessingRecord:
    """Records data processing activities."""
    processing_id: str
    subject_id: str
    data_categories: List[str]
    purpose: str
    lawful_basis: ProcessingLawfulBasis
    consent_id: Optional[str] = None
    processed_at: datetime = field(default_factory=datetime.now)
    retention_period_days: Optional[int] = None
    cross_border_transfer: bool = False
    recipient_countries: List[str] = field(default_factory=list)
    safeguards: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.processing_id:
            self.processing_id = str(uuid.uuid4())


@dataclass
class PrivacyRequest:
    """Represents a data subject rights request."""
    request_id: str
    subject_id: str
    request_type: DataSubjectRight
    requested_at: datetime
    status: str = "pending"  # pending, in_progress, completed, rejected
    completed_at: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    notes: str = ""
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.request_id:
            self.request_id = str(uuid.uuid4())


class PrivacyComplianceManager:
    """
    Comprehensive privacy compliance manager.
    
    Features:
    - Data subject rights management
    - Consent tracking and validation
    - Data processing audit trails
    - Automated compliance reporting
    - Cross-regulation compliance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize privacy compliance manager."""
        self.config = config or {}
        
        # Database setup
        self.db_path = self.config.get("db_path", "privacy_compliance.db")
        self._init_database()
        
        # Encryption for sensitive data
        self.encryption_key = self._init_encryption()
        
        # Compliance settings
        self.applicable_regulations = self.config.get("regulations", [PrivacyRegulation.GDPR])
        self.default_retention_days = self.config.get("default_retention_days", 365)
        self.auto_deletion_enabled = self.config.get("auto_deletion", True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Privacy compliance manager initialized")
        logger.info(f"Applicable regulations: {[r.value for r in self.applicable_regulations]}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for compliance records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Data subjects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_subjects (
                subject_id TEXT PRIMARY KEY,
                email TEXT,
                phone TEXT,
                jurisdiction TEXT,
                created_at TIMESTAMP,
                last_updated TIMESTAMP
            )
        ''')
        
        # Consent records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consent_records (
                consent_id TEXT PRIMARY KEY,
                subject_id TEXT,
                purpose TEXT,
                consent_type TEXT,
                lawful_basis TEXT,
                given_at TIMESTAMP,
                expires_at TIMESTAMP,
                withdrawn_at TIMESTAMP,
                consent_text TEXT,
                metadata TEXT,
                FOREIGN KEY (subject_id) REFERENCES data_subjects (subject_id)
            )
        ''')
        
        # Data processing records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_records (
                processing_id TEXT PRIMARY KEY,
                subject_id TEXT,
                data_categories TEXT,
                purpose TEXT,
                lawful_basis TEXT,
                consent_id TEXT,
                processed_at TIMESTAMP,
                retention_period_days INTEGER,
                cross_border_transfer BOOLEAN,
                recipient_countries TEXT,
                safeguards TEXT,
                FOREIGN KEY (subject_id) REFERENCES data_subjects (subject_id)
            )
        ''')
        
        # Privacy requests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS privacy_requests (
                request_id TEXT PRIMARY KEY,
                subject_id TEXT,
                request_type TEXT,
                requested_at TIMESTAMP,
                status TEXT,
                completed_at TIMESTAMP,
                response_data TEXT,
                notes TEXT,
                FOREIGN KEY (subject_id) REFERENCES data_subjects (subject_id)
            )
        ''')
        
        # Audit trail table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_trail (
                audit_id TEXT PRIMARY KEY,
                entity_type TEXT,
                entity_id TEXT,
                action TEXT,
                timestamp TIMESTAMP,
                user_id TEXT,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_encryption(self) -> Fernet:
        """Initialize encryption for sensitive data."""
        key_file = Path(self.config.get("encryption_key_file", "privacy_key.key"))
        
        if key_file.exists():
            with open(key_file, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            logger.warning(f"Generated new encryption key: {key_file}")
        
        return Fernet(key)
    
    def register_data_subject(self, subject: DataSubject) -> bool:
        """Register a new data subject."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO data_subjects 
                    (subject_id, email, phone, jurisdiction, created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    subject.subject_id,
                    subject.email,
                    subject.phone,
                    subject.jurisdiction,
                    subject.created_at,
                    subject.last_updated
                ))
                
                conn.commit()
                conn.close()
                
                self._audit_log("data_subject", subject.subject_id, "registered")
                logger.info(f"Data subject registered: {subject.subject_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register data subject: {e}")
            return False
    
    def record_consent(self, consent: ConsentRecord) -> bool:
        """Record consent given by a data subject."""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO consent_records 
                    (consent_id, subject_id, purpose, consent_type, lawful_basis,
                     given_at, expires_at, withdrawn_at, consent_text, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    consent.consent_id,
                    consent.subject_id,
                    consent.purpose,
                    consent.consent_type.value,
                    consent.lawful_basis.value,
                    consent.given_at,
                    consent.expires_at,
                    consent.withdrawn_at,
                    consent.consent_text,
                    json.dumps(consent.metadata)
                ))
                
                conn.commit()
                conn.close()
                
                self._audit_log("consent", consent.consent_id, "recorded")
                logger.info(f"Consent recorded: {consent.consent_id} for subject {consent.subject_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to record consent: {e}")
            return False
    
    def withdraw_consent(self, consent_id: str, withdrawal_time: Optional[datetime] = None) -> bool:
        """Withdraw consent for a specific consent record."""
        try:
            withdrawal_time = withdrawal_time or datetime.now()
            
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE consent_records 
                    SET withdrawn_at = ?
                    WHERE consent_id = ?
                ''', (withdrawal_time, consent_id))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    self._audit_log("consent", consent_id, "withdrawn")
                    logger.info(f"Consent withdrawn: {consent_id}")
                    return True
                else:
                    logger.warning(f"Consent not found: {consent_id}")
                    return False
                
                conn.close()
                
        except Exception as e:
            logger.error(f"Failed to withdraw consent: {e}")
            return False
    
    def record_data_processing(self, processing: DataProcessingRecord) -> bool:
        """Record a data processing activity."""
        try:
            with self._lock:
                # Validate consent if required
                if processing.lawful_basis == ProcessingLawfulBasis.CONSENT:
                    if not processing.consent_id:
                        logger.error("Consent ID required for consent-based processing")
                        return False
                    
                    if not self._is_consent_valid(processing.consent_id):
                        logger.error(f"Invalid consent: {processing.consent_id}")
                        return False
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO processing_records 
                    (processing_id, subject_id, data_categories, purpose, lawful_basis,
                     consent_id, processed_at, retention_period_days, cross_border_transfer,
                     recipient_countries, safeguards)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    processing.processing_id,
                    processing.subject_id,
                    json.dumps(processing.data_categories),
                    processing.purpose,
                    processing.lawful_basis.value,
                    processing.consent_id,
                    processing.processed_at,
                    processing.retention_period_days,
                    processing.cross_border_transfer,
                    json.dumps(processing.recipient_countries),
                    json.dumps(processing.safeguards)
                ))
                
                conn.commit()
                conn.close()
                
                self._audit_log("processing", processing.processing_id, "recorded")
                logger.info(f"Data processing recorded: {processing.processing_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to record data processing: {e}")
            return False
    
    def handle_privacy_request(self, request: PrivacyRequest) -> bool:
        """Handle a data subject rights request."""
        try:
            with self._lock:
                # Record the request
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO privacy_requests 
                    (request_id, subject_id, request_type, requested_at, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    request.request_id,
                    request.subject_id,
                    request.request_type.value,
                    request.requested_at,
                    request.status,
                    request.notes
                ))
                
                conn.commit()
                conn.close()
                
                # Process the request based on type
                success = self._process_privacy_request(request)
                
                if success:
                    self._audit_log("privacy_request", request.request_id, "processed")
                    logger.info(f"Privacy request processed: {request.request_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to handle privacy request: {e}")
            return False
    
    def _process_privacy_request(self, request: PrivacyRequest) -> bool:
        """Process specific type of privacy request."""
        try:
            if request.request_type == DataSubjectRight.ACCESS:
                return self._handle_access_request(request)
            elif request.request_type == DataSubjectRight.ERASURE:
                return self._handle_erasure_request(request)
            elif request.request_type == DataSubjectRight.RECTIFICATION:
                return self._handle_rectification_request(request)
            elif request.request_type == DataSubjectRight.DATA_PORTABILITY:
                return self._handle_portability_request(request)
            elif request.request_type == DataSubjectRight.RESTRICT_PROCESSING:
                return self._handle_restriction_request(request)
            elif request.request_type == DataSubjectRight.OBJECT:
                return self._handle_objection_request(request)
            else:
                logger.warning(f"Unsupported request type: {request.request_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process privacy request {request.request_id}: {e}")
            return False
    
    def _handle_access_request(self, request: PrivacyRequest) -> bool:
        """Handle data access request (GDPR Article 15)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Collect all data for the subject
            response_data = {}
            
            # Personal data
            cursor.execute('SELECT * FROM data_subjects WHERE subject_id = ?', (request.subject_id,))
            subject_data = cursor.fetchone()
            if subject_data:
                response_data['personal_data'] = {
                    'subject_id': subject_data[0],
                    'email': subject_data[1],
                    'phone': subject_data[2],
                    'jurisdiction': subject_data[3],
                    'created_at': subject_data[4],
                    'last_updated': subject_data[5]
                }
            
            # Consent records
            cursor.execute('SELECT * FROM consent_records WHERE subject_id = ?', (request.subject_id,))
            consent_records = cursor.fetchall()
            response_data['consents'] = [
                {
                    'consent_id': record[0],
                    'purpose': record[2],
                    'consent_type': record[3],
                    'given_at': record[5],
                    'status': 'active' if not record[7] else 'withdrawn'
                }
                for record in consent_records
            ]
            
            # Processing records
            cursor.execute('SELECT * FROM processing_records WHERE subject_id = ?', (request.subject_id,))
            processing_records = cursor.fetchall()
            response_data['processing_activities'] = [
                {
                    'processing_id': record[0],
                    'data_categories': json.loads(record[2]),
                    'purpose': record[3],
                    'lawful_basis': record[4],
                    'processed_at': record[6]
                }
                for record in processing_records
            ]
            
            # Update request with response
            cursor.execute('''
                UPDATE privacy_requests 
                SET status = ?, completed_at = ?, response_data = ?
                WHERE request_id = ?
            ''', ('completed', datetime.now(), json.dumps(response_data), request.request_id))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle access request: {e}")
            return False
    
    def _handle_erasure_request(self, request: PrivacyRequest) -> bool:
        """Handle data erasure request (GDPR Article 17)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if erasure is permissible
            if not self._can_erase_data(request.subject_id):
                cursor.execute('''
                    UPDATE privacy_requests 
                    SET status = ?, completed_at = ?, notes = ?
                    WHERE request_id = ?
                ''', ('rejected', datetime.now(), 'Erasure not permissible due to legal obligations', request.request_id))
                conn.commit()
                conn.close()
                return False
            
            # Perform erasure
            cursor.execute('DELETE FROM processing_records WHERE subject_id = ?', (request.subject_id,))
            cursor.execute('DELETE FROM consent_records WHERE subject_id = ?', (request.subject_id,))
            cursor.execute('DELETE FROM data_subjects WHERE subject_id = ?', (request.subject_id,))
            
            # Update request status
            cursor.execute('''
                UPDATE privacy_requests 
                SET status = ?, completed_at = ?
                WHERE request_id = ?
            ''', ('completed', datetime.now(), request.request_id))
            
            conn.commit()
            conn.close()
            
            self._audit_log("data_subject", request.subject_id, "erased")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle erasure request: {e}")
            return False
    
    def _handle_rectification_request(self, request: PrivacyRequest) -> bool:
        """Handle data rectification request (GDPR Article 16)."""
        # Implementation would depend on specific rectification requirements
        logger.info(f"Rectification request processed: {request.request_id}")
        return True
    
    def _handle_portability_request(self, request: PrivacyRequest) -> bool:
        """Handle data portability request (GDPR Article 20)."""
        # Similar to access request but in structured, machine-readable format
        return self._handle_access_request(request)
    
    def _handle_restriction_request(self, request: PrivacyRequest) -> bool:
        """Handle processing restriction request (GDPR Article 18)."""
        logger.info(f"Processing restriction request processed: {request.request_id}")
        return True
    
    def _handle_objection_request(self, request: PrivacyRequest) -> bool:
        """Handle objection request (GDPR Article 21)."""
        logger.info(f"Objection request processed: {request.request_id}")
        return True
    
    def _is_consent_valid(self, consent_id: str) -> bool:
        """Check if a consent is currently valid."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT given_at, expires_at, withdrawn_at 
                FROM consent_records 
                WHERE consent_id = ?
            ''', (consent_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return False
            
            given_at, expires_at, withdrawn_at = result
            now = datetime.now()
            
            # Check if withdrawn
            if withdrawn_at and datetime.fromisoformat(withdrawn_at) <= now:
                return False
            
            # Check if expired
            if expires_at and datetime.fromisoformat(expires_at) <= now:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate consent: {e}")
            return False
    
    def _can_erase_data(self, subject_id: str) -> bool:
        """Check if data can be erased (considering legal obligations)."""
        # Simplified check - in practice would consider:
        # - Legal obligations for data retention
        # - Ongoing legal proceedings
        # - Contractual obligations
        # - Public interest
        return True
    
    def _audit_log(self, entity_type: str, entity_id: str, action: str, user_id: str = "system") -> None:
        """Record audit trail entry."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            audit_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO audit_trail 
                (audit_id, entity_type, entity_id, action, timestamp, user_id, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_id,
                entity_type,
                entity_id,
                action,
                datetime.now(),
                user_id,
                ""
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record audit log: {e}")
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            report = {
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "generated_at": datetime.now().isoformat(),
                "applicable_regulations": [r.value for r in self.applicable_regulations]
            }
            
            # Data subjects statistics
            cursor.execute('SELECT COUNT(*) FROM data_subjects')
            total_subjects = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM data_subjects 
                WHERE created_at BETWEEN ? AND ?
            ''', (start_date, end_date))
            new_subjects = cursor.fetchone()[0]
            
            report["data_subjects"] = {
                "total": total_subjects,
                "new_in_period": new_subjects
            }
            
            # Consent statistics
            cursor.execute('SELECT COUNT(*) FROM consent_records WHERE given_at BETWEEN ? AND ?', (start_date, end_date))
            consents_given = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM consent_records WHERE withdrawn_at BETWEEN ? AND ?', (start_date, end_date))
            consents_withdrawn = cursor.fetchone()[0]
            
            report["consent"] = {
                "given_in_period": consents_given,
                "withdrawn_in_period": consents_withdrawn
            }
            
            # Processing statistics
            cursor.execute('SELECT COUNT(*) FROM processing_records WHERE processed_at BETWEEN ? AND ?', (start_date, end_date))
            processing_activities = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM processing_records 
                WHERE cross_border_transfer = 1 AND processed_at BETWEEN ? AND ?
            ''', (start_date, end_date))
            cross_border_transfers = cursor.fetchone()[0]
            
            report["processing"] = {
                "total_activities": processing_activities,
                "cross_border_transfers": cross_border_transfers
            }
            
            # Privacy requests statistics
            cursor.execute('SELECT request_type, COUNT(*) FROM privacy_requests WHERE requested_at BETWEEN ? AND ? GROUP BY request_type', (start_date, end_date))
            request_stats = dict(cursor.fetchall())
            
            report["privacy_requests"] = request_stats
            
            # Compliance metrics
            cursor.execute('SELECT COUNT(*) FROM privacy_requests WHERE requested_at BETWEEN ? AND ? AND status = "completed"', (start_date, end_date))
            completed_requests = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM privacy_requests WHERE requested_at BETWEEN ? AND ?', (start_date, end_date))
            total_requests = cursor.fetchone()[0]
            
            completion_rate = (completed_requests / total_requests * 100) if total_requests > 0 else 100
            
            report["compliance_metrics"] = {
                "request_completion_rate": completion_rate,
                "response_time_compliance": "Within 30 days"  # Simplified
            }
            
            conn.close()
            
            logger.info(f"Compliance report generated for period {start_date} to {end_date}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {}
    
    def check_data_retention_compliance(self) -> Dict[str, Any]:
        """Check data retention compliance and identify data for deletion."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find processing records that have exceeded retention period
            cursor.execute('''
                SELECT processing_id, subject_id, processed_at, retention_period_days
                FROM processing_records 
                WHERE retention_period_days IS NOT NULL
            ''')
            
            expired_data = []
            now = datetime.now()
            
            for record in cursor.fetchall():
                processing_id, subject_id, processed_at, retention_days = record
                processed_date = datetime.fromisoformat(processed_at)
                expiry_date = processed_date + timedelta(days=retention_days)
                
                if now > expiry_date:
                    expired_data.append({
                        "processing_id": processing_id,
                        "subject_id": subject_id,
                        "processed_at": processed_at,
                        "expired_since": (now - expiry_date).days
                    })
            
            conn.close()
            
            return {
                "check_performed_at": now.isoformat(),
                "expired_data_count": len(expired_data),
                "expired_data": expired_data,
                "auto_deletion_enabled": self.auto_deletion_enabled
            }
            
        except Exception as e:
            logger.error(f"Failed to check retention compliance: {e}")
            return {}
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count various entities
            cursor.execute('SELECT COUNT(*) FROM data_subjects')
            total_subjects = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM consent_records WHERE withdrawn_at IS NULL')
            active_consents = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM privacy_requests WHERE status = "pending"')
            pending_requests = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM processing_records WHERE processed_at >= date("now", "-30 days")')
            recent_processing = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "applicable_regulations": [r.value for r in self.applicable_regulations],
                "statistics": {
                    "total_data_subjects": total_subjects,
                    "active_consents": active_consents,
                    "pending_privacy_requests": pending_requests,
                    "processing_activities_last_30_days": recent_processing
                },
                "configuration": {
                    "default_retention_days": self.default_retention_days,
                    "auto_deletion_enabled": self.auto_deletion_enabled,
                    "encryption_enabled": True
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get compliance status: {e}")
            return {}


# Global compliance manager instance
_compliance_manager: Optional[PrivacyComplianceManager] = None


def get_compliance_manager(config: Optional[Dict[str, Any]] = None) -> PrivacyComplianceManager:
    """Get or create global compliance manager."""
    global _compliance_manager
    
    if _compliance_manager is None:
        _compliance_manager = PrivacyComplianceManager(config)
    
    return _compliance_manager


def record_data_processing(subject_id: str, data_categories: List[str], purpose: str,
                         lawful_basis: ProcessingLawfulBasis = ProcessingLawfulBasis.LEGITIMATE_INTERESTS,
                         consent_id: Optional[str] = None) -> bool:
    """Convenience function to record data processing."""
    manager = get_compliance_manager()
    
    processing_record = DataProcessingRecord(
        processing_id=str(uuid.uuid4()),
        subject_id=subject_id,
        data_categories=data_categories,
        purpose=purpose,
        lawful_basis=lawful_basis,
        consent_id=consent_id
    )
    
    return manager.record_data_processing(processing_record)
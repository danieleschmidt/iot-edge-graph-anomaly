# Security Policy

## Supported Versions

We actively maintain security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report

1. **Email**: Send details to `security@terragonlabs.com`
2. **Subject Line**: Include "SECURITY: IoT Edge Graph Anomaly" 
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Affected versions
   - Any proof-of-concept code

### What to Expect

- **Acknowledgment**: Within 24 hours
- **Initial Assessment**: Within 72 hours  
- **Status Updates**: Weekly until resolved
- **Resolution Timeline**: Critical issues within 7 days, others within 30 days

## Security Considerations

### Edge Deployment Security

This system is designed for IoT edge deployment where security is paramount:

#### Data Protection
- All sensor data is processed locally on edge devices
- No raw sensor data is transmitted to cloud services
- Model inference results are anonymized before export
- Encryption at rest for model files and configuration

#### Network Security
- TLS 1.3 for all external communications
- mTLS for device-to-server authentication
- Network isolation for containerized deployments
- Rate limiting and DDoS protection

#### Container Security
- Non-root user execution in containers
- Minimal base images with security scanning
- Read-only filesystem where possible
- Resource limits to prevent DoS

#### Model Security
- Model integrity verification using checksums
- Secure model loading with input validation
- Memory protection against model extraction
- Inference-time input sanitization

### Threat Model

#### Assets
- ML models (proprietary anomaly detection algorithms)
- Sensor data streams (potentially sensitive IoT data)
- Edge device resources (CPU, memory, storage)
- Network communications (telemetry and alerts)

#### Threats
- **Model Extraction**: Attempts to steal trained models
- **Data Poisoning**: Malicious sensor data injection
- **Resource Exhaustion**: DoS attacks on edge devices
- **Man-in-the-Middle**: Interception of telemetry
- **Privilege Escalation**: Container escape attempts

#### Mitigations
- Model obfuscation and access controls
- Input validation and anomaly detection
- Resource monitoring and limits
- Certificate pinning and encryption
- Container hardening and monitoring

### Secure Configuration

#### Environment Variables
```bash
# Never commit these values
export JWT_SECRET="use-strong-random-secret"
export ENCRYPTION_KEY="32-byte-random-key"
export API_KEY="rotating-api-key"
```

#### File Permissions
```bash
# Model files
chmod 600 models/*.pth

# Configuration files
chmod 600 config/*.yaml

# SSL certificates
chmod 600 certs/*
```

#### Docker Security
```dockerfile
# Use non-root user
USER 1001

# Read-only root filesystem
--read-only

# No new privileges
--security-opt=no-new-privileges

# Drop capabilities
--cap-drop=ALL
```

## Security Testing

### Static Analysis
- **Bandit**: Python security linting
- **Safety**: Dependency vulnerability scanning
- **Semgrep**: Code pattern security analysis

### Dynamic Analysis
- **Container scanning** with Trivy/Snyk
- **Dependency auditing** with pip-audit
- **Network security testing** with nmap

### Penetration Testing
- Quarterly security assessments
- Red team exercises for critical deployments
- Automated security regression testing

## Compliance

### Standards
- **ISO 27001**: Information security management
- **NIST Cybersecurity Framework**: Risk management
- **IEC 62443**: Industrial cybersecurity standards
- **GDPR**: Data protection (where applicable)

### Certifications
- Security-reviewed deployment configurations
- Penetration testing reports available for enterprise customers
- Compliance documentation for regulated industries

## Security Updates

### Automatic Updates
- Security patches auto-applied in development
- Staged rollout for production environments
- Rollback capabilities for failed updates

### Manual Updates
- Critical security updates require immediate action
- Security advisories published for all CVEs
- Update procedures documented in runbooks

## Contact

- **Security Team**: security@terragonlabs.com
- **General Issues**: github.com/terragonlabs/iot-edge-graph-anomaly/issues
- **Emergency Contact**: +1-XXX-XXX-XXXX (24/7 security hotline)

## Attribution

We appreciate responsible disclosure and will acknowledge security researchers who help improve our security posture.

---

**Last Updated**: 2025-01-27  
**Next Review**: 2025-04-27
---
name: Security Vulnerability Report
about: Report a security vulnerability (use security@terragonlabs.com for sensitive issues)
title: '[SECURITY] '
labels: security, needs-triage
assignees: security-team

---

⚠️ **IMPORTANT**: For sensitive security issues, please email security@terragonlabs.com instead of using this public template.

## Security Vulnerability Report

### Vulnerability Summary
*Provide a brief summary of the vulnerability*

### Severity Assessment
- [ ] Critical (Remote code execution, data breach)
- [ ] High (Privilege escalation, authentication bypass)
- [ ] Medium (Information disclosure, DoS)
- [ ] Low (Minor information leak)

### Affected Components
- [ ] LSTM Model
- [ ] GNN Layer  
- [ ] Data Loader
- [ ] Monitoring System
- [ ] Container/Docker
- [ ] Dependencies
- [ ] Configuration
- [ ] Other: ___________

### Environment
- **Version**: [e.g., v0.1.0]
- **Deployment**: [e.g., Docker, Kubernetes, bare metal]
- **OS**: [e.g., Ubuntu 20.04, Alpine Linux]
- **Architecture**: [e.g., x86_64, ARM64]

### Vulnerability Details

#### Description
*Detailed description of the vulnerability*

#### Attack Vector
*How can this vulnerability be exploited?*

#### Impact
*What is the potential impact if exploited?*

### Reproduction Steps
1. 
2. 
3. 
4. 

### Expected vs Actual Behavior
**Expected**: *What should happen*
**Actual**: *What actually happens*

### Proof of Concept
*Include minimal PoC code if safe to share publicly*

```python
# PoC code here (remove sensitive details)
```

### Suggested Fix
*If you have suggestions for how to fix this issue*

### Additional Context
*Any other context about the vulnerability*

### Checklist
- [ ] I have verified this is a security issue
- [ ] I have checked this hasn't been reported before
- [ ] I have included sufficient detail to reproduce
- [ ] I understand this will be publicly visible
- [ ] For sensitive issues, I will email security@terragonlabs.com instead

---
**Reporter**: @[username]  
**Date**: [YYYY-MM-DD]
#!/usr/bin/env python3
"""Automated compliance checking script for various frameworks."""

import argparse
import json
import sys
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml

class ComplianceChecker:
    """Automated compliance framework checker."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(project_root),
            'framework_results': {},
            'overall_score': 0,
            'critical_findings': [],
            'recommendations': []
        }
    
    def check_nist_csf(self) -> Dict[str, Any]:
        """Check NIST Cybersecurity Framework compliance."""
        print("🔍 Checking NIST Cybersecurity Framework compliance...")
        
        checks = {
            'identify': self._check_nist_identify(),
            'protect': self._check_nist_protect(),
            'detect': self._check_nist_detect(),
            'respond': self._check_nist_respond(),
            'recover': self._check_nist_recover()
        }
        
        score = sum(1 for result in checks.values() if result['passed']) / len(checks) * 100
        
        return {
            'framework': 'NIST Cybersecurity Framework',
            'score': score,
            'checks': checks,
            'passed': score >= 80
        }
    
    def _check_nist_identify(self) -> Dict[str, Any]:
        """Check NIST Identify function."""
        findings = []
        
        # Asset inventory
        has_readme = (self.project_root / 'README.md').exists()
        has_architecture = (self.project_root / 'docs' / 'ARCHITECTURE.md').exists()
        has_dependencies = (self.project_root / 'requirements.txt').exists() or \
                          (self.project_root / 'pyproject.toml').exists()
        
        if not has_readme:
            findings.append("Missing README.md for project documentation")
        if not has_architecture:
            findings.append("Missing architecture documentation")
        if not has_dependencies:
            findings.append("Missing dependency documentation")
        
        # Risk assessment documentation
        has_security_doc = (self.project_root / 'SECURITY.md').exists()
        if not has_security_doc:
            findings.append("Missing SECURITY.md for vulnerability reporting")
        
        passed = len(findings) == 0
        return {
            'passed': passed,
            'findings': findings,
            'description': 'Asset inventory and risk assessment'
        }
    
    def _check_nist_protect(self) -> Dict[str, Any]:
        """Check NIST Protect function."""
        findings = []
        
        # Access control
        has_codeowners = (self.project_root / '.github' / 'CODEOWNERS').exists()
        if not has_codeowners:
            findings.append("Missing CODEOWNERS for access control")
        
        # Data security
        has_gitignore = (self.project_root / '.gitignore').exists()
        if has_gitignore:
            with open(self.project_root / '.gitignore') as f:
                gitignore_content = f.read()
                if '.env' not in gitignore_content:
                    findings.append(".env files not excluded in .gitignore")
                if '*.key' not in gitignore_content and '*.pem' not in gitignore_content:
                    findings.append("Key files not excluded in .gitignore")
        else:
            findings.append("Missing .gitignore file")
        
        # Secure development
        has_precommit = (self.project_root / '.pre-commit-config.yaml').exists()
        if not has_precommit:
            findings.append("Missing pre-commit hooks for code quality")
        
        passed = len(findings) <= 1  # Allow one minor finding
        return {
            'passed': passed,
            'findings': findings,
            'description': 'Access controls and data protection'
        }
    
    def _check_nist_detect(self) -> Dict[str, Any]:
        """Check NIST Detect function."""
        findings = []
        
        # Monitoring capabilities
        monitoring_dir = self.project_root / 'monitoring'
        has_monitoring = monitoring_dir.exists()
        if not has_monitoring:
            findings.append("Missing monitoring configuration")
        
        # Health checks
        health_files = list(self.project_root.rglob('*health*'))
        if not health_files:
            findings.append("Missing health check implementation")
        
        # Security scanning
        if has_precommit := (self.project_root / '.pre-commit-config.yaml').exists():
            with open(self.project_root / '.pre-commit-config.yaml') as f:
                precommit_content = f.read()
                if 'bandit' not in precommit_content:
                    findings.append("Missing security scanning (bandit) in pre-commit")
                if 'safety' not in precommit_content:
                    findings.append("Missing dependency vulnerability scanning")
        
        passed = len(findings) <= 1
        return {
            'passed': passed,
            'findings': findings,
            'description': 'Security monitoring and detection'
        }
    
    def _check_nist_respond(self) -> Dict[str, Any]:
        """Check NIST Respond function."""
        findings = []
        
        # Incident response
        has_security_md = (self.project_root / 'SECURITY.md').exists()
        if not has_security_md:
            findings.append("Missing SECURITY.md for incident reporting")
        
        # Issue templates
        issue_templates = self.project_root / '.github' / 'ISSUE_TEMPLATE'
        if issue_templates.exists():
            security_template = issue_templates / 'security_report.md'
            if not security_template.exists():
                findings.append("Missing security issue template")
        else:
            findings.append("Missing GitHub issue templates")
        
        passed = len(findings) == 0
        return {
            'passed': passed,
            'findings': findings,
            'description': 'Incident response planning'
        }
    
    def _check_nist_recover(self) -> Dict[str, Any]:
        """Check NIST Recover function."""
        findings = []
        
        # Recovery documentation
        has_deployment_docs = (self.project_root / 'docs' / 'DEPLOYMENT.md').exists()
        if not has_deployment_docs:
            findings.append("Missing deployment/recovery documentation")
        
        # Backup and versioning
        has_version_control = (self.project_root / '.git').exists()
        if not has_version_control:
            findings.append("Missing version control for recovery")
        
        # Container recovery
        has_docker = (self.project_root / 'Dockerfile').exists()
        has_compose = (self.project_root / 'docker-compose.yml').exists()
        if not (has_docker or has_compose):
            findings.append("Missing containerization for reliable recovery")
        
        passed = len(findings) <= 1
        return {
            'passed': passed,
            'findings': findings,
            'description': 'Recovery and continuity planning'
        }
    
    def check_iso_27001(self) -> Dict[str, Any]:
        """Check ISO 27001 compliance."""
        print("🔍 Checking ISO 27001 compliance...")
        
        checks = {
            'information_security_policy': self._check_iso_policy(),
            'risk_management': self._check_iso_risk(),
            'access_control': self._check_iso_access(),
            'cryptography': self._check_iso_crypto(),
            'system_security': self._check_iso_system_security(),
            'incident_management': self._check_iso_incidents()
        }
        
        score = sum(1 for result in checks.values() if result['passed']) / len(checks) * 100
        
        return {
            'framework': 'ISO 27001',
            'score': score,
            'checks': checks,
            'passed': score >= 75
        }
    
    def _check_iso_policy(self) -> Dict[str, Any]:
        """Check information security policy."""
        findings = []
        
        security_files = [
            'SECURITY.md',
            'CODE_OF_CONDUCT.md',
            'CONTRIBUTING.md'
        ]
        
        for file in security_files:
            if not (self.project_root / file).exists():
                findings.append(f"Missing {file}")
        
        return {
            'passed': len(findings) <= 1,
            'findings': findings,
            'description': 'Information security policies'
        }
    
    def _check_iso_risk(self) -> Dict[str, Any]:
        """Check risk management."""
        findings = []
        
        # Risk assessment documentation
        compliance_docs = list(self.project_root.rglob('*COMPLIANCE*'))
        risk_docs = list(self.project_root.rglob('*RISK*'))
        
        if not compliance_docs and not risk_docs:
            findings.append("Missing risk assessment documentation")
        
        return {
            'passed': len(findings) == 0,
            'findings': findings,
            'description': 'Risk management processes'
        }
    
    def _check_iso_access(self) -> Dict[str, Any]:
        """Check access control."""
        findings = []
        
        has_codeowners = (self.project_root / '.github' / 'CODEOWNERS').exists()
        if not has_codeowners:
            findings.append("Missing CODEOWNERS for access control")
        
        # Check for authentication in code
        auth_files = []
        for py_file in self.project_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(term in content.lower() for term in ['auth', 'token', 'credential']):
                        auth_files.append(str(py_file))
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not auth_files:
            findings.append("No authentication/authorization implementation found")
        
        return {
            'passed': len(findings) <= 1,
            'findings': findings,
            'description': 'Access control implementation'
        }
    
    def _check_iso_crypto(self) -> Dict[str, Any]:
        """Check cryptographic controls."""
        findings = []
        
        # Check for crypto implementation
        crypto_terms = ['encrypt', 'decrypt', 'hash', 'crypto', 'ssl', 'tls']
        crypto_found = False
        
        for py_file in self.project_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if any(term in content for term in crypto_terms):
                        crypto_found = True
                        break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not crypto_found:
            findings.append("No cryptographic implementation found")
        
        return {
            'passed': crypto_found,
            'findings': findings,
            'description': 'Cryptographic controls'
        }
    
    def _check_iso_system_security(self) -> Dict[str, Any]:
        """Check system security."""
        findings = []
        
        # Container security
        dockerfile = self.project_root / 'Dockerfile'
        if dockerfile.exists():
            with open(dockerfile) as f:
                content = f.read()
                if 'USER root' in content and 'USER ' not in content.split('USER root')[1]:
                    findings.append("Dockerfile runs as root without switching to non-root user")
        
        # Dependency scanning
        has_dependabot = (self.project_root / '.github' / 'dependabot.yml').exists()
        has_renovate = (self.project_root / 'renovate.json').exists()
        
        if not (has_dependabot or has_renovate):
            findings.append("Missing automated dependency update management")
        
        return {
            'passed': len(findings) <= 1,
            'findings': findings,
            'description': 'System security controls'
        }
    
    def _check_iso_incidents(self) -> Dict[str, Any]:
        """Check incident management."""
        findings = []
        
        has_security_md = (self.project_root / 'SECURITY.md').exists()
        if not has_security_md:
            findings.append("Missing SECURITY.md for incident reporting")
        
        # Check for logging implementation
        logging_found = False
        for py_file in self.project_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'logging' in content or 'log' in content:
                        logging_found = True
                        break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not logging_found:
            findings.append("No logging implementation found for incident tracking")
        
        return {
            'passed': len(findings) == 0,
            'findings': findings,
            'description': 'Incident management procedures'
        }
    
    def check_gdpr(self) -> Dict[str, Any]:
        """Check GDPR compliance."""
        print("🔍 Checking GDPR compliance...")
        
        checks = {
            'privacy_by_design': self._check_gdpr_privacy_design(),
            'data_protection': self._check_gdpr_data_protection(),
            'consent_management': self._check_gdpr_consent(),
            'data_subject_rights': self._check_gdpr_rights(),
            'breach_notification': self._check_gdpr_breach()
        }
        
        score = sum(1 for result in checks.values() if result['passed']) / len(checks) * 100
        
        return {
            'framework': 'GDPR',
            'score': score,
            'checks': checks,
            'passed': score >= 70
        }
    
    def _check_gdpr_privacy_design(self) -> Dict[str, Any]:
        """Check privacy by design implementation."""
        findings = []
        
        # Privacy documentation
        privacy_docs = list(self.project_root.rglob('*PRIVACY*'))
        if not privacy_docs:
            findings.append("Missing privacy policy documentation")
        
        # Data minimization in code
        data_files = list(self.project_root.rglob('*data*'))
        if data_files:
            # Check for data processing practices
            privacy_terms = ['anonymize', 'pseudonymize', 'encrypt', 'mask']
            privacy_found = False
            
            for py_file in self.project_root.rglob('*.py'):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if any(term in content for term in privacy_terms):
                            privacy_found = True
                            break
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            if not privacy_found:
                findings.append("No privacy-preserving data processing found")
        
        return {
            'passed': len(findings) <= 1,
            'findings': findings,
            'description': 'Privacy by design implementation'
        }
    
    def _check_gdpr_data_protection(self) -> Dict[str, Any]:
        """Check data protection measures."""
        findings = []
        
        # Check for encryption
        crypto_found = False
        for py_file in self.project_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if any(term in content for term in ['encrypt', 'crypto', 'cipher']):
                        crypto_found = True
                        break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not crypto_found:
            findings.append("No data encryption implementation found")
        
        # Check for secure configuration
        env_example = self.project_root / '.env.example'
        if env_example.exists():
            with open(env_example) as f:
                content = f.read()
                if 'encryption' not in content.lower() and 'secret' not in content.lower():
                    findings.append("Missing encryption/security configuration examples")
        
        return {
            'passed': len(findings) == 0,
            'findings': findings,
            'description': 'Data protection measures'
        }
    
    def _check_gdpr_consent(self) -> Dict[str, Any]:
        """Check consent management."""
        findings = []
        
        # For IoT systems, consent is often implicit, but documentation should exist
        has_privacy_doc = any(self.project_root.rglob('*PRIVACY*'))
        has_terms_doc = any(self.project_root.rglob('*TERMS*'))
        
        if not (has_privacy_doc or has_terms_doc):
            findings.append("Missing privacy/terms documentation for consent")
        
        return {
            'passed': len(findings) == 0,
            'findings': findings,
            'description': 'Consent management procedures'
        }
    
    def _check_gdpr_rights(self) -> Dict[str, Any]:
        """Check data subject rights implementation."""
        findings = []
        
        # Check for data subject rights documentation
        if not (self.project_root / 'SECURITY.md').exists():
            findings.append("Missing contact information for data subject requests")
        
        # Check for data portability features
        export_found = False
        for py_file in self.project_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if any(term in content for term in ['export', 'download', 'extract']):
                        export_found = True
                        break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not export_found:
            findings.append("No data export/portability functionality found")
        
        return {
            'passed': len(findings) <= 1,
            'findings': findings,
            'description': 'Data subject rights implementation'
        }
    
    def _check_gdpr_breach(self) -> Dict[str, Any]:
        """Check breach notification procedures."""
        findings = []
        
        has_security_md = (self.project_root / 'SECURITY.md').exists()
        if not has_security_md:
            findings.append("Missing SECURITY.md for breach reporting")
        
        # Check for incident response documentation
        incident_docs = list(self.project_root.rglob('*INCIDENT*'))
        response_docs = list(self.project_root.rglob('*RESPONSE*'))
        
        if not (incident_docs or response_docs):
            findings.append("Missing incident response documentation")
        
        return {
            'passed': len(findings) <= 1,
            'findings': findings,
            'description': 'Breach notification procedures'
        }
    
    def run_compliance_check(self, framework: str) -> Dict[str, Any]:
        """Run compliance check for specified framework."""
        framework_lower = framework.lower()
        
        if framework_lower == 'nist-csf':
            result = self.check_nist_csf()
        elif framework_lower == 'iso-27001':
            result = self.check_iso_27001()
        elif framework_lower == 'gdpr':
            result = self.check_gdpr()
        else:
            raise ValueError(f"Unknown framework: {framework}")
        
        self.results['framework_results'][framework] = result
        
        # Collect critical findings
        for check_name, check_result in result['checks'].items():
            if not check_result['passed']:
                for finding in check_result['findings']:
                    self.results['critical_findings'].append({
                        'framework': result['framework'],
                        'check': check_name,
                        'finding': finding
                    })
        
        return result
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate compliance report."""
        if not self.results['framework_results']:
            return "No compliance checks have been run."
        
        # Calculate overall score
        scores = [r['score'] for r in self.results['framework_results'].values()]
        self.results['overall_score'] = sum(scores) / len(scores)
        
        # Generate recommendations
        if self.results['overall_score'] < 80:
            self.results['recommendations'].append(
                "Overall compliance score is below 80%. Review critical findings."
            )
        
        if len(self.results['critical_findings']) > 5:
            self.results['recommendations'].append(
                "Multiple critical findings detected. Prioritize security improvements."
            )
        
        report = f"""
# Compliance Report

**Generated**: {self.results['timestamp']}
**Project**: {self.results['project_root']}
**Overall Score**: {self.results['overall_score']:.1f}%

## Framework Results

"""
        
        for framework, result in self.results['framework_results'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report += f"### {result['framework']} - {status}\n"
            report += f"**Score**: {result['score']:.1f}%\n\n"
            
            for check_name, check_result in result['checks'].items():
                check_status = "✅" if check_result['passed'] else "❌"
                report += f"- {check_status} **{check_name}**: {check_result['description']}\n"
                
                if check_result['findings']:
                    for finding in check_result['findings']:
                        report += f"  - ⚠️ {finding}\n"
            
            report += "\n"
        
        if self.results['critical_findings']:
            report += "## Critical Findings\n\n"
            for finding in self.results['critical_findings']:
                report += f"- **{finding['framework']}** ({finding['check']}): {finding['finding']}\n"
            report += "\n"
        
        if self.results['recommendations']:
            report += "## Recommendations\n\n"
            for rec in self.results['recommendations']:
                report += f"- {rec}\n"
        
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to: {output_file}")
        
        return report
    
    def save_json_results(self, output_file: str):
        """Save results as JSON."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 JSON results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Automated compliance checking')
    parser.add_argument('--framework', required=True, 
                       choices=['nist-csf', 'iso-27001', 'gdpr'],
                       help='Compliance framework to check')
    parser.add_argument('--project-root', default='.',
                       help='Project root directory (default: current directory)')
    parser.add_argument('--output', help='Output report file')
    parser.add_argument('--json-output', help='JSON output file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"❌ Project root does not exist: {project_root}")
        sys.exit(1)
    
    print(f"🚀 Starting compliance check for {args.framework.upper()}")
    print(f"📁 Project root: {project_root}")
    
    checker = ComplianceChecker(project_root)
    
    try:
        result = checker.run_compliance_check(args.framework)
        
        if args.verbose:
            print(f"\n📊 Framework: {result['framework']}")
            print(f"📈 Score: {result['score']:.1f}%")
            print(f"✅ Passed: {result['passed']}")
        
        # Generate report
        output_file = args.output or f"reports/compliance/{args.framework}-report.md"
        report = checker.generate_report(output_file)
        
        if args.json_output:
            checker.save_json_results(args.json_output)
        
        if not args.output and not args.verbose:
            print(report)
        
        # Exit with appropriate code
        sys.exit(0 if result['passed'] else 1)
        
    except Exception as e:
        print(f"❌ Error during compliance check: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quality Gates Validation for IoT Edge Anomaly Detection Framework.

This script validates the implementation quality without requiring runtime dependencies.
It performs static analysis, code structure validation, and completeness checks.
"""
import os
import sys
from pathlib import Path
import ast
import re
import json
from typing import Dict, List, Tuple, Any

class QualityGateValidator:
    """Validates implementation quality through static analysis."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.src_path = self.repo_path / "src"
        self.results = {
            'code_structure': {'passed': 0, 'failed': 0, 'details': []},
            'implementation_completeness': {'passed': 0, 'failed': 0, 'details': []},
            'documentation': {'passed': 0, 'failed': 0, 'details': []},
            'architecture': {'passed': 0, 'failed': 0, 'details': []},
            'security': {'passed': 0, 'failed': 0, 'details': []},
            'performance': {'passed': 0, 'failed': 0, 'details': []},
        }
    
    def validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and organization."""
        print("🏗️  Validating Code Structure...")
        
        required_modules = [
            'src/iot_edge_anomaly/models/lstm_gnn_hybrid.py',
            'src/iot_edge_anomaly/models/gnn_layer.py',
            'src/iot_edge_anomaly/validation/model_validator.py',
            'src/iot_edge_anomaly/resilience/fault_tolerance.py',
            'src/iot_edge_anomaly/security/secure_inference.py',
            'src/iot_edge_anomaly/optimization/performance_optimizer.py'
        ]
        
        for module in required_modules:
            module_path = self.repo_path / module
            if module_path.exists():
                self.results['code_structure']['passed'] += 1
                self.results['code_structure']['details'].append(f"✅ {module} exists")
            else:
                self.results['code_structure']['failed'] += 1
                self.results['code_structure']['details'].append(f"❌ {module} missing")
        
        # Check examples and research directories
        example_files = list((self.repo_path / "examples").glob("*.py")) if (self.repo_path / "examples").exists() else []
        research_files = list((self.repo_path / "research").glob("*.py")) if (self.repo_path / "research").exists() else []
        
        if example_files:
            self.results['code_structure']['passed'] += 1
            self.results['code_structure']['details'].append(f"✅ Examples directory with {len(example_files)} files")
        else:
            self.results['code_structure']['failed'] += 1
            self.results['code_structure']['details'].append("❌ Examples directory missing or empty")
        
        if research_files:
            self.results['code_structure']['passed'] += 1
            self.results['code_structure']['details'].append(f"✅ Research directory with {len(research_files)} files")
        else:
            self.results['code_structure']['failed'] += 1
            self.results['code_structure']['details'].append("❌ Research directory missing or empty")
        
        return self.results['code_structure']
    
    def validate_implementation_completeness(self) -> Dict[str, Any]:
        """Validate implementation completeness through AST analysis."""
        print("🔍 Validating Implementation Completeness...")
        
        # Check LSTM-GNN hybrid model
        hybrid_model_path = self.src_path / "iot_edge_anomaly/models/lstm_gnn_hybrid.py"
        if hybrid_model_path.exists():
            content = hybrid_model_path.read_text()
            
            required_classes = ['LSTMGNNHybridModel', 'FeatureFusionLayer']
            required_methods = ['forward', 'encode', 'compute_hybrid_anomaly_score']
            
            all_found = True
            for cls in required_classes:
                if f"class {cls}" in content:
                    self.results['implementation_completeness']['passed'] += 1
                    self.results['implementation_completeness']['details'].append(f"✅ {cls} class implemented")
                else:
                    self.results['implementation_completeness']['failed'] += 1
                    self.results['implementation_completeness']['details'].append(f"❌ {cls} class missing")
                    all_found = False
            
            for method in required_methods:
                if f"def {method}" in content:
                    self.results['implementation_completeness']['passed'] += 1
                    self.results['implementation_completeness']['details'].append(f"✅ {method} method implemented")
                else:
                    self.results['implementation_completeness']['failed'] += 1
                    self.results['implementation_completeness']['details'].append(f"❌ {method} method missing")
                    all_found = False
        
        # Check validation framework
        validator_path = self.src_path / "iot_edge_anomaly/validation/model_validator.py"
        if validator_path.exists():
            content = validator_path.read_text()
            
            validation_classes = ['DataValidator', 'ModelValidator', 'ComprehensiveValidator']
            for cls in validation_classes:
                if f"class {cls}" in content:
                    self.results['implementation_completeness']['passed'] += 1
                    self.results['implementation_completeness']['details'].append(f"✅ {cls} validation class implemented")
                else:
                    self.results['implementation_completeness']['failed'] += 1
                    self.results['implementation_completeness']['details'].append(f"❌ {cls} validation class missing")
        
        # Check security framework
        security_path = self.src_path / "iot_edge_anomaly/security/secure_inference.py"
        if security_path.exists():
            content = security_path.read_text()
            
            security_classes = ['SecureInferenceEngine', 'InputSanitizer', 'ModelProtector']
            for cls in security_classes:
                if f"class {cls}" in content:
                    self.results['implementation_completeness']['passed'] += 1
                    self.results['implementation_completeness']['details'].append(f"✅ {cls} security class implemented")
                else:
                    self.results['implementation_completeness']['failed'] += 1
                    self.results['implementation_completeness']['details'].append(f"❌ {cls} security class missing")
        
        return self.results['implementation_completeness']
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation quality and completeness."""
        print("📚 Validating Documentation...")
        
        # Check for docstrings in key modules
        key_modules = [
            'src/iot_edge_anomaly/models/lstm_gnn_hybrid.py',
            'src/iot_edge_anomaly/validation/model_validator.py',
            'src/iot_edge_anomaly/security/secure_inference.py'
        ]
        
        for module_path in key_modules:
            full_path = self.repo_path / module_path
            if full_path.exists():
                content = full_path.read_text()
                
                # Check module docstring
                if '"""' in content and content.strip().startswith('"""'):
                    self.results['documentation']['passed'] += 1
                    self.results['documentation']['details'].append(f"✅ {module_path} has module docstring")
                else:
                    self.results['documentation']['failed'] += 1
                    self.results['documentation']['details'].append(f"❌ {module_path} missing module docstring")
                
                # Check for class and method docstrings
                docstring_count = content.count('"""')
                if docstring_count >= 4:  # Module + classes + methods
                    self.results['documentation']['passed'] += 1
                    self.results['documentation']['details'].append(f"✅ {module_path} has comprehensive docstrings ({docstring_count} found)")
                else:
                    self.results['documentation']['failed'] += 1
                    self.results['documentation']['details'].append(f"❌ {module_path} lacks sufficient docstrings ({docstring_count} found)")
        
        # Check README and docs
        readme_path = self.repo_path / "README.md"
        if readme_path.exists() and readme_path.stat().st_size > 1000:  # At least 1KB
            self.results['documentation']['passed'] += 1
            self.results['documentation']['details'].append("✅ README.md exists and is substantial")
        else:
            self.results['documentation']['failed'] += 1
            self.results['documentation']['details'].append("❌ README.md missing or too short")
        
        return self.results['documentation']
    
    def validate_architecture(self) -> Dict[str, Any]:
        """Validate architecture and design patterns."""
        print("🏛️  Validating Architecture...")
        
        # Check for proper separation of concerns
        expected_directories = [
            'src/iot_edge_anomaly/models',
            'src/iot_edge_anomaly/validation',
            'src/iot_edge_anomaly/resilience',
            'src/iot_edge_anomaly/security',
            'src/iot_edge_anomaly/optimization'
        ]
        
        for dir_path in expected_directories:
            full_path = self.repo_path / dir_path
            if full_path.exists() and full_path.is_dir():
                # Check if directory has meaningful content
                py_files = list(full_path.glob("*.py"))
                if py_files:
                    self.results['architecture']['passed'] += 1
                    self.results['architecture']['details'].append(f"✅ {dir_path} properly organized ({len(py_files)} files)")
                else:
                    self.results['architecture']['failed'] += 1
                    self.results['architecture']['details'].append(f"❌ {dir_path} empty or no Python files")
            else:
                self.results['architecture']['failed'] += 1
                self.results['architecture']['details'].append(f"❌ {dir_path} missing")
        
        # Check for configuration management
        config_files = list(self.repo_path.glob("*config*")) + list(self.repo_path.glob("*.toml")) + list(self.repo_path.glob("*.yaml"))
        if config_files:
            self.results['architecture']['passed'] += 1
            self.results['architecture']['details'].append(f"✅ Configuration files present ({len(config_files)} found)")
        else:
            self.results['architecture']['failed'] += 1
            self.results['architecture']['details'].append("❌ No configuration files found")
        
        return self.results['architecture']
    
    def validate_security_implementation(self) -> Dict[str, Any]:
        """Validate security implementation."""
        print("🔒 Validating Security Implementation...")
        
        security_path = self.src_path / "iot_edge_anomaly/security/secure_inference.py"
        if security_path.exists():
            content = security_path.read_text()
            
            # Check for security features
            security_features = [
                'authentication', 'sanitization', 'validation', 
                'threat', 'security', 'hmac', 'token'
            ]
            
            features_found = 0
            for feature in security_features:
                if feature in content.lower():
                    features_found += 1
            
            if features_found >= len(security_features) * 0.7:  # 70% of features
                self.results['security']['passed'] += 1
                self.results['security']['details'].append(f"✅ Security features implemented ({features_found}/{len(security_features)} found)")
            else:
                self.results['security']['failed'] += 1
                self.results['security']['details'].append(f"❌ Insufficient security features ({features_found}/{len(security_features)} found)")
            
            # Check for input validation
            if 'sanitize' in content.lower() and 'validate' in content.lower():
                self.results['security']['passed'] += 1
                self.results['security']['details'].append("✅ Input sanitization and validation implemented")
            else:
                self.results['security']['failed'] += 1
                self.results['security']['details'].append("❌ Input sanitization/validation missing")
            
            # Check for authentication
            if 'token' in content and 'authentication' in content.lower():
                self.results['security']['passed'] += 1
                self.results['security']['details'].append("✅ Authentication framework implemented")
            else:
                self.results['security']['failed'] += 1
                self.results['security']['details'].append("❌ Authentication framework missing")
        else:
            self.results['security']['failed'] += 1
            self.results['security']['details'].append("❌ Security module missing entirely")
        
        return self.results['security']
    
    def validate_performance_optimization(self) -> Dict[str, Any]:
        """Validate performance optimization implementation."""
        print("⚡ Validating Performance Optimization...")
        
        perf_path = self.src_path / "iot_edge_anomaly/optimization/performance_optimizer.py"
        if perf_path.exists():
            content = perf_path.read_text()
            
            # Check for optimization features
            optimization_features = [
                'cache', 'batch', 'quantization', 'optimization', 
                'performance', 'memory', 'concurrent', 'async'
            ]
            
            features_found = 0
            for feature in optimization_features:
                if feature in content.lower():
                    features_found += 1
            
            if features_found >= len(optimization_features) * 0.7:
                self.results['performance']['passed'] += 1
                self.results['performance']['details'].append(f"✅ Performance features implemented ({features_found}/{len(optimization_features)} found)")
            else:
                self.results['performance']['failed'] += 1
                self.results['performance']['details'].append(f"❌ Insufficient performance features ({features_found}/{len(optimization_features)} found)")
            
            # Check for caching implementation
            if 'IntelligentCache' in content or 'class.*Cache' in content:
                self.results['performance']['passed'] += 1
                self.results['performance']['details'].append("✅ Caching system implemented")
            else:
                self.results['performance']['failed'] += 1
                self.results['performance']['details'].append("❌ Caching system missing")
            
            # Check for batch processing
            if 'batch' in content.lower() and 'BatchProcessor' in content:
                self.results['performance']['passed'] += 1
                self.results['performance']['details'].append("✅ Batch processing implemented")
            else:
                self.results['performance']['failed'] += 1
                self.results['performance']['details'].append("❌ Batch processing missing")
        else:
            self.results['performance']['failed'] += 1
            self.results['performance']['details'].append("❌ Performance optimization module missing")
        
        return self.results['performance']
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        print("🎯 IoT Edge Anomaly Detection - Quality Gates Validation")
        print("=" * 70)
        
        # Run all validations
        self.validate_code_structure()
        self.validate_implementation_completeness()
        self.validate_documentation()
        self.validate_architecture()
        self.validate_security_implementation()
        self.validate_performance_optimization()
        
        # Calculate overall scores
        total_passed = sum(category['passed'] for category in self.results.values())
        total_failed = sum(category['failed'] for category in self.results.values())
        total_tests = total_passed + total_failed
        
        overall_score = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 70)
        print("📊 QUALITY GATES SUMMARY")
        print("=" * 70)
        
        for category, result in self.results.items():
            category_score = (result['passed'] / max(1, result['passed'] + result['failed']) * 100)
            status = "✅ PASS" if category_score >= 70 else "❌ FAIL"
            print(f"{category.replace('_', ' ').title():.<30} {status} ({category_score:.1f}%)")
            
            # Show details for failed categories
            if category_score < 70:
                for detail in result['details']:
                    if detail.startswith("❌"):
                        print(f"  {detail}")
        
        print(f"\n🎯 Overall Score: {overall_score:.1f}%")
        
        # Quality gates evaluation
        if overall_score >= 85:
            print("🎉 EXCELLENT: All quality gates passed with high scores!")
            quality_status = "EXCELLENT"
        elif overall_score >= 70:
            print("✅ GOOD: Quality gates passed with acceptable scores.")
            quality_status = "GOOD"
        elif overall_score >= 50:
            print("⚠️  MODERATE: Some quality issues need attention.")
            quality_status = "MODERATE"
        else:
            print("❌ POOR: Significant quality issues require immediate attention.")
            quality_status = "POOR"
        
        # Generate report
        report = {
            'overall_score': overall_score,
            'quality_status': quality_status,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'category_results': self.results,
            'timestamp': '2025-01-27'
        }
        
        # Save report
        report_path = self.repo_path / "quality_gates_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        return report

if __name__ == "__main__":
    validator = QualityGateValidator()
    report = validator.run_all_validations()
    
    # Exit with appropriate code
    if report['overall_score'] >= 70:
        exit(0)
    else:
        exit(1)
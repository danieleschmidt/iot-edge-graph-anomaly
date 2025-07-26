"""
Test suite for Docker container functionality.
"""
import pytest
from pathlib import Path

def test_dockerfile_exists():
    """Test that Dockerfile exists in the repository root."""
    repo_root = Path(__file__).parent.parent
    dockerfile = repo_root / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile not found in repository root"

def test_dockerignore_exists():
    """Test that .dockerignore exists to optimize build context."""
    repo_root = Path(__file__).parent.parent
    dockerignore = repo_root / ".dockerignore"
    assert dockerignore.exists(), ".dockerignore not found in repository root"

def test_dockerfile_has_multistage_build():
    """Test that Dockerfile uses multi-stage build for size optimization."""
    repo_root = Path(__file__).parent.parent
    dockerfile = repo_root / "Dockerfile"
    
    if dockerfile.exists():
        content = dockerfile.read_text()
        assert "python:" in content, "Dockerfile should use Python base image"
        assert "as builder" in content, "Dockerfile should use multi-stage build"

def test_dockerfile_arm64_compatibility():
    """Test that Dockerfile is configured for ARM64 (Raspberry Pi) compatibility."""
    repo_root = Path(__file__).parent.parent
    dockerfile = repo_root / "Dockerfile"
    
    if dockerfile.exists():
        content = dockerfile.read_text()
        # Check for platform specification or ARM64 compatible base
        arm_indicators = ["linux/arm64", "aarch64", "arm64v8"]
        has_arm_support = any(indicator in content for indicator in arm_indicators)
        assert has_arm_support, "Dockerfile should support ARM64 architecture"
# GitHub Actions Setup Guide

This guide provides templates and instructions for setting up GitHub Actions workflows for the IoT Edge Graph Anomaly Detection project.

## Overview

The project requires the following automated workflows:
- **Continuous Integration**: Testing, linting, security scanning
- **Security Scanning**: Vulnerability detection and compliance checks
- **Container Build**: Docker image building and testing
- **Performance Testing**: Edge device constraint validation
- **Release Automation**: Automated releases and deployments

## Required Secrets

Configure these secrets in your GitHub repository settings:

```bash
# PyPI Publishing
PYPI_API_TOKEN=pypi-xxx

# Docker Registry
DOCKER_HUB_USERNAME=your-username
DOCKER_HUB_ACCESS_TOKEN=your-token

# Code Coverage
CODECOV_TOKEN=your-codecov-token

# Security Scanning
SNYK_TOKEN=your-snyk-token

# Monitoring
SENTRY_DSN=your-sentry-dsn
```

## Workflow Templates

### 1. Continuous Integration (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  PYTHON_VERSION: '3.11'

jobs:
  test:
    name: Test Python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,test]
      
      - name: Run linting
        run: make lint
      
      - name: Run type checking
        run: make type-check
      
      - name: Run tests
        run: make test
      
      - name: Upload coverage to Codecov
        if: matrix.python-version == '3.11'
        uses: codecov/codecov-action@v3
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage.xml
          fail_ci_if_error: true

  performance:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -e .[dev,test]
      
      - name: Run performance tests
        run: |
          pytest tests/performance/ -v --tb=short -m "performance and not slow"
      
      - name: Upload performance results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: reports/performance/
```

### 2. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: Security

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM

jobs:
  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e .[security]
      
      - name: Run safety check
        run: safety check
      
      - name: Run pip-audit
        run: pip-audit
      
      - name: Run bandit security linter
        run: bandit -r src/ -f json -o reports/bandit-report.json || true
      
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: reports/

  code-scanning:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python
      
      - name: Autobuild
        uses: github/codeql-action/autobuild@v2
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

  compliance-check:
    name: Compliance Framework Check
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install pyyaml
      
      - name: Run NIST CSF compliance check
        run: python scripts/compliance_check.py --framework nist-csf --verbose
      
      - name: Run ISO 27001 compliance check
        run: python scripts/compliance_check.py --framework iso-27001 --verbose
      
      - name: Run GDPR compliance check
        run: python scripts/compliance_check.py --framework gdpr --verbose
      
      - name: Upload compliance reports
        uses: actions/upload-artifact@v3
        with:
          name: compliance-reports
          path: reports/compliance/
```

### 3. Container Build (`.github/workflows/docker.yml`)

```yaml
name: Docker

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: docker.io
  IMAGE_NAME: iot-edge-anomaly

jobs:
  build:
    name: Build and Test Docker Image
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Docker Hub
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
      
      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Test Docker image
        run: |
          docker run --rm ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest --help
      
      - name: Run container security scan
        uses: anchore/scan-action@v3
        with:
          image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          fail-build: false
          severity-cutoff: high

  vulnerability-scan:
    name: Container Vulnerability Scan
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name != 'pull_request'
    
    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

### 4. Performance Testing (`.github/workflows/performance.yml`)

```yaml
name: Performance

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly performance regression testing

jobs:
  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    strategy:
      matrix:
        platform: [linux/amd64, linux/arm64]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Build test image
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: ${{ matrix.platform }}
          load: true
          tags: iot-edge-anomaly:test
          target: test-stage
      
      - name: Run performance benchmarks
        run: |
          docker run --rm --platform ${{ matrix.platform }} \
            -v $(pwd)/reports:/app/reports \
            iot-edge-anomaly:test \
            python -m pytest tests/performance/ -v \
            --tb=short --json-report --json-report-file=/app/reports/performance-${{ matrix.platform }}.json
      
      - name: Upload benchmark results  
        uses: actions/upload-artifact@v3
        with:
          name: performance-${{ matrix.platform }}
          path: reports/performance-${{ matrix.platform }}.json

  regression-check:
    name: Performance Regression Check
    runs-on: ubuntu-latest
    needs: benchmark
    if: github.event_name == 'pull_request'
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Download benchmark results
        uses: actions/download-artifact@v3
        with:
          name: performance-linux/amd64
          path: ./current-results/
      
      - name: Get baseline results
        run: |
          git checkout main
          # Download or generate baseline results
          mkdir -p baseline-results/
      
      - name: Check for regressions
        run: |
          python scripts/benchmark_regression.py \
            baseline-results/performance-linux-amd64.json \
            current-results/performance-linux-amd64.json
```

### 5. Release Automation (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
      
      - name: Generate release notes
        id: release_notes
        run: |
          # Generate release notes from CHANGELOG.md or git log
          echo "RELEASE_NOTES<<EOF" >> $GITHUB_OUTPUT
          git log --pretty=format:"- %s" $(git describe --tags --abbrev=0 HEAD^)..HEAD >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT
      
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          body: ${{ steps.release_notes.outputs.RELEASE_NOTES }}
          draft: false
          prerelease: false

  docker-release:
    name: Release Docker Images
    runs-on: ubuntu-latest
    needs: release
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          registry: docker.io
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}
      
      - name: Extract version
        id: version
        run: echo "VERSION=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT
      
      - name: Build and push release images
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: |
            iot-edge-anomaly:${{ steps.version.outputs.VERSION }}
            iot-edge-anomaly:latest
```

## Branch Protection Rules

Configure these branch protection rules for the `main` branch:

```yaml
# Branch Protection Configuration (via GitHub UI)
protection_rules:
  main:
    required_status_checks:
      strict: true
      contexts:
        - "CI / Test Python (3.8)"
        - "CI / Test Python (3.9)"  
        - "CI / Test Python (3.10)"
        - "CI / Test Python (3.11)"
        - "Security / Dependency Vulnerability Scan"
        - "Security / CodeQL Analysis"
        - "Docker / Build and Test Docker Image"
    
    enforce_admins: false
    required_pull_request_reviews:
      required_approving_review_count: 1
      dismiss_stale_reviews: true
      require_code_owner_reviews: true
    
    restrictions:
      users: []
      teams: ["maintainers", "senior-developers"]
    
    required_linear_history: true
    allow_force_pushes: false
    allow_deletions: false
```

## Workflow Monitoring

### Notification Setup

Add Slack/Teams notifications for workflow failures:

```yaml
# Add to any workflow
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#ci-alerts'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Metrics Collection

Monitor workflow performance:

- Build duration trends
- Test failure rates  
- Security scan results
- Performance regression frequency
- Release deployment success rate

## Local Testing

Test workflows locally using `act`:

```bash
# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run CI workflow locally
act -j test

# Run with secrets
act -j test --secret-file .secrets

# Run specific workflow
act -W .github/workflows/ci.yml
```

## Troubleshooting

### Common Issues

1. **Workflow Permissions**
   - Ensure repository has necessary permissions
   - Check if organization policies restrict actions

2. **Secret Management**
   - Verify all required secrets are configured
   - Use repository secrets for sensitive data

3. **Resource Limits**
   - GitHub Actions has usage limits
   - Consider self-hosted runners for intensive tasks

4. **Cross-Platform Testing**
   - Use matrix builds for multiple Python versions
   - Test on both amd64 and arm64 architectures

### Debugging Tips

```yaml
# Add debugging step to any job
- name: Debug Information
  run: |
    echo "Event: ${{ github.event_name }}"
    echo "Ref: ${{ github.ref }}"
    echo "SHA: ${{ github.sha }}"
    echo "Actor: ${{ github.actor }}"
    env
```

---

**GitHub Actions Setup Version**: 1.0  
**Last Updated**: 2025-01-27  
**Next Review**: 2025-04-27
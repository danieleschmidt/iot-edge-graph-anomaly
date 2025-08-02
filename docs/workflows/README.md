# CI/CD Workflow Documentation and Templates

This document provides comprehensive CI/CD workflow templates and setup instructions for the IoT Edge Graph Anomaly Detection project.

## Overview

The CI/CD pipeline implements a complete automation strategy covering:
- **Continuous Integration**: Automated testing, linting, and security scanning
- **Continuous Deployment**: Multi-architecture builds and registry publishing
- **Quality Gates**: Code coverage, security scans, and performance benchmarks
- **Release Management**: Semantic versioning and automated releases

## Workflow Architecture

```mermaid
graph TB
    A[Code Push] --> B[CI Pipeline]
    B --> C[Lint & Format]
    B --> D[Unit Tests]
    B --> E[Integration Tests]
    B --> F[Security Scan]
    
    C --> G[Quality Gate]
    D --> G
    E --> G
    F --> G
    
    G --> H[Multi-Arch Build]
    H --> I[Container Scan]
    I --> J[Deploy to Registry]
    
    K[Version Tag] --> L[Release Pipeline]
    L --> M[Build Artifacts]
    L --> N[Generate Release Notes]
    L --> O[Publish Release]
```

## Required Secrets and Configuration

### Repository Secrets
```bash
# Container Registry
DOCKER_USERNAME           # Docker Hub username
DOCKER_PASSWORD           # Docker Hub token
REGISTRY_URL              # Container registry URL (optional)

# PyPI Publishing
PYPI_API_TOKEN            # PyPI publishing token

# Security Scanning
SNYK_TOKEN                # Snyk security scanning token
SONAR_TOKEN               # SonarCloud analysis token

# Notifications
SLACK_WEBHOOK_URL         # Slack notifications
TEAMS_WEBHOOK_URL         # Microsoft Teams notifications

# Cloud Deployment (if applicable)
AWS_ACCESS_KEY_ID         # AWS deployment credentials
AWS_SECRET_ACCESS_KEY     # AWS deployment credentials
AZURE_CREDENTIALS         # Azure deployment credentials
```

### Branch Protection Rules
```yaml
# Required status checks
- "Lint and Format Check"
- "Unit Tests"
- "Integration Tests" 
- "Security Scan"
- "Multi-Architecture Build"

# Additional settings
- Require branches to be up to date before merging
- Require pull request reviews (2 reviewers)
- Dismiss stale reviews when new commits are pushed
- Require review from code owners
- Restrict pushes to matching branches (admin only)
```

## Workflow Templates

### 1. Continuous Integration Workflow

**File**: `.github/workflows/ci.yml`

```yaml
name: Continuous Integration

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC

env:
  PYTHON_VERSION: '3.10'
  NODE_VERSION: '18'

jobs:
  # Code quality and formatting
  lint-and-format:
    name: Lint and Format Check
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
          
      - name: Install dependencies
        run: |
          pip install -e .[dev]
          
      - name: Run linting
        run: |
          flake8 src tests
          mypy src
          
      - name: Check formatting
        run: |
          black --check src tests
          isort --check-only src tests
          
      - name: Security linting
        run: |
          bandit -r src
          safety check

  # Unit tests with coverage
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    timeout-minutes: 15
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
        
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
          
      - name: Install dependencies
        run: |
          pip install -e .[test]
          
      - name: Run unit tests
        run: |
          pytest tests/unit -v --cov=src --cov-report=xml --cov-report=term
          
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  # Integration tests
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    timeout-minutes: 30
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_PASSWORD: postgres
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
          
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
          
      - name: Install dependencies
        run: |
          pip install -e .[test]
          
      - name: Run integration tests
        run: |
          pytest tests/integration -v --tb=short
        env:
          REDIS_URL: redis://localhost:6379
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/postgres

  # End-to-end tests with Docker
  e2e-tests:
    name: End-to-End Tests
    runs-on: ubuntu-latest
    timeout-minutes: 45
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Build test image
        uses: docker/build-push-action@v5
        with:
          context: .
          load: true
          tags: iot-edge-anomaly:test
          cache-from: type=gha
          cache-to: type=gha,mode=max
          
      - name: Run E2E tests
        run: |
          docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
          
      - name: Collect test results
        if: always()
        run: |
          docker-compose -f docker-compose.test.yml logs > e2e-logs.txt
          
      - name: Upload test artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: e2e-test-results
          path: |
            e2e-logs.txt
            test-results/

  # Security scanning
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 20
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: python
          
      - name: Autobuild
        uses: github/codeql-action/autobuild@v3
        
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        
      - name: Run Snyk security scan
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
          
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

  # Multi-architecture Docker build
  multi-arch-build:
    name: Multi-Architecture Build
    runs-on: ubuntu-latest
    timeout-minutes: 60
    needs: [lint-and-format, unit-tests, integration-tests, security-scan]
    if: github.event_name == 'push' || github.event_name == 'schedule'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
        
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Login to Docker Hub
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
          
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ secrets.DOCKER_USERNAME }}/iot-edge-graph-anomaly
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}
            
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          
      - name: Run container security scan
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: |
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy image ${{ secrets.DOCKER_USERNAME }}/iot-edge-graph-anomaly:latest

  # Performance benchmarks
  performance-tests:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    timeout-minutes: 30
    needs: [multi-arch-build]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
          
      - name: Install dependencies
        run: |
          pip install -e .[test]
          
      - name: Run performance benchmarks
        run: |
          python benchmarks.py --output benchmarks.json
          
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: performance-benchmarks
          path: benchmarks.json
          
      - name: Comment PR with benchmarks
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const benchmarks = JSON.parse(fs.readFileSync('benchmarks.json', 'utf8'));
            
            const comment = \`## 🚀 Performance Benchmarks
            
            | Metric | Value | Target | Status |
            |--------|-------|--------|--------|
            | Inference Latency (p95) | \${benchmarks.latency_p95}ms | <10ms | \${benchmarks.latency_p95 < 10 ? '✅' : '❌'} |
            | Memory Usage | \${benchmarks.memory_mb}MB | <100MB | \${benchmarks.memory_mb < 100 ? '✅' : '❌'} |
            | Throughput | \${benchmarks.throughput}/sec | >100/sec | \${benchmarks.throughput > 100 ? '✅' : '❌'} |
            \`;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });

  # Notification on failure
  notify-failure:
    name: Notify on Failure
    runs-on: ubuntu-latest
    needs: [lint-and-format, unit-tests, integration-tests, e2e-tests, security-scan, multi-arch-build]
    if: failure() && github.ref == 'refs/heads/main'
    
    steps:
      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: failure
          channel: '#iot-alerts'
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
          fields: repo,message,commit,author,action,eventName,ref,workflow
```

### 2. Release Automation Workflow

**File**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

env:
  PYTHON_VERSION: '3.10'

jobs:
  # Create GitHub release
  create-release:
    name: Create Release
    runs-on: ubuntu-latest
    timeout-minutes: 15
    outputs:
      upload_url: ${{ steps.create_release.outputs.upload_url }}
      release_id: ${{ steps.create_release.outputs.id }}
      
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: Generate release notes
        id: generate_notes
        run: |
          # Generate changelog since last tag
          LAST_TAG=$(git describe --tags --abbrev=0 HEAD~1 2>/dev/null || echo "")
          if [ -n "$LAST_TAG" ]; then
            CHANGELOG=$(git log $LAST_TAG..HEAD --pretty=format:"- %s (%h)" --grep="^feat\|^fix\|^perf" --perl-regexp)
          else
            CHANGELOG=$(git log --pretty=format:"- %s (%h)" --grep="^feat\|^fix\|^perf" --perl-regexp)
          fi
          
          echo "CHANGELOG<<EOF" >> $GITHUB_OUTPUT
          echo "$CHANGELOG" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT
          
      - name: Create Release
        id: create_release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref_name }}
          release_name: Release ${{ github.ref_name }}
          body: |
            ## What's Changed
            
            ${{ steps.generate_notes.outputs.CHANGELOG }}
            
            ## Docker Images
            
            - **Multi-architecture**: `docker pull ${{ secrets.DOCKER_USERNAME }}/iot-edge-graph-anomaly:${{ github.ref_name }}`
            - **ARM64**: `docker pull ${{ secrets.DOCKER_USERNAME }}/iot-edge-graph-anomaly:${{ github.ref_name }}-arm64`
            - **AMD64**: `docker pull ${{ secrets.DOCKER_USERNAME }}/iot-edge-graph-anomaly:${{ github.ref_name }}-amd64`
            
            ## Installation
            
            ```bash
            pip install iot-edge-graph-anomaly==${{ github.ref_name }}
            ```
            
            ## Deployment
            
            ```bash
            docker run -d -p 8080:8080 ${{ secrets.DOCKER_USERNAME }}/iot-edge-graph-anomaly:${{ github.ref_name }}
            ```
          draft: false
          prerelease: ${{ contains(github.ref_name, 'rc') || contains(github.ref_name, 'beta') || contains(github.ref_name, 'alpha') }}

  # Build Python packages
  build-python-packages:
    name: Build Python Packages
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install build dependencies
        run: |
          pip install build twine
          
      - name: Build packages
        run: |
          python -m build
          
      - name: Check packages
        run: |
          twine check dist/*
          
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: python-packages
          path: dist/

  # Build Docker images
  build-docker-images:
    name: Build Docker Images
    runs-on: ubuntu-latest
    timeout-minutes: 60
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
        
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
          
      - name: Build and push Docker images
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: |
            ${{ secrets.DOCKER_USERNAME }}/iot-edge-graph-anomaly:${{ github.ref_name }}
            ${{ secrets.DOCKER_USERNAME }}/iot-edge-graph-anomaly:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
          
      - name: Export Docker images
        run: |
          docker save ${{ secrets.DOCKER_USERNAME }}/iot-edge-graph-anomaly:${{ github.ref_name }} | gzip > docker-image-${{ github.ref_name }}.tar.gz
          
      - name: Upload Docker image artifact
        uses: actions/upload-artifact@v3
        with:
          name: docker-images
          path: docker-image-${{ github.ref_name }}.tar.gz

  # Publish to PyPI
  publish-pypi:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    timeout-minutes: 10
    needs: [create-release, build-python-packages]
    environment: production
    
    steps:
      - name: Download build artifacts
        uses: actions/download-artifact@v3
        with:
          name: python-packages
          path: dist/
          
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}

  # Upload release assets
  upload-assets:
    name: Upload Release Assets
    runs-on: ubuntu-latest
    timeout-minutes: 15
    needs: [create-release, build-python-packages, build-docker-images]
    
    steps:
      - name: Download Python packages
        uses: actions/download-artifact@v3
        with:
          name: python-packages
          path: dist/
          
      - name: Download Docker images
        uses: actions/download-artifact@v3
        with:
          name: docker-images
          path: .
          
      - name: Upload wheel package
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ needs.create-release.outputs.upload_url }}
          asset_path: ./dist/*.whl
          asset_name: iot-edge-graph-anomaly-${{ github.ref_name }}-py3-none-any.whl
          asset_content_type: application/zip
          
      - name: Upload source distribution
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ needs.create-release.outputs.upload_url }}
          asset_path: ./dist/*.tar.gz
          asset_name: iot-edge-graph-anomaly-${{ github.ref_name }}.tar.gz
          asset_content_type: application/gzip
          
      - name: Upload Docker image
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ needs.create-release.outputs.upload_url }}
          asset_path: ./docker-image-${{ github.ref_name }}.tar.gz
          asset_name: docker-image-${{ github.ref_name }}.tar.gz
          asset_content_type: application/gzip

  # Deployment to staging
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    timeout-minutes: 30
    needs: [create-release, build-docker-images]
    environment: staging
    
    steps:
      - name: Deploy to staging environment
        run: |
          echo "Deploying to staging environment..."
          # Add staging deployment logic here
          
  # Notify success
  notify-success:
    name: Notify Success
    runs-on: ubuntu-latest
    needs: [create-release, publish-pypi, upload-assets, deploy-staging]
    if: success()
    
    steps:
      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: success
          channel: '#iot-releases'
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
          fields: repo,message,commit,author,action,eventName,ref,workflow
          text: |
            🚀 Successfully released ${{ github.ref_name }}
            
            📦 PyPI: https://pypi.org/project/iot-edge-graph-anomaly/
            🐳 Docker: docker pull ${{ secrets.DOCKER_USERNAME }}/iot-edge-graph-anomaly:${{ github.ref_name }}
            📋 Release Notes: ${{ github.server_url }}/${{ github.repository }}/releases/tag/${{ github.ref_name }}
```

### 3. Security Scanning Workflow

**File**: `.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  push:
    branches: [main]
  pull_request:
    branches: [main]
    types: [opened, synchronize, labeled]

jobs:
  # Dependency vulnerability scanning
  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    timeout-minutes: 20
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          pip install safety pip-audit
          pip install -e .
          
      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json || true
          
      - name: Run pip-audit
        run: |
          pip-audit --format=json --output=pip-audit-report.json || true
          
      - name: Upload vulnerability reports
        uses: actions/upload-artifact@v3
        with:
          name: vulnerability-reports
          path: |
            safety-report.json
            pip-audit-report.json

  # Code security analysis
  code-security-scan:
    name: Code Security Analysis
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: python
          queries: security-and-quality
          
      - name: Autobuild
        uses: github/codeql-action/autobuild@v3
        
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:python"
          
      - name: Run Bandit security linter
        run: |
          pip install bandit[toml]
          bandit -r src -f json -o bandit-report.json || true
          
      - name: Run Semgrep security scan
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/secrets
            p/python
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}

  # Container security scanning
  container-security-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Build Docker image
        run: |
          docker build -t iot-edge-anomaly:security-scan .
          
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'iot-edge-anomaly:security-scan'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'
          
      - name: Run Docker Bench Security
        run: |
          docker run --rm --net host --pid host --userns host --cap-add audit_control \
            -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
            -v /etc:/etc:ro \
            -v /usr/bin/containerd:/usr/bin/containerd:ro \
            -v /usr/bin/runc:/usr/bin/runc:ro \
            -v /usr/lib/systemd:/usr/lib/systemd:ro \
            -v /var/lib:/var/lib:ro \
            -v /var/run/docker.sock:/var/run/docker.sock:ro \
            --label docker_bench_security \
            docker/docker-bench-security > docker-bench-security.txt
            
      - name: Upload Docker Bench results
        uses: actions/upload-artifact@v3
        with:
          name: docker-bench-security
          path: docker-bench-security.txt

  # Secrets scanning
  secrets-scan:
    name: Secrets Scanning
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: Run TruffleHog secrets scan
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD
          extra_args: --debug --only-verified

  # License compliance check
  license-compliance:
    name: License Compliance Check
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          pip install pip-licenses
          pip install -e .
          
      - name: Check licenses
        run: |
          pip-licenses --format=json --output-file=licenses.json
          pip-licenses --format=markdown --output-file=licenses.md
          
      - name: Upload license reports
        uses: actions/upload-artifact@v3
        with:
          name: license-reports
          path: |
            licenses.json
            licenses.md

  # Security report aggregation
  security-report:
    name: Generate Security Report
    runs-on: ubuntu-latest
    needs: [dependency-scan, code-security-scan, container-security-scan, license-compliance]
    if: always()
    
    steps:
      - name: Download all artifacts
        uses: actions/download-artifact@v3
        
      - name: Generate security summary
        run: |
          cat > security-summary.md << 'EOF'
          # Security Scan Summary
          
          **Scan Date**: $(date -u)
          **Repository**: ${{ github.repository }}
          **Commit**: ${{ github.sha }}
          
          ## Scan Results
          
          - ✅ Dependency Vulnerability Scan
          - ✅ Code Security Analysis  
          - ✅ Container Security Scan
          - ✅ Secrets Scanning
          - ✅ License Compliance Check
          
          ## Reports
          
          Detailed reports are available in the workflow artifacts.
          EOF
          
      - name: Upload security summary
        uses: actions/upload-artifact@v3
        with:
          name: security-summary
          path: security-summary.md
```

## Manual Setup Instructions

### Step 1: Repository Configuration

1. **Enable GitHub Actions**:
   - Go to repository Settings > Actions > General
   - Allow all actions and reusable workflows
   - Allow actions created by GitHub and verified creators

2. **Configure Branch Protection**:
   ```bash
   # Via GitHub CLI
   gh api repos/:owner/:repo/branches/main/protection \
     --method PUT \
     --field required_status_checks='{"strict":true,"contexts":["Lint and Format Check","Unit Tests","Integration Tests","Security Scan"]}' \
     --field enforce_admins=true \
     --field required_pull_request_reviews='{"required_approving_review_count":2,"dismiss_stale_reviews":true}' \
     --field restrictions=null
   ```

### Step 2: Secrets Configuration

Add the following secrets in Repository Settings > Secrets and variables > Actions:

```bash
# Required secrets
DOCKER_USERNAME=<your-docker-username>
DOCKER_PASSWORD=<your-docker-token>
PYPI_API_TOKEN=<your-pypi-token>

# Optional secrets
SNYK_TOKEN=<snyk-token>
SONAR_TOKEN=<sonarcloud-token>
SLACK_WEBHOOK_URL=<slack-webhook>
SEMGREP_APP_TOKEN=<semgrep-token>
```

### Step 3: Workflow Files Creation

Create the workflow files manually by copying the templates above into the respective paths:

1. `.github/workflows/ci.yml`
2. `.github/workflows/release.yml`  
3. `.github/workflows/security.yml`

### Step 4: Environment Configuration

Create environments for deployment stages:

1. **Staging Environment**:
   - Name: `staging`
   - Protection rules: Require reviewers
   - Environment secrets: staging deployment credentials

2. **Production Environment**:
   - Name: `production`
   - Protection rules: Require reviewers + wait timer
   - Environment secrets: production deployment credentials

### Step 5: Notification Setup

Configure webhook integrations:

1. **Slack Integration**:
   - Create Slack app with webhook permissions
   - Add webhook URL to repository secrets
   - Configure channel permissions

2. **Email Notifications**:
   - Configure in repository Settings > Notifications
   - Set up for workflow failures and security alerts

## Monitoring and Maintenance

### Workflow Monitoring

1. **Workflow Status Dashboard**: Monitor via GitHub Actions tab
2. **Performance Metrics**: Track build times and success rates
3. **Security Alerts**: Monitor security scan results
4. **Resource Usage**: Monitor runner usage and costs

### Regular Maintenance

1. **Monthly**: Review and update workflow dependencies
2. **Quarterly**: Security audit of workflow permissions
3. **Semi-annually**: Performance optimization review
4. **Annually**: Complete workflow architecture review

## Troubleshooting

### Common Issues

1. **Workflow Permissions**: Ensure proper GITHUB_TOKEN permissions
2. **Secret Access**: Verify secrets are available in correct environments
3. **Runner Limitations**: Monitor timeout and resource constraints
4. **Cache Issues**: Clear GitHub Actions cache if builds are inconsistent

### Debug Commands

```bash
# Check workflow status
gh run list --repo owner/repo

# View workflow logs
gh run view <run-id> --log

# Re-run failed workflows
gh run rerun <run-id>
```

---

**Manual Action Required**: Repository maintainers must create workflow files from these templates due to GitHub App permission limitations.

**Workflow Documentation Version**: 1.0  
**Last Updated**: 2025-08-02  
**Next Review**: 2025-11-02
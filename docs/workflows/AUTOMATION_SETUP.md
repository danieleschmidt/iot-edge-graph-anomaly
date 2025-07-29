# Automation Setup Guide

This document provides instructions for setting up automated workflows and processes for the IoT Edge Graph Anomaly Detection project.

## Overview

The project uses several automation tools to maintain code quality, security, and dependency management:

- **GitHub Actions**: CI/CD pipelines for testing, building, and deployment
- **Dependabot**: Automated dependency updates
- **Renovate**: Advanced dependency management (alternative to Dependabot)
- **Pre-commit**: Local code quality checks
- **Bumpversion**: Automated version management

## GitHub Actions Workflows

### Required Workflows

The following workflows should be created in `.github/workflows/`:

#### 1. Continuous Integration (`ci.yml`)

```yaml
name: CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
      - name: Run tests
        run: make test
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

#### 2. Security Scanning (`security.yml`)

```yaml
name: Security
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e .[security]
      - name: Run security checks
        run: make security-check
```

#### 3. Container Build (`docker.yml`)

```yaml
name: Docker
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Build Docker image
        run: make docker-build
      - name: Test Docker image
        run: |
          docker run --rm iot-edge-graph-anomaly:latest --help
```

#### 4. Release (`release.yml`)

```yaml
name: Release
on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Build package
        run: make build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### Workflow Secrets

Configure the following secrets in GitHub:

- `PYPI_API_TOKEN`: For publishing to PyPI
- `DOCKER_HUB_USERNAME`: For Docker Hub registry
- `DOCKER_HUB_ACCESS_TOKEN`: For Docker Hub registry
- `CODECOV_TOKEN`: For code coverage reporting

## Dependency Management

### Dependabot Configuration

Dependabot is configured in `.github/dependabot.yml` to:

- Update Python dependencies weekly on Mondays
- Update GitHub Actions weekly on Mondays
- Update Docker base images weekly on Sundays
- Automatically assign reviewers and labels
- Limit concurrent pull requests

### Renovate Configuration (Alternative)

Renovate provides more advanced dependency management:

- Semantic commit messages
- Grouped updates for related packages
- Vulnerability alerts with security team assignment
- Lock file maintenance
- Custom scheduling and approval rules

To use Renovate instead of Dependabot:

1. Install the Renovate GitHub App
2. Configure using `renovate.json`
3. Disable Dependabot in repository settings

## Pre-commit Hooks

Pre-commit hooks are configured to run:

- Code formatting (Black, isort)
- Linting (flake8)
- Type checking (mypy)
- Security scanning (bandit, safety)
- YAML validation
- Trailing whitespace removal

### Setup

```bash
# Install pre-commit hooks
make install-hooks

# Run manually on all files
make pre-commit
```

## Version Management

### Automated Version Bumping

Use bumpversion for automated version management:

```bash
# Patch version (0.1.0 -> 0.1.1)
make release-patch

# Minor version (0.1.0 -> 0.2.0)
make release-minor

# Major version (0.1.0 -> 1.0.0)
make release-major
```

Bumpversion will:

1. Increment version numbers in all configured files
2. Create a git commit with the version change
3. Create a git tag for the new version
4. Push changes and tags to remote repository

### Release Process

1. **Prepare Release**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Update Version**
   ```bash
   make release-minor  # or patch/major
   ```

3. **Push to GitHub**
   ```bash
   git push origin main --tags
   ```

4. **GitHub Actions will automatically:**
   - Run tests and security checks
   - Build and publish Docker images
   - Create GitHub release
   - Publish to PyPI (if configured)

## Quality Gates

### Branch Protection Rules

Configure these branch protection rules for `main`:

- Require pull request reviews (at least 1)
- Require status checks to pass:
  - `ci / test (3.8)`
  - `ci / test (3.9)`
  - `ci / test (3.10)`
  - `ci / test (3.11)`
  - `security / security`
  - `docker / build`
- Require branches to be up to date
- Restrict pushes that create files matching `.github/workflows/*`

### Required Checks

All pull requests must pass:

- ✅ Unit tests with >80% coverage
- ✅ Integration tests
- ✅ Linting (flake8, black, isort)
- ✅ Type checking (mypy)
- ✅ Security scanning (bandit, safety)
- ✅ Docker build verification
- ✅ Performance tests (memory <100MB, CPU <25%)

## Monitoring and Alerting

### GitHub Actions Notifications

Configure Slack/Teams notifications for:

- Failed builds on main branch
- Security vulnerability alerts
- Deployment successes/failures

### Dependency Alerts

Configure notifications for:

- High-severity security vulnerabilities
- Dependency update failures
- License compliance issues

## Performance Monitoring

### Automated Performance Testing

CI pipeline includes performance tests to ensure:

- Memory usage stays under 100MB
- CPU usage stays under 25% on Raspberry Pi 4
- Inference latency remains under 10ms
- Docker image size stays under 500MB

### Benchmarking

Regular benchmarking runs:

- Model inference performance
- Memory consumption over time
- Container startup time
- Resource utilization patterns

## Compliance and Auditing

### Automated Compliance Checks

- License compatibility scanning
- SBOM (Software Bill of Materials) generation
- Security policy compliance
- Code quality metrics tracking

### Audit Trail

All automation maintains audit trails for:

- Dependency changes and approvals
- Security scan results
- Performance test results
- Deployment history
- Version changes

## Troubleshooting

### Common Issues

1. **Failed Security Scans**
   - Check for new vulnerabilities in dependencies
   - Update vulnerable packages
   - Add exceptions for false positives

2. **Performance Test Failures**
   - Profile memory usage with `memory_profiler`
   - Optimize model loading and inference
   - Check for memory leaks

3. **Docker Build Failures**
   - Verify base image compatibility
   - Check for dependency conflicts
   - Update multi-arch build configuration

### Getting Help

- Check GitHub Actions logs for detailed error messages
- Review pre-commit hook output for code quality issues
- Consult team documentation for project-specific setup
- Contact DevOps team for infrastructure-related issues

---

**Automation Setup Version**: 1.0  
**Last Updated**: 2025-01-27  
**Next Review**: 2025-04-27
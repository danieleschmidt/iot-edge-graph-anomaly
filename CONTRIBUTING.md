# Contributing to IoT Edge Graph Anomaly Detection

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Docker (optional but recommended)

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/iot-edge-graph-anomaly.git
   cd iot-edge-graph-anomaly
   ```

3. Set up development environment:
   ```bash
   # Using make
   make dev-setup
   
   # Or manually
   python -m venv venv
   source venv/bin/activate
   pip install -e .[dev]
   pre-commit install
   ```

4. Verify setup:
   ```bash
   make test
   ```

## Development Process

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes

### Workflow
1. Create feature branch from `develop`
2. Make changes with tests
3. Run quality checks
4. Submit pull request
5. Address review feedback
6. Merge after approval

### Code Standards
- Follow PEP 8 style guide
- Use type hints
- Write docstrings (Google style)
- Maintain test coverage >80%
- Add security considerations

### Commit Messages
Use conventional commits:
```
type(scope): description

feat(models): add LSTM-GNN hybrid architecture
fix(data): resolve sensor data normalization issue
docs(readme): update installation instructions
test(integration): add end-to-end pipeline tests
```

## Pull Request Guidelines

### Before Submitting
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Security implications considered

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests (if applicable)

## Security
- [ ] Security implications reviewed
- [ ] No sensitive data exposed
- [ ] Dependencies updated

## Checklist
- [ ] Code review completed
- [ ] Tests passing
- [ ] Documentation updated
```

## Testing

### Test Types
- **Unit Tests**: `pytest tests/unit/`
- **Integration Tests**: `pytest tests/integration/`
- **Performance Tests**: `pytest tests/performance/`
- **End-to-End Tests**: `pytest tests/e2e/`

### Running Tests
```bash
# All tests
make test

# Specific test types
make test-unit
make test-integration
make test-performance

# With coverage
make test-cov
```

### Test Guidelines
- Write tests for new features
- Maintain high test coverage
- Use fixtures for common setup
- Mock external dependencies
- Include performance benchmarks

## Documentation

### Types
- **API Documentation**: Inline docstrings
- **User Guides**: `docs/guides/`
- **Developer Docs**: `docs/development/`
- **Architecture**: `ARCHITECTURE.md`

### Writing Guidelines
- Clear and concise language
- Include code examples
- Provide troubleshooting steps
- Update with code changes

## Review Process

### Code Review Checklist
- [ ] Code quality and style
- [ ] Test coverage and quality
- [ ] Performance implications
- [ ] Security considerations
- [ ] Documentation completeness
- [ ] Breaking change impact

### Approval Requirements
- [ ] 2 approving reviews
- [ ] All CI checks passing
- [ ] No conflicts with target branch
- [ ] Security scan passed

## Issue Guidelines

### Bug Reports
Include:
- Environment details
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs
- Minimal reproduction case

### Feature Requests
Include:
- Use case description
- Proposed solution
- Alternative approaches
- Breaking change impact

## Release Process

### Version Numbering
Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

### Release Checklist
- [ ] Version updated
- [ ] CHANGELOG.md updated
- [ ] Documentation reviewed
- [ ] Security scan passed
- [ ] Performance benchmarks met
- [ ] Deployment tested

## Community

### Communication Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Email: security@terragon-labs.com (security issues)

### Getting Help
- Check existing issues and documentation
- Search GitHub discussions
- Create detailed issue if needed
- Join community discussions

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation
- Community highlights

Thank you for contributing to IoT Edge Graph Anomaly Detection!
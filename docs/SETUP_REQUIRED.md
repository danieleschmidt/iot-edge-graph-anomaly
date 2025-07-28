# Manual Setup Requirements

## GitHub Repository Configuration

### Branch Protection Rules
```bash
# Configure via GitHub UI: Settings > Branches
# Protect 'main' branch with:
- Require pull request reviews (2 reviewers)
- Require status checks to pass
- Require branches to be up to date
- Restrict pushes to admins only
```

### Repository Settings
```bash
# Configure via GitHub UI: Settings > General
- Enable Issues and Projects
- Set default branch to 'main'
- Enable vulnerability alerts
- Enable Dependabot security updates
```

### Secrets Configuration
```bash
# Add via GitHub UI: Settings > Secrets and variables
DOCKER_USERNAME=<docker-hub-username>
DOCKER_PASSWORD=<docker-hub-token>
PYPI_API_TOKEN=<pypi-publishing-token>
```

## External Integrations

### Code Quality Tools
- **CodeClimate**: Connect repository for quality metrics
- **Codecov**: Set up coverage reporting
- **SonarCloud**: Configure code analysis

### Monitoring and Alerts
- **Sentry**: Error tracking integration
- **DataDog/New Relic**: Performance monitoring
- **PagerDuty**: Critical alert routing

### Documentation
- **Read the Docs**: Automated documentation builds
- **GitHub Pages**: Static site deployment

## Security Configuration

### Access Controls
- Review and audit repository collaborators
- Configure team permissions appropriately
- Enable two-factor authentication requirement

### Vulnerability Management
- Enable Dependabot alerts and updates
- Configure security advisories
- Set up automated security scanning

## Deployment Prerequisites

### Container Registry
- Docker Hub or AWS ECR setup
- Registry authentication configured
- Multi-architecture build support

### Edge Device Management
- SSH key distribution to edge devices
- Device inventory and monitoring
- Update and rollback procedures

For detailed setup instructions, see [Development Guide](DEVELOPMENT.md).
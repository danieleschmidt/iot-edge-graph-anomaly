# ADR-0002: Edge Deployment Strategy with Containerd OTA

## Status
Accepted

## Context
IoT edge devices have strict resource constraints (RAM, CPU, storage) and require reliable deployment mechanisms for security-critical anomaly detection models. Traditional deployment approaches often exceed edge device capabilities or lack proper update mechanisms.

## Decision
We will deploy the anomaly detection system using:
1. Multi-stage Docker containers optimized for ARM64 architecture
2. Containerd-based Over-the-Air (OTA) updates
3. Resource envelope targeting <100MB RAM, <25% CPU on Raspberry Pi 4
4. Non-root container execution for enhanced security

## Rationale
- **Resource Optimization**: Multi-stage builds minimize runtime image size by excluding build dependencies
- **OTA Updates**: Containerd provides reliable container image updates without full system restarts
- **Security**: Non-root execution reduces attack surface and follows security best practices
- **Compatibility**: ARM64 targeting ensures broad compatibility with modern edge devices
- **Predictable Performance**: Defined resource envelope enables capacity planning and SLA guarantees

## Consequences

### Positive
- Minimal resource footprint suitable for constrained edge devices
- Reliable update mechanism for security patches and model improvements
- Enhanced security through container isolation and non-root execution
- Standardized deployment across different edge device types
- Clear performance boundaries for operational planning

### Negative
- Container overhead compared to native deployment
- Dependency on container runtime availability
- Additional complexity in OTA update orchestration
- Limited debugging capabilities in production containers

## Implementation Details

### Container Architecture
```dockerfile
# Multi-stage build
FROM python:3.10-slim as builder  # Build dependencies
FROM python:3.10-slim as runtime  # Minimal runtime
```

### Resource Limits
- Memory: <100MB RAM (enforced via container limits)
- CPU: <25% utilization (2 cores on Raspberry Pi 4)
- Storage: <500MB container image
- Network: Minimal bandwidth for metrics export

### Security Measures
- Non-root user execution (`iotuser`)
- Minimal runtime dependencies
- Health check endpoints
- No sensitive data in container layers

### OTA Update Process
1. New container image built and pushed to registry
2. Edge devices poll registry for updates
3. Containerd pulls new image
4. Rolling restart with health checks
5. Rollback capability on failure

## Alternatives Considered
1. **Native Python Deployment**: Lower overhead but complex dependency management and updates
2. **Snap Packages**: Good for updates but higher resource overhead and limited ARM64 support
3. **Docker Swarm**: Excellent orchestration but too complex for single-device edge deployments
4. **Kubernetes**: Powerful but resource overhead exceeds edge device capabilities
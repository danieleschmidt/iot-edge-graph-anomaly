---
name: Performance Regression Report
about: Report performance degradation in ML model or edge deployment
title: '[PERF] Performance regression in [component]'
labels: 'performance, regression, edge-deployment'
assignees: 'terragon-labs/performance-team'

---

## Performance Regression Details

### Component Affected
- [ ] LSTM Autoencoder model
- [ ] GNN layer processing  
- [ ] Data loading pipeline
- [ ] Inference pipeline
- [ ] Metrics collection
- [ ] Container runtime
- [ ] Memory management
- [ ] Other: ___________

### Environment
- **Deployment Target**: [Raspberry Pi 4 / x86_64 / Cloud]
- **Python Version**: 
- **PyTorch Version**:
- **Container Runtime**: [Docker / Containerd]
- **Hardware**: [ARM64 / x86_64]

### Performance Metrics

#### Before (Baseline)
- **Memory Usage**: _____ MB
- **CPU Usage**: _____ %
- **Inference Time**: _____ ms
- **Throughput**: _____ samples/sec
- **Battery Impact**: _____ (if applicable)

#### After (Current)
- **Memory Usage**: _____ MB  
- **CPU Usage**: _____ %
- **Inference Time**: _____ ms
- **Throughput**: _____ samples/sec
- **Battery Impact**: _____ (if applicable)

#### Edge Device Constraints
- [ ] Memory usage exceeds 100MB limit
- [ ] CPU usage exceeds 25% limit  
- [ ] Inference time exceeds 10ms target
- [ ] Throughput below 100 samples/sec
- [ ] Battery drain increased significantly

### Reproduction Steps
1. 
2. 
3. 

### Performance Profile
<!-- Include profiling data if available -->
```
Paste profiling output here
```

### Suspected Cause
<!-- What changes might have caused this regression? -->

### Impact Assessment
- [ ] Critical - blocks edge deployment
- [ ] High - affects user experience
- [ ] Medium - noticeable but acceptable
- [ ] Low - minimal impact

### Additional Context
<!-- Add any other context about the performance issue -->

### Checklist
- [ ] Performance regression is reproducible
- [ ] Baseline measurements are available
- [ ] Hardware constraints are documented
- [ ] Profiling data is collected (if possible)
- [ ] Impact on edge deployment is assessed
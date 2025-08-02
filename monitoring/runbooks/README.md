# Operational Runbooks

This directory contains operational runbooks for common scenarios and incident response procedures.

## Alert Response Procedures

### Critical Alerts

#### Application Down
**Alert**: `ApplicationDown`  
**Severity**: Critical  
**Response Time**: Immediate (<5 minutes)

**Immediate Actions**:
1. Check container status: `docker ps | grep iot-edge-anomaly`
2. Check container logs: `docker logs iot-edge-anomaly --tail 50`
3. Verify resource availability: `docker stats iot-edge-anomaly`
4. Restart container if needed: `docker restart iot-edge-anomaly`

**Root Cause Investigation**:
- Review application logs for errors
- Check system resource utilization
- Verify network connectivity
- Examine recent deployments or changes

**Prevention**:
- Implement health checks with automatic restart
- Monitor resource trends
- Set up proactive alerting

---

#### High Memory Usage
**Alert**: `HighMemoryUsage`  
**Severity**: Critical  
**Response Time**: <10 minutes

**Immediate Actions**:
1. Check current memory usage: `docker stats iot-edge-anomaly`
2. Look for memory leaks in logs
3. Restart container if memory usage >95%: `docker restart iot-edge-anomaly`
4. Scale down non-essential processes if possible

**Investigation Steps**:
- Review memory allocation patterns
- Check for memory leaks in application code
- Analyze memory usage trends over time
- Verify garbage collection effectiveness

**Long-term Solutions**:
- Optimize memory usage in application
- Increase memory limits if justified
- Implement memory monitoring and alerting

---

### Warning Alerts

#### High Inference Latency
**Alert**: `HighInferenceLatency`  
**Severity**: Warning  
**Response Time**: <30 minutes

**Investigation Steps**:
1. Check CPU usage: `docker stats iot-edge-anomaly`
2. Review inference performance metrics
3. Analyze model complexity and input data size
4. Check for resource contention

**Optimization Actions**:
- Optimize model inference code
- Consider model quantization
- Review batch processing efficiency
- Scale resources if needed

---

#### Model Drift Detected
**Alert**: `ModelDriftDetected`  
**Severity**: Warning  
**Response Time**: <1 hour

**Response Actions**:
1. Validate drift detection accuracy
2. Analyze recent data patterns
3. Review model performance metrics
4. Prepare for model retraining if confirmed

**Long-term Actions**:
- Implement automated model retraining pipeline
- Enhance drift detection algorithms
- Set up model versioning and rollback procedures

---

## Maintenance Procedures

### Routine Maintenance

#### Weekly Health Checks
```bash
#!/bin/bash
# Weekly health check script

echo "=== IoT Edge Anomaly Detection Health Check ==="
echo "Date: $(date)"
echo

# Check container status
echo "Container Status:"
docker ps | grep iot-edge-anomaly

# Check resource usage
echo -e "\nResource Usage:"
docker stats --no-stream iot-edge-anomaly

# Check logs for errors
echo -e "\nRecent Errors:"
docker logs iot-edge-anomaly --since 168h | grep -i error | tail -10

# Check metrics endpoint
echo -e "\nMetrics Endpoint:"
curl -s http://localhost:8080/health | jq .

# Check model performance
echo -e "\nModel Performance:"
curl -s http://localhost:9090/api/v1/query?query=model_accuracy_ratio | jq '.data.result[0].value[1]'
```

#### Monthly Performance Review
1. Review alert frequency and accuracy
2. Analyze resource utilization trends
3. Check model performance degradation
4. Update capacity planning
5. Review and update alert thresholds

### Deployment Procedures

#### Model Update Deployment
```bash
#!/bin/bash
# Model update deployment script

MODEL_VERSION="$1"
BACKUP_PATH="/opt/model-backups"

# Backup current model
docker cp iot-edge-anomaly:/app/models/model.pth "$BACKUP_PATH/model-$(date +%Y%m%d-%H%M%S).pth"

# Deploy new model
docker cp "models/model-${MODEL_VERSION}.pth" iot-edge-anomaly:/app/models/model.pth

# Restart application to load new model
docker restart iot-edge-anomaly

# Wait for startup
sleep 30

# Validate deployment
if curl -f http://localhost:8080/health; then
    echo "Model deployment successful"
else
    echo "Model deployment failed, rolling back"
    docker cp "$BACKUP_PATH/model-$(ls -t $BACKUP_PATH | head -1)" iot-edge-anomaly:/app/models/model.pth
    docker restart iot-edge-anomaly
fi
```

#### Configuration Update
```bash
#!/bin/bash
# Configuration update script

CONFIG_FILE="$1"

# Validate configuration
if ! python -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))"; then
    echo "Invalid configuration file"
    exit 1
fi

# Update configuration
docker cp "$CONFIG_FILE" iot-edge-anomaly:/app/config/config.yaml

# Reload configuration (send SIGHUP)
docker kill --signal=HUP iot-edge-anomaly

# Verify configuration reload
sleep 5
if curl -f http://localhost:8080/health; then
    echo "Configuration update successful"
else
    echo "Configuration update failed"
    exit 1
fi
```

## Troubleshooting Guide

### Common Issues

#### Container Won't Start
**Symptoms**: Container exits immediately or fails to start

**Investigation Steps**:
1. Check container logs: `docker logs iot-edge-anomaly`
2. Verify image integrity: `docker image inspect iot-edge-anomaly:latest`
3. Check resource constraints: `docker info`
4. Validate environment variables and volumes

**Common Solutions**:
- Fix configuration errors
- Ensure required volumes are mounted
- Check file permissions
- Verify resource availability

#### High Error Rate
**Symptoms**: Increased HTTP 5xx errors or application exceptions

**Investigation Steps**:
1. Review application logs for error patterns
2. Check external service connectivity
3. Analyze request patterns and load
4. Verify model and configuration integrity

**Common Solutions**:
- Fix application bugs
- Improve error handling
- Scale resources
- Update configurations

#### Model Performance Degradation
**Symptoms**: Decreased accuracy or increased false positives

**Investigation Steps**:
1. Analyze recent data quality
2. Check for data distribution changes
3. Review model metrics and drift detection
4. Compare with baseline performance

**Common Solutions**:
- Retrain model with recent data
- Adjust detection thresholds
- Improve data preprocessing
- Implement data quality checks

### Performance Optimization

#### Memory Optimization
```python
# Memory usage monitoring
import psutil
import logging

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    logging.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB ({memory_percent:.1f}%)")
    
    if memory_percent > 80:
        logging.warning("High memory usage detected")
        # Trigger garbage collection
        import gc
        gc.collect()
```

#### CPU Optimization
```python
# CPU usage monitoring
import time
import threading

def monitor_cpu():
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            logging.warning(f"High CPU usage: {cpu_percent}%")
        time.sleep(60)

# Start monitoring thread
cpu_monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
cpu_monitor_thread.start()
```

## Emergency Procedures

### Complete System Failure

#### Recovery Steps
1. **Immediate Response** (0-5 minutes):
   - Stop all containers: `docker stop $(docker ps -q)`
   - Check system resources: `free -h && df -h`
   - Review system logs: `journalctl -xe`

2. **System Recovery** (5-15 minutes):
   - Restart Docker daemon: `sudo systemctl restart docker`
   - Pull latest images: `docker-compose pull`
   - Start essential services: `docker-compose up -d iot-edge-anomaly prometheus`

3. **Validation** (15-30 minutes):
   - Verify all services are running
   - Check health endpoints
   - Validate data processing
   - Test alert systems

### Data Corruption

#### Recovery Steps
1. **Stop Data Processing**: `docker stop iot-edge-anomaly`
2. **Backup Current State**: `cp -r /data /data-backup-$(date +%Y%m%d)`
3. **Restore from Backup**: Restore from latest clean backup
4. **Validate Data Integrity**: Run data validation scripts
5. **Resume Processing**: `docker start iot-edge-anomaly`

### Security Incident

#### Response Steps
1. **Isolate System**: Disconnect from network if necessary
2. **Preserve Evidence**: Capture logs and system state
3. **Notify Security Team**: Follow incident response procedures
4. **Investigate**: Analyze logs for unauthorized access
5. **Remediate**: Apply security patches and update credentials
6. **Monitor**: Enhanced monitoring for follow-up attacks

## Escalation Procedures

### Alert Escalation Matrix

| Severity | Response Time | Primary Contact | Secondary Contact | Manager |
|----------|---------------|-----------------|-------------------|---------|
| Critical | 5 minutes | On-call Engineer | DevOps Lead | Engineering Manager |
| Warning | 30 minutes | DevOps Team | Platform Team | DevOps Manager |
| Info | 4 hours | Platform Team | - | - |

### Contact Information

#### On-Call Rotation
- **Primary**: Slack #iot-oncall, PagerDuty
- **Secondary**: Slack #iot-devops
- **Manager**: Slack #iot-leadership

#### Vendor Contacts
- **Cloud Provider**: [Support Portal URL]
- **Monitoring Vendor**: [Support Contact]
- **Hardware Vendor**: [Support Contact]

---

**Runbook Version**: 1.0  
**Last Updated**: 2025-08-02  
**Next Review**: 2025-11-02
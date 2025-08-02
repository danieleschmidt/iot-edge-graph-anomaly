#!/bin/bash
# Entrypoint script for IoT Edge Graph Anomaly Detection container
# Handles initialization, configuration, and graceful shutdown

set -euo pipefail

# Default values
LOG_LEVEL="${LOG_LEVEL:-INFO}"
HEALTH_CHECK_PORT="${HEALTH_CHECK_PORT:-8080}"
METRICS_PORT="${METRICS_PORT:-9090}"

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1"
}

# Signal handlers for graceful shutdown
cleanup() {
    log_info "Received shutdown signal, cleaning up..."
    
    # Kill background processes
    if [[ -n "${APP_PID:-}" ]]; then
        log_info "Stopping application (PID: $APP_PID)..."
        kill -TERM "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    
    # Additional cleanup tasks
    cleanup_temp_files
    cleanup_resources
    
    log_success "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Cleanup temporary files
cleanup_temp_files() {
    log_info "Cleaning up temporary files..."
    rm -rf /tmp/iot-anomaly-* 2>/dev/null || true
}

# Cleanup application resources
cleanup_resources() {
    log_info "Cleaning up application resources..."
    # Close any open connections, flush buffers, etc.
    # This is application-specific
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check required directories
    local required_dirs=("/app/logs" "/app/models" "/app/data")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_warning "Creating missing directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Check required environment variables
    local required_vars=("DEVICE_ID")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_warning "Environment variable $var is not set, using default"
        fi
    done
    
    # Check model files
    local model_path="${MODEL_PATH:-/app/models/model.pth}"
    if [[ ! -f "$model_path" ]]; then
        log_warning "Model file not found at $model_path"
        log_warning "Application will run in training mode or use default model"
    else
        log_info "Model file found at $model_path"
    fi
    
    # Check permissions
    if [[ ! -w "/app/logs" ]]; then
        log_error "No write permission for logs directory"
        exit 1
    fi
    
    # Test network connectivity (if required)
    if [[ "${OTLP_ENDPOINT:-}" ]]; then
        local endpoint_host
        endpoint_host=$(echo "$OTLP_ENDPOINT" | sed -E 's|.*://([^:/]+).*|\1|')
        if command -v curl >/dev/null 2>&1; then
            if ! curl -s --max-time 5 "$endpoint_host" >/dev/null 2>&1; then
                log_warning "Cannot reach OTLP endpoint: $OTLP_ENDPOINT"
                log_warning "Metrics export may not work properly"
            fi
        fi
    fi
    
    log_success "Pre-flight checks completed"
}

# Initialize configuration
init_config() {
    log_info "Initializing configuration..."
    
    # Set default values for unset variables
    export DEVICE_ID="${DEVICE_ID:-edge-device-$(hostname)}"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    export ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-0.5}"
    export MEMORY_LIMIT="${MEMORY_LIMIT:-95}"
    export CPU_LIMIT="${CPU_LIMIT:-25}"
    
    # Create configuration file if it doesn't exist
    local config_file="/app/config/runtime.conf"
    if [[ ! -f "$config_file" ]]; then
        log_info "Creating runtime configuration file..."
        mkdir -p "$(dirname "$config_file")"
        cat > "$config_file" << EOF
# Runtime configuration for IoT Edge Anomaly Detection
device_id=$DEVICE_ID
log_level=$LOG_LEVEL
anomaly_threshold=$ANOMALY_THRESHOLD
memory_limit=$MEMORY_LIMIT
cpu_limit=$CPU_LIMIT
health_check_port=$HEALTH_CHECK_PORT
metrics_port=$METRICS_PORT
startup_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
    fi
    
    log_success "Configuration initialized"
}

# Health check service
start_health_check() {
    log_info "Starting health check service on port $HEALTH_CHECK_PORT..."
    
    # Simple HTTP health check server
    while true; do
        echo -e "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK" | nc -l -p "$HEALTH_CHECK_PORT" -q 1 2>/dev/null || true
        sleep 1
    done &
    
    HEALTH_CHECK_PID=$!
    log_success "Health check service started (PID: $HEALTH_CHECK_PID)"
}

# Monitor resources
monitor_resources() {
    log_info "Starting resource monitoring..."
    
    while true; do
        # Check memory usage
        local memory_usage
        memory_usage=$(ps -o pid,ppid,cmd,%mem,%cpu --sort=-%mem -p $$ | awk 'NR>1 {print $4}' | head -1)
        
        if (( $(echo "$memory_usage > ${MEMORY_LIMIT:-95}" | bc -l) )); then
            log_warning "Memory usage high: ${memory_usage}%"
        fi
        
        # Check if application is still running
        if [[ -n "${APP_PID:-}" ]] && ! kill -0 "$APP_PID" 2>/dev/null; then
            log_error "Application process died unexpectedly"
            exit 1
        fi
        
        sleep 30
    done &
    
    MONITOR_PID=$!
}

# Start the main application
start_application() {
    log_info "Starting IoT Edge Graph Anomaly Detection application..."
    
    # Export environment variables for the application
    export PYTHONUNBUFFERED=1
    export PYTHONDONTWRITEBYTECODE=1
    
    # Start the application in background
    "$@" &
    APP_PID=$!
    
    log_success "Application started (PID: $APP_PID)"
    
    # Wait for application to be ready
    local max_wait=30
    local wait_time=0
    
    while [[ $wait_time -lt $max_wait ]]; do
        if kill -0 "$APP_PID" 2>/dev/null; then
            log_success "Application is running"
            break
        fi
        
        sleep 1
        ((wait_time++))
    done
    
    if [[ $wait_time -ge $max_wait ]]; then
        log_error "Application failed to start within $max_wait seconds"
        exit 1
    fi
}

# Main execution
main() {
    log_info "IoT Edge Graph Anomaly Detection - Container Starting"
    log_info "Version: ${VERSION:-unknown}"
    log_info "Device ID: ${DEVICE_ID:-unknown}"
    log_info "Log Level: $LOG_LEVEL"
    
    # Run initialization
    preflight_checks
    init_config
    
    # Start background services
    start_health_check
    monitor_resources
    
    # Start main application
    start_application "$@"
    
    log_success "All services started successfully"
    log_info "Container is ready to accept requests"
    
    # Wait for the main application
    wait "$APP_PID"
    
    # If we reach here, the application exited
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        log_info "Application exited successfully"
    else
        log_error "Application exited with code: $exit_code"
    fi
    
    # Cleanup and exit
    cleanup
    exit $exit_code
}

# Check if running as PID 1 (in container)
if [[ $$ -eq 1 ]]; then
    # Running as init process, handle signals properly
    main "$@"
else
    # Running in shell, just execute the command
    exec "$@"
fi
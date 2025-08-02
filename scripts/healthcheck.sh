#!/bin/bash
# Health check script for IoT Edge Graph Anomaly Detection container
# Performs comprehensive health checks for containerized deployment

set -euo pipefail

# Configuration
HEALTH_CHECK_PORT="${HEALTH_CHECK_PORT:-8080}"
METRICS_PORT="${METRICS_PORT:-9090}"
APP_PORT="${APP_PORT:-8000}"
TIMEOUT="${HEALTH_CHECK_TIMEOUT:-10}"
MAX_MEMORY_PERCENT="${MAX_MEMORY_PERCENT:-90}"
MAX_CPU_PERCENT="${MAX_CPU_PERCENT:-80}"

# Health check results
HEALTH_STATUS="healthy"
HEALTH_DETAILS=()
HEALTH_METRICS=()

# Colors (only use if output is a terminal)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Logging functions
log_info() {
    echo -e "${BLUE}[HEALTH]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[HEALTH]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[HEALTH]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[HEALTH]${NC} $1" >&2
}

# Add health detail
add_detail() {
    HEALTH_DETAILS+=("$1")
}

# Add health metric
add_metric() {
    HEALTH_METRICS+=("$1")
}

# Mark as unhealthy
mark_unhealthy() {
    HEALTH_STATUS="unhealthy"
    add_detail "$1"
    log_error "$1"
}

# Mark as degraded
mark_degraded() {
    if [[ "$HEALTH_STATUS" == "healthy" ]]; then
        HEALTH_STATUS="degraded"
    fi
    add_detail "$1"
    log_warning "$1"
}

# Check if port is listening
check_port() {
    local port=$1
    local service=$2
    
    if command -v nc >/dev/null 2>&1; then
        if ! nc -z -w5 localhost "$port" 2>/dev/null; then
            mark_unhealthy "$service port $port is not accessible"
            return 1
        fi
    elif command -v curl >/dev/null 2>&1; then
        if ! curl -f -s -m5 "http://localhost:$port/" >/dev/null 2>&1; then
            mark_unhealthy "$service port $port is not accessible"
            return 1
        fi
    else
        mark_degraded "Cannot check $service port $port (no nc or curl available)"
        return 1
    fi
    
    add_detail "$service port $port is accessible"
    return 0
}

# Check HTTP endpoint
check_http_endpoint() {
    local url=$1
    local service=$2
    local expected_status=${3:-200}
    
    if ! command -v curl >/dev/null 2>&1; then
        mark_degraded "Cannot check $service endpoint (curl not available)"
        return 1
    fi
    
    local http_code
    http_code=$(curl -f -s -m"$TIMEOUT" -w "%{http_code}" -o /dev/null "$url" 2>/dev/null || echo "000")
    
    if [[ "$http_code" != "$expected_status" ]]; then
        mark_unhealthy "$service endpoint $url returned HTTP $http_code (expected $expected_status)"
        return 1
    fi
    
    add_detail "$service endpoint $url is healthy"
    return 0
}

# Check process health
check_process_health() {
    local process_name="python"
    
    # Check if main process is running
    if ! pgrep -f "iot_edge_anomaly.main" >/dev/null 2>&1; then
        mark_unhealthy "Main application process is not running"
        return 1
    fi
    
    # Get process info
    local pid
    pid=$(pgrep -f "iot_edge_anomaly.main" | head -1)
    
    if [[ -n "$pid" ]]; then
        add_detail "Main process is running (PID: $pid)"
        add_metric "process_pid=$pid"
        
        # Check process status
        if [[ -f "/proc/$pid/stat" ]]; then
            local stat
            stat=$(cat "/proc/$pid/stat")
            local state
            state=$(echo "$stat" | awk '{print $3}')
            
            case $state in
                S|R)
                    add_detail "Process state is healthy ($state)"
                    ;;
                D)
                    mark_degraded "Process is in uninterruptible sleep"
                    ;;
                Z)
                    mark_unhealthy "Process is a zombie"
                    ;;
                T)
                    mark_unhealthy "Process is stopped"
                    ;;
                *)
                    mark_degraded "Process state unknown ($state)"
                    ;;
            esac
        fi
    fi
    
    return 0
}

# Check memory usage
check_memory_usage() {
    local memory_info
    
    if [[ -f "/proc/meminfo" ]]; then
        memory_info=$(cat /proc/meminfo)
        local mem_total
        mem_total=$(echo "$memory_info" | grep "^MemTotal:" | awk '{print $2}')
        local mem_available
        mem_available=$(echo "$memory_info" | grep "^MemAvailable:" | awk '{print $2}')
        
        if [[ -n "$mem_total" && -n "$mem_available" ]]; then
            local mem_used=$((mem_total - mem_available))
            local mem_percent=$((mem_used * 100 / mem_total))
            
            add_metric "memory_used_percent=$mem_percent"
            add_metric "memory_used_kb=$mem_used"
            add_metric "memory_total_kb=$mem_total"
            
            if [[ $mem_percent -gt $MAX_MEMORY_PERCENT ]]; then
                mark_unhealthy "Memory usage too high: ${mem_percent}% (limit: ${MAX_MEMORY_PERCENT}%)"
            else
                add_detail "Memory usage is acceptable: ${mem_percent}%"
            fi
        fi
    else
        mark_degraded "Cannot check memory usage (/proc/meminfo not available)"
    fi
}

# Check CPU usage
check_cpu_usage() {
    if [[ -f "/proc/loadavg" ]]; then
        local loadavg
        loadavg=$(cat /proc/loadavg)
        local load1
        load1=$(echo "$loadavg" | awk '{print $1}')
        
        add_metric "load_average_1min=$load1"
        
        # Get number of CPU cores
        local cpu_cores
        cpu_cores=$(nproc 2>/dev/null || echo "1")
        add_metric "cpu_cores=$cpu_cores"
        
        # Calculate CPU usage percentage (rough estimate)
        local cpu_percent
        cpu_percent=$(echo "$load1 * 100 / $cpu_cores" | bc -l 2>/dev/null | cut -d. -f1 || echo "0")
        
        add_metric "cpu_usage_percent=$cpu_percent"
        
        if [[ $cpu_percent -gt $MAX_CPU_PERCENT ]]; then
            mark_degraded "CPU usage high: ${cpu_percent}% (limit: ${MAX_CPU_PERCENT}%)"
        else
            add_detail "CPU usage is acceptable: ${cpu_percent}%"
        fi
    else
        mark_degraded "Cannot check CPU usage (/proc/loadavg not available)"
    fi
}

# Check disk space
check_disk_space() {
    local disk_usage
    
    if command -v df >/dev/null 2>&1; then
        # Check root filesystem
        disk_usage=$(df / 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//')
        
        if [[ -n "$disk_usage" ]]; then
            add_metric "disk_usage_percent=$disk_usage"
            
            if [[ $disk_usage -gt 90 ]]; then
                mark_unhealthy "Disk usage too high: ${disk_usage}%"
            elif [[ $disk_usage -gt 80 ]]; then
                mark_degraded "Disk usage high: ${disk_usage}%"
            else
                add_detail "Disk usage is acceptable: ${disk_usage}%"
            fi
        fi
        
        # Check logs directory if it exists
        if [[ -d "/app/logs" ]]; then
            local logs_usage
            logs_usage=$(du -sh /app/logs 2>/dev/null | awk '{print $1}' || echo "unknown")
            add_metric "logs_size=$logs_usage"
        fi
    else
        mark_degraded "Cannot check disk usage (df command not available)"
    fi
}

# Check file permissions
check_file_permissions() {
    local required_dirs=("/app/logs")
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            if [[ ! -w "$dir" ]]; then
                mark_unhealthy "No write permission for $dir"
            else
                add_detail "Write permission OK for $dir"
            fi
        else
            mark_degraded "Required directory missing: $dir"
        fi
    done
}

# Check configuration
check_configuration() {
    # Check required environment variables
    local required_vars=("DEVICE_ID")
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            mark_degraded "Required environment variable $var is not set"
        else
            add_detail "Environment variable $var is set"
        fi
    done
    
    # Check model file
    local model_path="${MODEL_PATH:-/app/models/model.pth}"
    if [[ -f "$model_path" ]]; then
        add_detail "Model file exists: $model_path"
        
        # Check if model file is readable
        if [[ ! -r "$model_path" ]]; then
            mark_unhealthy "Model file is not readable: $model_path"
        fi
    else
        mark_degraded "Model file not found: $model_path"
    fi
}

# Generate health report
generate_report() {
    local timestamp
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    echo "{"
    echo "  \"status\": \"$HEALTH_STATUS\","
    echo "  \"timestamp\": \"$timestamp\","
    echo "  \"checks\": {"
    
    # Add check details
    if [[ ${#HEALTH_DETAILS[@]} -gt 0 ]]; then
        echo "    \"details\": ["
        for i in "${!HEALTH_DETAILS[@]}"; do
            echo -n "      \"${HEALTH_DETAILS[$i]}\""
            if [[ $i -lt $((${#HEALTH_DETAILS[@]} - 1)) ]]; then
                echo ","
            else
                echo ""
            fi
        done
        echo "    ],"
    fi
    
    # Add metrics
    if [[ ${#HEALTH_METRICS[@]} -gt 0 ]]; then
        echo "    \"metrics\": {"
        for i in "${!HEALTH_METRICS[@]}"; do
            local metric="${HEALTH_METRICS[$i]}"
            local key="${metric%=*}"
            local value="${metric#*=}"
            
            echo -n "      \"$key\": \"$value\""
            if [[ $i -lt $((${#HEALTH_METRICS[@]} - 1)) ]]; then
                echo ","
            else
                echo ""
            fi
        done
        echo "    }"
    fi
    
    echo "  }"
    echo "}"
}

# Main health check
main() {
    log_info "Starting health check..."
    
    # Run all health checks
    check_process_health
    check_memory_usage
    check_cpu_usage
    check_disk_space
    check_file_permissions
    check_configuration
    
    # Check network endpoints
    check_port "$HEALTH_CHECK_PORT" "health check" || true
    check_port "$APP_PORT" "application" || true
    
    # Check HTTP endpoints if curl is available
    if command -v curl >/dev/null 2>&1; then
        check_http_endpoint "http://localhost:$HEALTH_CHECK_PORT/health" "health check" || true
        check_http_endpoint "http://localhost:$METRICS_PORT/metrics" "metrics" || true
    fi
    
    # Generate and output report
    if [[ "${HEALTH_CHECK_JSON:-false}" == "true" ]]; then
        generate_report
    else
        # Simple output for Docker HEALTHCHECK
        case $HEALTH_STATUS in
            healthy)
                log_success "Health check passed"
                echo "healthy"
                ;;
            degraded)
                log_warning "Health check passed with warnings"
                echo "degraded"
                ;;
            unhealthy)
                log_error "Health check failed"
                echo "unhealthy"
                exit 1
                ;;
        esac
    fi
    
    # Exit with appropriate code
    case $HEALTH_STATUS in
        healthy)
            exit 0
            ;;
        degraded)
            exit 0  # Still considered passing for Docker
            ;;
        unhealthy)
            exit 1
            ;;
    esac
}

# Handle different invocation modes
case "${1:-}" in
    --json)
        HEALTH_CHECK_JSON=true
        main
        ;;
    --verbose)
        set -x
        main
        ;;
    *)
        main
        ;;
esac
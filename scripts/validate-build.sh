#!/bin/bash
# Build validation script for IoT Edge Anomaly Detection
# Validates Docker images, security, and performance requirements

set -euo pipefail

# Configuration
IMAGE_NAME="${IMAGE_NAME:-iot-edge-graph-anomaly}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
VALIDATION_TIMEOUT="${VALIDATION_TIMEOUT:-300}"
MEMORY_LIMIT="${MEMORY_LIMIT:-100}"  # MB
CPU_LIMIT="${CPU_LIMIT:-0.25}"       # CPU cores
LATENCY_REQUIREMENT="${LATENCY_REQUIREMENT:-10}"  # milliseconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
VALIDATION_RESULTS=()

# Helper function to record test results
record_result() {
    local test_name="$1"
    local status="$2"
    local message="$3"
    
    if [[ "$status" == "PASS" ]]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        VALIDATION_RESULTS+=("✅ $test_name: $message")
        log_success "$test_name: $message"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        VALIDATION_RESULTS+=("❌ $test_name: $message")
        log_error "$test_name: $message"
    fi
}

# Test 1: Image existence and basic properties
test_image_properties() {
    log_info "Testing image properties..."
    
    # Check if image exists
    if docker image inspect "$IMAGE_NAME:$IMAGE_TAG" &> /dev/null; then
        record_result "Image Existence" "PASS" "Image $IMAGE_NAME:$IMAGE_TAG exists"
    else
        record_result "Image Existence" "FAIL" "Image $IMAGE_NAME:$IMAGE_TAG not found"
        return 1
    fi
    
    # Check image size
    local image_size_bytes=$(docker image inspect "$IMAGE_NAME:$IMAGE_TAG" --format='{{.Size}}')
    local image_size_mb=$((image_size_bytes / 1024 / 1024))
    
    if [[ $image_size_mb -le 500 ]]; then
        record_result "Image Size" "PASS" "Image size: ${image_size_mb}MB (≤500MB)"
    else
        record_result "Image Size" "FAIL" "Image size: ${image_size_mb}MB (>500MB)"
    fi
    
    # Check image layers
    local layer_count=$(docker image inspect "$IMAGE_NAME:$IMAGE_TAG" --format='{{len .RootFS.Layers}}')
    
    if [[ $layer_count -le 20 ]]; then
        record_result "Image Layers" "PASS" "Layer count: $layer_count (≤20)"
    else
        record_result "Image Layers" "FAIL" "Layer count: $layer_count (>20)"
    fi
}

# Test 2: Security validation
test_security() {
    log_info "Testing security configuration..."
    
    # Check if running as non-root
    local user_info=$(docker run --rm "$IMAGE_NAME:$IMAGE_TAG" id 2>/dev/null || echo "uid=0(root)")
    
    if [[ "$user_info" != *"uid=0(root)"* ]]; then
        record_result "Non-root User" "PASS" "Container runs as non-root user"
    else
        record_result "Non-root User" "FAIL" "Container runs as root user"
    fi
    
    # Security scan with Trivy (if available)
    if command -v trivy &> /dev/null; then
        log_info "Running security scan with Trivy..."
        local critical_vulns=$(trivy image --format json "$IMAGE_NAME:$IMAGE_TAG" 2>/dev/null | jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' 2>/dev/null || echo "unknown")
        
        if [[ "$critical_vulns" == "0" ]]; then
            record_result "Security Scan" "PASS" "No critical vulnerabilities found"
        elif [[ "$critical_vulns" == "unknown" ]]; then
            record_result "Security Scan" "FAIL" "Security scan failed"
        else
            record_result "Security Scan" "FAIL" "$critical_vulns critical vulnerabilities found"
        fi
    else
        log_warning "Trivy not available, skipping security scan"
        record_result "Security Scan" "FAIL" "Security scanner not available"
    fi
}

# Test 3: Container startup and health
test_container_startup() {
    log_info "Testing container startup and health..."
    
    local container_name="validation-test-$$"
    
    # Start container with resource limits
    log_info "Starting container with resource limits..."
    if docker run -d \
        --name "$container_name" \
        --memory="${MEMORY_LIMIT}m" \
        --cpus="$CPU_LIMIT" \
        -p 8080:8080 \
        "$IMAGE_NAME:$IMAGE_TAG" &> /dev/null; then
        record_result "Container Start" "PASS" "Container started successfully"
    else
        record_result "Container Start" "FAIL" "Container failed to start"
        return 1
    fi
    
    # Wait for container to be ready
    log_info "Waiting for container to be ready..."
    local ready=false
    local attempts=0
    local max_attempts=30  # 30 seconds timeout
    
    while [[ $attempts -lt $max_attempts ]]; do
        if docker exec "$container_name" python -c "import iot_edge_anomaly; print('OK')" &> /dev/null; then
            ready=true
            break
        fi
        sleep 1
        attempts=$((attempts + 1))
    done
    
    if [[ "$ready" == "true" ]]; then
        record_result "Container Ready" "PASS" "Container ready in ${attempts}s (≤30s)"
    else
        record_result "Container Ready" "FAIL" "Container not ready after 30s"
        docker logs "$container_name" | tail -20
    fi
    
    # Test health check endpoint (if available)
    local health_attempts=0
    local health_max_attempts=10
    local health_ok=false
    
    while [[ $health_attempts -lt $health_max_attempts ]]; do
        if curl -f -s http://localhost:8080/health &> /dev/null; then
            health_ok=true
            break
        fi
        sleep 2
        health_attempts=$((health_attempts + 1))
    done
    
    if [[ "$health_ok" == "true" ]]; then
        record_result "Health Check" "PASS" "Health endpoint responds correctly"
    else
        record_result "Health Check" "FAIL" "Health endpoint not responding"
    fi
    
    # Cleanup
    docker stop "$container_name" &> /dev/null || true
    docker rm "$container_name" &> /dev/null || true
}

# Test 4: Resource consumption validation
test_resource_consumption() {
    log_info "Testing resource consumption..."
    
    local container_name="resource-test-$$"
    
    # Start container with monitoring
    docker run -d \
        --name "$container_name" \
        --memory="${MEMORY_LIMIT}m" \
        --cpus="$CPU_LIMIT" \
        "$IMAGE_NAME:$IMAGE_TAG" &> /dev/null || return 1
    
    # Wait for startup
    sleep 10
    
    # Check memory usage
    local memory_usage=$(docker stats --no-stream --format "table {{.MemUsage}}" "$container_name" | tail -n +2 | cut -d'/' -f1 | sed 's/MiB//' | sed 's/ //g')
    
    if [[ -n "$memory_usage" ]] && (( $(echo "$memory_usage < $MEMORY_LIMIT" | bc -l) )); then
        record_result "Memory Usage" "PASS" "Memory usage: ${memory_usage}MB (≤${MEMORY_LIMIT}MB)"
    else
        record_result "Memory Usage" "FAIL" "Memory usage: ${memory_usage}MB (>${MEMORY_LIMIT}MB)"
    fi
    
    # Check CPU usage
    local cpu_usage=$(docker stats --no-stream --format "table {{.CPUPerc}}" "$container_name" | tail -n +2 | sed 's/%//')
    local cpu_limit_percent=$(echo "$CPU_LIMIT * 100" | bc)
    
    if [[ -n "$cpu_usage" ]] && (( $(echo "$cpu_usage < $cpu_limit_percent" | bc -l) )); then
        record_result "CPU Usage" "PASS" "CPU usage: ${cpu_usage}% (≤${cpu_limit_percent}%)"
    else
        record_result "CPU Usage" "FAIL" "CPU usage: ${cpu_usage}% (>${cpu_limit_percent}%)"
    fi
    
    # Cleanup
    docker stop "$container_name" &> /dev/null || true
    docker rm "$container_name" &> /dev/null || true
}

# Test 5: API functionality (if applicable)
test_api_functionality() {
    log_info "Testing API functionality..."
    
    local container_name="api-test-$$"
    
    # Start container
    docker run -d \
        --name "$container_name" \
        -p 8000:8000 \
        -p 8080:8080 \
        "$IMAGE_NAME:$IMAGE_TAG" &> /dev/null || return 1
    
    # Wait for startup
    sleep 15
    
    # Test health endpoint
    if curl -f -s http://localhost:8080/health | grep -q "healthy\|ok\|OK"; then
        record_result "Health API" "PASS" "Health API responds correctly"
    else
        record_result "Health API" "FAIL" "Health API not responding correctly"
    fi
    
    # Test metrics endpoint (if available)
    if curl -f -s http://localhost:8080/metrics &> /dev/null; then
        record_result "Metrics API" "PASS" "Metrics endpoint available"
    else
        record_result "Metrics API" "FAIL" "Metrics endpoint not available"
    fi
    
    # Cleanup
    docker stop "$container_name" &> /dev/null || true
    docker rm "$container_name" &> /dev/null || true
}

# Generate validation report
generate_report() {
    local total_tests=$((TESTS_PASSED + TESTS_FAILED))
    local success_rate=$((TESTS_PASSED * 100 / total_tests))
    
    log_info "Generating validation report..."
    
    local report_file="validation-report-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$report_file" << EOF
IoT Edge Anomaly Detection - Build Validation Report
====================================================

Validation Summary:
  Image: $IMAGE_NAME:$IMAGE_TAG
  Date: $(date -u)
  Total Tests: $total_tests
  Passed: $TESTS_PASSED
  Failed: $TESTS_FAILED
  Success Rate: ${success_rate}%

Test Results:
EOF

    for result in "${VALIDATION_RESULTS[@]}"; do
        echo "  $result" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF

Resource Requirements Validation:
  Memory Limit: ${MEMORY_LIMIT}MB
  CPU Limit: ${CPU_LIMIT} cores
  Latency Requirement: <${LATENCY_REQUIREMENT}ms

Recommendations:
EOF

    if [[ $TESTS_FAILED -eq 0 ]]; then
        cat >> "$report_file" << EOF
  ✅ All validation tests passed
  ✅ Image is ready for production deployment
  ✅ Resource requirements are met
  ✅ Security requirements are satisfied
EOF
    else
        cat >> "$report_file" << EOF
  ❌ $TESTS_FAILED validation test(s) failed
  ⚠️  Review failed tests before production deployment
  ⚠️  Address security and resource issues
  ⚠️  Re-run validation after fixes
EOF
    fi
    
    log_success "Validation report saved to: $report_file"
    
    # Print summary
    echo ""
    log_info "=== VALIDATION SUMMARY ==="
    for result in "${VALIDATION_RESULTS[@]}"; do
        echo "  $result"
    done
    echo ""
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "🎉 All validation tests PASSED! Image is ready for deployment."
        return 0
    else
        log_error "❌ $TESTS_FAILED test(s) FAILED. Image needs fixes before deployment."
        return 1
    fi
}

# Cleanup function
cleanup_on_exit() {
    log_info "Cleaning up test containers..."
    docker ps -q --filter "name=validation-test-*" | xargs -r docker stop &> /dev/null || true
    docker ps -q --filter "name=resource-test-*" | xargs -r docker stop &> /dev/null || true
    docker ps -q --filter "name=api-test-*" | xargs -r docker stop &> /dev/null || true
    docker ps -aq --filter "name=validation-test-*" | xargs -r docker rm &> /dev/null || true
    docker ps -aq --filter "name=resource-test-*" | xargs -r docker rm &> /dev/null || true
    docker ps -aq --filter "name=api-test-*" | xargs -r docker rm &> /dev/null || true
}

# Set trap for cleanup
trap cleanup_on_exit EXIT

# Main execution
main() {
    log_info "Starting build validation for $IMAGE_NAME:$IMAGE_TAG"
    
    # Check prerequisites
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed or not in PATH"
        exit 1
    fi
    
    # Run validation tests
    test_image_properties
    test_security
    test_container_startup
    test_resource_consumption
    test_api_functionality
    
    # Generate and display report
    generate_report
}

# Run main function
main "$@"
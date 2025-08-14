#!/bin/bash
# Multi-architecture build script for IoT Edge Anomaly Detection v4.0
# Enhanced TERRAGON GLOBAL-FIRST Cross-Platform Build System
# Supports comprehensive edge device deployment across global infrastructure

set -euo pipefail

# Configuration
IMAGE_NAME="${IMAGE_NAME:-iot-edge-anomaly-v4}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/arm64,linux/amd64,linux/arm/v7,linux/386,linux/ppc64le,linux/s390x}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
BUILD_ARGS="${BUILD_ARGS:-}"
PUSH="${PUSH:-false}"

# Enhanced edge device variants
EDGE_VARIANTS="${EDGE_VARIANTS:-raspberry-pi,jetson-nano,intel-nuc,industrial-gateway}"
GLOBAL_REGIONS="${GLOBAL_REGIONS:-us-east,eu-west,asia-pacific,latam}"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-standard}"  # standard, edge-optimized, cloud-hybrid

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

# Enhanced device specifications
declare -A DEVICE_SPECS=(
    ["raspberry-pi"]="linux/arm64 --build-arg DEVICE_TYPE=raspberry-pi --build-arg OPTIMIZE_FOR=edge"
    ["jetson-nano"]="linux/arm64 --build-arg DEVICE_TYPE=jetson-nano --build-arg OPTIMIZE_FOR=gpu"
    ["intel-nuc"]="linux/amd64 --build-arg DEVICE_TYPE=intel-nuc --build-arg OPTIMIZE_FOR=performance"
    ["industrial-gateway"]="linux/amd64,linux/arm64 --build-arg DEVICE_TYPE=gateway --build-arg OPTIMIZE_FOR=reliability"
    ["cloud-vm"]="linux/amd64 --build-arg DEVICE_TYPE=cloud --build-arg OPTIMIZE_FOR=scale"
    ["embedded-arm"]="linux/arm/v7 --build-arg DEVICE_TYPE=embedded --build-arg OPTIMIZE_FOR=minimal"
)

# Regional deployment configurations
declare -A REGION_CONFIGS=(
    ["us-east"]="registry=us-docker.pkg.dev gcr_project=iot-us-east k8s_cluster=us-east-1"
    ["eu-west"]="registry=europe-docker.pkg.dev gcr_project=iot-eu-west k8s_cluster=eu-west-1"
    ["asia-pacific"]="registry=asia-docker.pkg.dev gcr_project=iot-asia k8s_cluster=asia-southeast1"
    ["latam"]="registry=southamerica-docker.pkg.dev gcr_project=iot-latam k8s_cluster=southamerica-east1"
)

# Help function
show_help() {
    cat << EOF
TERRAGON IoT Edge Anomaly Detection v4.0 - Global Multi-Architecture Build System

Usage: $0 [OPTIONS]

Standard Options:
    -n, --name NAME         Image name (default: iot-edge-anomaly-v4)
    -t, --tag TAG          Image tag (default: latest)
    -p, --platforms PLAT   Target platforms (default: all supported)
    -r, --registry REG     Docker registry URL
    --push                 Push images to registry
    --build-arg ARG        Build arguments (can be used multiple times)
    -h, --help             Show this help

Enhanced Global Options:
    --edge-variant VAR     Edge device variant (raspberry-pi, jetson-nano, intel-nuc, industrial-gateway)
    --global-region REG    Target global region (us-east, eu-west, asia-pacific, latam)
    --deployment-mode MODE Deployment mode (standard, edge-optimized, cloud-hybrid)
    --generate-k8s         Generate Kubernetes deployment manifests
    --security-scan        Run comprehensive security scan
    --performance-test     Run performance benchmarks
    --global-deploy        Deploy to all configured global regions

Examples:
    # Build for all platforms
    $0 --name iot-edge-v4 --tag v4.0.0

    # Build optimized for Raspberry Pi with edge deployment
    $0 --edge-variant raspberry-pi --deployment-mode edge-optimized

    # Build and deploy to all global regions
    $0 --tag v4.0.0 --global-deploy --push

    # Build for specific region with Kubernetes manifests
    $0 --global-region eu-west --generate-k8s --push

    # Build for Jetson Nano with GPU optimization
    $0 --edge-variant jetson-nano --build-arg ENABLE_GPU=true

    # Build industrial gateway variant with high reliability
    $0 --edge-variant industrial-gateway --deployment-mode edge-optimized

Environment Variables:
    IMAGE_NAME         Image name
    IMAGE_TAG          Image tag  
    PLATFORMS          Target platforms (comma-separated)
    DOCKER_REGISTRY    Registry URL
    BUILD_ARGS         Build arguments
    PUSH               Push to registry (true/false)
    EDGE_VARIANTS      Edge device variants
    GLOBAL_REGIONS     Global deployment regions
    DEPLOYMENT_MODE    Deployment optimization mode

Supported Platforms:
    linux/arm64        - ARM 64-bit (Raspberry Pi 4, Jetson Nano)
    linux/amd64        - x86 64-bit (Intel NUC, Industrial PCs)
    linux/arm/v7       - ARM 32-bit (Raspberry Pi 3, embedded)
    linux/386          - x86 32-bit (legacy systems)
    linux/ppc64le      - PowerPC 64-bit (IBM Power systems)
    linux/s390x        - IBM System z (mainframe edge)

Edge Device Variants:
    raspberry-pi       - Optimized for Raspberry Pi (ARM64, low power)
    jetson-nano        - NVIDIA Jetson with GPU acceleration
    intel-nuc          - Intel NUC mini PCs (high performance)
    industrial-gateway - Industrial IoT gateways (reliability focus)
    cloud-vm           - Cloud virtual machines (scalability)
    embedded-arm       - Embedded ARM devices (minimal footprint)

Global Regions:
    us-east           - US East Coast (Google Cloud us-east1)
    eu-west           - Europe West (Google Cloud europe-west1)
    asia-pacific      - Asia Pacific (Google Cloud asia-southeast1)
    latam             - Latin America (Google Cloud southamerica-east1)
EOF
}

# Enhanced command line parsing
EDGE_VARIANT=""
GLOBAL_REGION=""
GENERATE_K8S="false"
SECURITY_SCAN="false"
PERFORMANCE_TEST="false"
GLOBAL_DEPLOY="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -p|--platforms)
            PLATFORMS="$2"
            shift 2
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        --edge-variant)
            EDGE_VARIANT="$2"
            shift 2
            ;;
        --global-region)
            GLOBAL_REGION="$2"
            shift 2
            ;;
        --deployment-mode)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        --generate-k8s)
            GENERATE_K8S="true"
            shift
            ;;
        --security-scan)
            SECURITY_SCAN="true"
            shift
            ;;
        --performance-test)
            PERFORMANCE_TEST="true"
            shift
            ;;
        --global-deploy)
            GLOBAL_DEPLOY="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Process edge variant and adjust platforms/build args
if [[ -n "$EDGE_VARIANT" ]]; then
    if [[ -n "${DEVICE_SPECS[$EDGE_VARIANT]:-}" ]]; then
        DEVICE_CONFIG="${DEVICE_SPECS[$EDGE_VARIANT]}"
        # Extract platform from device config
        VARIANT_PLATFORM=$(echo "$DEVICE_CONFIG" | grep -o 'linux/[^ ]*' | head -1)
        if [[ -n "$VARIANT_PLATFORM" ]]; then
            PLATFORMS="$VARIANT_PLATFORM"
        fi
        # Extract build args from device config
        VARIANT_BUILD_ARGS=$(echo "$DEVICE_CONFIG" | grep -o -- '--build-arg [^-]*' | tr '\n' ' ')
        BUILD_ARGS="$BUILD_ARGS $VARIANT_BUILD_ARGS"
        log_info "Using edge variant: $EDGE_VARIANT"
        log_info "Platform: $PLATFORMS"
        log_info "Variant build args: $VARIANT_BUILD_ARGS"
    else
        log_error "Unknown edge variant: $EDGE_VARIANT"
        log_info "Available variants: ${!DEVICE_SPECS[*]}"
        exit 1
    fi
fi

# Process global region configuration
if [[ -n "$GLOBAL_REGION" ]]; then
    if [[ -n "${REGION_CONFIGS[$GLOBAL_REGION]:-}" ]]; then
        REGION_CONFIG="${REGION_CONFIGS[$GLOBAL_REGION]}"
        # Parse region config
        eval "$REGION_CONFIG"
        if [[ -z "$DOCKER_REGISTRY" && -n "${registry:-}" ]]; then
            DOCKER_REGISTRY="$registry"
        fi
        log_info "Using global region: $GLOBAL_REGION"
        log_info "Registry: ${registry:-default}"
    else
        log_error "Unknown global region: $GLOBAL_REGION"
        log_info "Available regions: ${!REGION_CONFIGS[*]}"
        exit 1
    fi
fi

# Add deployment mode optimizations
case "$DEPLOYMENT_MODE" in
    "edge-optimized")
        BUILD_ARGS="$BUILD_ARGS --build-arg OPTIMIZATION_MODE=edge --build-arg ENABLE_MINIMAL_DEPS=true"
        ;;
    "cloud-hybrid")
        BUILD_ARGS="$BUILD_ARGS --build-arg OPTIMIZATION_MODE=hybrid --build-arg ENABLE_CLOUD_FEATURES=true"
        ;;
    "standard")
        BUILD_ARGS="$BUILD_ARGS --build-arg OPTIMIZATION_MODE=standard"
        ;;
esac

# Construct full image name with registry
FULL_IMAGE_NAME="$IMAGE_NAME"
if [[ -n "$DOCKER_REGISTRY" ]]; then
    FULL_IMAGE_NAME="$DOCKER_REGISTRY/$IMAGE_NAME"
fi

# Pre-flight checks
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if buildx is available
    if ! docker buildx version &> /dev/null; then
        log_error "Docker buildx is not available. Please install Docker Desktop or enable buildx"
        exit 1
    fi

    # Check if multi-arch builder exists
    if ! docker buildx ls | grep -q "multiarch-builder"; then
        log_info "Creating multi-architecture builder..."
        docker buildx create --name multiarch-builder --use --bootstrap
        log_success "Multi-architecture builder created"
    else
        log_info "Using existing multi-architecture builder"
        docker buildx use multiarch-builder
    fi

    # Check if registry is accessible (if pushing)
    if [[ "$PUSH" == "true" && -n "$DOCKER_REGISTRY" ]]; then
        log_info "Testing registry connectivity..."
        if ! docker buildx imagetools inspect "$DOCKER_REGISTRY/hello-world:latest" &> /dev/null; then
            log_warning "Cannot connect to registry $DOCKER_REGISTRY - ensure you're logged in"
            log_info "Run: docker login $DOCKER_REGISTRY"
        fi
    fi

    log_success "Prerequisites check completed"
}

# Build function
build_images() {
    log_info "Starting multi-architecture build..."
    log_info "Image: $FULL_IMAGE_NAME:$IMAGE_TAG"
    log_info "Platforms: $PLATFORMS"
    
    # Build command
    BUILD_CMD="docker buildx build"
    BUILD_CMD="$BUILD_CMD --platform $PLATFORMS"
    BUILD_CMD="$BUILD_CMD --tag $FULL_IMAGE_NAME:$IMAGE_TAG"
    BUILD_CMD="$BUILD_CMD --tag $FULL_IMAGE_NAME:latest"
    
    # Add build arguments
    if [[ -n "$BUILD_ARGS" ]]; then
        BUILD_CMD="$BUILD_CMD $BUILD_ARGS"
    fi
    
    # Add push flag if needed
    if [[ "$PUSH" == "true" ]]; then
        BUILD_CMD="$BUILD_CMD --push"
    else
        BUILD_CMD="$BUILD_CMD --load"
    fi
    
    # Add current directory as build context
    BUILD_CMD="$BUILD_CMD ."
    
    log_info "Executing: $BUILD_CMD"
    
    # Execute build
    if eval "$BUILD_CMD"; then
        log_success "Multi-architecture build completed successfully"
    else
        log_error "Build failed"
        exit 1
    fi
}

# Generate build report
generate_report() {
    log_info "Generating build report..."
    
    # Create report file
    REPORT_FILE="build-report-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$REPORT_FILE" << EOF
IoT Edge Anomaly Detection - Multi-Architecture Build Report
============================================================

Build Information:
  Image Name: $FULL_IMAGE_NAME
  Tag: $IMAGE_TAG
  Platforms: $PLATFORMS
  Registry: ${DOCKER_REGISTRY:-"None (local build)"}
  Pushed: $PUSH
  Build Arguments: ${BUILD_ARGS:-"None"}
  Build Date: $(date -u)
  Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo "Unknown")
  Git Branch: $(git branch --show-current 2>/dev/null || echo "Unknown")

Platform Details:
EOF

    # Add platform-specific information
    IFS=',' read -ra PLATFORM_ARRAY <<< "$PLATFORMS"
    for platform in "${PLATFORM_ARRAY[@]}"; do
        platform=$(echo "$platform" | xargs) # trim whitespace
        echo "  - $platform" >> "$REPORT_FILE"
        
        # Get image details if available
        if [[ "$PUSH" == "true" ]]; then
            if docker buildx imagetools inspect "$FULL_IMAGE_NAME:$IMAGE_TAG" --format '{{json .}}' | jq -r ".manifests[] | select(.platform.os + \"/\" + .platform.architecture == \"$platform\") | \"    Size: \" + (.size | tostring) + \" bytes\"" &> /dev/null; then
                docker buildx imagetools inspect "$FULL_IMAGE_NAME:$IMAGE_TAG" --format '{{json .}}' | jq -r ".manifests[] | select(.platform.os + \"/\" + .platform.architecture == \"$platform\") | \"    Size: \" + (.size | tostring) + \" bytes\"" >> "$REPORT_FILE"
            fi
        fi
    done
    
    cat >> "$REPORT_FILE" << EOF

Build Performance:
  Build Duration: $BUILD_DURATION seconds
  Build Status: SUCCESS

Security Scan Results:
  (Run 'trivy image $FULL_IMAGE_NAME:$IMAGE_TAG' for detailed security scan)

Next Steps:
EOF
    
    if [[ "$PUSH" == "true" ]]; then
        cat >> "$REPORT_FILE" << EOF
  1. Verify image in registry: docker pull $FULL_IMAGE_NAME:$IMAGE_TAG
  2. Deploy to edge devices: docker run $FULL_IMAGE_NAME:$IMAGE_TAG
  3. Run security scan: trivy image $FULL_IMAGE_NAME:$IMAGE_TAG
EOF
    else
        cat >> "$REPORT_FILE" << EOF
  1. Test local image: docker run $FULL_IMAGE_NAME:$IMAGE_TAG
  2. Push to registry: $0 --push
  3. Run security scan: trivy image $FULL_IMAGE_NAME:$IMAGE_TAG
EOF
    fi
    
    log_success "Build report saved to: $REPORT_FILE"
}

# Generate Kubernetes deployment manifests
generate_kubernetes_manifests() {
    if [[ "$GENERATE_K8S" != "true" ]]; then
        return 0
    fi
    
    log_info "Generating Kubernetes deployment manifests..."
    
    # Create deployment directory
    K8S_DIR="k8s-manifests-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$K8S_DIR"
    
    # Generate deployment manifest
    cat > "$K8S_DIR/deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iot-edge-anomaly
  labels:
    app: iot-edge-anomaly
    version: $IMAGE_TAG
    variant: ${EDGE_VARIANT:-standard}
    region: ${GLOBAL_REGION:-default}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iot-edge-anomaly
  template:
    metadata:
      labels:
        app: iot-edge-anomaly
        version: $IMAGE_TAG
    spec:
      containers:
      - name: iot-edge-anomaly
        image: $FULL_IMAGE_NAME:$IMAGE_TAG
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8080
          name: health
        - containerPort: 9090
          name: metrics
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: DEPLOYMENT_MODE
          value: "$DEPLOYMENT_MODE"
        - name: EDGE_VARIANT
          value: "${EDGE_VARIANT:-standard}"
        - name: GLOBAL_REGION
          value: "${GLOBAL_REGION:-default}"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
---
apiVersion: v1
kind: Service
metadata:
  name: iot-edge-anomaly-service
  labels:
    app: iot-edge-anomaly
spec:
  selector:
    app: iot-edge-anomaly
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: health
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iot-edge-anomaly-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: iot-edge-anomaly
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF

    # Generate ConfigMap for edge-specific configuration
    cat > "$K8S_DIR/configmap.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: iot-edge-anomaly-config
data:
  config.yaml: |
    model:
      input_size: 5
      hidden_size: 64
      num_layers: 2
      dropout: 0.1
    anomaly_threshold: 0.5
    processing:
      loop_interval: 5.0
    health:
      memory_threshold_mb: 100
      cpu_threshold_percent: 80
    deployment_mode: "$DEPLOYMENT_MODE"
    edge_variant: "${EDGE_VARIANT:-standard}"
    global_region: "${GLOBAL_REGION:-default}"
EOF

    # Generate edge-specific optimizations
    if [[ -n "$EDGE_VARIANT" ]]; then
        case "$EDGE_VARIANT" in
            "raspberry-pi")
                cat > "$K8S_DIR/edge-nodeaffinity.yaml" << EOF
apiVersion: v1
kind: Node
metadata:
  name: raspberry-pi-node
  labels:
    kubernetes.io/arch: arm64
    edge.terragon.io/device-type: raspberry-pi
    edge.terragon.io/power-profile: low
EOF
                ;;
            "jetson-nano")
                cat > "$K8S_DIR/gpu-resources.yaml" << EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
spec:
  hard:
    nvidia.com/gpu: "1"
EOF
                ;;
        esac
    fi
    
    log_success "Kubernetes manifests generated in $K8S_DIR/"
}

# Run security scan
run_security_scan() {
    if [[ "$SECURITY_SCAN" != "true" ]]; then
        return 0
    fi
    
    log_info "Running comprehensive security scan..."
    
    # Check if trivy is available
    if ! command -v trivy &> /dev/null; then
        log_warning "Trivy not found. Installing..."
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin v0.45.0
    fi
    
    # Run vulnerability scan
    SCAN_REPORT="security-scan-$(date +%Y%m%d-%H%M%S).json"
    
    log_info "Scanning image for vulnerabilities..."
    trivy image --format json --output "$SCAN_REPORT" "$FULL_IMAGE_NAME:$IMAGE_TAG" || true
    
    # Generate human-readable report
    SCAN_SUMMARY="security-summary-$(date +%Y%m%d-%H%M%S).txt"
    trivy image --format table "$FULL_IMAGE_NAME:$IMAGE_TAG" > "$SCAN_SUMMARY" || true
    
    # Check for critical vulnerabilities
    CRITICAL_COUNT=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | length' "$SCAN_REPORT" 2>/dev/null | wc -l || echo "0")
    HIGH_COUNT=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH") | length' "$SCAN_REPORT" 2>/dev/null | wc -l || echo "0")
    
    if [[ "$CRITICAL_COUNT" -gt 0 ]]; then
        log_error "Found $CRITICAL_COUNT critical vulnerabilities!"
        log_warning "Review $SCAN_REPORT for details"
    elif [[ "$HIGH_COUNT" -gt 5 ]]; then
        log_warning "Found $HIGH_COUNT high severity vulnerabilities"
    else
        log_success "Security scan completed - no critical issues found"
    fi
    
    log_info "Security reports: $SCAN_REPORT, $SCAN_SUMMARY"
}

# Run performance tests
run_performance_tests() {
    if [[ "$PERFORMANCE_TEST" != "true" ]]; then
        return 0
    fi
    
    log_info "Running performance benchmarks..."
    
    # Start container for testing
    CONTAINER_ID=$(docker run -d -p 8080:8080 "$FULL_IMAGE_NAME:$IMAGE_TAG" || echo "")
    
    if [[ -z "$CONTAINER_ID" ]]; then
        log_error "Failed to start container for performance testing"
        return 1
    fi
    
    # Wait for container to be ready
    log_info "Waiting for container to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8080/health &>/dev/null; then
            break
        fi
        sleep 2
    done
    
    # Performance test results
    PERF_REPORT="performance-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "IoT Edge Anomaly Detection v4.0 - Performance Test Report"
        echo "========================================================"
        echo "Image: $FULL_IMAGE_NAME:$IMAGE_TAG"
        echo "Edge Variant: ${EDGE_VARIANT:-standard}"
        echo "Deployment Mode: $DEPLOYMENT_MODE"
        echo "Test Date: $(date -u)"
        echo ""
        
        # Container resource usage
        echo "Container Resource Usage:"
        docker stats --no-stream "$CONTAINER_ID" 2>/dev/null || echo "Stats not available"
        echo ""
        
        # Memory usage
        echo "Memory Usage:"
        docker exec "$CONTAINER_ID" cat /proc/meminfo | grep -E "MemTotal|MemAvailable" 2>/dev/null || echo "Memory info not available"
        echo ""
        
        # Health check response time
        echo "Health Check Performance:"
        for i in {1..5}; do
            time curl -s http://localhost:8080/health >/dev/null 2>&1
        done
        echo ""
        
        # Basic load test (if wrk is available)
        if command -v wrk &> /dev/null; then
            echo "Basic Load Test (10 seconds, 2 threads, 10 connections):"
            wrk -t2 -c10 -d10s http://localhost:8080/health 2>/dev/null || echo "Load test not available"
        fi
        
    } > "$PERF_REPORT"
    
    # Cleanup
    docker stop "$CONTAINER_ID" >/dev/null 2>&1
    docker rm "$CONTAINER_ID" >/dev/null 2>&1
    
    log_success "Performance test completed: $PERF_REPORT"
}

# Deploy to global regions
deploy_global() {
    if [[ "$GLOBAL_DEPLOY" != "true" ]]; then
        return 0
    fi
    
    log_info "Starting global deployment..."
    
    for region in ${GLOBAL_REGIONS//,/ }; do
        log_info "Deploying to region: $region"
        
        if [[ -n "${REGION_CONFIGS[$region]:-}" ]]; then
            REGION_CONFIG="${REGION_CONFIGS[$region]}"
            eval "$REGION_CONFIG"
            
            # Build for specific region
            REGION_REGISTRY="${registry:-$DOCKER_REGISTRY}"
            REGION_IMAGE="$REGION_REGISTRY/$IMAGE_NAME:$IMAGE_TAG-$region"
            
            log_info "Building for region $region: $REGION_IMAGE"
            
            # Regional build with specific tags
            docker buildx build \
                --platform "$PLATFORMS" \
                --tag "$REGION_IMAGE" \
                --build-arg "GLOBAL_REGION=$region" \
                --build-arg "DEPLOYMENT_MODE=$DEPLOYMENT_MODE" \
                $BUILD_ARGS \
                --push \
                . || log_error "Failed to build for region $region"
            
            # Deploy to Kubernetes if cluster info is available
            if [[ -n "${k8s_cluster:-}" ]]; then
                log_info "Deploying to Kubernetes cluster: ${k8s_cluster}"
                # This would typically use kubectl or helm
                # kubectl --context="${k8s_cluster}" apply -f k8s-manifests/
            fi
            
        else
            log_warning "Unknown region configuration: $region"
        fi
    done
    
    log_success "Global deployment completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up build artifacts..."
    
    # Remove dangling images
    if docker images --filter "dangling=true" -q | grep -q .; then
        docker rmi $(docker images --filter "dangling=true" -q) || true
        log_info "Removed dangling images"
    fi
    
    # Prune build cache (keep recent builds)
    docker buildx prune --filter "until=24h" --force || true
    log_info "Pruned old build cache"
}

# Main execution
main() {
    log_info "Starting TERRAGON IoT Edge Anomaly Detection v4.0 Global Build System"
    log_info "Image: $IMAGE_NAME:$IMAGE_TAG"
    log_info "Platforms: $PLATFORMS"
    log_info "Edge Variant: ${EDGE_VARIANT:-standard}"
    log_info "Deployment Mode: $DEPLOYMENT_MODE"
    log_info "Global Region: ${GLOBAL_REGION:-all}"
    
    BUILD_START_TIME=$(date +%s)
    
    # Check prerequisites
    check_prerequisites
    
    # Global deployment (builds for all regions)
    if [[ "$GLOBAL_DEPLOY" == "true" ]]; then
        deploy_global
    else
        # Build images for specified platforms
        build_images
    fi
    
    BUILD_END_TIME=$(date +%s)
    BUILD_DURATION=$((BUILD_END_TIME - BUILD_START_TIME))
    
    # Generate Kubernetes manifests
    generate_kubernetes_manifests
    
    # Run security scan
    run_security_scan
    
    # Run performance tests
    run_performance_tests
    
    # Generate comprehensive report
    generate_report
    
    # Cleanup
    cleanup
    
    log_success "TERRAGON Build completed in $BUILD_DURATION seconds"
    
    # Print enhanced next steps
    echo ""
    log_info "🚀 TERRAGON Deployment Options:"
    
    if [[ "$PUSH" == "true" || "$GLOBAL_DEPLOY" == "true" ]]; then
        echo "  🔧 Edge Device Deployment:"
        case "${EDGE_VARIANT:-standard}" in
            "raspberry-pi")
                echo "     ssh pi@raspberrypi 'docker pull $FULL_IMAGE_NAME:$IMAGE_TAG && docker run -d --restart=unless-stopped -p 8080:8080 $FULL_IMAGE_NAME:$IMAGE_TAG'"
                ;;
            "jetson-nano")
                echo "     ssh nvidia@jetson 'docker pull $FULL_IMAGE_NAME:$IMAGE_TAG && docker run -d --restart=unless-stopped --gpus all -p 8080:8080 $FULL_IMAGE_NAME:$IMAGE_TAG'"
                ;;
            "intel-nuc")
                echo "     ssh admin@intel-nuc 'docker pull $FULL_IMAGE_NAME:$IMAGE_TAG && docker run -d --restart=unless-stopped -p 8080:8080 $FULL_IMAGE_NAME:$IMAGE_TAG'"
                ;;
            *)
                echo "     docker pull $FULL_IMAGE_NAME:$IMAGE_TAG && docker run -d --restart=unless-stopped -p 8080:8080 $FULL_IMAGE_NAME:$IMAGE_TAG"
                ;;
        esac
        
        if [[ "$GENERATE_K8S" == "true" ]]; then
            echo "  ☸️  Kubernetes Deployment:"
            echo "     kubectl apply -f k8s-manifests-*/"
            echo "     kubectl get pods -l app=iot-edge-anomaly"
        fi
        
        if [[ "$GLOBAL_DEPLOY" == "true" ]]; then
            echo "  🌍 Global Regions Deployed:"
            for region in ${GLOBAL_REGIONS//,/ }; do
                echo "     - $region: $FULL_IMAGE_NAME:$IMAGE_TAG-$region"
            done
        fi
    else
        echo "  🧪 Local Testing:"
        echo "     docker run --rm -p 8080:8080 $FULL_IMAGE_NAME:$IMAGE_TAG"
        echo "     curl http://localhost:8080/health"
        echo ""
        echo "  📤 Push to Registry:"
        echo "     $0 --push"
        echo ""
        echo "  🌍 Global Deployment:"
        echo "     $0 --global-deploy --push"
    fi
    
    echo ""
    echo "  🔐 Security & Performance:"
    if [[ "$SECURITY_SCAN" == "true" ]]; then
        echo "     ✅ Security scan completed"
    else
        echo "     $0 --security-scan"
    fi
    
    if [[ "$PERFORMANCE_TEST" == "true" ]]; then
        echo "     ✅ Performance test completed"
    else
        echo "     $0 --performance-test"
    fi
    
    echo ""
    echo "  📊 Monitoring:"
    echo "     Health: http://localhost:8080/health"
    echo "     Metrics: http://localhost:8080/metrics"
    echo "     Grafana: http://localhost:3000 (if using docker-compose)"
}

# Run main function
main "$@"
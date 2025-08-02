#!/bin/bash
# Multi-architecture build script for IoT Edge Anomaly Detection
# Supports ARM64 (Raspberry Pi) and x86_64 (Industrial Gateways)

set -euo pipefail

# Configuration
IMAGE_NAME="${IMAGE_NAME:-iot-edge-graph-anomaly}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/arm64,linux/amd64}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
BUILD_ARGS="${BUILD_ARGS:-}"
PUSH="${PUSH:-false}"

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

# Help function
show_help() {
    cat << EOF
Multi-Architecture Build Script for IoT Edge Anomaly Detection

Usage: $0 [OPTIONS]

Options:
    -n, --name NAME         Image name (default: iot-edge-graph-anomaly)
    -t, --tag TAG          Image tag (default: latest)
    -p, --platforms PLAT   Target platforms (default: linux/arm64,linux/amd64)
    -r, --registry REG     Docker registry URL
    --push                 Push images to registry
    --build-arg ARG        Build arguments (can be used multiple times)
    -h, --help             Show this help

Examples:
    # Basic build for ARM64 and AMD64
    $0 --name myapp --tag v1.0.0

    # Build and push to registry
    $0 --name myapp --tag v1.0.0 --registry docker.io/myuser --push

    # Build for Raspberry Pi only
    $0 --platforms linux/arm64 --build-arg OPTIMIZE_FOR=raspberry_pi

    # Build with custom Python version
    $0 --build-arg PYTHON_VERSION=3.11

Environment Variables:
    IMAGE_NAME      Image name
    IMAGE_TAG       Image tag  
    PLATFORMS       Target platforms
    DOCKER_REGISTRY Registry URL
    BUILD_ARGS      Build arguments
    PUSH            Push to registry (true/false)
EOF
}

# Parse command line arguments
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
    log_info "Starting IoT Edge Anomaly Detection multi-arch build"
    log_info "Image: $IMAGE_NAME:$IMAGE_TAG"
    log_info "Platforms: $PLATFORMS"
    
    BUILD_START_TIME=$(date +%s)
    
    # Check prerequisites
    check_prerequisites
    
    # Build images
    build_images
    
    BUILD_END_TIME=$(date +%s)
    BUILD_DURATION=$((BUILD_END_TIME - BUILD_START_TIME))
    
    # Generate report
    generate_report
    
    # Cleanup
    cleanup
    
    log_success "Build completed in $BUILD_DURATION seconds"
    
    # Print next steps
    echo ""
    log_info "Next steps:"
    if [[ "$PUSH" == "true" ]]; then
        echo "  1. Deploy to Raspberry Pi: ssh pi@raspberrypi 'docker pull $FULL_IMAGE_NAME:$IMAGE_TAG && docker run -d $FULL_IMAGE_NAME:$IMAGE_TAG'"
        echo "  2. Deploy to Industrial Gateway: ssh gateway 'docker pull $FULL_IMAGE_NAME:$IMAGE_TAG && docker run -d $FULL_IMAGE_NAME:$IMAGE_TAG'"
    else
        echo "  1. Test locally: docker run --rm -p 8080:8080 $FULL_IMAGE_NAME:$IMAGE_TAG"
        echo "  2. Push to registry: $0 --push"
    fi
    echo "  3. Run security scan: trivy image $FULL_IMAGE_NAME:$IMAGE_TAG"
}

# Run main function
main "$@"
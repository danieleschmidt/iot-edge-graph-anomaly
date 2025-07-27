#!/bin/bash

# IoT Edge Graph Anomaly Detection - Release Script
# Automates version bumping, tagging, and release preparation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_REGISTRY="terragonlabs"
IMAGE_NAME="iot-edge-anomaly"

# Functions
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

usage() {
    cat << EOF
Usage: $0 <version_type> [options]

Version Types:
  patch    - Bump patch version (0.1.0 -> 0.1.1)
  minor    - Bump minor version (0.1.0 -> 0.2.0) 
  major    - Bump major version (0.1.0 -> 1.0.0)

Options:
  --dry-run       Show what would be done without making changes
  --skip-tests    Skip running tests before release
  --skip-build    Skip building Docker images
  --skip-push     Skip pushing to registry
  --help          Show this help message

Examples:
  $0 patch                    # Create patch release
  $0 minor --dry-run          # Preview minor release
  $0 major --skip-tests       # Major release without testing
EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log_error "Working directory has uncommitted changes"
        exit 1
    fi
    
    # Check if on main/master branch
    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != "main" && "$CURRENT_BRANCH" != "master" ]]; then
        log_warning "Not on main/master branch (current: $CURRENT_BRANCH)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check required tools
    for tool in bumpversion docker make; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

get_current_version() {
    # Extract version from pyproject.toml
    grep -E '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'
}

run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests (--skip-tests specified)"
        return
    fi
    
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run linting
    log_info "Running linting checks..."
    make lint || {
        log_error "Linting failed"
        exit 1
    }
    
    # Run type checking
    log_info "Running type checks..."
    make type-check || {
        log_error "Type checking failed"
        exit 1
    }
    
    # Run security checks
    log_info "Running security checks..."
    make security-check || {
        log_error "Security checks failed"
        exit 1
    }
    
    # Run all tests
    log_info "Running test suite..."
    make test || {
        log_error "Tests failed"
        exit 1
    }
    
    log_success "All tests passed"
}

bump_version() {
    local version_type=$1
    
    log_info "Bumping $version_type version..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would bump $version_type version"
        bumpversion --dry-run --verbose "$version_type"
        return
    fi
    
    # Create version update
    bumpversion "$version_type"
    
    NEW_VERSION=$(get_current_version)
    log_success "Version bumped to $NEW_VERSION"
}

build_docker_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_warning "Skipping Docker build (--skip-build specified)"
        return
    fi
    
    local version=$(get_current_version)
    
    log_info "Building Docker images for version $version..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would build Docker images"
        return
    fi
    
    # Build multi-arch images
    log_info "Building x86_64 image..."
    docker build --platform linux/amd64 -t "$IMAGE_NAME:$version" -t "$IMAGE_NAME:latest" .
    
    log_info "Building ARM64 image..."
    docker build --platform linux/arm64 -t "$IMAGE_NAME:$version-arm64" .
    
    # Tag for registry
    docker tag "$IMAGE_NAME:$version" "$DOCKER_REGISTRY/$IMAGE_NAME:$version"
    docker tag "$IMAGE_NAME:latest" "$DOCKER_REGISTRY/$IMAGE_NAME:latest"
    docker tag "$IMAGE_NAME:$version-arm64" "$DOCKER_REGISTRY/$IMAGE_NAME:$version-arm64"
    
    log_success "Docker images built successfully"
}

push_docker_images() {
    if [[ "$SKIP_PUSH" == "true" ]]; then
        log_warning "Skipping Docker push (--skip-push specified)"
        return
    fi
    
    local version=$(get_current_version)
    
    log_info "Pushing Docker images..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would push Docker images to registry"
        return
    fi
    
    # Push images to registry
    docker push "$DOCKER_REGISTRY/$IMAGE_NAME:$version"
    docker push "$DOCKER_REGISTRY/$IMAGE_NAME:latest"
    docker push "$DOCKER_REGISTRY/$IMAGE_NAME:$version-arm64"
    
    log_success "Docker images pushed to registry"
}

generate_release_notes() {
    local version=$(get_current_version)
    local previous_tag=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")
    
    log_info "Generating release notes for version $version..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would generate release notes"
        return
    fi
    
    # Create release notes
    cat > "release-notes-$version.md" << EOF
# Release v$version

## Changes

$(git log --pretty=format:"- %s" $previous_tag..HEAD)

## Docker Images

\`\`\`bash
# x86_64
docker pull $DOCKER_REGISTRY/$IMAGE_NAME:$version

# ARM64 (Raspberry Pi)
docker pull $DOCKER_REGISTRY/$IMAGE_NAME:$version-arm64
\`\`\`

## Deployment

\`\`\`bash
docker run -d \\
  --name iot-edge-anomaly \\
  --restart unless-stopped \\
  --memory=256m \\
  --cpus=0.5 \\
  -p 8000:8000 \\
  -p 8080:8080 \\
  -p 9090:9090 \\
  $DOCKER_REGISTRY/$IMAGE_NAME:$version
\`\`\`

---
Generated on $(date)
EOF
    
    log_success "Release notes saved to release-notes-$version.md"
}

create_github_release() {
    local version=$(get_current_version)
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create GitHub release"
        return
    fi
    
    if command -v gh &> /dev/null; then
        log_info "Creating GitHub release..."
        gh release create "v$version" \
            --title "Release v$version" \
            --notes-file "release-notes-$version.md" \
            --draft
        log_success "GitHub release created (draft)"
    else
        log_warning "GitHub CLI not found, skipping release creation"
        log_info "Manual steps:"
        log_info "1. Go to GitHub releases page"
        log_info "2. Create new release with tag v$version"
        log_info "3. Copy content from release-notes-$version.md"
    fi
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f release-notes-*.md
}

main() {
    # Parse arguments
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi
    
    VERSION_TYPE=""
    DRY_RUN="false"
    SKIP_TESTS="false"
    SKIP_BUILD="false"
    SKIP_PUSH="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            patch|minor|major)
                VERSION_TYPE="$1"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            --skip-build)
                SKIP_BUILD="true"
                shift
                ;;
            --skip-push)
                SKIP_PUSH="true"
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$VERSION_TYPE" ]]; then
        log_error "Version type is required"
        usage
        exit 1
    fi
    
    # Confirm dry run
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN MODE - No changes will be made"
    fi
    
    # Get current version
    CURRENT_VERSION=$(get_current_version)
    log_info "Current version: $CURRENT_VERSION"
    
    # Confirm release
    if [[ "$DRY_RUN" == "false" ]]; then
        echo
        log_warning "This will create a $VERSION_TYPE release"
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Release cancelled"
            exit 0
        fi
    fi
    
    # Execute release steps
    check_prerequisites
    run_tests
    bump_version "$VERSION_TYPE"
    build_docker_images
    push_docker_images
    generate_release_notes
    create_github_release
    
    if [[ "$DRY_RUN" == "false" ]]; then
        NEW_VERSION=$(get_current_version)
        log_success "Release $NEW_VERSION completed successfully!"
        log_info "Next steps:"
        log_info "1. Review the GitHub release draft"
        log_info "2. Publish the release when ready"
        log_info "3. Update deployment documentation"
        log_info "4. Notify stakeholders"
    else
        log_info "Dry run completed - no changes made"
    fi
    
    cleanup
}

# Execute main function
main "$@"
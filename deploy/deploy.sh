#!/bin/bash
# ResearchCrew Vertex AI Agent Engine Deployment Script
#
# Usage:
#   ./deploy/deploy.sh [command] [options]
#
# Commands:
#   build      Build the Docker image
#   push       Push image to Google Container Registry
#   deploy     Deploy to Vertex AI Agent Engine
#   all        Build, push, and deploy (default)
#   local      Run locally in Docker
#   validate   Validate configuration

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables if .env exists
if [[ -f "$PROJECT_DIR/.env" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
fi

# Default values (override via environment or .env)
GCP_PROJECT_ID="${GCP_PROJECT_ID:-}"
GCP_REGION="${GCP_REGION:-us-central1}"
IMAGE_NAME="${IMAGE_NAME:-researchcrew}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Full image path
IMAGE_PATH="gcr.io/${GCP_PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_success() {
    echo "[SUCCESS] $*"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check for required tools
    local missing_tools=()

    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi

    if ! command -v gcloud &> /dev/null; then
        missing_tools+=("gcloud")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install them and try again."
        exit 1
    fi

    # Check GCP project is set
    if [[ -z "$GCP_PROJECT_ID" ]]; then
        log_error "GCP_PROJECT_ID is not set"
        log_error "Set it via environment variable or .env file"
        exit 1
    fi

    log_success "All prerequisites met"
}

# ============================================================================
# Build Command
# ============================================================================

cmd_build() {
    log_info "Building Docker image: $IMAGE_PATH"

    docker build \
        --tag "$IMAGE_PATH" \
        --tag "${IMAGE_NAME}:latest" \
        --file "$PROJECT_DIR/Dockerfile" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        "$PROJECT_DIR"

    log_success "Image built successfully"
}

# ============================================================================
# Push Command
# ============================================================================

cmd_push() {
    log_info "Pushing image to GCR: $IMAGE_PATH"

    # Configure Docker for GCR
    gcloud auth configure-docker --quiet

    docker push "$IMAGE_PATH"

    log_success "Image pushed successfully"
}

# ============================================================================
# Deploy Command
# ============================================================================

cmd_deploy() {
    log_info "Deploying to Vertex AI Agent Engine..."

    # Check if ADK is installed
    if ! command -v adk &> /dev/null; then
        log_error "ADK CLI not found. Install with: pip install google-adk"
        exit 1
    fi

    # Deploy using ADK
    adk deploy \
        --project="$GCP_PROJECT_ID" \
        --region="$GCP_REGION" \
        --config="$SCRIPT_DIR/config.yaml"

    log_success "Deployment completed"
    log_info "View deployment at: https://console.cloud.google.com/vertex-ai/agents"
}

# ============================================================================
# Local Command
# ============================================================================

cmd_local() {
    log_info "Running locally in Docker..."

    # Check for GOOGLE_API_KEY
    if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
        log_error "GOOGLE_API_KEY is not set"
        log_error "Set it via environment variable or .env file"
        exit 1
    fi

    # Build if image doesn't exist
    if ! docker image inspect "${IMAGE_NAME}:latest" &> /dev/null; then
        cmd_build
    fi

    # Run container
    docker run \
        --rm \
        --interactive \
        --tty \
        --publish 8080:8080 \
        --env GOOGLE_API_KEY="$GOOGLE_API_KEY" \
        --env LOG_LEVEL="${LOG_LEVEL:-INFO}" \
        "${IMAGE_NAME}:latest"
}

# ============================================================================
# Validate Command
# ============================================================================

cmd_validate() {
    log_info "Validating configuration..."

    # Check config file exists
    if [[ ! -f "$SCRIPT_DIR/config.yaml" ]]; then
        log_error "Config file not found: $SCRIPT_DIR/config.yaml"
        exit 1
    fi

    # Validate Dockerfile
    if [[ ! -f "$PROJECT_DIR/Dockerfile" ]]; then
        log_error "Dockerfile not found"
        exit 1
    fi

    # Check Dockerfile syntax with hadolint if available
    if command -v hadolint &> /dev/null; then
        log_info "Running Dockerfile linter..."
        hadolint "$PROJECT_DIR/Dockerfile" || true
    fi

    # Validate YAML syntax
    if command -v python3 &> /dev/null; then
        python3 -c "import yaml; yaml.safe_load(open('$SCRIPT_DIR/config.yaml'))" 2>/dev/null
        if [[ $? -ne 0 ]]; then
            log_error "Invalid YAML in config.yaml"
            exit 1
        fi
    fi

    log_success "Configuration is valid"
}

# ============================================================================
# All Command
# ============================================================================

cmd_all() {
    check_prerequisites
    cmd_validate
    cmd_build
    cmd_push
    cmd_deploy
}

# ============================================================================
# Help
# ============================================================================

show_help() {
    cat << EOF
ResearchCrew Deployment Script

Usage:
    ./deploy/deploy.sh [command] [options]

Commands:
    build      Build the Docker image
    push       Push image to Google Container Registry
    deploy     Deploy to Vertex AI Agent Engine
    all        Build, push, and deploy (default)
    local      Run locally in Docker
    validate   Validate configuration
    help       Show this help message

Environment Variables:
    GCP_PROJECT_ID     Google Cloud project ID (required)
    GCP_REGION         GCP region (default: us-central1)
    IMAGE_NAME         Docker image name (default: researchcrew)
    IMAGE_TAG          Docker image tag (default: latest)
    GOOGLE_API_KEY     Google API key (required for local run)

Examples:
    # Build and deploy
    GCP_PROJECT_ID=my-project ./deploy/deploy.sh all

    # Run locally
    GOOGLE_API_KEY=xxx ./deploy/deploy.sh local

    # Just build
    GCP_PROJECT_ID=my-project ./deploy/deploy.sh build

EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    local command="${1:-all}"

    case "$command" in
        build)
            check_prerequisites
            cmd_build
            ;;
        push)
            check_prerequisites
            cmd_push
            ;;
        deploy)
            check_prerequisites
            cmd_deploy
            ;;
        all)
            cmd_all
            ;;
        local)
            cmd_local
            ;;
        validate)
            cmd_validate
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

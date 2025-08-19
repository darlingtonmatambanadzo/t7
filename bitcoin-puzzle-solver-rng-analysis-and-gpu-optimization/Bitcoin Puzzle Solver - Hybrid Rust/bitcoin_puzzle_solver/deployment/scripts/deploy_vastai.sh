#!/bin/bash

# Bitcoin Puzzle Solver - Vast.ai Deployment Script
# Automated deployment and management for vast.ai instances
#
# Author: Manus AI
# Version: 1.0.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Configuration
DOCKER_IMAGE="bitcoin-puzzle-solver:latest"
CONTAINER_NAME="puzzle-solver"
VAST_API_KEY="${VAST_API_KEY:-}"
INSTANCE_TYPE="A100"
MIN_GPU_COUNT=4
MAX_PRICE_PER_HOUR=2.0

# Default puzzle configuration
DEFAULT_PUZZLE=71
DEFAULT_MAX_TIME=3600
DEFAULT_TARGET_ADDRESS=""

# Usage information
usage() {
    cat << EOF
Bitcoin Puzzle Solver - Vast.ai Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build           Build Docker image
    deploy          Deploy to vast.ai
    start           Start puzzle solving
    stop            Stop puzzle solving
    status          Check status
    logs            View logs
    cleanup         Cleanup resources
    benchmark       Run performance benchmark

Options:
    --puzzle NUM            Target puzzle number (default: $DEFAULT_PUZZLE)
    --address ADDR          Target Bitcoin address
    --max-time SECONDS      Maximum solving time (default: $DEFAULT_MAX_TIME)
    --price PRICE           Maximum price per hour (default: $MAX_PRICE_PER_HOUR)
    --gpu-count COUNT       Minimum GPU count (default: $MIN_GPU_COUNT)
    --instance-id ID        Specific vast.ai instance ID
    --help                  Show this help message

Environment Variables:
    VAST_API_KEY           Your vast.ai API key (required for deployment)
    PUZZLE_NUMBER          Target puzzle number
    TARGET_ADDRESS         Target Bitcoin address
    MAX_TIME               Maximum solving time in seconds

Examples:
    # Build and deploy to vast.ai
    $0 build
    $0 deploy --puzzle 71 --max-time 7200

    # Start solving on existing instance
    $0 start --puzzle 72 --address "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ"

    # Check status and view logs
    $0 status
    $0 logs

    # Run benchmark
    $0 benchmark --instance-id 12345

EOF
}

# Parse command line arguments
parse_args() {
    COMMAND=""
    PUZZLE_NUMBER="$DEFAULT_PUZZLE"
    TARGET_ADDRESS="$DEFAULT_TARGET_ADDRESS"
    MAX_TIME="$DEFAULT_MAX_TIME"
    MAX_PRICE="$MAX_PRICE_PER_HOUR"
    GPU_COUNT="$MIN_GPU_COUNT"
    INSTANCE_ID=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            build|deploy|start|stop|status|logs|cleanup|benchmark)
                COMMAND="$1"
                shift
                ;;
            --puzzle)
                PUZZLE_NUMBER="$2"
                shift 2
                ;;
            --address)
                TARGET_ADDRESS="$2"
                shift 2
                ;;
            --max-time)
                MAX_TIME="$2"
                shift 2
                ;;
            --price)
                MAX_PRICE="$2"
                shift 2
                ;;
            --gpu-count)
                GPU_COUNT="$2"
                shift 2
                ;;
            --instance-id)
                INSTANCE_ID="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    if [[ -z "$COMMAND" ]]; then
        error "No command specified"
        usage
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is required but not installed"
        exit 1
    fi

    # Check vast.ai CLI (if needed)
    if [[ "$COMMAND" == "deploy" ]] && ! command -v vastai &> /dev/null; then
        warning "vast.ai CLI not found. Installing..."
        pip install vastai || {
            error "Failed to install vast.ai CLI"
            exit 1
        }
    fi

    # Check API key for vast.ai operations
    if [[ "$COMMAND" == "deploy" ]] && [[ -z "$VAST_API_KEY" ]]; then
        error "VAST_API_KEY environment variable is required for deployment"
        error "Get your API key from https://vast.ai/console/account/"
        exit 1
    fi

    success "Dependencies verified"
}

# Build Docker image
build_image() {
    log "Building Docker image: $DOCKER_IMAGE"

    cd "$(dirname "$0")/../.."

    docker build \
        -f deployment/docker/Dockerfile \
        -t "$DOCKER_IMAGE" \
        . || {
        error "Docker build failed"
        exit 1
    }

    success "Docker image built successfully"
}

# Find suitable vast.ai instance
find_instance() {
    log "Searching for suitable vast.ai instances..."

    # Search for instances with required specifications
    vastai search offers \
        --type=on-demand \
        --gpu_name="$INSTANCE_TYPE" \
        --num_gpus="$GPU_COUNT" \
        --max_price="$MAX_PRICE" \
        --sort_by=price \
        --limit=10 || {
        error "Failed to search vast.ai offers"
        exit 1
    }
}

# Deploy to vast.ai
deploy_vastai() {
    log "Deploying to vast.ai..."

    # Set vast.ai API key
    vastai set api-key "$VAST_API_KEY"

    # Find and rent instance
    log "Finding suitable instance..."
    OFFERS=$(vastai search offers \
        --type=on-demand \
        --gpu_name="$INSTANCE_TYPE" \
        --num_gpus="$GPU_COUNT" \
        --max_price="$MAX_PRICE" \
        --sort_by=price \
        --raw)

    if [[ -z "$OFFERS" ]]; then
        error "No suitable instances found"
        error "Try increasing --price or reducing --gpu-count"
        exit 1
    fi

    # Get the best offer
    OFFER_ID=$(echo "$OFFERS" | head -n1 | cut -d' ' -f1)
    log "Selected offer ID: $OFFER_ID"

    # Create instance
    log "Creating instance..."
    INSTANCE_RESULT=$(vastai create instance \
        --image="$DOCKER_IMAGE" \
        --disk=50 \
        --env="PUZZLE_NUMBER=$PUZZLE_NUMBER" \
        --env="TARGET_ADDRESS=$TARGET_ADDRESS" \
        --env="MAX_TIME=$MAX_TIME" \
        "$OFFER_ID")

    INSTANCE_ID=$(echo "$INSTANCE_RESULT" | grep -o 'instance [0-9]*' | cut -d' ' -f2)

    if [[ -z "$INSTANCE_ID" ]]; then
        error "Failed to create instance"
        exit 1
    fi

    success "Instance created: $INSTANCE_ID"
    log "Waiting for instance to start..."

    # Wait for instance to be ready
    for i in {1..30}; do
        STATUS=$(vastai show instance "$INSTANCE_ID" --raw | cut -d' ' -f3)
        if [[ "$STATUS" == "running" ]]; then
            success "Instance is running"
            break
        fi
        log "Instance status: $STATUS (attempt $i/30)"
        sleep 10
    done

    # Save instance ID for future operations
    echo "$INSTANCE_ID" > .vastai_instance_id

    success "Deployment completed successfully"
    log "Instance ID: $INSTANCE_ID"
    log "Use '$0 status' to check progress"
}

# Start puzzle solving
start_solving() {
    log "Starting puzzle solving..."

    if [[ -z "$INSTANCE_ID" ]] && [[ -f ".vastai_instance_id" ]]; then
        INSTANCE_ID=$(cat .vastai_instance_id)
    fi

    if [[ -z "$INSTANCE_ID" ]]; then
        error "No instance ID specified. Use --instance-id or deploy first"
        exit 1
    fi

    # Execute solver on the instance
    vastai ssh "$INSTANCE_ID" \
        "cd /app && ./scripts/start_solver.sh" || {
        error "Failed to start solver"
        exit 1
    }

    success "Puzzle solving started on instance $INSTANCE_ID"
}

# Check status
check_status() {
    log "Checking system status..."

    if [[ -z "$INSTANCE_ID" ]] && [[ -f ".vastai_instance_id" ]]; then
        INSTANCE_ID=$(cat .vastai_instance_id)
    fi

    if [[ -z "$INSTANCE_ID" ]]; then
        error "No instance ID found"
        exit 1
    fi

    # Get instance status
    log "Instance status:"
    vastai show instance "$INSTANCE_ID"

    # Get solver status
    log "Solver status:"
    vastai ssh "$INSTANCE_ID" \
        "cd /app && python3.11 -c 'from python_gpu.src.coordination.system_coordinator import SystemCoordinator; import asyncio; print(asyncio.run(SystemCoordinator.get_status()))'" 2>/dev/null || {
        warning "Could not retrieve solver status"
    }
}

# View logs
view_logs() {
    log "Retrieving logs..."

    if [[ -z "$INSTANCE_ID" ]] && [[ -f ".vastai_instance_id" ]]; then
        INSTANCE_ID=$(cat .vastai_instance_id)
    fi

    if [[ -z "$INSTANCE_ID" ]]; then
        error "No instance ID found"
        exit 1
    fi

    # Show recent logs
    vastai ssh "$INSTANCE_ID" \
        "tail -f /app/logs/solver.log" || {
        error "Failed to retrieve logs"
        exit 1
    }
}

# Run benchmark
run_benchmark() {
    log "Running performance benchmark..."

    if [[ -z "$INSTANCE_ID" ]] && [[ -f ".vastai_instance_id" ]]; then
        INSTANCE_ID=$(cat .vastai_instance_id)
    fi

    if [[ -z "$INSTANCE_ID" ]]; then
        error "No instance ID found"
        exit 1
    fi

    # Run benchmark
    vastai ssh "$INSTANCE_ID" \
        "cd /app && python3.11 -c 'from python_gpu.src.coordination.system_coordinator import SystemCoordinator; import asyncio; coordinator = SystemCoordinator(); asyncio.run(coordinator.benchmark_system(60))'" || {
        error "Benchmark failed"
        exit 1
    }
}

# Stop instance
stop_instance() {
    log "Stopping instance..."

    if [[ -z "$INSTANCE_ID" ]] && [[ -f ".vastai_instance_id" ]]; then
        INSTANCE_ID=$(cat .vastai_instance_id)
    fi

    if [[ -z "$INSTANCE_ID" ]]; then
        error "No instance ID found"
        exit 1
    fi

    vastai destroy instance "$INSTANCE_ID" || {
        error "Failed to stop instance"
        exit 1
    }

    rm -f .vastai_instance_id
    success "Instance stopped"
}

# Cleanup resources
cleanup() {
    log "Cleaning up resources..."

    # Stop instance if running
    if [[ -f ".vastai_instance_id" ]]; then
        stop_instance
    fi

    # Remove local files
    rm -f .vastai_instance_id

    success "Cleanup completed"
}

# Main execution
main() {
    log "=== Bitcoin Puzzle Solver - Vast.ai Deployment ==="

    parse_args "$@"
    check_dependencies

    case "$COMMAND" in
        build)
            build_image
            ;;
        deploy)
            build_image
            deploy_vastai
            ;;
        start)
            start_solving
            ;;
        stop)
            stop_instance
            ;;
        status)
            check_status
            ;;
        logs)
            view_logs
            ;;
        benchmark)
            run_benchmark
            ;;
        cleanup)
            cleanup
            ;;
        *)
            error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac

    success "Operation completed successfully"
}

# Execute main function
main "$@"


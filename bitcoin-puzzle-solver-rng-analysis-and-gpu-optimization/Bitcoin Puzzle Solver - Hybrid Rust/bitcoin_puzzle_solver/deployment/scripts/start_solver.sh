#!/bin/bash

# Bitcoin Puzzle Solver - Startup Script
# Optimized for vast.ai deployment with 4x A100 GPUs
#
# Author: Manus AI
# Version: 1.0.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
APP_DIR="/app"
LOG_DIR="/app/logs"
DATA_DIR="/app/data"
RESULTS_DIR="/app/results"
CONFIG_FILE="/app/configs/solver_config.toml"

# Create necessary directories
mkdir -p "$LOG_DIR" "$DATA_DIR" "$RESULTS_DIR"

# Set up logging
exec 1> >(tee -a "$LOG_DIR/solver.log")
exec 2> >(tee -a "$LOG_DIR/solver_error.log" >&2)

log "Starting Bitcoin Puzzle Solver System..."
log "Application directory: $APP_DIR"
log "Log directory: $LOG_DIR"

# Check GPU availability
check_gpus() {
    log "Checking GPU availability..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. NVIDIA drivers may not be installed."
        exit 1
    fi
    
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    log "Detected $GPU_COUNT GPU(s)"
    
    if [ "$GPU_COUNT" -lt 1 ]; then
        error "No GPUs detected. This system requires NVIDIA GPUs."
        exit 1
    fi
    
    # Display GPU information
    log "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits | while read line; do
        log "  GPU $line"
    done
    
    # Check for A100 GPUs (recommended)
    A100_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c "A100" || true)
    if [ "$A100_COUNT" -gt 0 ]; then
        success "Detected $A100_COUNT A100 GPU(s) - optimal for this workload"
    else
        warning "No A100 GPUs detected. Performance may be suboptimal."
    fi
}

# Check CUDA installation
check_cuda() {
    log "Checking CUDA installation..."
    
    if ! command -v nvcc &> /dev/null; then
        error "CUDA compiler (nvcc) not found."
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    log "CUDA version: $CUDA_VERSION"
    
    # Check CUDA runtime
    python3.11 -c "import cupy; print(f'CuPy CUDA version: {cupy.cuda.runtime.runtimeGetVersion()}')" 2>/dev/null || {
        error "CuPy CUDA runtime check failed"
        exit 1
    }
    
    success "CUDA installation verified"
}

# Check Python environment
check_python() {
    log "Checking Python environment..."
    
    PYTHON_VERSION=$(python3.11 --version)
    log "Python version: $PYTHON_VERSION"
    
    # Check critical packages
    REQUIRED_PACKAGES=("numpy" "cupy" "numba" "sklearn" "pandas")
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python3.11 -c "import $package" 2>/dev/null; then
            log "  ✓ $package installed"
        else
            error "  ✗ $package not found"
            exit 1
        fi
    done
    
    success "Python environment verified"
}

# Check Rust components
check_rust() {
    log "Checking Rust components..."
    
    if [ ! -f "$APP_DIR/rust_core/target/release/libbitcoin_puzzle_core.so" ]; then
        error "Rust core library not found. Building..."
        cd "$APP_DIR/rust_core"
        cargo build --release --features python
        cd "$APP_DIR"
    fi
    
    # Test Rust-Python integration
    python3.11 -c "
import sys
sys.path.append('/app/rust_core/target/release')
try:
    import bitcoin_puzzle_core
    print('Rust-Python integration: OK')
except ImportError as e:
    print(f'Rust-Python integration failed: {e}')
    exit(1)
" || exit 1
    
    success "Rust components verified"
}

# Load configuration
load_config() {
    log "Loading configuration..."
    
    if [ ! -f "$CONFIG_FILE" ]; then
        warning "Configuration file not found. Creating default configuration..."
        cat > "$CONFIG_FILE" << EOF
[system]
puzzle_number = 71
max_time_seconds = 3600
log_level = "info"

[gpu]
enabled = true
batch_size = 1000000
max_concurrent_batches = 8
threads_per_block = 256

[ml]
enabled = true
model_path = "/app/models/hot_zone_model.joblib"
training_data_path = "/app/data/puzzle_data.csv"
max_hot_zones = 5
min_confidence = 0.1

[monitoring]
enabled = true
prometheus_port = 9090
grafana_port = 3000
update_interval = 10
EOF
    fi
    
    log "Configuration loaded from: $CONFIG_FILE"
}

# Start monitoring services
start_monitoring() {
    log "Starting monitoring services..."
    
    # Start Prometheus node exporter
    if command -v prometheus-node-exporter &> /dev/null; then
        prometheus-node-exporter --web.listen-address=":9100" &
        log "Prometheus node exporter started on port 9100"
    fi
    
    # Start GPU monitoring
    python3.11 "$APP_DIR/monitoring/gpu_monitor.py" &
    log "GPU monitoring started"
    
    # Start system monitor
    python3.11 "$APP_DIR/monitoring/system_monitor.py" &
    log "System monitoring started"
}

# Download training data if not present
setup_training_data() {
    log "Setting up training data..."
    
    TRAINING_DATA_FILE="$DATA_DIR/puzzle_data.csv"
    
    if [ ! -f "$TRAINING_DATA_FILE" ]; then
        log "Training data not found. Checking for uploaded data..."
        
        # Check if data was uploaded to the container
        if [ -f "/app/upload/916_BTC.csv" ]; then
            log "Found uploaded training data, copying..."
            cp "/app/upload/916_BTC.csv" "$TRAINING_DATA_FILE"
            success "Training data copied from upload"
        else
            warning "No training data found. ML predictions will use default model."
        fi
    else
        log "Training data found: $TRAINING_DATA_FILE"
    fi
}

# Main solver function
run_solver() {
    log "Starting Bitcoin Puzzle Solver..."
    
    # Get puzzle number from environment or config
    PUZZLE_NUMBER=${PUZZLE_NUMBER:-71}
    TARGET_ADDRESS=${TARGET_ADDRESS:-""}
    MAX_TIME=${MAX_TIME:-3600}
    
    log "Target puzzle: #$PUZZLE_NUMBER"
    log "Max time: $MAX_TIME seconds"
    
    if [ -z "$TARGET_ADDRESS" ]; then
        # Get address for puzzle from data
        case $PUZZLE_NUMBER in
            71) TARGET_ADDRESS="1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9" ;;
            72) TARGET_ADDRESS="1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ" ;;
            73) TARGET_ADDRESS="19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG" ;;
            *) 
                error "No target address specified for puzzle #$PUZZLE_NUMBER"
                error "Set TARGET_ADDRESS environment variable"
                exit 1
                ;;
        esac
    fi
    
    log "Target address: $TARGET_ADDRESS"
    
    # Run the solver
    cd "$APP_DIR"
    python3.11 -m python_gpu.src.main \
        --puzzle-number "$PUZZLE_NUMBER" \
        --target-address "$TARGET_ADDRESS" \
        --max-time "$MAX_TIME" \
        --config "$CONFIG_FILE" \
        --log-dir "$LOG_DIR" \
        --results-dir "$RESULTS_DIR"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Kill background processes
    jobs -p | xargs -r kill
    
    # Save final logs
    log "Solver session ended at $(date)"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    log "=== Bitcoin Puzzle Solver - Vast.ai Deployment ==="
    log "Starting system initialization..."
    
    # System checks
    check_gpus
    check_cuda
    check_python
    check_rust
    
    # Setup
    load_config
    setup_training_data
    start_monitoring
    
    success "System initialization completed successfully!"
    log "=== Starting Puzzle Solving Process ==="
    
    # Run the solver
    run_solver
}

# Execute main function
main "$@"


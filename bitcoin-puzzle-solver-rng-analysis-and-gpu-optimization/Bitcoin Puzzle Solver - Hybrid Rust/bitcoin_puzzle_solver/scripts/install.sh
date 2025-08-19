#!/bin/bash

# Bitcoin Puzzle Solver - Installation Script
# Installs all dependencies and sets up the optimized solving environment

set -e  # Exit on any error

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

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    log_info "Detected OS: $OS"
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    case $OS in
        "debian")
            sudo apt update
            sudo apt install -y \
                curl wget git build-essential \
                pkg-config libssl-dev \
                python3 python3-pip python3-venv \
                nvidia-cuda-toolkit \
                cmake clang llvm \
                libgmp-dev libmpfr-dev \
                htop nvtop
            ;;
        "redhat")
            sudo yum update -y
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                curl wget git \
                openssl-devel \
                python3 python3-pip \
                cuda-toolkit \
                cmake clang llvm \
                gmp-devel mpfr-devel
            ;;
        "macos")
            # Check if Homebrew is installed
            if ! command -v brew &> /dev/null; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew update
            brew install \
                curl wget git \
                openssl \
                python3 \
                cmake llvm \
                gmp mpfr
            ;;
    esac
    
    log_success "System dependencies installed"
}

# Install Rust
install_rust() {
    log_info "Installing Rust..."
    
    if ! command -v rustc &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    else
        log_info "Rust already installed, updating..."
        rustup update
    fi
    
    # Install required components
    rustup component add clippy rustfmt
    rustup target add x86_64-unknown-linux-gnu
    
    # Verify installation
    rustc --version
    cargo --version
    
    log_success "Rust installed and configured"
}

# Install Python dependencies
install_python_deps() {
    log_info "Setting up Python environment..."
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA support
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected, installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_warning "No NVIDIA GPU detected, installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other ML dependencies
    pip install \
        tensorflow \
        scikit-learn \
        xgboost \
        lightgbm \
        pandas \
        numpy \
        scipy \
        matplotlib \
        seaborn \
        jupyter \
        networkx \
        pymc3 \
        arviz \
        lifelines \
        joblib \
        tqdm \
        psutil
    
    # Install CUDA-specific packages if available
    if command -v nvidia-smi &> /dev/null; then
        pip install \
            cupy-cuda11x \
            cudf-cu11 \
            cuml-cu11 \
            cugraph-cu11 || log_warning "Some CUDA packages failed to install"
    fi
    
    log_success "Python environment configured"
}

# Build Rust components
build_rust() {
    log_info "Building Rust components..."
    
    cd rust_core
    
    # Build in release mode with optimizations
    RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" \
    cargo build --release --features="gpu,simd,parallel"
    
    # Run tests
    cargo test --release
    
    # Build benchmarks
    cargo bench --no-run
    
    cd ..
    
    log_success "Rust components built successfully"
}

# Setup configuration
setup_config() {
    log_info "Setting up configuration..."
    
    # Create config directory
    mkdir -p ~/.config/bitcoin-puzzle-solver
    
    # Generate default configuration
    ./rust_core/target/release/puzzle-solver config -o ~/.config/bitcoin-puzzle-solver/solver_config.toml
    
    # Create data directories
    mkdir -p data/{models,logs,results,puzzles}
    
    # Download puzzle data if available
    if [ ! -f "data/puzzles/bitcoin_puzzles.csv" ]; then
        log_info "Creating sample puzzle data..."
        cat > data/puzzles/bitcoin_puzzles.csv << 'EOF'
puzzle,private_key,address,status
1,1,1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH,SOLVED
2,3,1CUTxxqjWCn3KjDMC4FfwgVHytHxqSKmw6,SOLVED
3,7,1DEP8i3QJCsomS4BSMY2RpU1upv62aGvhD,SOLVED
4,8,1KhbWlHfa-redacted-for-security,SOLVED
5,15,1Jq6MksXQVWzrznvZzxkV6oY57oWXD9TXB,SOLVED
EOF
    fi
    
    log_success "Configuration setup complete"
}

# Setup vast.ai integration
setup_vastai() {
    log_info "Setting up vast.ai integration..."
    
    # Install vast.ai CLI
    pip install vastai
    
    # Create vast.ai configuration template
    cat > scripts/vastai_deploy.sh << 'EOF'
#!/bin/bash
# vast.ai deployment script
# Usage: ./vastai_deploy.sh <api_key>

if [ -z "$1" ]; then
    echo "Usage: $0 <vastai_api_key>"
    exit 1
fi

export VASTAI_API_KEY="$1"

# Search for suitable instances
vastai search offers 'reliability > 0.95 gpu_name=RTX_4090 num_gpus>=4 inet_down>=100'

echo "To create an instance, run:"
echo "vastai create instance <instance_id> --image pytorch/pytorch:latest --disk 50"
EOF
    
    chmod +x scripts/vastai_deploy.sh
    
    log_success "vast.ai integration setup complete"
}

# Performance optimization
optimize_system() {
    log_info "Applying system optimizations..."
    
    # CPU governor settings
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null || true
    fi
    
    # Memory settings
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf > /dev/null || true
    echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf > /dev/null || true
    
    # GPU settings (if NVIDIA)
    if command -v nvidia-smi &> /dev/null; then
        # Set persistence mode
        sudo nvidia-smi -pm 1 || true
        
        # Set power limit to maximum
        sudo nvidia-smi -pl $(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | head -1) || true
    fi
    
    log_success "System optimizations applied"
}

# Create monitoring scripts
create_monitoring() {
    log_info "Creating monitoring scripts..."
    
    cat > scripts/monitor.sh << 'EOF'
#!/bin/bash
# System monitoring script for puzzle solving

echo "=== Bitcoin Puzzle Solver Monitor ==="
echo "Date: $(date)"
echo

# CPU usage
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2 $4}'

# Memory usage
echo "Memory Usage:"
free -h | grep "Mem:"

# GPU usage (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Usage:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
fi

# Disk usage
echo "Disk Usage:"
df -h | grep -E "/$|/home"

# Network usage
echo "Network Interfaces:"
ip -s link show | grep -E "eth0|wlan0" -A 1

# Process monitoring
echo "Puzzle Solver Processes:"
ps aux | grep -E "puzzle-solver|python.*hot_zone" | grep -v grep
EOF
    
    chmod +x scripts/monitor.sh
    
    # Create log rotation script
    cat > scripts/rotate_logs.sh << 'EOF'
#!/bin/bash
# Log rotation script

LOG_DIR="data/logs"
MAX_SIZE="100M"
MAX_FILES=10

find $LOG_DIR -name "*.log" -size +$MAX_SIZE -exec gzip {} \;
find $LOG_DIR -name "*.log.gz" | sort -r | tail -n +$((MAX_FILES + 1)) | xargs rm -f
EOF
    
    chmod +x scripts/rotate_logs.sh
    
    log_success "Monitoring scripts created"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check Rust build
    if [ -f "rust_core/target/release/puzzle-solver" ]; then
        log_success "Rust binary built successfully"
        ./rust_core/target/release/puzzle-solver info | head -5
    else
        log_error "Rust binary not found"
        return 1
    fi
    
    # Check Python environment
    source venv/bin/activate
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name --format=csv,noheader
    fi
    
    # Run quick benchmark
    log_info "Running quick benchmark..."
    timeout 10s ./rust_core/target/release/puzzle-solver benchmark -d 5 -p 71 || true
    
    log_success "Installation verification complete"
}

# Main installation function
main() {
    echo "=========================================="
    echo "Bitcoin Puzzle Solver Installation Script"
    echo "=========================================="
    echo
    
    check_root
    detect_os
    
    log_info "Starting installation process..."
    
    install_system_deps
    install_rust
    install_python_deps
    build_rust
    setup_config
    setup_vastai
    optimize_system
    create_monitoring
    verify_installation
    
    echo
    echo "=========================================="
    log_success "Installation completed successfully!"
    echo "=========================================="
    echo
    echo "Next steps:"
    echo "1. Activate Python environment: source venv/bin/activate"
    echo "2. Configure solver: edit ~/.config/bitcoin-puzzle-solver/solver_config.toml"
    echo "3. Train ML models: ./rust_core/target/release/puzzle-solver train -d data/puzzles/bitcoin_puzzles.csv"
    echo "4. Start solving: ./rust_core/target/release/puzzle-solver solve -p 71 -a <target_address>"
    echo
    echo "For help: ./rust_core/target/release/puzzle-solver --help"
    echo "Monitor system: ./scripts/monitor.sh"
    echo
}

# Run main function
main "$@"


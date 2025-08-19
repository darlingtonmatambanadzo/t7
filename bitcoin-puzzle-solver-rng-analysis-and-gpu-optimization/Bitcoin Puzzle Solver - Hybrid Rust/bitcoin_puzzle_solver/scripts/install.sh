#!/bin/bash

# Bitcoin Puzzle Solver - vast.ai Deployment Script
# Automated deployment to vast.ai GPU instances

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
DOCKER_IMAGE="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
MIN_GPU_COUNT=4
MIN_GPU_MEMORY=24  # GB
MIN_RELIABILITY=0.95
MAX_PRICE_PER_HOUR=2.0  # USD
DISK_SIZE=100  # GB

# Check if vast.ai CLI is installed
check_vastai_cli() {
    if ! command -v vastai &> /dev/null; then
        log_error "vast.ai CLI not found. Install with: pip install vastai"
        exit 1
    fi
}

# Check API key
check_api_key() {
    if [ -z "$VASTAI_API_KEY" ]; then
        log_error "VASTAI_API_KEY environment variable not set"
        echo "Get your API key from: https://vast.ai/console/account/"
        echo "Then run: export VASTAI_API_KEY='your_api_key_here'"
        exit 1
    fi
}

# Search for suitable instances
search_instances() {
    log_info "Searching for suitable GPU instances..."
    
    local search_query="reliability > $MIN_RELIABILITY gpu_ram >= $MIN_GPU_MEMORY num_gpus >= $MIN_GPU_COUNT dph < $MAX_PRICE_PER_HOUR gpu_name=RTX_4090 OR gpu_name=A100 OR gpu_name=RTX_3090"
    
    log_info "Search criteria:"
    log_info "- Minimum GPUs: $MIN_GPU_COUNT"
    log_info "- Minimum GPU RAM: ${MIN_GPU_MEMORY}GB"
    log_info "- Minimum reliability: $MIN_RELIABILITY"
    log_info "- Maximum price: \$${MAX_PRICE_PER_HOUR}/hour"
    
    vastai search offers "$search_query" --raw | head -20
}

# Create instance
create_instance() {
    local instance_id=$1
    
    if [ -z "$instance_id" ]; then
        log_error "Instance ID required"
        echo "Usage: $0 create <instance_id>"
        exit 1
    fi
    
    log_info "Creating instance $instance_id..."
    
    # Create startup script
    cat > /tmp/startup_script.sh << 'EOF'
#!/bin/bash
set -e

# Update system
apt-get update && apt-get upgrade -y

# Install system dependencies
apt-get install -y \
    curl wget git build-essential \
    pkg-config libssl-dev \
    cmake clang llvm \
    libgmp-dev libmpfr-dev \
    htop nvtop screen tmux

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Clone repository
cd /workspace
git clone https://github.com/darlingtonmatambanadzo/t7.git
cd t7

# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh

# Create systemd service for auto-start
cat > /etc/systemd/system/puzzle-solver.service << 'EOSERVICE'
[Unit]
Description=Bitcoin Puzzle Solver
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace/t7
ExecStart=/workspace/t7/rust_core/target/release/puzzle-solver solve -p 71 -a 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU --spr
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOSERVICE

systemctl enable puzzle-solver
systemctl start puzzle-solver

echo "Bitcoin Puzzle Solver deployed and started!"
EOF
    
    # Create instance with startup script
    vastai create instance $instance_id \
        --image $DOCKER_IMAGE \
        --disk $DISK_SIZE \
        --onstart-cmd "$(cat /tmp/startup_script.sh)" \
        --env "CUDA_VISIBLE_DEVICES=all"
    
    rm /tmp/startup_script.sh
    
    log_success "Instance $instance_id created successfully"
}

# Monitor instance
monitor_instance() {
    local instance_id=$1
    
    if [ -z "$instance_id" ]; then
        log_error "Instance ID required"
        echo "Usage: $0 monitor <instance_id>"
        exit 1
    fi
    
    log_info "Monitoring instance $instance_id..."
    
    while true; do
        clear
        echo "=== Instance $instance_id Status ==="
        echo "Time: $(date)"
        echo
        
        # Get instance status
        vastai show instance $instance_id
        
        echo
        echo "=== GPU Utilization ==="
        vastai ssh $instance_id "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" 2>/dev/null || echo "GPU info unavailable"
        
        echo
        echo "=== Puzzle Solver Status ==="
        vastai ssh $instance_id "systemctl status puzzle-solver --no-pager -l" 2>/dev/null || echo "Service status unavailable"
        
        echo
        echo "=== Recent Logs ==="
        vastai ssh $instance_id "journalctl -u puzzle-solver --no-pager -n 10" 2>/dev/null || echo "Logs unavailable"
        
        echo
        echo "Press Ctrl+C to exit monitoring"
        sleep 30
    done
}

# Connect to instance
connect_instance() {
    local instance_id=$1
    
    if [ -z "$instance_id" ]; then
        log_error "Instance ID required"
        echo "Usage: $0 connect <instance_id>"
        exit 1
    fi
    
    log_info "Connecting to instance $instance_id..."
    vastai ssh $instance_id
}

# Stop instance
stop_instance() {
    local instance_id=$1
    
    if [ -z "$instance_id" ]; then
        log_error "Instance ID required"
        echo "Usage: $0 stop <instance_id>"
        exit 1
    fi
    
    log_info "Stopping instance $instance_id..."
    vastai stop instance $instance_id
    log_success "Instance $instance_id stopped"
}

# Destroy instance
destroy_instance() {
    local instance_id=$1
    
    if [ -z "$instance_id" ]; then
        log_error "Instance ID required"
        echo "Usage: $0 destroy <instance_id>"
        exit 1
    fi
    
    log_warning "This will permanently destroy instance $instance_id"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        vastai destroy instance $instance_id
        log_success "Instance $instance_id destroyed"
    else
        log_info "Operation cancelled"
    fi
}

# List instances
list_instances() {
    log_info "Your vast.ai instances:"
    vastai show instances
}

# Download results
download_results() {
    local instance_id=$1
    local local_dir=${2:-"./results"}
    
    if [ -z "$instance_id" ]; then
        log_error "Instance ID required"
        echo "Usage: $0 download <instance_id> [local_directory]"
        exit 1
    fi
    
    log_info "Downloading results from instance $instance_id to $local_dir..."
    
    mkdir -p "$local_dir"
    
    # Download solution files
    vastai rsync $instance_id:/workspace/t7/data/results/ "$local_dir/" --download
    
    # Download logs
    vastai rsync $instance_id:/workspace/t7/data/logs/ "$local_dir/logs/" --download
    
    log_success "Results downloaded to $local_dir"
}

# Setup auto-scaling
setup_autoscaling() {
    log_info "Setting up auto-scaling configuration..."
    
    cat > autoscale_config.json << 'EOF'
{
    "min_instances": 1,
    "max_instances": 10,
    "target_gpu_utilization": 80,
    "scale_up_threshold": 90,
    "scale_down_threshold": 30,
    "cooldown_period": 300,
    "instance_config": {
        "image": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
        "disk_size": 100,
        "min_gpu_count": 4,
        "min_gpu_memory": 24,
        "max_price_per_hour": 2.0
    }
}
EOF
    
    log_success "Auto-scaling configuration created: autoscale_config.json"
}

# Cost estimation
estimate_costs() {
    local hours=${1:-24}
    local instances=${2:-1}
    
    log_info "Cost estimation for $instances instance(s) running $hours hour(s):"
    
    # Get current pricing
    local avg_price=$(vastai search offers "num_gpus >= $MIN_GPU_COUNT gpu_ram >= $MIN_GPU_MEMORY" --raw | awk '{print $4}' | sort -n | head -10 | awk '{sum+=$1} END {print sum/NR}')
    
    local total_cost=$(echo "$avg_price * $hours * $instances" | bc -l)
    
    echo "Average price per hour: \$$(printf "%.2f" $avg_price)"
    echo "Total estimated cost: \$$(printf "%.2f" $total_cost)"
    echo
    echo "Note: Prices fluctuate based on demand. Check current pricing with 'search' command."
}

# Show help
show_help() {
    echo "Bitcoin Puzzle Solver - vast.ai Deployment Script"
    echo
    echo "Usage: $0 <command> [arguments]"
    echo
    echo "Commands:"
    echo "  search                    - Search for suitable GPU instances"
    echo "  create <instance_id>      - Create and deploy to instance"
    echo "  monitor <instance_id>     - Monitor instance status and logs"
    echo "  connect <instance_id>     - SSH into instance"
    echo "  stop <instance_id>        - Stop instance"
    echo "  destroy <instance_id>     - Destroy instance (permanent)"
    echo "  list                      - List your instances"
    echo "  download <instance_id>    - Download results from instance"
    echo "  autoscale                 - Setup auto-scaling configuration"
    echo "  estimate [hours] [count]  - Estimate costs"
    echo "  help                      - Show this help"
    echo
    echo "Environment Variables:"
    echo "  VASTAI_API_KEY           - Your vast.ai API key (required)"
    echo
    echo "Examples:"
    echo "  $0 search"
    echo "  $0 create 1234567"
    echo "  $0 monitor 1234567"
    echo "  $0 download 1234567 ./my_results"
    echo "  $0 estimate 24 2"
}

# Main function
main() {
    local command=$1
    shift || true
    
    case $command in
        "search")
            check_vastai_cli
            check_api_key
            search_instances
            ;;
        "create")
            check_vastai_cli
            check_api_key
            create_instance "$@"
            ;;
        "monitor")
            check_vastai_cli
            check_api_key
            monitor_instance "$@"
            ;;
        "connect")
            check_vastai_cli
            check_api_key
            connect_instance "$@"
            ;;
        "stop")
            check_vastai_cli
            check_api_key
            stop_instance "$@"
            ;;
        "destroy")
            check_vastai_cli
            check_api_key
            destroy_instance "$@"
            ;;
        "list")
            check_vastai_cli
            check_api_key
            list_instances
            ;;
        "download")
            check_vastai_cli
            check_api_key
            download_results "$@"
            ;;
        "autoscale")
            setup_autoscaling
            ;;
        "estimate")
            estimate_costs "$@"
            ;;
        "help"|"--help"|"-h"|"")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

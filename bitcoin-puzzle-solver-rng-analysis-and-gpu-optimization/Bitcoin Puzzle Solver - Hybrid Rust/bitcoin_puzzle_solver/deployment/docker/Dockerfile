# Bitcoin Puzzle Solver - Vast.ai Deployment
# Optimized for 4x NVIDIA A100 GPUs with CUDA 12.x support
# 
# Author: Manus AI
# Version: 1.0.0

FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV RUST_BACKTRACE=1
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    # Python and development
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Cryptographic libraries
    libssl-dev \
    libgmp-dev \
    libsecp256k1-dev \
    # System utilities
    htop \
    nvtop \
    tmux \
    vim \
    nano \
    # Networking
    openssh-client \
    rsync \
    # Monitoring
    prometheus-node-exporter \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Update Rust to latest stable
RUN rustup update stable
RUN rustup default stable

# Install additional Rust components
RUN rustup component add clippy rustfmt

# Create application directory
WORKDIR /app

# Copy Rust project files
COPY rust_core/ ./rust_core/
COPY python_gpu/ ./python_gpu/

# Build Rust core components
WORKDIR /app/rust_core
RUN cargo build --release --features python

# Install Python dependencies
WORKDIR /app/python_gpu
RUN python3.11 -m pip install --upgrade pip setuptools wheel
RUN python3.11 -m pip install -r requirements.txt

# Install additional GPU-specific packages
RUN python3.11 -m pip install \
    cupy-cuda12x \
    numba[cuda] \
    pycuda \
    nvidia-ml-py3

# Create Python package
RUN python3.11 -m pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/models /app/results

# Copy deployment scripts
COPY deployment/scripts/ ./scripts/
COPY deployment/configs/ ./configs/

# Make scripts executable
RUN chmod +x ./scripts/*.sh

# Set up environment
ENV PYTHONPATH="/app/python_gpu/src:${PYTHONPATH}"
ENV RUST_LOG=info
ENV CUDA_CACHE_PATH=/app/cache/cuda

# Create cache directories
RUN mkdir -p /app/cache/cuda /app/cache/ml

# Install monitoring tools
RUN python3.11 -m pip install \
    prometheus-client \
    grafana-api \
    psutil \
    nvidia-ml-py3

# Copy monitoring configuration
COPY deployment/monitoring/ ./monitoring/

# Health check script
COPY deployment/scripts/healthcheck.sh ./healthcheck.sh
RUN chmod +x ./healthcheck.sh

# Set up health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ./healthcheck.sh

# Expose ports for monitoring
EXPOSE 8080 9090 3000

# Default command
CMD ["./scripts/start_solver.sh"]

# Labels for vast.ai
LABEL maintainer="Manus AI"
LABEL description="Bitcoin Puzzle Solver optimized for 4x A100 GPUs"
LABEL version="1.0.0"
LABEL gpu.required="true"
LABEL gpu.count="4"
LABEL gpu.memory="40GB"
LABEL gpu.type="A100"


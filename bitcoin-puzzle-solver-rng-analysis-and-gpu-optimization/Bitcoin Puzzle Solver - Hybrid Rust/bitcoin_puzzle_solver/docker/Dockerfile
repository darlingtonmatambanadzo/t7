# Bitcoin Puzzle Solver - Optimized Docker Container
# Multi-stage build for minimal production image

# Build stage
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH
ENV RUST_VERSION=1.75.0

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    clang \
    llvm \
    libgmp-dev \
    libmpfr-dev \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION
RUN rustup component add clippy rustfmt

# Set working directory
WORKDIR /build

# Copy Rust source code
COPY rust_core/ ./rust_core/
COPY Cargo.lock ./

# Build Rust components with optimizations
WORKDIR /build/rust_core
RUN RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" \
    cargo build --release --features="gpu,simd,parallel"

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libssl3 \
    libgmp10 \
    libmpfr6 \
    curl \
    wget \
    git \
    htop \
    nvtop \
    screen \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -m -u 1000 solver && \
    mkdir -p /app /data && \
    chown -R solver:solver /app /data

# Set working directory
WORKDIR /app

# Copy built Rust binary
COPY --from=builder /build/rust_core/target/release/puzzle-solver /usr/local/bin/

# Copy Python source code
COPY python_gpu/ ./python_gpu/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY docs/ ./docs/
COPY *.md ./
COPY *.toml ./

# Create Python virtual environment and install dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
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
    networkx \
    pymc3 \
    arviz \
    lifelines \
    joblib \
    tqdm \
    psutil \
    vastai

# Install CUDA-specific packages
RUN pip install cupy-cuda11x || echo "CuPy installation failed, continuing..."

# Set up configuration
RUN mkdir -p /data/{models,logs,results,puzzles} && \
    puzzle-solver config -o /app/solver_config.toml

# Create startup script
RUN cat > /app/start.sh << 'EOF' && \
#!/bin/bash
set -e

echo "=== Bitcoin Puzzle Solver Container ==="
echo "Starting at: $(date)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo
fi

# Check Python environment
echo "Python Environment:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || echo "GPU count: 0"
echo

# Show solver info
echo "Solver Information:"
puzzle-solver info | head -10
echo

# Default action based on environment variables
if [ -n "$PUZZLE_NUMBER" ] && [ -n "$TARGET_ADDRESS" ]; then
    echo "Starting puzzle solving..."
    echo "Puzzle: $PUZZLE_NUMBER"
    echo "Address: $TARGET_ADDRESS"
    echo "SPR: ${USE_SPR:-true}"
    echo
    
    exec puzzle-solver solve \
        -p "$PUZZLE_NUMBER" \
        -a "$TARGET_ADDRESS" \
        ${USE_SPR:+--spr} \
        ${GPU_DEVICES:+--gpu-devices "$GPU_DEVICES"} \
        ${CPU_CORES:+--cpu-cores "$CPU_CORES"} \
        -v
elif [ -n "$BENCHMARK_MODE" ]; then
    echo "Running benchmark mode..."
    exec puzzle-solver benchmark -d "${BENCHMARK_DURATION:-300}" -p "${BENCHMARK_PUZZLE:-71}"
else
    echo "No specific task specified. Available commands:"
    echo "  puzzle-solver solve -p <puzzle> -a <address>"
    echo "  puzzle-solver benchmark"
    echo "  puzzle-solver train -d data/puzzles/bitcoin_puzzles.csv"
    echo "  puzzle-solver info"
    echo
    echo "Environment variables:"
    echo "  PUZZLE_NUMBER - Puzzle to solve"
    echo "  TARGET_ADDRESS - Target Bitcoin address"
    echo "  USE_SPR - Enable SPR optimization (true/false)"
    echo "  GPU_DEVICES - GPU device IDs (comma-separated)"
    echo "  CPU_CORES - Number of CPU cores to use"
    echo "  BENCHMARK_MODE - Run in benchmark mode"
    echo
    exec /bin/bash
fi
EOF
chmod +x /app/start.sh

# Set ownership
RUN chown -R solver:solver /app /data /opt/venv

# Switch to application user
USER solver

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD puzzle-solver info > /dev/null || exit 1

# Expose ports (if needed for monitoring)
EXPOSE 8080 8888

# Set entrypoint
ENTRYPOINT ["/app/start.sh"]

# Labels
LABEL maintainer="Manus AI <manus@example.com>"
LABEL version="2.0.0"
LABEL description="Bitcoin Puzzle Solver with 35 Mathematical Optimizations"
LABEL gpu.required="true"
LABEL gpu.min_memory="24GB"
LABEL gpu.min_count="4"


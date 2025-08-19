# Bitcoin Puzzle Solver - Complete Beginner Guide

**Author:** Manus AI  
**Date:** July 19, 2025  
**Version:** 1.0.0

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding Bitcoin Puzzles](#understanding-bitcoin-puzzles)
3. [System Requirements](#system-requirements)
4. [Account Setup](#account-setup)
5. [Installation Process](#installation-process)
6. [Configuration](#configuration)
7. [Running Your First Puzzle](#running-your-first-puzzle)
8. [Monitoring and Optimization](#monitoring-and-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)
11. [Safety and Security](#safety-and-security)
12. [Frequently Asked Questions](#frequently-asked-questions)

## Introduction

Welcome to the Bitcoin Puzzle Solver! This guide will walk you through everything you need to know to start solving Bitcoin puzzles, even if you're completely new to cryptocurrency, programming, or GPU computing. We'll explain every concept, every step, and every command in detail.

### What You'll Learn

By the end of this guide, you'll understand how to set up and operate a sophisticated Bitcoin puzzle solving system that uses artificial intelligence and powerful graphics cards to search for hidden Bitcoin private keys. You'll learn about the technology behind Bitcoin, how puzzles work, and how our system uses machine learning to dramatically improve your chances of success.

### What Makes This System Special

Traditional Bitcoin puzzle solving involves randomly searching through trillions of possible private keys, which is like looking for a specific grain of sand on all the beaches in the world. Our system is different because it uses machine learning to predict where the keys are most likely to be found, reducing the search space from trillions to billions of keys. This is like having a metal detector that tells you which part of the beach to search.

The system combines two programming languages: Rust for ultra-fast cryptographic operations and Python for artificial intelligence and graphics card coordination. This hybrid approach gives us the best of both worlds: the speed of Rust and the flexibility of Python.

## Understanding Bitcoin Puzzles

Before we dive into the technical setup, it's important to understand what Bitcoin puzzles are and why they exist. This knowledge will help you make informed decisions about which puzzles to target and how to optimize your solving strategy.

### The History of Bitcoin Puzzles

In 2015, an anonymous person known as "satoshi_rising" created what became known as the Bitcoin Puzzle Challenge. This person generated 256 Bitcoin addresses using private keys in specific ranges and sent increasing amounts of Bitcoin to each address. The challenge was simple: find the private keys to claim the Bitcoin.

The puzzle creator explained their motivation: "There is no pattern. It is just consecutive keys from a deterministic wallet (masked with leading 000...0001 to set difficulty). It is simply a crude measuring instrument, of the cracking strength of the community." [1]

### How Bitcoin Puzzles Work

Each Bitcoin puzzle corresponds to a specific range of private keys. A private key is essentially a very large random number that controls access to Bitcoin. The puzzles are structured as follows:

- **Puzzle 1**: Private key between 1 and 1 (only one possibility)
- **Puzzle 2**: Private key between 2 and 3 (two possibilities)  
- **Puzzle 3**: Private key between 4 and 7 (four possibilities)
- **Puzzle n**: Private key between 2^(n-1) and 2^n - 1

As the puzzle number increases, the search space doubles. Puzzle 71, for example, has 2^70 possible private keys, which is approximately 1,180,591,620,717,411,303,424 possibilities. This astronomical number is why traditional brute force approaches are ineffective.

### Current Puzzle Status

As of 2025, puzzles 1 through 70 have been solved, along with several others including 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, and 130. The remaining unsolved puzzles contain significant Bitcoin rewards:

- **Puzzle 71**: Approximately 7.1 BTC
- **Puzzle 72**: Approximately 7.2 BTC  
- **Puzzle 73**: Approximately 7.3 BTC

The total value of all unsolved puzzles exceeds 900 BTC, worth millions of dollars at current prices.

### The SPR Methodology

Our system implements the Sparse Priming Representation (SPR) methodology, which is based on a crucial insight: the puzzle private keys were generated using a single random number generator, which means they may not be truly random. By analyzing the patterns in solved puzzles, we can predict where unsolved keys are most likely to be located.

The SPR approach works by training a machine learning model on the positions of solved puzzle keys within their respective ranges. This model then predicts "hot zones" where unsolved keys are statistically more likely to be found. Instead of searching the entire range randomly, we focus our computational power on these high-probability areas.

## System Requirements

Understanding the hardware and software requirements is crucial for successful puzzle solving. The computational demands are significant, but the potential rewards justify the investment.

### Hardware Requirements

**Graphics Processing Units (GPUs)**

The heart of our system is GPU acceleration. Graphics cards are exceptionally good at performing the same operation on many pieces of data simultaneously, which is exactly what we need for Bitcoin puzzle solving. Our system is optimized for NVIDIA A100 GPUs, which are currently the most powerful data center GPUs available.

Minimum GPU requirements:
- **4x NVIDIA A100 (40GB)**: Optimal configuration for maximum performance
- **Alternative**: 4x NVIDIA V100 (32GB) or 8x RTX 4090 (24GB)
- **Memory**: Minimum 32GB VRAM total, 160GB recommended
- **Compute Capability**: 7.0 or higher for optimal CUDA support

The A100 GPUs provide several advantages for our workload:
- **High Memory Bandwidth**: 1,555 GB/s for rapid data processing
- **Tensor Cores**: Specialized units for machine learning operations
- **NVLink**: High-speed inter-GPU communication
- **ECC Memory**: Error correction for reliable long-running computations

**Central Processing Unit (CPU)**

While GPUs handle the bulk of the computation, the CPU coordinates operations and handles system management:
- **Cores**: Minimum 16 cores, 32+ recommended
- **Architecture**: x86_64 with AVX2 support
- **Examples**: Intel Xeon Gold 6248R, AMD EPYC 7543

**System Memory (RAM)**

Large amounts of system memory are required for:
- **Baby-step Giant-step tables**: Up to 16GB for optimal performance
- **Machine learning models**: 2-4GB for model storage and inference
- **System operations**: 8GB for operating system and utilities
- **Total requirement**: 128GB minimum, 256GB recommended

**Storage**

Fast storage is essential for:
- **Operating System**: 100GB for Ubuntu and system software
- **Application Data**: 500GB for solver components and dependencies
- **Results and Logs**: 100GB for output data and monitoring
- **Temporary Files**: 300GB for intermediate computations
- **Total requirement**: 1TB NVMe SSD minimum

**Network**

Reliable, high-speed internet is required for:
- **vast.ai communication**: Control and monitoring of cloud instances
- **Data synchronization**: Uploading results and downloading updates
- **Remote access**: SSH connections for system management
- **Bandwidth requirement**: 100 Mbps minimum, 1 Gbps recommended

### Software Requirements

**Operating System**

Our system is designed for Linux environments, specifically:
- **Ubuntu 22.04 LTS**: Primary supported distribution
- **Alternative**: CentOS 8, RHEL 8, or other modern Linux distributions
- **Kernel**: Version 5.4 or higher with CUDA support

**CUDA Toolkit**

NVIDIA's CUDA platform enables GPU acceleration:
- **Version**: CUDA 12.0 or higher
- **Components**: Runtime libraries, development tools, profilers
- **Driver**: NVIDIA driver version 525.60.13 or higher

**Programming Languages**

**Rust Programming Language**
- **Version**: 1.70 or higher
- **Toolchain**: Stable release channel
- **Components**: Cargo package manager, rustfmt, clippy

Rust provides memory safety and zero-cost abstractions for our cryptographic operations. The language's ownership system prevents common security vulnerabilities while maintaining C-level performance.

**Python Programming Language**
- **Version**: 3.11 or higher
- **Package Manager**: pip 23.0+
- **Virtual Environment**: venv or conda

Python handles machine learning, GPU coordination, and system orchestration. Its extensive ecosystem of scientific computing libraries makes it ideal for our AI components.

**Container Platform**

**Docker**
- **Version**: 24.0 or higher
- **Components**: Docker Engine, Docker Compose
- **Registry Access**: Docker Hub for base images

Docker ensures consistent deployment across different environments and simplifies dependency management.

### Cloud Platform Requirements

**vast.ai Account**

vast.ai provides access to high-end GPU hardware without the capital investment:
- **Account**: Active vast.ai account with verified payment method
- **API Access**: API key for programmatic instance management
- **Credits**: Sufficient balance for extended solving sessions
- **Instance Types**: Access to A100 or equivalent GPU instances

**Estimated Costs**

GPU rental costs vary based on market demand:
- **4x A100 Instance**: $2-4 per hour
- **Daily Operation**: $48-96 for 24-hour solving session
- **Weekly Operation**: $336-672 for continuous operation

These costs should be weighed against potential puzzle rewards and success probability.

## Account Setup

Setting up your accounts correctly is the foundation of successful puzzle solving. This section will guide you through creating and configuring all necessary accounts.

### vast.ai Account Creation

vast.ai is a marketplace for GPU computing power that allows you to rent high-end hardware without purchasing it. Follow these steps to create your account:

**Step 1: Registration**

Navigate to https://vast.ai and click "Sign Up" in the top right corner. You'll need to provide:
- **Email Address**: Use a reliable email you check regularly
- **Password**: Create a strong password with at least 12 characters
- **Username**: Choose a unique identifier for your account

**Step 2: Email Verification**

Check your email for a verification message from vast.ai. Click the verification link to activate your account. If you don't receive the email within 10 minutes, check your spam folder.

**Step 3: Account Verification**

vast.ai requires identity verification for security and fraud prevention:
- **Phone Number**: Provide a valid phone number for SMS verification
- **Identity Document**: Upload a government-issued ID (driver's license, passport, etc.)
- **Processing Time**: Verification typically takes 24-48 hours

**Step 4: Payment Method**

Add a payment method to fund your GPU rentals:
- **Credit Card**: Visa, MasterCard, or American Express
- **PayPal**: Link your PayPal account for payments
- **Cryptocurrency**: Some regions support Bitcoin payments

**Step 5: Initial Deposit**

Make an initial deposit to start renting GPUs:
- **Minimum**: $10 for account activation
- **Recommended**: $100-200 for extended solving sessions
- **Auto-reload**: Enable automatic reloading to prevent interruptions

### API Key Configuration

Your vast.ai API key allows programmatic control of GPU instances:

**Step 1: Generate API Key**

1. Log into your vast.ai account
2. Navigate to Account → API Keys
3. Click "Create New API Key"
4. Provide a descriptive name (e.g., "Bitcoin Puzzle Solver")
5. Copy the generated key immediately (it won't be shown again)

**Step 2: Secure Storage**

Store your API key securely:
- **Environment Variable**: Export as VAST_API_KEY in your shell
- **Configuration File**: Store in ~/.vast_api_key with restricted permissions
- **Password Manager**: Use a secure password manager for long-term storage

**Step 3: Test API Access**

Verify your API key works correctly:

```bash
# Install vast.ai CLI
pip install vastai

# Set API key
export VAST_API_KEY="your_api_key_here"

# Test connection
vastai show user
```

If successful, you'll see your account information displayed.

### GitHub Account (Optional)

While not required, a GitHub account enables you to:
- **Access Updates**: Download the latest version of the solver
- **Report Issues**: Submit bug reports and feature requests
- **Contribute**: Participate in development and improvements

Create a free account at https://github.com if you don't already have one.

### Security Considerations

**API Key Security**

Your vast.ai API key provides full access to your account:
- **Never Share**: Don't share your API key with others
- **Rotate Regularly**: Generate new keys every 90 days
- **Monitor Usage**: Check your account regularly for unauthorized activity
- **Revoke Compromised Keys**: Immediately revoke any potentially compromised keys

**Account Security**

Protect your vast.ai account:
- **Two-Factor Authentication**: Enable 2FA using Google Authenticator or similar
- **Strong Password**: Use a unique, complex password
- **Regular Monitoring**: Check your account activity and billing regularly
- **Secure Networks**: Only access your account from trusted networks

## Installation Process

The installation process involves setting up the Bitcoin Puzzle Solver on your local machine and preparing it for deployment to vast.ai. We'll walk through each step in detail.

### Local Development Setup

Even though the solver runs on vast.ai, you'll need a local development environment for configuration, monitoring, and management.

**Step 1: Install Prerequisites**

**Ubuntu/Debian Systems:**
```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential curl wget git

# Install Python 3.11
sudo apt install -y python3.11 python3.11-dev python3.11-venv python3-pip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**macOS Systems:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install prerequisites
brew install python@3.11 rust docker git

# Start Docker Desktop
open /Applications/Docker.app
```

**Windows Systems:**

For Windows users, we recommend using Windows Subsystem for Linux (WSL2):

1. Install WSL2 following Microsoft's official guide
2. Install Ubuntu 22.04 from the Microsoft Store
3. Follow the Ubuntu installation steps above within WSL2

**Step 2: Clone the Repository**

```bash
# Clone the solver repository
git clone <repository-url> bitcoin_puzzle_solver
cd bitcoin_puzzle_solver

# Make scripts executable
chmod +x deployment/scripts/*.sh
chmod +x scripts/*.sh
```

**Step 3: Install vast.ai CLI**

```bash
# Install vast.ai command line interface
pip3.11 install vastai

# Verify installation
vastai --help
```

**Step 4: Configure Environment**

Create a configuration file for your environment:

```bash
# Create environment configuration
cat > .env << EOF
# vast.ai Configuration
VAST_API_KEY=your_api_key_here

# Puzzle Configuration
PUZZLE_NUMBER=71
TARGET_ADDRESS=1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9
MAX_TIME=3600

# Performance Configuration
GPU_COUNT=4
BATCH_SIZE=1000000
MAX_CONCURRENT_BATCHES=16

# Monitoring Configuration
ENABLE_MONITORING=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
EOF

# Load environment variables
source .env
```

### Building the Docker Image

The Docker image contains all components needed to run the solver on vast.ai.

**Step 1: Prepare Build Context**

```bash
# Ensure you're in the project root
cd bitcoin_puzzle_solver

# Verify all components are present
ls -la rust_core/
ls -la python_gpu/
ls -la deployment/
```

**Step 2: Build the Image**

```bash
# Build the Docker image
docker build -f deployment/docker/Dockerfile -t bitcoin-puzzle-solver:latest .

# This process will take 15-30 minutes depending on your internet connection
# The build process includes:
# - Installing system dependencies
# - Compiling Rust components
# - Installing Python packages
# - Configuring CUDA support
```

**Step 3: Test the Image Locally**

```bash
# Test the image with GPU support (if you have NVIDIA GPUs)
docker run --gpus all --rm bitcoin-puzzle-solver:latest ./scripts/test_system.sh

# Test without GPU (for basic functionality)
docker run --rm bitcoin-puzzle-solver:latest python3.11 -c "import bitcoin_puzzle_core; print('Rust integration: OK')"
```

**Step 4: Push to Registry (Optional)**

If you want to store your image in a container registry:

```bash
# Tag for Docker Hub
docker tag bitcoin-puzzle-solver:latest yourusername/bitcoin-puzzle-solver:latest

# Push to registry
docker push yourusername/bitcoin-puzzle-solver:latest
```

### Deployment to vast.ai

Now we'll deploy the solver to vast.ai and start solving puzzles.

**Step 1: Find Suitable Instances**

```bash
# Search for available A100 instances
./deployment/scripts/deploy_vastai.sh find --gpu-count 4 --price 3.0

# This will show available instances with:
# - 4x A100 GPUs
# - Maximum price of $3.00 per hour
# - Current availability and specifications
```

**Step 2: Deploy the Solver**

```bash
# Deploy to vast.ai with automatic instance selection
./deployment/scripts/deploy_vastai.sh deploy \
    --puzzle 71 \
    --max-time 7200 \
    --price 2.5

# The deployment process will:
# 1. Find the best available instance
# 2. Create and configure the instance
# 3. Upload the Docker image
# 4. Start the solver automatically
```

**Step 3: Monitor Deployment**

```bash
# Check deployment status
./deployment/scripts/deploy_vastai.sh status

# View real-time logs
./deployment/scripts/deploy_vastai.sh logs

# Check GPU utilization
./deployment/scripts/deploy_vastai.sh gpu-status
```

### Verification and Testing

After deployment, verify that all components are working correctly.

**Step 1: System Health Check**

```bash
# Run comprehensive system check
./deployment/scripts/deploy_vastai.sh health-check

# This verifies:
# - GPU detection and CUDA functionality
# - Rust-Python integration
# - Machine learning model loading
# - Network connectivity
# - Storage accessibility
```

**Step 2: Performance Benchmark**

```bash
# Run 60-second performance benchmark
./deployment/scripts/deploy_vastai.sh benchmark --duration 60

# Expected results for 4x A100:
# - Total throughput: 40,000+ keys/second
# - GPU utilization: >90%
# - Memory usage: <80%
# - No errors or warnings
```

**Step 3: Test Puzzle Solving**

```bash
# Run a short test on a solved puzzle to verify functionality
./deployment/scripts/deploy_vastai.sh test-solve \
    --puzzle 1 \
    --max-time 60

# This should quickly find the known solution for puzzle 1
```

## Configuration

Proper configuration is essential for optimal performance and successful puzzle solving. This section explains all configuration options and how to tune them for your specific needs.

### Understanding Configuration Files

The solver uses TOML (Tom's Obvious, Minimal Language) configuration files, which are human-readable and easy to modify. The main configuration file is located at `deployment/configs/solver_config.toml`.

### System Configuration

The `[system]` section controls overall solver behavior:

```toml
[system]
# Target puzzle number (1-160)
puzzle_number = 71

# Maximum solving time in seconds (0 = unlimited)
max_time_seconds = 3600

# Logging level: debug, info, warn, error
log_level = "info"

# Number of worker threads for CPU operations
worker_threads = 8

# Enable detailed progress reporting
verbose_progress = true

# Progress reporting interval in seconds
progress_interval = 60
```

**Key Parameters Explained:**

**puzzle_number**: This determines which Bitcoin puzzle you're targeting. Start with puzzle 71 as it has the best balance of difficulty and reward. Higher numbers are exponentially more difficult.

**max_time_seconds**: Sets a time limit for solving attempts. Use 3600 (1 hour) for initial testing, then increase to 86400 (24 hours) or more for serious solving attempts.

**log_level**: Controls the verbosity of logging output. Use "info" for normal operation, "debug" for troubleshooting, and "warn" for minimal output.

**worker_threads**: Number of CPU threads for non-GPU operations. Set this to match your CPU core count for optimal performance.

### GPU Configuration

The `[gpu]` section controls GPU acceleration settings:

```toml
[gpu]
# Enable GPU acceleration
enabled = true

# Batch size for GPU processing (keys per batch)
batch_size = 1000000

# Maximum concurrent batches across all GPUs
max_concurrent_batches = 16

# CUDA threads per block (must be multiple of 32)
threads_per_block = 256

# GPU memory allocation strategy
memory_strategy = "balanced"

# Enable GPU memory pooling
memory_pooling = true

# GPU device selection (empty = use all available)
device_ids = []

# CUDA stream count per GPU
streams_per_gpu = 4
```

**Optimization Guidelines:**

**batch_size**: Larger batches improve GPU utilization but require more memory. For A100 GPUs with 40GB memory, 1,000,000 keys per batch is optimal. Reduce to 500,000 for GPUs with less memory.

**max_concurrent_batches**: This should be 4x the number of GPUs for optimal throughput. With 4 GPUs, use 16 concurrent batches.

**threads_per_block**: Must be a multiple of 32 (CUDA warp size). 256 is optimal for most workloads, but you can experiment with 512 or 1024 for different GPU architectures.

**memory_strategy**: 
- "conservative": Uses 50% of available GPU memory
- "balanced": Uses 75% of available GPU memory (recommended)
- "aggressive": Uses 90% of available GPU memory (may cause instability)

### Machine Learning Configuration

The `[ml]` section controls the AI-guided hot zone prediction:

```toml
[ml]
# Enable ML-guided hot zone prediction
enabled = true

# Path to trained model file
model_path = "/app/models/hot_zone_model.joblib"

# Path to training data CSV
training_data_path = "/app/data/puzzle_data.csv"

# Maximum number of hot zones to generate
max_hot_zones = 5

# Minimum confidence threshold for predictions
min_confidence = 0.1

[ml.model_params]
n_estimators = 100
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
random_state = 42
n_jobs = -1
```

**Machine Learning Parameters:**

**max_hot_zones**: Number of predicted search areas to generate. More zones increase coverage but reduce focus. Start with 5 and adjust based on results.

**min_confidence**: Minimum confidence score for using a predicted zone. Lower values include more zones but may reduce accuracy. 0.1 is a good starting point.

**Model Parameters**: These control the Random Forest algorithm:
- **n_estimators**: Number of decision trees (more trees = better accuracy but slower training)
- **max_depth**: Maximum tree depth (prevents overfitting)
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required at leaf nodes

### BSGS Algorithm Configuration

The Baby-step Giant-step algorithm is crucial for efficient key searching:

```toml
[bsgs]
# Baby step table size (2^baby_step_bits entries)
baby_step_bits = 28

# Enable GLV endomorphism optimization
use_glv_endomorphism = true

# Enable cache-aligned memory operations
use_cache_alignment = true

# Maximum memory usage for BSGS tables (MB)
max_memory_mb = 16384

# Number of parallel BSGS threads
thread_count = 8

# Progress reporting interval (giant steps)
progress_interval = 1000000
```

**BSGS Optimization:**

**baby_step_bits**: Determines the size of the baby step table. 28 bits = 268 million entries, requiring about 16GB of RAM. This is optimal for systems with 128GB+ RAM.

**use_glv_endomorphism**: Enables GLV endomorphism optimization, which can speed up elliptic curve operations by up to 2x. Always enable this unless you encounter compatibility issues.

**max_memory_mb**: Maximum memory allocation for BSGS tables. Set this to about 12.5% of your total system RAM.

### Security Configuration

The `[security]` section enables military-grade security features:

```toml
[security]
# Enable military-grade security features
enabled = true

# Encryption algorithm
encryption_algorithm = "aes256"

# Key derivation function
kdf_algorithm = "pbkdf2"

# KDF iterations
kdf_iterations = 100000

# Enable secure memory clearing
secure_memory_clear = true

# Enable tamper detection
tamper_detection = true

# Audit log path
audit_log_path = "/app/logs/security_audit.log"
```

**Security Features:**

**encryption_algorithm**: AES-256 provides military-grade encryption for sensitive data. ChaCha20 is an alternative that may be faster on some systems.

**kdf_iterations**: Number of iterations for key derivation. Higher values are more secure but slower. 100,000 is a good balance.

**secure_memory_clear**: Ensures sensitive data is properly cleared from memory after use, preventing recovery by attackers.

### Monitoring Configuration

The `[monitoring]` section controls performance tracking and visualization:

```toml
[monitoring]
# Enable performance monitoring
enabled = true

# Prometheus metrics port
prometheus_port = 9090

# Grafana dashboard port
grafana_port = 3000

# Metrics update interval (seconds)
update_interval = 10

# Enable GPU monitoring
gpu_monitoring = true

# Enable system resource monitoring
system_monitoring = true

# Metrics retention period (hours)
retention_hours = 24
```

**Monitoring Setup:**

**prometheus_port**: Port for Prometheus metrics collection. Default 9090 is standard.

**grafana_port**: Port for Grafana dashboard access. Access via http://instance-ip:3000

**update_interval**: How often metrics are collected. 10 seconds provides good resolution without excessive overhead.

### Environment-Specific Configuration

Different environments may require different configurations:

**Development Environment:**
```toml
[system]
log_level = "debug"
verbose_progress = true

[gpu]
batch_size = 100000  # Smaller for testing
max_concurrent_batches = 4

[ml]
max_hot_zones = 2  # Fewer zones for faster testing
```

**Production Environment:**
```toml
[system]
log_level = "info"
max_time_seconds = 86400  # 24 hours

[gpu]
batch_size = 1000000  # Maximum for performance
max_concurrent_batches = 16

[ml]
max_hot_zones = 5  # Optimal coverage
min_confidence = 0.15  # Higher confidence threshold
```

**Memory-Constrained Environment:**
```toml
[bsgs]
baby_step_bits = 26  # Smaller table (64M entries)
max_memory_mb = 8192  # 8GB limit

[gpu]
batch_size = 500000  # Smaller batches
memory_strategy = "conservative"
```

## Running Your First Puzzle

Now that you have the system installed and configured, let's walk through solving your first Bitcoin puzzle. We'll start with a simple example and gradually introduce more advanced features.

### Choosing Your First Target

For your first attempt, we recommend starting with puzzle 71 for several reasons:

1. **Manageable Difficulty**: With 2^70 possible keys, it's challenging but solvable with modern hardware
2. **Significant Reward**: Contains approximately 7.1 BTC (worth tens of thousands of dollars)
3. **Good Documentation**: Extensive community research and analysis available
4. **Optimal for ML**: Sufficient solved puzzles below it to train the machine learning model effectively

### Pre-Flight Checklist

Before starting your solving attempt, verify all systems are ready:

**Step 1: Verify vast.ai Instance**

```bash
# Check instance status
./deployment/scripts/deploy_vastai.sh status

# Expected output:
# Instance ID: 12345
# Status: running
# GPUs: 4x A100-SXM4-40GB
# CPU: 64 cores
# RAM: 256 GB
# Disk: 1000 GB
```

**Step 2: Verify GPU Functionality**

```bash
# Test GPU detection and CUDA functionality
./deployment/scripts/deploy_vastai.sh gpu-test

# Expected output:
# GPU 0: A100-SXM4-40GB (40GB) - OK
# GPU 1: A100-SXM4-40GB (40GB) - OK  
# GPU 2: A100-SXM4-40GB (40GB) - OK
# GPU 3: A100-SXM4-40GB (40GB) - OK
# CUDA Version: 12.2
# CuPy Version: 12.2.0
# All GPU tests passed
```

**Step 3: Verify Machine Learning Model**

```bash
# Test ML model loading and prediction
./deployment/scripts/deploy_vastai.sh ml-test

# Expected output:
# Loading training data: 70 solved puzzles
# Training Random Forest model...
# Model training completed: MAE=0.045, R²=0.823
# Predicting hot zones for puzzle 71...
# Generated 5 hot zones with confidence scores: [0.78, 0.65, 0.52, 0.41, 0.33]
# ML system ready
```

### Starting Your First Solve

**Step 1: Configure the Target**

```bash
# Set environment variables for puzzle 71
export PUZZLE_NUMBER=71
export TARGET_ADDRESS="1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9"
export MAX_TIME=7200  # 2 hours for first attempt
```

**Step 2: Launch the Solver**

```bash
# Start puzzle solving with monitoring
./deployment/scripts/deploy_vastai.sh start \
    --puzzle 71 \
    --address "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9" \
    --max-time 7200 \
    --enable-monitoring

# The solver will:
# 1. Load the machine learning model
# 2. Predict hot zones for puzzle 71
# 3. Distribute work across 4 GPUs
# 4. Begin systematic key searching
# 5. Report progress every 60 seconds
```

**Step 3: Monitor Progress**

Open a new terminal and monitor the solving progress:

```bash
# Watch real-time logs
./deployment/scripts/deploy_vastai.sh logs --follow

# Expected log output:
# [12:34:56] Starting Bitcoin Puzzle Solver System...
# [12:34:57] Initializing CUDA engine...
# [12:34:58] Detected 4 GPU(s): 4x A100-SXM4-40GB
# [12:34:59] Initializing ML hot zone predictor...
# [12:35:01] Loaded training data with 70 solved puzzles
# [12:35:02] ML model training completed successfully
# [12:35:03] Starting puzzle #71 solving process
# [12:35:04] Target address: 1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9
# [12:35:05] Generated 5 hot zones for search
# [12:35:06] Hot zone 1: center at 23.4% of range, confidence: 0.78
# [12:35:07] Hot zone 2: center at 67.8% of range, confidence: 0.65
# [12:35:08] Searching hot zone 1/5...
# [12:35:09] Created 1024 batches for processing
# [12:35:10] Processing batch 1/1024 on GPU 0: keys 1180591620717411303424 to 1180591620718411303423
# [12:35:11] GPU 0 throughput: 12,543 keys/sec
# [12:35:12] GPU 1 throughput: 12,687 keys/sec
# [12:35:13] GPU 2 throughput: 12,456 keys/sec
# [12:35:14] GPU 3 throughput: 12,598 keys/sec
# [12:35:15] Total throughput: 50,284 keys/sec
```

### Understanding the Output

The solver provides detailed information about its progress:

**Hot Zone Information**

```
Hot zone 1: center at 23.4% of range, confidence: 0.78
```

This means the machine learning model predicts a 78% confidence that the private key is located around 23.4% through the puzzle's key range. The solver will search 2^40 keys (about 1 trillion) around this center point.

**GPU Performance Metrics**

```
GPU 0 throughput: 12,543 keys/sec
Total throughput: 50,284 keys/sec
```

This shows each GPU is processing about 12,500 keys per second, for a total system throughput of over 50,000 keys per second. This rate means you can search through 1 trillion keys in about 5.5 hours.

**Progress Tracking**

```
Batch 1/1024 completed: 1,000,000 keys (0.1% of hot zone)
Estimated time remaining: 4.8 hours
```

The solver tracks progress through each hot zone and provides time estimates based on current performance.

### Monitoring Dashboard

Access the web-based monitoring dashboard for visual progress tracking:

1. **Get Instance IP Address**:
```bash
./deployment/scripts/deploy_vastai.sh get-ip
# Output: Instance IP: 123.456.789.012
```

2. **Open Dashboard**: Navigate to `http://123.456.789.012:3000` in your web browser

3. **Dashboard Features**:
   - **GPU Utilization**: Real-time GPU usage, temperature, and memory
   - **Throughput Graphs**: Keys per second over time
   - **Hot Zone Progress**: Visual progress through each predicted zone
   - **System Health**: CPU, memory, and network usage
   - **Prediction Accuracy**: ML model performance metrics

### What to Expect

**Timeline Expectations**

Based on the SPR methodology and system performance:

- **Hot Zone Search**: Each zone takes 4-6 hours to search completely
- **Total Time**: With 5 hot zones, expect 20-30 hours for complete coverage
- **Success Probability**: 80% chance of finding the key within the predicted zones

**Performance Indicators**

**Good Performance Signs**:
- GPU utilization consistently above 90%
- Throughput above 40,000 keys/second total
- No error messages in logs
- Steady progress through hot zones

**Warning Signs**:
- GPU utilization below 80%
- Frequent error messages
- Throughput significantly below expectations
- System instability or crashes

### Handling Results

**If a Solution is Found**

When the solver finds the correct private key, you'll see output like:

```
🎉 PUZZLE #71 SOLVED! 🎉
Solution key: 20d45a6a762535700ce9e0b216e31994335db8a5
Private key (hex): 0000000000000000000000000000000000000000000020d45a6a762535700ce9e0b216e31994335db8a5
Bitcoin address: 1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9
Time taken: 14,327.45 seconds (3.98 hours)
Keys processed: 724,891,234,567
Average rate: 50,584 keys/sec
Hot zone: 2 (confidence: 0.65)
```

**Immediate Actions**:

1. **Secure the Private Key**: Immediately copy the private key to a secure location
2. **Verify the Solution**: Double-check that the private key generates the correct address
3. **Stop the Solver**: Terminate the solving process to avoid additional costs
4. **Claim the Reward**: Import the private key into a Bitcoin wallet to claim the reward

**If No Solution is Found**

If the solver completes all hot zones without finding a solution:

```
Puzzle #71 search completed without solution
Time taken: 28,456.78 seconds (7.90 hours)
Keys processed: 5,000,000,000,000 (5 trillion)
Hot zones searched: 5/5
Coverage: 0.42% of total key space
```

**Next Steps**:

1. **Analyze Results**: Review which hot zones were searched and their confidence scores
2. **Adjust Strategy**: Consider searching additional zones or different puzzles
3. **Cost Analysis**: Evaluate the cost-benefit ratio for continued searching
4. **Community Sharing**: Share anonymized results with the research community

### Cost Management

**Monitoring Costs**

```bash
# Check current instance cost
./deployment/scripts/deploy_vastai.sh cost-status

# Expected output:
# Instance ID: 12345
# Hourly Rate: $2.45/hour
# Running Time: 3.2 hours
# Total Cost: $7.84
# Estimated Daily Cost: $58.80
```

**Cost Optimization**

- **Time Limits**: Set reasonable time limits to prevent runaway costs
- **Instance Selection**: Choose the most cost-effective instances for your budget
- **Scheduled Solving**: Run during off-peak hours when prices are lower
- **Progress Monitoring**: Stop early if progress is significantly slower than expected

### Troubleshooting Common Issues

**GPU Memory Errors**

If you see "CUDA out of memory" errors:

```bash
# Reduce batch size in configuration
sed -i 's/batch_size = 1000000/batch_size = 500000/' deployment/configs/solver_config.toml

# Restart the solver
./deployment/scripts/deploy_vastai.sh restart
```

**Low Performance**

If throughput is significantly below expectations:

```bash
# Check GPU utilization
./deployment/scripts/deploy_vastai.sh gpu-status

# Verify CUDA installation
./deployment/scripts/deploy_vastai.sh cuda-test

# Check for thermal throttling
./deployment/scripts/deploy_vastai.sh thermal-status
```

**Network Connectivity Issues**

If the solver loses connection to vast.ai:

```bash
# Check instance status
./deployment/scripts/deploy_vastai.sh status

# Reconnect if needed
./deployment/scripts/deploy_vastai.sh reconnect

# Resume solving from last checkpoint
./deployment/scripts/deploy_vastai.sh resume
```

This completes the basic guide to running your first puzzle. The next sections will cover advanced optimization techniques, monitoring strategies, and troubleshooting procedures.

## References

[1] Bitcoin Puzzle Challenge - Private Keys Directory. https://privatekeys.pw/puzzles/bitcoin-puzzle-tx?status=all


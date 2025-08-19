# Bitcoin Puzzle Solver - vast.ai Deployment Guide

This guide provides step-by-step instructions for deploying the Bitcoin Puzzle Solver on vast.ai infrastructure with GPU acceleration.

## 🎯 Overview

The Bitcoin Puzzle Solver is optimized for vast.ai's GPU instances, supporting:
- **4x NVIDIA A100** (80GB VRAM each) - Maximum performance
- **4x NVIDIA V100** (32GB VRAM each) - High performance  
- **4x NVIDIA RTX5090** (24GB VRAM each) - Consumer-grade performance

## 📋 Prerequisites

### vast.ai Account Setup
1. Create account at [vast.ai](https://vast.ai)
2. Add payment method and credits
3. Verify email and complete account setup

### Local Preparation
1. Download the complete Bitcoin Puzzle Solver package
2. Ensure you have the deployment files:
   - `vast_ai_package.tar.gz` (created by the deployment script)
   - All Python and Rust source files
   - Training data (`bitcoin_puzzle_data.csv`)

## 🚀 Deployment Steps

### Step 1: Create Deployment Package

On your local machine:

```bash
# Create the deployment package
python3 vast_ai_deployment.py --create-package

# This creates vast_ai_package.tar.gz with all necessary files
```

### Step 2: Choose vast.ai Instance

1. **Log into vast.ai console**
2. **Browse available instances**
   - Filter by GPU type (A100, V100, RTX5090)
   - Look for 4x GPU configurations
   - Check for sufficient RAM (32GB+ recommended)
   - Verify CUDA compatibility

3. **Recommended Instance Specifications**
   ```
   GPU: 4x A100 (80GB) or 4x V100 (32GB) or 4x RTX5090 (24GB)
   RAM: 64GB+ (128GB preferred)
   Storage: 100GB+ SSD
   Network: 1Gbps+
   CUDA: 11.8+
   ```

### Step 3: Launch Instance

1. **Select instance and click "Rent"**
2. **Configure instance settings:**
   ```
   Image: pytorch/pytorch:latest or nvidia/cuda:11.8-devel-ubuntu22.04
   Disk Space: 100GB minimum
   SSH Key: Upload your public key
   ```

3. **Launch instance and wait for "Running" status**

### Step 4: Connect to Instance

```bash
# SSH into your vast.ai instance
ssh -p [PORT] root@[INSTANCE_IP]

# Example:
ssh -p 12345 root@123.456.789.012
```

### Step 5: Upload and Setup

1. **Upload deployment package:**
   ```bash
   # From your local machine
   scp -P [PORT] vast_ai_package.tar.gz root@[INSTANCE_IP]:/root/
   
   # On the vast.ai instance
   cd /root
   tar -xzf vast_ai_package.tar.gz
   cd bitcoin_puzzle_solver
   ```

2. **Run setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This script will:
   - Install system dependencies
   - Setup Python environment
   - Install Rust toolchain
   - Compile the Rust core
   - Install Python packages
   - Configure GPU drivers

### Step 6: Verify Installation

```bash
# Test the installation
python3 test_complete_solution.py

# Check GPU availability
nvidia-smi

# Verify CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

## 🔧 Configuration

### GPU Configuration

Edit the configuration for your specific hardware:

```python
# In gpu_optimization.py
optimizer = VastAIGPUOptimizer()

# Configure for your GPU type
if "A100" in gpu_name:
    optimizer.configure_for_hardware("A100")
elif "V100" in gpu_name:
    optimizer.configure_for_hardware("V100")
elif "RTX5090" in gpu_name:
    optimizer.configure_for_hardware("RTX5090")
```

### Security Configuration

Set up secure password for encryption:

```bash
# Set environment variable for master password
export BITCOIN_PUZZLE_MASTER_PASSWORD="your_very_secure_password_here"

# Or edit the configuration file
nano config.json
```

### Memory Optimization

For different GPU configurations:

```python
# A100 (80GB VRAM each)
batch_size = 1000000
memory_limit = "75GB"

# V100 (32GB VRAM each)  
batch_size = 400000
memory_limit = "30GB"

# RTX5090 (24GB VRAM each)
batch_size = 300000
memory_limit = "22GB"
```

## 🎮 Running the Solver

### Basic Execution

```bash
# Start the solver with default settings
python3 run_solver.py

# Or with specific puzzle target
python3 run_solver.py --puzzle 71 --target-address 1By8rxztgeJeUX7qQjhAdmzQtAeqcE8Kd1
```

### Advanced Options

```bash
# Full command with all options
python3 run_solver.py \
    --puzzle 71 \
    --target-address 1By8rxztgeJeUX7qQjhAdmzQtAeqcE8Kd1 \
    --max-iterations 10000000 \
    --batch-size 1000000 \
    --gpu-count 4 \
    --security-level HIGH \
    --output-file solved_puzzle_71.csv
```

### Monitoring Progress

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor solver progress
tail -f solver.log

# Check system resources
htop
```

## 📊 Performance Optimization

### GPU Memory Management

```python
# Optimize for your available VRAM
def optimize_batch_size(available_vram_gb):
    if available_vram_gb >= 75:  # A100
        return 1000000
    elif available_vram_gb >= 30:  # V100
        return 400000
    elif available_vram_gb >= 22:  # RTX5090
        return 300000
    else:
        return 100000  # Fallback
```

### Network Optimization

```bash
# Optimize network settings for vast.ai
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
sysctl -p
```

### Storage Optimization

```bash
# Use tmpfs for temporary files
mkdir /tmp/solver_temp
mount -t tmpfs -o size=8G tmpfs /tmp/solver_temp
export TEMP_DIR=/tmp/solver_temp
```

## 🔐 Security Best Practices

### Data Protection

1. **Encrypt all outputs:**
   ```python
   # Always use encrypted export
   csv_manager.export_csv("results.csv", ExportConfig(
       encrypt_output=True,
       include_private_keys=True
   ))
   ```

2. **Secure password management:**
   ```bash
   # Use environment variables
   export BITCOIN_PUZZLE_MASTER_PASSWORD="$(openssl rand -base64 32)"
   ```

3. **Regular backups:**
   ```bash
   # Backup results every hour
   crontab -e
   # Add: 0 * * * * /root/backup_results.sh
   ```

### Network Security

```bash
# Configure firewall
ufw enable
ufw allow ssh
ufw allow from 10.0.0.0/8  # vast.ai internal network
ufw deny incoming
```

## 📈 Monitoring and Logging

### Performance Monitoring

```python
# Monitor key generation rate
def monitor_performance():
    start_time = time.time()
    keys_generated = 0
    
    while True:
        # ... key generation loop ...
        keys_generated += batch_size
        
        if time.time() - start_time >= 60:  # Every minute
            rate = keys_generated / (time.time() - start_time)
            print(f"Rate: {rate:.0f} keys/second")
            start_time = time.time()
            keys_generated = 0
```

### Log Management

```bash
# Setup log rotation
cat > /etc/logrotate.d/bitcoin-solver << EOF
/root/bitcoin_puzzle_solver/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   batch_size = batch_size // 2
   torch.cuda.empty_cache()
   ```

2. **SSH Connection Lost**
   ```bash
   # Use screen or tmux
   screen -S solver
   python3 run_solver.py
   # Ctrl+A, D to detach
   
   # Reconnect later
   screen -r solver
   ```

3. **Slow Performance**
   ```bash
   # Check GPU utilization
   nvidia-smi
   
   # Check CPU usage
   htop
   
   # Check memory usage
   free -h
   ```

### Error Recovery

```python
# Implement checkpoint system
def save_checkpoint(state, filename):
    with open(filename, 'wb') as f:
        pickle.dump(state, f)

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
```

## 💰 Cost Optimization

### Instance Selection

```
GPU Type    | Cost/hour | Performance | Recommended Use
A100 (4x)   | $3-5      | Maximum     | Production solving
V100 (4x)   | $2-3      | High        | Development/testing
RTX5090(4x) | $1-2      | Good        | Budget solving
```

### Cost Management

```bash
# Auto-shutdown after completion
echo "shutdown -h +60" | at now  # Shutdown in 1 hour

# Monitor costs
python3 cost_monitor.py --max-spend 100  # Stop at $100
```

## 📊 Expected Performance

### Key Generation Rates

```
Hardware     | Keys/Second/GPU | Total Rate (4x GPUs)
A100         | 50,000         | 200,000
V100         | 30,000         | 120,000  
RTX5090      | 20,000         | 80,000
```

### Solving Estimates

```
Puzzle | Search Space | Est. Time (A100 4x) | Est. Time (V100 4x)
71     | 2^70        | 2-5 days           | 4-8 days
72     | 2^71        | 4-10 days          | 8-16 days
73     | 2^72        | 8-20 days          | 16-32 days
```

## 🎯 Success Metrics

### Key Performance Indicators

1. **Keys/Second Rate**: Target >10,000 per core
2. **GPU Utilization**: Target >90%
3. **Memory Usage**: Target <90% of available
4. **Network Efficiency**: Minimal data transfer
5. **Cost Efficiency**: Optimize $/key generated

### Monitoring Dashboard

```python
# Create monitoring dashboard
def create_dashboard():
    metrics = {
        'keys_per_second': get_key_rate(),
        'gpu_utilization': get_gpu_usage(),
        'memory_usage': get_memory_usage(),
        'estimated_completion': estimate_completion(),
        'cost_so_far': calculate_cost()
    }
    return metrics
```

## 📞 Support

### Getting Help

1. **Check logs first**: `tail -f solver.log`
2. **Run diagnostics**: `python3 diagnostics.py`
3. **Check vast.ai status**: vast.ai console
4. **Review documentation**: This guide and README.md

### Emergency Procedures

```bash
# Save current state and shutdown
python3 emergency_save.py
shutdown -h now

# Quick restart
python3 run_solver.py --resume-from-checkpoint
```

---

**Note**: This deployment guide assumes familiarity with Linux command line and basic GPU computing concepts. Always test with small batches before running full-scale solving attempts.


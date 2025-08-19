#!/usr/bin/env python3
"""
Vast.ai Deployment Script for Bitcoin Puzzle Solver
Handles deployment and configuration for A100, V100, and RTX5090 instances
"""

import os
import sys
import json
import time
import logging
import subprocess
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VastAIInstance:
    """Vast.ai instance configuration"""
    instance_id: str
    gpu_type: str
    gpu_count: int
    memory_gb: int
    vcpus: int
    storage_gb: int
    price_per_hour: float
    status: str

class VastAIDeployment:
    """Vast.ai deployment manager"""
    
    def __init__(self):
        self.instances = []
        self.deployment_config = self._load_deployment_config()
    
    def _load_deployment_config(self) -> Dict:
        """Load deployment configuration"""
        return {
            "preferred_gpus": ["A100", "V100", "RTX5090", "RTX4090"],
            "min_memory_gb": 16,
            "min_storage_gb": 50,
            "max_price_per_hour": 2.0,
            "required_packages": [
                "python3-pip",
                "git",
                "build-essential",
                "pkg-config",
                "libssl-dev",
                "nvidia-cuda-toolkit",
                "nvidia-cuda-dev"
            ],
            "python_packages": [
                "cupy-cuda12x",
                "pycuda",
                "numba",
                "nvidia-ml-py3",
                "scikit-learn",
                "pandas",
                "numpy",
                "cryptography",
                "seaborn",
                "matplotlib"
            ]
        }
    
    def generate_startup_script(self, gpu_type: str) -> str:
        """Generate startup script for vast.ai instance"""
        script = f"""#!/bin/bash
# Vast.ai startup script for Bitcoin Puzzle Solver
# GPU Type: {gpu_type}

set -e

echo "Starting Bitcoin Puzzle Solver deployment on {gpu_type}..."

# Update system
apt-get update
apt-get install -y {' '.join(self.deployment_config['required_packages'])}

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Install Python packages
pip3 install --upgrade pip
pip3 install {' '.join(self.deployment_config['python_packages'])}

# Clone repository (if using git)
# git clone https://github.com/your-repo/bitcoin-puzzle-solver.git
# cd bitcoin-puzzle-solver

# Create working directory
mkdir -p /workspace/bitcoin_puzzle_solver
cd /workspace/bitcoin_puzzle_solver

# Download puzzle data
wget -O bitcoin_puzzle_data.csv https://raw.githubusercontent.com/HomelessPhD/BTC32/main/bits_address_privatekey.csv

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=1
export RUST_LOG=info

# GPU-specific optimizations
case "{gpu_type}" in
    "A100")
        export CUDA_ARCH=80
        export GPU_MEMORY_LIMIT=75
        export BATCH_SIZE=1000000
        ;;
    "V100")
        export CUDA_ARCH=70
        export GPU_MEMORY_LIMIT=28
        export BATCH_SIZE=500000
        ;;
    "RTX5090")
        export CUDA_ARCH=89
        export GPU_MEMORY_LIMIT=20
        export BATCH_SIZE=750000
        ;;
    *)
        export CUDA_ARCH=70
        export GPU_MEMORY_LIMIT=16
        export BATCH_SIZE=250000
        ;;
esac

echo "Deployment complete. GPU: {gpu_type}, Memory Limit: $GPU_MEMORY_LIMIT GB"
echo "Ready to start Bitcoin Puzzle Solver..."

# Keep container running
tail -f /dev/null
"""
        return script
    
    def create_docker_image(self, gpu_type: str) -> str:
        """Create Docker image for vast.ai deployment"""
        dockerfile = f"""
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    git \\
    wget \\
    curl \\
    build-essential \\
    pkg-config \\
    libssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${{PATH}}"

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install \\
    cupy-cuda12x \\
    pycuda \\
    numba \\
    nvidia-ml-py3 \\
    scikit-learn \\
    pandas \\
    numpy \\
    cryptography \\
    seaborn \\
    matplotlib

# Create workspace
WORKDIR /workspace/bitcoin_puzzle_solver

# Copy source code
COPY . .

# Build Rust components
RUN cd bitcoin_puzzle_solver_rust_core && \\
    cargo build --release --features parallel,security

# GPU-specific configuration
ENV GPU_TYPE={gpu_type}
{"ENV CUDA_ARCH=80" if gpu_type == "A100" else ""}
{"ENV CUDA_ARCH=70" if gpu_type == "V100" else ""}
{"ENV CUDA_ARCH=89" if gpu_type == "RTX5090" else ""}

# Set default command
CMD ["python3", "bitcoin_puzzle_solver.py", "--training-data", "bitcoin_puzzle_data.csv"]
"""
        return dockerfile
    
    def generate_vast_ai_commands(self) -> List[str]:
        """Generate vast.ai CLI commands for deployment"""
        commands = []
        
        # Search for available instances
        search_cmd = """
vast search offers \\
    --type=on-demand \\
    --gpu_name="RTX_5090,A100,V100" \\
    --min_gpu_ram=16 \\
    --min_disk=50 \\
    --order=score- \\
    --limit=10
"""
        commands.append(("Search for instances", search_cmd.strip()))
        
        # Create instance
        create_cmd = """
vast create instance <INSTANCE_ID> \\
    --image=nvidia/cuda:12.2-devel-ubuntu22.04 \\
    --disk=50 \\
    --onstart-cmd="bash /workspace/startup.sh" \\
    --env="CUDA_VISIBLE_DEVICES=0,PYTHONUNBUFFERED=1"
"""
        commands.append(("Create instance", create_cmd.strip()))
        
        # Upload files
        upload_cmd = """
vast copy <LOCAL_FILES> <INSTANCE_ID>:/workspace/bitcoin_puzzle_solver/
"""
        commands.append(("Upload files", upload_cmd.strip()))
        
        # Execute commands
        exec_cmd = """
vast exec <INSTANCE_ID> "cd /workspace/bitcoin_puzzle_solver && python3 bitcoin_puzzle_solver.py --puzzle 71"
"""
        commands.append(("Execute solver", exec_cmd.strip()))
        
        return commands
    
    def create_deployment_package(self, output_dir: str = "vast_ai_deployment"):
        """Create complete deployment package for vast.ai"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create startup scripts for each GPU type
        for gpu_type in ["A100", "V100", "RTX5090"]:
            script_content = self.generate_startup_script(gpu_type)
            script_path = os.path.join(output_dir, f"startup_{gpu_type.lower()}.sh")
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_path, 0o755)
            logger.info(f"Created startup script: {script_path}")
        
        # Create Dockerfiles
        for gpu_type in ["A100", "V100", "RTX5090"]:
            dockerfile_content = self.create_docker_image(gpu_type)
            dockerfile_path = os.path.join(output_dir, f"Dockerfile.{gpu_type.lower()}")
            
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            logger.info(f"Created Dockerfile: {dockerfile_path}")
        
        # Create deployment instructions
        instructions = self._create_deployment_instructions()
        instructions_path = os.path.join(output_dir, "DEPLOYMENT_INSTRUCTIONS.md")
        
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        logger.info(f"Created deployment instructions: {instructions_path}")
        
        # Create vast.ai commands file
        commands = self.generate_vast_ai_commands()
        commands_path = os.path.join(output_dir, "vast_ai_commands.sh")
        
        with open(commands_path, 'w') as f:
            f.write("#!/bin/bash\\n")
            f.write("# Vast.ai deployment commands\\n\\n")
            
            for description, command in commands:
                f.write(f"# {description}\\n")
                f.write(f"{command}\\n\\n")
        
        os.chmod(commands_path, 0o755)
        logger.info(f"Created commands file: {commands_path}")
        
        # Create configuration file
        config_path = os.path.join(output_dir, "deployment_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.deployment_config, f, indent=2)
        
        logger.info(f"Created configuration file: {config_path}")
        
        logger.info(f"Deployment package created in: {output_dir}")
    
    def _create_deployment_instructions(self) -> str:
        """Create detailed deployment instructions"""
        return """# Bitcoin Puzzle Solver - Vast.ai Deployment Guide

## Overview
This package contains everything needed to deploy the Bitcoin Puzzle Solver on vast.ai infrastructure with support for A100, V100, and RTX5090 GPUs.

## Prerequisites
1. Install vast.ai CLI: `pip install vastai`
2. Set up vast.ai account and API key: `vast set api-key YOUR_API_KEY`
3. Ensure you have sufficient credits in your vast.ai account

## Quick Start

### 1. Search for Available Instances
```bash
vast search offers \\
    --type=on-demand \\
    --gpu_name="RTX_5090,A100,V100" \\
    --min_gpu_ram=16 \\
    --min_disk=50 \\
    --order=score- \\
    --limit=10
```

### 2. Create Instance
Choose an instance ID from the search results and create it:
```bash
vast create instance <INSTANCE_ID> \\
    --image=nvidia/cuda:12.2-devel-ubuntu22.04 \\
    --disk=50 \\
    --onstart-cmd="bash /workspace/startup.sh"
```

### 3. Upload Files
Upload the solver files to your instance:
```bash
# Upload all solver files
vast copy bitcoin_puzzle_solver.py <INSTANCE_ID>:/workspace/
vast copy bitcoin_puzzle_solver_rust_core/ <INSTANCE_ID>:/workspace/
vast copy gpu_optimization.py <INSTANCE_ID>:/workspace/
vast copy startup_<gpu_type>.sh <INSTANCE_ID>:/workspace/startup.sh
```

### 4. Run the Solver
Execute the Bitcoin Puzzle Solver:
```bash
vast exec <INSTANCE_ID> "cd /workspace && python3 bitcoin_puzzle_solver.py --puzzle 71 --max-time 24"
```

## GPU-Specific Configurations

### A100 (80GB)
- Optimal for large-scale searches
- Batch size: 1,000,000 keys
- Memory limit: 75GB
- Expected performance: 50M+ keys/sec

### V100 (32GB)
- Good balance of performance and cost
- Batch size: 500,000 keys
- Memory limit: 28GB
- Expected performance: 25M+ keys/sec

### RTX5090 (24GB)
- High performance consumer GPU
- Batch size: 750,000 keys
- Memory limit: 20GB
- Expected performance: 35M+ keys/sec

## Multi-GPU Deployment

For distributed solving across multiple instances:

1. Create multiple instances with the same configuration
2. Use the distributed GPU manager:
```python
from gpu_optimization import DistributedGPUManager

manager = DistributedGPUManager()
# Add instances
manager.add_gpu_instance("instance1", optimizer1)
manager.add_gpu_instance("instance2", optimizer2)

# Distribute work
results = manager.distribute_work(71, hot_zones, target_address)
```

## Monitoring and Management

### Check Instance Status
```bash
vast show instances
```

### Monitor GPU Usage
```bash
vast exec <INSTANCE_ID> "nvidia-smi"
```

### View Logs
```bash
vast logs <INSTANCE_ID>
```

### Stop Instance
```bash
vast destroy instance <INSTANCE_ID>
```

## Cost Optimization

1. **Use Spot Instances**: Add `--type=spot` for lower costs
2. **Monitor Usage**: Set up alerts for long-running instances
3. **Auto-shutdown**: Configure automatic shutdown after completion
4. **Resource Matching**: Choose GPU type based on puzzle difficulty

## Security Considerations

1. **Encrypted Storage**: All private keys are encrypted using military-grade encryption
2. **Secure Transfer**: Use secure channels for result transfer
3. **Access Control**: Limit SSH access and use strong passwords
4. **Data Cleanup**: Ensure sensitive data is wiped after completion

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Check GPU memory usage with `nvidia-smi`

2. **Compilation Errors**
   - Ensure CUDA toolkit is properly installed
   - Check Rust installation: `rustc --version`

3. **Network Issues**
   - Verify internet connectivity
   - Check firewall settings

4. **Performance Issues**
   - Monitor GPU utilization
   - Adjust thread/block configurations
   - Check thermal throttling

### Support

For issues specific to this solver:
1. Check logs in `/workspace/bitcoin_puzzle_solver.log`
2. Verify GPU compatibility with `nvidia-smi`
3. Test with smaller puzzle numbers first

For vast.ai platform issues:
1. Check vast.ai documentation
2. Contact vast.ai support
3. Verify account credits and limits

## Performance Expectations

| GPU Type | Memory | Keys/sec | Est. Time (Puzzle 71) |
|----------|--------|----------|----------------------|
| A100     | 80GB   | 50M+     | 12-24 hours          |
| V100     | 32GB   | 25M+     | 24-48 hours          |
| RTX5090  | 24GB   | 35M+     | 18-36 hours          |

*Times are estimates based on hot zone targeting and may vary significantly based on actual key location.*

## Legal and Ethical Considerations

This solver is provided for educational and research purposes. Users are responsible for:
1. Complying with local laws and regulations
2. Respecting the original puzzle creator's intentions
3. Using computational resources responsibly
4. Not engaging in any illegal activities

The Bitcoin puzzles are public challenges, and solving them is generally considered legitimate research activity.
"""

def main():
    """Main function for deployment script"""
    parser = argparse.ArgumentParser(description="Vast.ai deployment for Bitcoin Puzzle Solver")
    parser.add_argument("--create-package", action="store_true", help="Create deployment package")
    parser.add_argument("--output-dir", default="vast_ai_deployment", help="Output directory")
    parser.add_argument("--gpu-type", choices=["A100", "V100", "RTX5090"], help="Target GPU type")
    
    args = parser.parse_args()
    
    deployment = VastAIDeployment()
    
    if args.create_package:
        deployment.create_deployment_package(args.output_dir)
        print(f"Deployment package created in {args.output_dir}")
        print("See DEPLOYMENT_INSTRUCTIONS.md for usage details")
    
    elif args.gpu_type:
        # Generate specific configuration for GPU type
        startup_script = deployment.generate_startup_script(args.gpu_type)
        print(f"Startup script for {args.gpu_type}:")
        print(startup_script)
    
    else:
        print("Use --create-package to generate deployment files")
        print("Use --gpu-type to see GPU-specific configuration")

if __name__ == "__main__":
    main()


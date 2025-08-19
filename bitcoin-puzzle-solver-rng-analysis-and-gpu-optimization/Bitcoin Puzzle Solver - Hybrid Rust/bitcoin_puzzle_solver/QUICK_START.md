# Bitcoin Puzzle Solver - Quick Start Guide

**🚀 Get solving Bitcoin puzzles in 5 minutes!**

## ⚡ One-Command Installation

```bash
# Download and run the automated installer
curl -sSL https://raw.githubusercontent.com/your-repo/bitcoin-puzzle-solver/main/scripts/install.sh | bash
```

## 🎯 Immediate Usage

### 1. Solve a Puzzle (Basic)
```bash
# Activate environment
source venv/bin/activate

# Solve puzzle 71 with SPR optimization
./rust_core/target/release/puzzle-solver solve \
  -p 71 \
  -a 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH \
  --spr \
  -v
```

### 2. Solve with ML Hot Zone Prediction
```bash
# First, train ML models (one-time setup)
./rust_core/target/release/puzzle-solver train -d data/puzzles/bitcoin_puzzles.csv

# Predict hot zones for puzzles 71-75
./rust_core/target/release/puzzle-solver predict -p "71,72,73,74,75"

# Solve using predicted hot zone
./rust_core/target/release/puzzle-solver solve \
  -p 71 \
  -a 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH \
  --hot-zone 0x1a838b13505b26867 \
  --spr \
  --gpu-devices "0,1,2,3" \
  -v
```

### 3. Deploy to vast.ai (Cloud GPUs)
```bash
# Set your vast.ai API key
export VASTAI_API_KEY="your_api_key_here"

# Search for suitable instances
./scripts/deploy_vastai.sh search

# Create instance (replace 1234567 with actual instance ID)
./scripts/deploy_vastai.sh create 1234567

# Monitor progress
./scripts/deploy_vastai.sh monitor 1234567
```

## 📊 Performance Expectations

| Configuration | Expected Performance | Target Puzzles |
|---------------|---------------------|----------------|
| Single RTX 4090 | 40,000+ keys/sec | 71-80 |
| 4x RTX 4090 | 160,000+ keys/sec | 71-90 |
| 4x A100 | 200,000+ keys/sec | 71-100 |
| 8x A100 | 400,000+ keys/sec | 71-110 |

## 🎯 Success Rate Targets

- **Puzzles 71-80**: 80% solve rate in <5 days
- **Puzzles 81-90**: 60% solve rate in <30 days  
- **Puzzles 91-100**: 40% solve rate in <90 days

## 🔧 Quick Configuration

### Essential Settings (solver_config.toml)
```toml
[optimization]
use_spr = true
gpu_device_count = 4
enable_ml_prediction = true
search_algorithm = "auto"  # Automatically selects best algorithm

[hardware]
gpu_devices = [0, 1, 2, 3]
cpu_cores = 16
memory_limit_gb = 64

[search_range]
use_spr = true
spr_reduction_factor = 1000  # Reduces search space by 1000x
hot_zone_expansion = 2.0     # ±2^40 around predicted center
```

## 🚨 Important Notes

1. **GPU Memory**: Ensure each GPU has at least 8GB VRAM (24GB recommended)
2. **Power**: 4x RTX 4090 setup requires ~2000W power supply
3. **Cooling**: High-performance GPUs generate significant heat
4. **Legal**: Only solve puzzles you own or have permission to solve
5. **Backup**: Always backup your configuration and results

## 📈 Monitoring

```bash
# Real-time system monitoring
./scripts/monitor.sh

# Check GPU utilization
nvidia-smi -l 1

# View solver logs
tail -f data/logs/solver.log
```

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA not found | Install NVIDIA drivers + CUDA toolkit |
| Out of memory | Reduce batch size in config |
| Low GPU usage | Check GPU device IDs in config |
| Slow performance | Enable all optimizations in config |
| No solutions found | Verify target address and range |

## 📚 Next Steps

1. Read the [Complete Implementation Guide](COMPLETE_IMPLEMENTATION_GUIDE.md)
2. Review [35 Mathematical Optimizations](35_mathematical_optimizations.md)
3. Join the community for support and updates
4. Consider contributing improvements back to the project

**Happy puzzle solving! 🧩⚡**


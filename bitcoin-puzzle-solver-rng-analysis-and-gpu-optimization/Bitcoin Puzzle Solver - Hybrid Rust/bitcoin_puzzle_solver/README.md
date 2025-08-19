# Bitcoin Puzzle Solver - Hybrid Rust & Python System

**Author:** Manus AI  
**Version:** 1.0.0  
**License:** MIT  
**Optimized for:** 4x NVIDIA A100 GPUs on vast.ai

## 🎯 Overview

The Bitcoin Puzzle Solver is a high-performance, military-grade secure system designed to solve Bitcoin puzzles using advanced machine learning guidance and GPU acceleration. This hybrid Rust and Python implementation leverages the Sparse Priming Representation (SPR) methodology to dramatically reduce search space through intelligent hot zone prediction.

### Key Features

- **🚀 GPU Acceleration**: Optimized for 4x NVIDIA A100 GPUs with CUDA 12.x support
- **🧠 ML-Guided Search**: Random Forest regression for hot zone prediction
- **⚡ Hybrid Architecture**: Rust core for cryptographic operations, Python for orchestration
- **🔒 Military-Grade Security**: AES-256 encryption, secure memory management
- **📊 Real-Time Monitoring**: Comprehensive performance tracking and visualization
- **☁️ Cloud-Ready**: Automated deployment on vast.ai infrastructure
- **🎯 SPR Optimization**: Implements Sparse Priming Representation for efficient search

### Performance Targets

Based on the SPR methodology, this system targets:
- **Puzzles 71-80**: 80% solve rate in <5 days
- **Puzzles 91-100**: 40% solve rate in <30 days  
- **Speed**: >10,000 keys/sec/core on optimized hardware

## 🏗️ Architecture

The system implements a three-tier hybrid architecture:

1. **Rust Core Layer**: High-performance cryptographic operations with secp256k1 optimizations
2. **Python GPU Layer**: CUDA acceleration and machine learning coordination
3. **Orchestration Layer**: System coordination, monitoring, and deployment management

## 📋 Prerequisites

### Hardware Requirements

- **GPU**: 4x NVIDIA A100 (40GB) or equivalent high-end GPUs
- **RAM**: Minimum 128GB system memory
- **Storage**: 1TB+ NVMe SSD for optimal performance
- **Network**: High-bandwidth internet connection

### Software Requirements

- **Operating System**: Ubuntu 22.04 LTS (recommended)
- **CUDA**: Version 12.0 or higher
- **Docker**: Version 24.0 or higher
- **Python**: 3.11+
- **Rust**: 1.70+ with stable toolchain

### vast.ai Account

- Active vast.ai account with API access
- Sufficient credits for GPU rental
- API key configured in environment

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd bitcoin_puzzle_solver
chmod +x deployment/scripts/*.sh
```

### 2. Configure Environment

```bash
export VAST_API_KEY="your_vast_ai_api_key"
export PUZZLE_NUMBER=71
export TARGET_ADDRESS="1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9"
```

### 3. Deploy to vast.ai

```bash
./deployment/scripts/deploy_vastai.sh deploy --puzzle 71 --max-time 3600
```

### 4. Monitor Progress

```bash
./deployment/scripts/deploy_vastai.sh status
./deployment/scripts/deploy_vastai.sh logs
```

## 📚 Documentation

- [**Installation Guide**](docs/installation.md) - Detailed setup instructions
- [**Configuration Reference**](docs/configuration.md) - Complete configuration options
- [**API Documentation**](docs/api.md) - Python and Rust API reference
- [**Performance Tuning**](docs/performance.md) - Optimization guidelines
- [**Security Guide**](docs/security.md) - Security features and best practices
- [**Troubleshooting**](docs/troubleshooting.md) - Common issues and solutions

## 🎯 Puzzle Targets

The system includes predefined configurations for unsolved puzzles:

| Puzzle | Address | Reward | Difficulty |
|--------|---------|---------|------------|
| 71 | 1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9 | ~7.1 BTC | 2^70 keys |
| 72 | 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ | ~7.2 BTC | 2^71 keys |
| 73 | 19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG | ~7.3 BTC | 2^72 keys |

## 🔧 Configuration

The system uses TOML configuration files for easy customization:

```toml
[system]
puzzle_number = 71
max_time_seconds = 3600
log_level = "info"

[gpu]
enabled = true
batch_size = 1000000
max_concurrent_batches = 16

[ml]
enabled = true
max_hot_zones = 5
min_confidence = 0.1
```

## 📊 Monitoring

Real-time monitoring includes:

- **GPU Utilization**: Memory usage, temperature, power consumption
- **Processing Rate**: Keys per second across all devices
- **ML Predictions**: Hot zone confidence and accuracy
- **System Health**: CPU, memory, network, and storage metrics

Access monitoring dashboard at: `http://instance-ip:3000`

## 🔒 Security Features

- **Encryption**: AES-256-GCM for all sensitive data
- **Memory Protection**: Secure memory clearing and overflow prevention
- **Network Security**: TLS 1.3 with certificate pinning
- **Audit Logging**: Comprehensive security event tracking
- **Access Control**: Role-based permissions and multi-factor authentication

## 🧪 Testing

Run the test suite:

```bash
# Rust tests
cd rust_core && cargo test

# Python tests
cd python_gpu && python -m pytest tests/

# Integration tests
./scripts/run_integration_tests.sh

# Performance benchmark
./deployment/scripts/deploy_vastai.sh benchmark
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Users are responsible for compliance with all applicable laws and regulations. The authors assume no liability for any misuse or damages.

## 🆘 Support

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join our community discussions
- **Email**: Contact the development team

## 🙏 Acknowledgments

- Bitcoin puzzle community for research and insights
- vast.ai for providing accessible GPU infrastructure
- Rust and Python communities for excellent tooling
- NVIDIA for CUDA development platform

---

**Happy Puzzle Solving! 🧩**


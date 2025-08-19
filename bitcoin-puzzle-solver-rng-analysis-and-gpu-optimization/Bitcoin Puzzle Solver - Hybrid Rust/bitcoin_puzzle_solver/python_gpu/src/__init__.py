"""
Bitcoin Puzzle Solver - Python GPU Acceleration Layer

High-performance GPU-accelerated Bitcoin puzzle solving system with ML-guided optimization.
Designed for 4x NVIDIA A100 GPUs on vast.ai infrastructure.

Author: Manus AI
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Manus AI"

# Core imports
from .coordination.system_coordinator import SystemCoordinator
from .gpu_acceleration.cuda_engine import CUDAEngine
from .ml_models.hot_zone_predictor import HotZonePredictor
from .monitoring.performance_monitor import PerformanceMonitor

# Configuration
from .coordination.config import SystemConfig, GPUConfig, MLConfig

__all__ = [
    "SystemCoordinator",
    "CUDAEngine", 
    "HotZonePredictor",
    "PerformanceMonitor",
    "SystemConfig",
    "GPUConfig",
    "MLConfig",
]


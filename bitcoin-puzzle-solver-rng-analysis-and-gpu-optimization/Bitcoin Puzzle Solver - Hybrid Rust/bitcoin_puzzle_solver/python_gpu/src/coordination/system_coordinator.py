"""
System Coordinator for Bitcoin Puzzle Solving

Orchestrates the entire puzzle solving process by coordinating:
- Rust core cryptographic operations
- GPU acceleration via CUDA
- ML-guided hot zone prediction
- Multi-GPU workload distribution
- Progress monitoring and reporting
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

from ..gpu_acceleration.cuda_engine import CUDAEngine, BatchResult
from ..ml_models.hot_zone_predictor import HotZonePredictor, PredictionResult
from ..monitoring.performance_monitor import PerformanceMonitor
from .config import SystemConfig
from .rust_interface import RustCoreInterface

logger = logging.getLogger(__name__)


@dataclass
class PuzzleTarget:
    """Target puzzle information"""
    puzzle_number: int
    bitcoin_address: str
    address_hash160: bytes
    range_start: int
    range_end: int
    range_size: int
    reward_btc: float


@dataclass
class SolvingSession:
    """Information about a puzzle solving session"""
    session_id: str
    puzzle_target: PuzzleTarget
    start_time: float
    end_time: Optional[float] = None
    total_keys_processed: int = 0
    hot_zones_searched: int = 0
    solution_found: bool = False
    solution_key: Optional[int] = None
    gpu_utilization: Dict[int, float] = None
    performance_stats: Dict[str, Any] = None


@dataclass
class SolvingResult:
    """Final result of puzzle solving attempt"""
    success: bool
    puzzle_number: int
    solution_key: Optional[int] = None
    private_key_hex: Optional[str] = None
    bitcoin_address: str = ""
    total_time_seconds: float = 0.0
    total_keys_processed: int = 0
    average_keys_per_second: float = 0.0
    hot_zones_used: int = 0
    gpu_devices_used: int = 0
    session_data: Optional[SolvingSession] = None


class SystemCoordinator:
    """
    Main coordinator for the Bitcoin puzzle solving system
    
    Orchestrates all components to efficiently solve Bitcoin puzzles using:
    - ML-guided hot zone prediction
    - Multi-GPU parallel processing
    - Rust core cryptographic operations
    - Real-time performance monitoring
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Core components
        self.cuda_engine: Optional[CUDAEngine] = None
        self.hot_zone_predictor: Optional[HotZonePredictor] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.rust_interface: Optional[RustCoreInterface] = None
        
        # State management
        self.is_initialized = False
        self.current_session: Optional[SolvingSession] = None
        self.solving_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.total_sessions = 0
        self.total_solutions_found = 0
        self.total_keys_processed = 0
        
    async def initialize(self) -> None:
        """Initialize all system components"""
        logger.info("Initializing Bitcoin Puzzle Solver System...")
        
        try:
            # Initialize Rust core interface
            logger.info("Initializing Rust core interface...")
            self.rust_interface = RustCoreInterface(self.config.rust_config)
            await self.rust_interface.initialize()
            
            # Initialize CUDA engine
            if self.config.gpu_config.enabled:
                logger.info("Initializing CUDA engine...")
                self.cuda_engine = CUDAEngine(self.config.gpu_config)
                await self.cuda_engine.initialize()
                
                gpu_info = self.cuda_engine.get_device_info()
                logger.info(f"Initialized {len(gpu_info)} GPU device(s):")
                for gpu in gpu_info:
                    logger.info(f"  GPU {gpu['device_id']}: {gpu['name']} "
                              f"({gpu['total_memory_gb']:.1f} GB)")
            
            # Initialize ML predictor
            if self.config.ml_config.enabled:
                logger.info("Initializing ML hot zone predictor...")
                self.hot_zone_predictor = HotZonePredictor(self.config.ml_config.model_params)
                
                # Load or train model
                if self.config.ml_config.model_path and Path(self.config.ml_config.model_path).exists():
                    await self.hot_zone_predictor.load_model(self.config.ml_config.model_path)
                    logger.info("Loaded pre-trained ML model")
                elif self.config.ml_config.training_data_path:
                    await self.hot_zone_predictor.load_training_data(self.config.ml_config.training_data_path)
                    await self.hot_zone_predictor.train_model()
                    logger.info("Trained new ML model")
                    
                    # Save the trained model
                    if self.config.ml_config.model_path:
                        await self.hot_zone_predictor.save_model(self.config.ml_config.model_path)
                else:
                    logger.warning("No ML model or training data specified")
            
            # Initialize performance monitor
            logger.info("Initializing performance monitor...")
            self.performance_monitor = PerformanceMonitor(self.config.monitoring_config)
            await self.performance_monitor.initialize()
            
            self.is_initialized = True
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            await self.cleanup()
            raise
    
    async def solve_puzzle(
        self, 
        puzzle_number: int, 
        bitcoin_address: str,
        max_time_seconds: Optional[int] = None
    ) -> SolvingResult:
        """
        Solve a specific Bitcoin puzzle
        
        Args:
            puzzle_number: Target puzzle number (1-160)
            bitcoin_address: Target Bitcoin address
            max_time_seconds: Maximum solving time (None for unlimited)
            
        Returns:
            SolvingResult with outcome and statistics
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        logger.info(f"Starting puzzle #{puzzle_number} solving process")
        logger.info(f"Target address: {bitcoin_address}")
        
        # Create puzzle target
        puzzle_target = await self._create_puzzle_target(puzzle_number, bitcoin_address)
        
        # Create solving session
        session_id = f"puzzle_{puzzle_number}_{int(time.time())}"
        session = SolvingSession(
            session_id=session_id,
            puzzle_target=puzzle_target,
            start_time=time.time(),
            gpu_utilization={}
        )
        self.current_session = session
        self.total_sessions += 1
        
        try:
            # Start performance monitoring
            if self.performance_monitor:
                await self.performance_monitor.start_session(session_id)
            
            # Generate hot zones using ML prediction
            hot_zones = await self._generate_hot_zones(puzzle_number)
            session.hot_zones_searched = len(hot_zones)
            
            logger.info(f"Generated {len(hot_zones)} hot zones for search")
            
            # Execute search across hot zones
            result = await self._execute_search(session, hot_zones, max_time_seconds)
            
            # Finalize session
            session.end_time = time.time()
            session.solution_found = result.success
            session.solution_key = result.solution_key
            
            # Update global statistics
            self.total_keys_processed += session.total_keys_processed
            if result.success:
                self.total_solutions_found += 1
            
            # Stop performance monitoring
            if self.performance_monitor:
                await self.performance_monitor.stop_session(session_id)
                session.performance_stats = await self.performance_monitor.get_session_stats(session_id)
            
            result.session_data = session
            
            if result.success:
                logger.info(f"🎉 PUZZLE #{puzzle_number} SOLVED! 🎉")
                logger.info(f"Solution key: {result.private_key_hex}")
                logger.info(f"Time taken: {result.total_time_seconds:.2f} seconds")
                logger.info(f"Keys processed: {result.total_keys_processed:,}")
                logger.info(f"Average rate: {result.average_keys_per_second:,.0f} keys/sec")
            else:
                logger.info(f"Puzzle #{puzzle_number} search completed without solution")
                logger.info(f"Time taken: {result.total_time_seconds:.2f} seconds")
                logger.info(f"Keys processed: {result.total_keys_processed:,}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during puzzle solving: {e}")
            session.end_time = time.time()
            
            # Create error result
            return SolvingResult(
                success=False,
                puzzle_number=puzzle_number,
                bitcoin_address=bitcoin_address,
                total_time_seconds=session.end_time - session.start_time,
                session_data=session
            )
        
        finally:
            self.current_session = None
    
    async def _create_puzzle_target(self, puzzle_number: int, bitcoin_address: str) -> PuzzleTarget:
        """Create puzzle target information"""
        
        # Calculate range bounds
        range_start = 2**(puzzle_number - 1)
        range_end = 2**puzzle_number - 1
        range_size = range_end - range_start + 1
        
        # Get address hash160 from Rust core
        address_hash160 = await self.rust_interface.get_address_hash160(bitcoin_address)
        
        # Estimate reward (simplified calculation)
        if puzzle_number <= 63:
            reward_btc = puzzle_number * 0.001  # Historical pattern
        else:
            reward_btc = (puzzle_number - 50) * 0.1  # Current pattern
        
        return PuzzleTarget(
            puzzle_number=puzzle_number,
            bitcoin_address=bitcoin_address,
            address_hash160=address_hash160,
            range_start=range_start,
            range_end=range_end,
            range_size=range_size,
            reward_btc=reward_btc
        )
    
    async def _generate_hot_zones(self, puzzle_number: int) -> List[PredictionResult]:
        """Generate hot zones for searching"""
        
        if self.hot_zone_predictor and self.config.ml_config.enabled:
            # Use ML prediction
            logger.info("Generating ML-guided hot zones...")
            hot_zones = await self.hot_zone_predictor.predict_hot_zones(
                puzzle_number=puzzle_number,
                num_zones=self.config.ml_config.max_hot_zones
            )
            
            # Filter by confidence threshold
            filtered_zones = [
                zone for zone in hot_zones 
                if zone.confidence >= self.config.ml_config.min_confidence
            ]
            
            if filtered_zones:
                logger.info(f"Using {len(filtered_zones)} high-confidence hot zones")
                return filtered_zones
            else:
                logger.warning("No high-confidence zones found, using all predictions")
                return hot_zones
        
        else:
            # Fallback: create zones based on statistical analysis
            logger.info("Generating statistical hot zones...")
            return await self._generate_statistical_zones(puzzle_number)
    
    async def _generate_statistical_zones(self, puzzle_number: int) -> List[PredictionResult]:
        """Generate hot zones based on statistical analysis"""
        
        range_start = 2**(puzzle_number - 1)
        range_end = 2**puzzle_number - 1
        range_size = range_end - range_start + 1
        
        # Create zones at common statistical positions
        positions = [0.1, 0.25, 0.5, 0.75, 0.9]  # 10%, 25%, 50%, 75%, 90%
        zones = []
        
        for i, pos in enumerate(positions):
            center = int(range_start + pos * range_size)
            radius = 2**40  # 1 trillion keys
            
            zone = PredictionResult(
                puzzle_number=puzzle_number,
                predicted_position=pos,
                confidence=0.2,  # Low confidence for statistical zones
                search_center=center,
                search_radius=radius,
                search_start=max(range_start, center - radius),
                search_end=min(range_end, center + radius),
                model_features={},
                metadata={'strategy': 'statistical', 'position': pos}
            )
            zones.append(zone)
        
        return zones
    
    async def _execute_search(
        self, 
        session: SolvingSession, 
        hot_zones: List[PredictionResult],
        max_time_seconds: Optional[int]
    ) -> SolvingResult:
        """Execute the search across hot zones"""
        
        start_time = time.time()
        total_keys_processed = 0
        
        # Process each hot zone
        for zone_idx, zone in enumerate(hot_zones):
            logger.info(f"Searching hot zone {zone_idx + 1}/{len(hot_zones)}")
            logger.info(f"  Range: {zone.search_start} to {zone.search_end}")
            logger.info(f"  Confidence: {zone.confidence:.3f}")
            
            # Check time limit
            if max_time_seconds and (time.time() - start_time) > max_time_seconds:
                logger.info("Time limit reached, stopping search")
                break
            
            # Create batches for this zone
            batches = self._create_batches(zone)
            logger.info(f"  Created {len(batches)} batches for processing")
            
            # Process batches on GPUs
            if self.cuda_engine:
                batch_results = await self.cuda_engine.process_multiple_batches(
                    key_ranges=batches,
                    target_hash160=session.puzzle_target.address_hash160,
                    max_concurrent=self.config.gpu_config.max_concurrent_batches
                )
            else:
                # Fallback to CPU processing via Rust
                batch_results = await self._process_batches_cpu(batches, session.puzzle_target)
            
            # Check results
            for batch_result in batch_results:
                total_keys_processed += batch_result.keys_processed
                
                if batch_result.found_solution:
                    # Solution found!
                    elapsed_time = time.time() - start_time
                    
                    # Verify solution with Rust core
                    is_valid = await self.rust_interface.verify_solution(
                        batch_result.solution_key,
                        session.puzzle_target.bitcoin_address
                    )
                    
                    if is_valid:
                        private_key_hex = hex(batch_result.solution_key)[2:].zfill(64)
                        
                        return SolvingResult(
                            success=True,
                            puzzle_number=session.puzzle_target.puzzle_number,
                            solution_key=batch_result.solution_key,
                            private_key_hex=private_key_hex,
                            bitcoin_address=session.puzzle_target.bitcoin_address,
                            total_time_seconds=elapsed_time,
                            total_keys_processed=total_keys_processed,
                            average_keys_per_second=total_keys_processed / elapsed_time,
                            hot_zones_used=zone_idx + 1,
                            gpu_devices_used=len(self.cuda_engine.devices) if self.cuda_engine else 0
                        )
                    else:
                        logger.warning(f"False positive solution detected: {batch_result.solution_key}")
            
            # Update session statistics
            session.total_keys_processed = total_keys_processed
            
            # Report progress
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                rate = total_keys_processed / elapsed_time
                logger.info(f"  Zone {zone_idx + 1} completed: {total_keys_processed:,} keys "
                          f"({rate:,.0f} keys/sec)")
        
        # No solution found
        elapsed_time = time.time() - start_time
        session.total_keys_processed = total_keys_processed
        
        return SolvingResult(
            success=False,
            puzzle_number=session.puzzle_target.puzzle_number,
            bitcoin_address=session.puzzle_target.bitcoin_address,
            total_time_seconds=elapsed_time,
            total_keys_processed=total_keys_processed,
            average_keys_per_second=total_keys_processed / elapsed_time if elapsed_time > 0 else 0,
            hot_zones_used=len(hot_zones),
            gpu_devices_used=len(self.cuda_engine.devices) if self.cuda_engine else 0
        )
    
    def _create_batches(self, zone: PredictionResult) -> List[Tuple[int, int]]:
        """Create processing batches for a hot zone"""
        
        batch_size = self.config.gpu_config.batch_size
        batches = []
        
        current_start = zone.search_start
        while current_start < zone.search_end:
            current_end = min(current_start + batch_size - 1, zone.search_end)
            batches.append((current_start, current_end))
            current_start = current_end + 1
        
        return batches
    
    async def _process_batches_cpu(
        self, 
        batches: List[Tuple[int, int]], 
        puzzle_target: PuzzleTarget
    ) -> List[BatchResult]:
        """Process batches using CPU via Rust interface"""
        
        logger.info("Processing batches on CPU via Rust core")
        results = []
        
        for i, (start_key, end_key) in enumerate(batches):
            batch_size = end_key - start_key + 1
            
            # Process batch via Rust
            result = await self.rust_interface.search_key_range(
                start_key=start_key,
                end_key=end_key,
                target_address=puzzle_target.bitcoin_address
            )
            
            batch_result = BatchResult(
                device_id=-1,  # CPU
                batch_id=i,
                start_key=start_key,
                end_key=end_key,
                keys_processed=batch_size,
                found_solution=result['found'],
                solution_key=result.get('solution_key'),
                processing_time=result.get('processing_time', 0.0),
                throughput=result.get('throughput', 0.0)
            )
            
            results.append(batch_result)
            
            if batch_result.found_solution:
                break  # Stop on first solution
        
        return results
    
    async def benchmark_system(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Benchmark system performance"""
        logger.info(f"Starting {duration_seconds}s system benchmark...")
        
        benchmark_results = {}
        
        # Benchmark GPU performance
        if self.cuda_engine:
            gpu_metrics = await self.cuda_engine.benchmark_performance(duration_seconds)
            benchmark_results['gpu'] = gpu_metrics
        
        # Benchmark Rust core performance
        if self.rust_interface:
            rust_metrics = await self.rust_interface.benchmark_performance(duration_seconds)
            benchmark_results['rust_core'] = rust_metrics
        
        # Benchmark ML prediction
        if self.hot_zone_predictor:
            ml_start = time.time()
            for puzzle_num in range(71, 81):  # Test puzzles 71-80
                await self.hot_zone_predictor.predict_hot_zones(puzzle_num)
            ml_time = time.time() - ml_start
            benchmark_results['ml_prediction'] = {
                'predictions_per_second': 10 / ml_time,
                'average_prediction_time': ml_time / 10
            }
        
        logger.info("Benchmark completed")
        return benchmark_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'initialized': self.is_initialized,
            'current_session': self.current_session.session_id if self.current_session else None,
            'total_sessions': self.total_sessions,
            'total_solutions_found': self.total_solutions_found,
            'total_keys_processed': self.total_keys_processed,
            'components': {
                'cuda_engine': self.cuda_engine is not None,
                'ml_predictor': self.hot_zone_predictor is not None,
                'performance_monitor': self.performance_monitor is not None,
                'rust_interface': self.rust_interface is not None
            }
        }
        
        # Add component-specific status
        if self.cuda_engine:
            status['gpu_info'] = self.cuda_engine.get_device_info()
            status['gpu_performance'] = self.cuda_engine.get_performance_stats()
        
        if self.hot_zone_predictor:
            status['ml_model_info'] = self.hot_zone_predictor.get_model_info()
        
        return status
    
    async def cleanup(self) -> None:
        """Cleanup system resources"""
        logger.info("Cleaning up system resources...")
        
        # Cancel any running tasks
        for task in self.solving_tasks:
            if not task.done():
                task.cancel()
        
        # Cleanup components
        if self.cuda_engine:
            await self.cuda_engine.shutdown()
        
        if self.performance_monitor:
            await self.performance_monitor.shutdown()
        
        if self.rust_interface:
            await self.rust_interface.shutdown()
        
        self.is_initialized = False
        logger.info("System cleanup completed")


# Example usage
async def main():
    """Example usage of the SystemCoordinator"""
    from .config import SystemConfig
    
    # Create configuration
    config = SystemConfig()
    
    # Initialize system
    coordinator = SystemCoordinator(config)
    await coordinator.initialize()
    
    try:
        # Solve puzzle #71
        result = await coordinator.solve_puzzle(
            puzzle_number=71,
            bitcoin_address="1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",
            max_time_seconds=3600  # 1 hour limit
        )
        
        print(f"Result: {result}")
        
    finally:
        await coordinator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())


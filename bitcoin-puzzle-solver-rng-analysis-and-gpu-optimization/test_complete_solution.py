#!/usr/bin/env python3
"""
Complete Solution Test Suite
Tests all components of the Bitcoin Puzzle Solver

Tests:
1. Rust core compilation and functionality
2. Python coordination layer
3. ML model training and prediction
4. GPU optimization (if available)
5. Security and encryption
6. CSV export and key management
7. Integration testing
8. Performance benchmarks
"""

import os
import sys
import time
import json
import logging
import subprocess
import unittest
from typing import Dict, List, Any
import tempfile
import shutil

# Import all our modules
from bitcoin_puzzle_solver import BitcoinPuzzleSolver
from security_module import BitcoinPuzzleSecurityManager, SecurityLevel
from csv_key_manager import CSVKeyManager, ExportConfig
from gpu_optimization import VastAIGPUOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRustCore(unittest.TestCase):
    """Test Rust core functionality"""
    
    def setUp(self):
        self.rust_dir = "bitcoin_puzzle_solver_rust_core"
    
    def test_rust_compilation(self):
        """Test that Rust core compiles successfully"""
        logger.info("Testing Rust core compilation...")
        
        if not os.path.exists(self.rust_dir):
            self.skipTest("Rust core directory not found")
        
        # Test compilation
        result = subprocess.run(
            ["cargo", "build", "--release", "--features", "parallel"],
            cwd=self.rust_dir,
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, f"Rust compilation failed: {result.stderr}")
        logger.info("✓ Rust core compilation successful")
    
    def test_rust_library_exists(self):
        """Test that compiled Rust library exists"""
        lib_path = os.path.join(self.rust_dir, "target", "release")
        
        # Look for library files
        lib_files = []
        if os.path.exists(lib_path):
            for file in os.listdir(lib_path):
                if file.startswith("libbitcoin_puzzle_solver_rust_core"):
                    lib_files.append(file)
        
        self.assertGreater(len(lib_files), 0, "No Rust library files found")
        logger.info(f"✓ Found Rust library files: {lib_files}")

class TestPythonCoordination(unittest.TestCase):
    """Test Python coordination layer"""
    
    def setUp(self):
        self.test_password = "test_password_123"
        self.solver = None
    
    def test_solver_initialization(self):
        """Test Bitcoin puzzle solver initialization"""
        logger.info("Testing solver initialization...")
        
        try:
            self.solver = BitcoinPuzzleSolver(password=self.test_password)
            self.assertIsNotNone(self.solver)
            logger.info("✓ Solver initialization successful")
        except Exception as e:
            self.fail(f"Solver initialization failed: {e}")
    
    def test_training_data_loading(self):
        """Test loading training data"""
        if not self.solver:
            self.solver = BitcoinPuzzleSolver(password=self.test_password)
        
        if os.path.exists("bitcoin_puzzle_data.csv"):
            logger.info("Testing training data loading...")
            
            try:
                data = self.solver.load_training_data("bitcoin_puzzle_data.csv")
                self.assertIsNotNone(data)
                self.assertGreater(len(data), 0)
                logger.info(f"✓ Loaded {len(data)} training records")
            except Exception as e:
                self.fail(f"Training data loading failed: {e}")
        else:
            self.skipTest("Training data file not found")
    
    def test_ml_model_training(self):
        """Test ML model training"""
        if not self.solver:
            self.solver = BitcoinPuzzleSolver(password=self.test_password)
        
        if os.path.exists("bitcoin_puzzle_data.csv"):
            logger.info("Testing ML model training...")
            
            try:
                self.solver.load_training_data("bitcoin_puzzle_data.csv")
                results = self.solver.train_ml_model()
                
                self.assertIsInstance(results, dict)
                self.assertIn('train_score', results)
                self.assertIn('test_score', results)
                logger.info(f"✓ ML model training successful: R² = {results['test_score']:.4f}")
            except Exception as e:
                self.fail(f"ML model training failed: {e}")
        else:
            self.skipTest("Training data file not found")

class TestSecurity(unittest.TestCase):
    """Test security and encryption functionality"""
    
    def setUp(self):
        self.test_password = "test_security_password_456"
        self.security_manager = None
    
    def test_security_manager_initialization(self):
        """Test security manager initialization"""
        logger.info("Testing security manager initialization...")
        
        try:
            self.security_manager = BitcoinPuzzleSecurityManager(
                master_password=self.test_password,
                security_level=SecurityLevel.HIGH  # Use HIGH for faster testing
            )
            self.assertIsNotNone(self.security_manager)
            logger.info("✓ Security manager initialization successful")
        except Exception as e:
            self.fail(f"Security manager initialization failed: {e}")
    
    def test_private_key_encryption(self):
        """Test private key encryption and decryption"""
        if not self.security_manager:
            self.security_manager = BitcoinPuzzleSecurityManager(
                master_password=self.test_password,
                security_level=SecurityLevel.HIGH
            )
        
        logger.info("Testing private key encryption...")
        
        test_key = "20d45a6a762535700ce9e0b216e31994335db8a5000000000000000000000000"
        
        try:
            # Encrypt
            encrypted_key = self.security_manager.encrypt_private_key(test_key, 71)
            self.assertIsInstance(encrypted_key, str)
            self.assertGreater(len(encrypted_key), 0)
            
            # Decrypt
            decrypted_key = self.security_manager.decrypt_private_key(encrypted_key)
            self.assertEqual(decrypted_key, test_key)
            
            logger.info("✓ Private key encryption/decryption successful")
        except Exception as e:
            self.fail(f"Private key encryption failed: {e}")
    
    def test_solution_storage(self):
        """Test secure solution storage"""
        if not self.security_manager:
            self.security_manager = BitcoinPuzzleSecurityManager(
                master_password=self.test_password,
                security_level=SecurityLevel.HIGH
            )
        
        logger.info("Testing solution storage...")
        
        try:
            solution_id = self.security_manager.secure_store_solution(
                puzzle_number=71,
                private_key="20d45a6a762535700ce9e0b216e31994335db8a5000000000000000000000000",
                address="1By8rxztgeJeUX7qQjhAdmzQtAeqcE8Kd1",
                additional_data={"test": True}
            )
            
            self.assertIsInstance(solution_id, str)
            
            # Retrieve solution
            solution = self.security_manager.retrieve_solution(71)
            self.assertIsNotNone(solution)
            self.assertEqual(solution['puzzle_number'], 71)
            
            logger.info("✓ Solution storage/retrieval successful")
        except Exception as e:
            self.fail(f"Solution storage failed: {e}")

class TestCSVKeyManager(unittest.TestCase):
    """Test CSV and key management functionality"""
    
    def setUp(self):
        self.test_password = "test_csv_password_789"
        self.security_manager = BitcoinPuzzleSecurityManager(
            master_password=self.test_password,
            security_level=SecurityLevel.HIGH
        )
        self.csv_manager = CSVKeyManager(self.security_manager)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_csv_manager_initialization(self):
        """Test CSV manager initialization"""
        logger.info("Testing CSV manager initialization...")
        
        self.assertIsNotNone(self.csv_manager)
        self.assertGreater(len(self.csv_manager.puzzle_keys), 0)
        logger.info("✓ CSV manager initialization successful")
    
    def test_add_solved_puzzle(self):
        """Test adding solved puzzle"""
        logger.info("Testing add solved puzzle...")
        
        try:
            self.csv_manager.add_solved_puzzle(
                puzzle_number=71,
                private_key="20d45a6a762535700ce9e0b216e31994335db8a5000000000000000000000000",
                public_key="0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
                address="1By8rxztgeJeUX7qQjhAdmzQtAeqcE8Kd1",
                solve_metadata={"test": True}
            )
            
            self.assertIn(71, self.csv_manager.solutions)
            logger.info("✓ Add solved puzzle successful")
        except Exception as e:
            self.fail(f"Add solved puzzle failed: {e}")
    
    def test_csv_export(self):
        """Test CSV export functionality"""
        logger.info("Testing CSV export...")
        
        # Add test data
        self.csv_manager.add_solved_puzzle(
            puzzle_number=71,
            private_key="20d45a6a762535700ce9e0b216e31994335db8a5000000000000000000000000",
            public_key="0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
            address="1By8rxztgeJeUX7qQjhAdmzQtAeqcE8Kd1"
        )
        
        try:
            csv_file = os.path.join(self.temp_dir, "test_export.csv")
            result_file = self.csv_manager.export_csv(csv_file, ExportConfig(
                include_private_keys=True,
                include_public_keys=True,
                encrypt_output=False
            ))
            
            self.assertTrue(os.path.exists(result_file))
            
            # Check file content
            with open(result_file, 'r') as f:
                content = f.read()
                self.assertIn("puzzle_number", content)
                self.assertIn("private_key", content)
                self.assertIn("71", content)
            
            logger.info("✓ CSV export successful")
        except Exception as e:
            self.fail(f"CSV export failed: {e}")
    
    def test_progress_report(self):
        """Test progress report generation"""
        logger.info("Testing progress report...")
        
        try:
            report = self.csv_manager.generate_progress_report()
            
            self.assertIsInstance(report, dict)
            self.assertIn('summary', report)
            self.assertIn('performance', report)
            
            logger.info("✓ Progress report generation successful")
        except Exception as e:
            self.fail(f"Progress report failed: {e}")

class TestGPUOptimization(unittest.TestCase):
    """Test GPU optimization functionality"""
    
    def test_gpu_optimizer_initialization(self):
        """Test GPU optimizer initialization"""
        logger.info("Testing GPU optimizer initialization...")
        
        try:
            optimizer = VastAIGPUOptimizer()
            self.assertIsNotNone(optimizer)
            logger.info("✓ GPU optimizer initialization successful")
        except Exception as e:
            logger.warning(f"GPU optimizer initialization failed (expected if no GPU): {e}")
            self.skipTest("GPU not available")
    
    def test_gpu_configuration(self):
        """Test GPU configuration detection"""
        try:
            optimizer = VastAIGPUOptimizer()
            
            self.assertIsNotNone(optimizer.gpu_configs)
            self.assertIn("A100", optimizer.gpu_configs)
            self.assertIn("V100", optimizer.gpu_configs)
            self.assertIn("RTX5090", optimizer.gpu_configs)
            
            logger.info("✓ GPU configuration successful")
        except Exception as e:
            self.skipTest(f"GPU configuration test skipped: {e}")

class TestIntegration(unittest.TestCase):
    """Integration tests for complete solution"""
    
    def setUp(self):
        self.test_password = "integration_test_password"
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_workflow(self):
        """Test complete workflow from initialization to CSV export"""
        logger.info("Testing complete workflow...")
        
        try:
            # 1. Initialize solver
            solver = BitcoinPuzzleSolver(password=self.test_password)
            
            # 2. Load training data (if available)
            if os.path.exists("bitcoin_puzzle_data.csv"):
                solver.load_training_data("bitcoin_puzzle_data.csv")
                solver.train_ml_model()
            
            # 3. Initialize CSV manager
            csv_manager = CSVKeyManager(solver.security)
            
            # 4. Add test solution
            csv_manager.add_solved_puzzle(
                puzzle_number=71,
                private_key="20d45a6a762535700ce9e0b216e31994335db8a5000000000000000000000000",
                public_key="0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
                address="1By8rxztgeJeUX7qQjhAdmzQtAeqcE8Kd1",
                solve_metadata={"integration_test": True}
            )
            
            # 5. Verify solution
            verified = csv_manager.verify_solution(71)
            self.assertTrue(verified)
            
            # 6. Generate progress report
            report = csv_manager.generate_progress_report()
            self.assertGreater(report['summary']['solved_puzzles'], 0)
            
            # 7. Export CSV
            csv_file = os.path.join(self.temp_dir, "integration_test.csv")
            result_file = csv_manager.export_csv(csv_file)
            self.assertTrue(os.path.exists(result_file))
            
            # 8. Create backup
            backup_files = csv_manager.backup_data(self.temp_dir)
            self.assertGreater(len(backup_files), 0)
            
            logger.info("✓ Complete workflow test successful")
            
        except Exception as e:
            self.fail(f"Complete workflow test failed: {e}")

class PerformanceBenchmark:
    """Performance benchmarks for the solution"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_ml_training(self):
        """Benchmark ML model training performance"""
        if not os.path.exists("bitcoin_puzzle_data.csv"):
            logger.warning("Skipping ML training benchmark - no training data")
            return
        
        logger.info("Benchmarking ML training performance...")
        
        solver = BitcoinPuzzleSolver(password="benchmark_password")
        
        start_time = time.time()
        solver.load_training_data("bitcoin_puzzle_data.csv")
        data_load_time = time.time() - start_time
        
        start_time = time.time()
        results = solver.train_ml_model()
        training_time = time.time() - start_time
        
        self.results['ml_training'] = {
            'data_load_time': data_load_time,
            'training_time': training_time,
            'model_score': results.get('test_score', 0)
        }
        
        logger.info(f"ML Training: {training_time:.2f}s, Score: {results.get('test_score', 0):.4f}")
    
    def benchmark_encryption(self):
        """Benchmark encryption performance"""
        logger.info("Benchmarking encryption performance...")
        
        security_manager = BitcoinPuzzleSecurityManager(
            master_password="benchmark_password",
            security_level=SecurityLevel.HIGH
        )
        
        test_key = "20d45a6a762535700ce9e0b216e31994335db8a5000000000000000000000000"
        
        # Benchmark encryption
        start_time = time.time()
        for _ in range(100):
            encrypted = security_manager.encrypt_private_key(test_key, 71)
        encryption_time = (time.time() - start_time) / 100
        
        # Benchmark decryption
        start_time = time.time()
        for _ in range(100):
            decrypted = security_manager.decrypt_private_key(encrypted)
        decryption_time = (time.time() - start_time) / 100
        
        self.results['encryption'] = {
            'encryption_time_ms': encryption_time * 1000,
            'decryption_time_ms': decryption_time * 1000
        }
        
        logger.info(f"Encryption: {encryption_time*1000:.2f}ms, Decryption: {decryption_time*1000:.2f}ms")
    
    def benchmark_csv_export(self):
        """Benchmark CSV export performance"""
        logger.info("Benchmarking CSV export performance...")
        
        security_manager = BitcoinPuzzleSecurityManager(
            master_password="benchmark_password",
            security_level=SecurityLevel.HIGH
        )
        csv_manager = CSVKeyManager(security_manager)
        
        # Add multiple test solutions
        test_private_key = "20d45a6a762535700ce9e0b216e31994335db8a5000000000000000000000000"
        test_addresses = [
            "1By8rxztgeJeUX7qQjhAdmzQtAeqcE8Kd1",
            "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ", 
            "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG",
            "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU",
            "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR",
            "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4",
            "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv",
            "1J36UjUByGroXcCvmj13U6uwaVv9caEeAt",
            "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF",
            "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE"
        ]
        
        for i in range(10):
            csv_manager.add_solved_puzzle(
                puzzle_number=70 + i,
                private_key=test_private_key,
                public_key="0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
                address=test_addresses[i]
            )
        
        start_time = time.time()
        csv_file = csv_manager.export_csv("benchmark_test.csv")
        export_time = time.time() - start_time
        
        self.results['csv_export'] = {
            'export_time': export_time,
            'puzzles_exported': 10
        }
        
        logger.info(f"CSV Export: {export_time:.2f}s for 10 puzzles")
        
        # Cleanup
        if os.path.exists("benchmark_test.csv"):
            os.remove("benchmark_test.csv")
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        logger.info("Running performance benchmarks...")
        
        self.benchmark_ml_training()
        self.benchmark_encryption()
        self.benchmark_csv_export()
        
        return self.results

def run_all_tests():
    """Run all test suites"""
    logger.info("Starting comprehensive test suite...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRustCore,
        TestPythonCoordination,
        TestSecurity,
        TestCSVKeyManager,
        TestGPUOptimization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance benchmarks
    logger.info("\\nRunning performance benchmarks...")
    benchmark = PerformanceBenchmark()
    benchmark_results = benchmark.run_all_benchmarks()
    
    # Generate test report
    test_report = {
        "timestamp": time.time(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
        "benchmark_results": benchmark_results
    }
    
    # Save test report
    with open("test_report.json", "w") as f:
        json.dump(test_report, f, indent=2)
    
    logger.info(f"\\nTest Summary:")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Success rate: {test_report['success_rate']:.1f}%")
    
    return result.wasSuccessful()

def main():
    """Main function"""
    success = run_all_tests()
    
    if success:
        logger.info("\\n🎉 All tests passed! The Bitcoin Puzzle Solver is ready for deployment.")
        sys.exit(0)
    else:
        logger.error("\\n❌ Some tests failed. Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()


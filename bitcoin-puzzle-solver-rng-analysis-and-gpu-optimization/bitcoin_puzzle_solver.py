#!/usr/bin/env python3
"""
Bitcoin Puzzle Solver - Python Coordination Layer
Hybrid Rust/Python implementation for solving Bitcoin puzzles
"""

import json
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import subprocess
import threading
import queue
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bitcoin_puzzle_solver.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PuzzleTarget:
    """Target puzzle information"""
    puzzle_number: int
    address: str
    reward_btc: float
    status: str = "unsolved"
    private_key: Optional[str] = None

@dataclass
class SearchConfig:
    """Search configuration parameters"""
    puzzle_number: int
    max_iterations: int = 1000000
    batch_size: int = 10000
    parallel_workers: int = 4
    search_strategy: str = "hot_zone_targeted"
    hot_zones: List[Dict] = None
    ml_model_path: Optional[str] = None

@dataclass
class SearchResult:
    """Search operation result"""
    found: bool
    private_key: Optional[str]
    address: Optional[str]
    search_time_ms: int
    keys_tested: int
    hot_zone_hit: Optional[int]

class MLPredictor:
    """Machine Learning predictor for hot zone identification"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, puzzle_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        features = []
        
        for _, row in puzzle_data.iterrows():
            puzzle_num = row['puzzle_num']
            private_key_int = row['private_key_int']
            alpha = row['alpha']
            
            # Basic features
            feature_vector = [
                puzzle_num,
                alpha,
                np.log2(puzzle_num),  # Log scale
                alpha ** 2,  # Quadratic term
                alpha ** 3,  # Cubic term
            ]
            
            # Bit pattern features
            key_bin = bin(private_key_int)[2:].zfill(puzzle_num)
            bit_density = key_bin.count('1') / len(key_bin)
            leading_zeros = len(key_bin) - len(key_bin.lstrip('0'))
            trailing_zeros = len(key_bin) - len(key_bin.rstrip('0'))
            
            feature_vector.extend([
                bit_density,
                leading_zeros / puzzle_num,
                trailing_zeros / puzzle_num,
            ])
            
            # Position-based features
            range_start = 2**(puzzle_num - 1) if puzzle_num > 1 else 1
            range_end = 2**puzzle_num - 1
            relative_position = (private_key_int - range_start) / (range_end - range_start) if range_end > range_start else 0
            
            feature_vector.extend([
                relative_position,
                np.sin(2 * np.pi * relative_position),  # Periodic features
                np.cos(2 * np.pi * relative_position),
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, puzzle_data: pd.DataFrame) -> Dict:
        """Train the ML model on solved puzzle data"""
        logger.info("Training ML model on solved puzzle data...")
        
        # Prepare features and targets
        X = self.prepare_features(puzzle_data)
        y = puzzle_data['alpha'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        logger.info(f"ML model trained - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': self.model.feature_importances_.tolist()
        }
    
    def predict_alpha(self, puzzle_number: int) -> Tuple[float, float]:
        """Predict alpha value for a given puzzle number"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Create dummy data point for prediction
        dummy_data = pd.DataFrame({
            'puzzle_num': [puzzle_number],
            'private_key_int': [2**(puzzle_number-1)],  # Dummy value
            'alpha': [0.5]  # Dummy value
        })
        
        X = self.prepare_features(dummy_data)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        # Estimate uncertainty using ensemble variance
        predictions = []
        for estimator in self.model.estimators_:
            pred = estimator.predict(X_scaled)[0]
            predictions.append(pred)
        
        uncertainty = np.std(predictions)
        
        return prediction, uncertainty

class SecurityManager:
    """Military-grade security and encryption manager"""
    
    def __init__(self, password: str):
        self.password = password.encode()
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Setup encryption keys"""
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'bitcoin_puzzle_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        self.fernet = Fernet(key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def secure_hash(self, data: str) -> str:
        """Create secure hash of data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_integrity(self, data: str, expected_hash: str) -> bool:
        """Verify data integrity"""
        return self.secure_hash(data) == expected_hash

class BitcoinPuzzleSolver:
    """Main Bitcoin Puzzle Solver class"""
    
    def __init__(self, password: str = "default_password"):
        self.security = SecurityManager(password)
        self.ml_predictor = MLPredictor()
        self.puzzle_data = None
        self.targets = {}
        self.results = []
        
        # Load known puzzle targets
        self._load_puzzle_targets()
        
        logger.info("Bitcoin Puzzle Solver initialized")
    
    def _load_puzzle_targets(self):
        """Load known puzzle targets and addresses"""
        # Known unsolved puzzles with their addresses
        known_targets = {
            71: PuzzleTarget(71, "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", 7.1),
            72: PuzzleTarget(72, "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR", 7.2),
            73: PuzzleTarget(73, "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4", 7.3),
            74: PuzzleTarget(74, "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv", 7.4),
            76: PuzzleTarget(76, "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF", 7.6),
            77: PuzzleTarget(77, "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE", 7.7),
            78: PuzzleTarget(78, "15qF6X51huDjqTmF9BJgxXdt1xcj46Jmhb", 7.8),
            79: PuzzleTarget(79, "1ARk8HWJMn8js8tQmGUJeQHjSE7KRkn2t8", 7.9),
        }
        
        for puzzle_num, target in known_targets.items():
            self.targets[puzzle_num] = target
        
        logger.info(f"Loaded {len(known_targets)} puzzle targets")
    
    def load_training_data(self, csv_path: str):
        """Load solved puzzle data for training"""
        logger.info(f"Loading training data from {csv_path}")
        
        df = pd.read_csv(csv_path, names=['puzzle_num', 'address', 'private_key_hex'])
        df['private_key_int'] = df['private_key_hex'].apply(lambda x: int(x, 16))
        
        # Calculate alpha values
        results = []
        for _, row in df.iterrows():
            puzzle_num = row['puzzle_num']
            private_key = row['private_key_int']
            
            if puzzle_num == 1:
                alpha = 0.0
            else:
                range_start = 2**(puzzle_num - 1)
                range_end = 2**puzzle_num - 1
                alpha = (private_key - range_start) / (range_end - range_start)
            
            results.append({
                'puzzle_num': puzzle_num,
                'address': row['address'],
                'private_key_hex': row['private_key_hex'],
                'private_key_int': private_key,
                'alpha': alpha
            })
        
        self.puzzle_data = pd.DataFrame(results)
        logger.info(f"Loaded {len(self.puzzle_data)} solved puzzles")
        
        return self.puzzle_data
    
    def train_ml_model(self) -> Dict:
        """Train the machine learning model"""
        if self.puzzle_data is None:
            raise ValueError("Training data must be loaded first")
        
        return self.ml_predictor.train(self.puzzle_data)
    
    def generate_hot_zones(self, puzzle_number: int) -> List[Dict]:
        """Generate hot zones for a puzzle using ML predictions"""
        if self.ml_predictor.is_trained:
            # Use ML prediction
            predicted_alpha, uncertainty = self.ml_predictor.predict_alpha(puzzle_number)
            
            # Create hot zones around prediction
            hot_zones = [
                {
                    "start_percent": max(0, (predicted_alpha - uncertainty) * 100),
                    "end_percent": min(100, (predicted_alpha + uncertainty) * 100),
                    "probability": 0.4,
                    "priority": 1
                }
            ]
            
            # Add default hot zones from analysis
            default_zones = [
                {"start_percent": 30.0, "end_percent": 40.0, "probability": 0.12, "priority": 2},
                {"start_percent": 40.0, "end_percent": 50.0, "probability": 0.12, "priority": 2},
                {"start_percent": 60.0, "end_percent": 70.0, "probability": 0.173, "priority": 1},
                {"start_percent": 90.0, "end_percent": 100.0, "probability": 0.12, "priority": 2},
            ]
            
            hot_zones.extend(default_zones)
        else:
            # Use default hot zones from analysis
            hot_zones = [
                {"start_percent": 30.0, "end_percent": 40.0, "probability": 0.12, "priority": 2},
                {"start_percent": 40.0, "end_percent": 50.0, "probability": 0.12, "priority": 2},
                {"start_percent": 60.0, "end_percent": 70.0, "probability": 0.173, "priority": 1},
                {"start_percent": 90.0, "end_percent": 100.0, "probability": 0.12, "priority": 2},
            ]
        
        return hot_zones
    
    def create_search_config(self, puzzle_number: int, **kwargs) -> SearchConfig:
        """Create search configuration for a puzzle"""
        hot_zones = self.generate_hot_zones(puzzle_number)
        
        config = SearchConfig(
            puzzle_number=puzzle_number,
            hot_zones=hot_zones,
            **kwargs
        )
        
        return config
    
    def run_rust_search(self, config: SearchConfig) -> SearchResult:
        """Run search using Rust core (simulated for now)"""
        logger.info(f"Starting search for puzzle {config.puzzle_number}")
        
        start_time = time.time()
        
        # For now, simulate the Rust search
        # In a real implementation, this would call the Rust library
        time.sleep(1)  # Simulate search time
        
        # Simulate result
        result = SearchResult(
            found=False,
            private_key=None,
            address=None,
            search_time_ms=int((time.time() - start_time) * 1000),
            keys_tested=config.batch_size,
            hot_zone_hit=None
        )
        
        logger.info(f"Search completed - Keys tested: {result.keys_tested}, Time: {result.search_time_ms}ms")
        
        return result
    
    def solve_puzzle(self, puzzle_number: int, max_time_hours: int = 24) -> Optional[SearchResult]:
        """Attempt to solve a specific puzzle"""
        if puzzle_number not in self.targets:
            logger.error(f"Puzzle {puzzle_number} not in target list")
            return None
        
        target = self.targets[puzzle_number]
        logger.info(f"Starting solve attempt for puzzle {puzzle_number} - Target: {target.address}")
        
        config = self.create_search_config(
            puzzle_number,
            max_iterations=1000000,
            batch_size=10000
        )
        
        start_time = time.time()
        max_time_seconds = max_time_hours * 3600
        
        while time.time() - start_time < max_time_seconds:
            result = self.run_rust_search(config)
            
            if result.found:
                logger.info(f"PUZZLE {puzzle_number} SOLVED!")
                logger.info(f"Private Key: {result.private_key}")
                
                # Encrypt and store result
                encrypted_key = self.security.encrypt_data(result.private_key)
                
                # Save to secure storage
                self._save_solution(puzzle_number, result, encrypted_key)
                
                return result
            
            # Progress update
            elapsed_hours = (time.time() - start_time) / 3600
            logger.info(f"Puzzle {puzzle_number} - {elapsed_hours:.2f}h elapsed, {result.keys_tested} keys tested")
        
        logger.info(f"Search timeout reached for puzzle {puzzle_number}")
        return None
    
    def _save_solution(self, puzzle_number: int, result: SearchResult, encrypted_key: str):
        """Save solution to secure storage"""
        solution_data = {
            'puzzle_number': puzzle_number,
            'address': result.address,
            'encrypted_private_key': encrypted_key,
            'timestamp': time.time(),
            'search_time_ms': result.search_time_ms,
            'keys_tested': result.keys_tested,
            'hot_zone_hit': result.hot_zone_hit
        }
        
        # Save to JSON file
        filename = f"solution_puzzle_{puzzle_number}.json"
        with open(filename, 'w') as f:
            json.dump(solution_data, f, indent=2)
        
        logger.info(f"Solution saved to {filename}")
    
    def export_results_csv(self, filename: str = "puzzle_solutions.csv"):
        """Export all results to CSV format"""
        if not self.results:
            logger.warning("No results to export")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        logger.info(f"Results exported to {filename}")
    
    def run_multi_puzzle_search(self, puzzle_numbers: List[int], max_time_hours: int = 24):
        """Run search on multiple puzzles simultaneously"""
        logger.info(f"Starting multi-puzzle search for puzzles: {puzzle_numbers}")
        
        threads = []
        results_queue = queue.Queue()
        
        def worker(puzzle_num):
            result = self.solve_puzzle(puzzle_num, max_time_hours)
            results_queue.put((puzzle_num, result))
        
        # Start threads for each puzzle
        for puzzle_num in puzzle_numbers:
            thread = threading.Thread(target=worker, args=(puzzle_num,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        while not results_queue.empty():
            puzzle_num, result = results_queue.get()
            if result and result.found:
                self.results.append(result)
                logger.info(f"Puzzle {puzzle_num} solved successfully!")
        
        logger.info(f"Multi-puzzle search completed. {len(self.results)} puzzles solved.")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bitcoin Puzzle Solver")
    parser.add_argument("--puzzle", type=int, help="Puzzle number to solve")
    parser.add_argument("--puzzles", nargs="+", type=int, help="Multiple puzzle numbers to solve")
    parser.add_argument("--training-data", default="bitcoin_puzzle_data.csv", help="Path to training data CSV")
    parser.add_argument("--max-time", type=int, default=24, help="Maximum time in hours")
    parser.add_argument("--password", default="secure_password_123", help="Encryption password")
    
    args = parser.parse_args()
    
    # Initialize solver
    solver = BitcoinPuzzleSolver(password=args.password)
    
    # Load training data and train ML model
    if os.path.exists(args.training_data):
        solver.load_training_data(args.training_data)
        training_results = solver.train_ml_model()
        logger.info(f"ML model training results: {training_results}")
    else:
        logger.warning(f"Training data file {args.training_data} not found")
    
    # Run search
    if args.puzzle:
        result = solver.solve_puzzle(args.puzzle, args.max_time)
        if result and result.found:
            print(f"SUCCESS: Puzzle {args.puzzle} solved!")
        else:
            print(f"Puzzle {args.puzzle} not solved within time limit")
    
    elif args.puzzles:
        solver.run_multi_puzzle_search(args.puzzles, args.max_time)
        solver.export_results_csv()
    
    else:
        # Default: try to solve puzzles 71-75
        default_puzzles = [71, 72, 73, 74, 75]
        solver.run_multi_puzzle_search(default_puzzles, args.max_time)
        solver.export_results_csv()

if __name__ == "__main__":
    main()


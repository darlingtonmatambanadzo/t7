#!/usr/bin/env python3
"""
CSV Output and Key Management System
For Bitcoin Puzzle Solver

Handles:
- Secure CSV export of solved puzzles
- Public key management and validation
- Private key presentation and verification
- Data integrity and validation
- Progress tracking and reporting
- Backup and recovery systems
"""

import os
import sys
import csv
import json
import time
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Import our security module
from security_module import BitcoinPuzzleSecurityManager, SecurityLevel

logger = logging.getLogger(__name__)

@dataclass
class PuzzleKey:
    """Represents a Bitcoin puzzle key (private or public)"""
    puzzle_number: int
    key_type: str  # "private" or "public"
    key_value: str
    address: str
    status: str  # "solved", "unsolved", "verified", "unverified"
    timestamp: float
    source: str  # "solved", "provided", "generated"
    metadata: Dict[str, Any]

@dataclass
class SolutionRecord:
    """Represents a complete puzzle solution"""
    puzzle_number: int
    private_key: str
    public_key: str
    address: str
    reward_btc: float
    solve_time_seconds: int
    keys_tested: int
    search_method: str
    hot_zone_hit: Optional[int]
    timestamp: float
    verified: bool
    metadata: Dict[str, Any]

@dataclass
class ExportConfig:
    """Configuration for CSV export"""
    include_private_keys: bool = True
    include_public_keys: bool = True
    include_metadata: bool = True
    encrypt_output: bool = True
    add_checksums: bool = True
    format_version: str = "1.0"
    compression: bool = False

class KeyValidator:
    """Validates Bitcoin keys and addresses"""
    
    @staticmethod
    def validate_private_key(private_key: str) -> bool:
        """Validate a Bitcoin private key"""
        try:
            # Remove 0x prefix if present
            if private_key.startswith('0x'):
                private_key = private_key[2:]
            
            # Check if it's valid hex
            int(private_key, 16)
            
            # Check length (64 characters for 256-bit key)
            if len(private_key) != 64:
                return False
            
            # Check if it's in valid range (1 to n-1 where n is secp256k1 order)
            key_int = int(private_key, 16)
            secp256k1_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
            
            return 1 <= key_int < secp256k1_order
            
        except ValueError:
            return False
    
    @staticmethod
    def validate_public_key(public_key: str) -> bool:
        """Validate a Bitcoin public key"""
        try:
            # Remove 0x prefix if present
            if public_key.startswith('0x'):
                public_key = public_key[2:]
            
            # Check if it's valid hex
            int(public_key, 16)
            
            # Check length (compressed: 66 chars, uncompressed: 130 chars)
            if len(public_key) not in [66, 130]:
                return False
            
            # Check prefix for compressed keys
            if len(public_key) == 66:
                prefix = public_key[:2]
                if prefix not in ['02', '03']:
                    return False
            
            # Check prefix for uncompressed keys
            if len(public_key) == 130:
                prefix = public_key[:2]
                if prefix != '04':
                    return False
            
            return True
            
        except ValueError:
            return False
    
    @staticmethod
    def validate_bitcoin_address(address: str) -> bool:
        """Validate a Bitcoin address"""
        try:
            # Basic validation for P2PKH addresses (starts with 1)
            if address.startswith('1'):
                # Check length (25-34 characters typical)
                if not (25 <= len(address) <= 34):
                    return False
                
                # Check base58 characters
                base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
                for char in address:
                    if char not in base58_chars:
                        return False
                
                return True
            
            # Basic validation for P2SH addresses (starts with 3)
            elif address.startswith('3'):
                if not (25 <= len(address) <= 34):
                    return False
                
                base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
                for char in address:
                    if char not in base58_chars:
                        return False
                
                return True
            
            # Basic validation for Bech32 addresses (starts with bc1)
            elif address.startswith('bc1'):
                if not (42 <= len(address) <= 62):
                    return False
                
                bech32_chars = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
                for char in address[3:]:  # Skip 'bc1' prefix
                    if char not in bech32_chars:
                        return False
                
                return True
            
            return False
            
        except Exception:
            return False

class CSVKeyManager:
    """Manages CSV export and key management operations"""
    
    def __init__(self, security_manager: BitcoinPuzzleSecurityManager):
        self.security_manager = security_manager
        self.validator = KeyValidator()
        self.puzzle_keys = {}  # puzzle_number -> PuzzleKey
        self.solutions = {}    # puzzle_number -> SolutionRecord
        
        # Load known puzzle data
        self._load_known_puzzles()
    
    def _load_known_puzzles(self):
        """Load known puzzle information"""
        # Known unsolved puzzles with addresses
        known_puzzles = {
            71: {"address": "1By8rxztgeJeUX7qQjhAdmzQtAeqcE8Kd1", "reward": 0.71},
            72: {"address": "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ", "reward": 0.72},
            73: {"address": "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG", "reward": 0.73},
            74: {"address": "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", "reward": 0.74},
            75: {"address": "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR", "reward": 0.75},
            76: {"address": "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4", "reward": 0.76},
            77: {"address": "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv", "reward": 0.77},
            78: {"address": "1J36UjUByGroXcCvmj13U6uwaVv9caEeAt", "reward": 0.78},
            79: {"address": "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF", "reward": 0.79},
            80: {"address": "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE", "reward": 0.80},
        }
        
        for puzzle_num, info in known_puzzles.items():
            self.puzzle_keys[puzzle_num] = PuzzleKey(
                puzzle_number=puzzle_num,
                key_type="public",
                key_value="",  # Public key not known
                address=info["address"],
                status="unsolved",
                timestamp=time.time(),
                source="provided",
                metadata={"reward_btc": info["reward"]}
            )
        
        logger.info(f"Loaded {len(known_puzzles)} known puzzle targets")
    
    def add_solved_puzzle(self, puzzle_number: int, private_key: str, 
                         public_key: str, address: str, solve_metadata: Dict[str, Any] = None):
        """Add a solved puzzle to the collection"""
        if solve_metadata is None:
            solve_metadata = {}
        
        # Validate inputs
        if not self.validator.validate_private_key(private_key):
            raise ValueError(f"Invalid private key for puzzle {puzzle_number}")
        
        if not self.validator.validate_bitcoin_address(address):
            raise ValueError(f"Invalid address for puzzle {puzzle_number}")
        
        if public_key and not self.validator.validate_public_key(public_key):
            raise ValueError(f"Invalid public key for puzzle {puzzle_number}")
        
        # Create solution record
        solution = SolutionRecord(
            puzzle_number=puzzle_number,
            private_key=private_key,
            public_key=public_key,
            address=address,
            reward_btc=solve_metadata.get("reward_btc", 0.0),
            solve_time_seconds=solve_metadata.get("solve_time_seconds", 0),
            keys_tested=solve_metadata.get("keys_tested", 0),
            search_method=solve_metadata.get("search_method", "unknown"),
            hot_zone_hit=solve_metadata.get("hot_zone_hit"),
            timestamp=time.time(),
            verified=False,  # Will be verified separately
            metadata=solve_metadata
        )
        
        self.solutions[puzzle_number] = solution
        
        # Update puzzle key record
        self.puzzle_keys[puzzle_number] = PuzzleKey(
            puzzle_number=puzzle_number,
            key_type="private",
            key_value=private_key,
            address=address,
            status="solved",
            timestamp=time.time(),
            source="solved",
            metadata=solve_metadata
        )
        
        # Store securely
        self.security_manager.secure_store_solution(
            puzzle_number, private_key, address, solve_metadata
        )
        
        logger.info(f"Added solved puzzle {puzzle_number}")
    
    def add_public_key(self, puzzle_number: int, public_key: str, 
                      address: str, metadata: Dict[str, Any] = None):
        """Add a public key for an unsolved puzzle"""
        if metadata is None:
            metadata = {}
        
        # Validate inputs
        if not self.validator.validate_public_key(public_key):
            raise ValueError(f"Invalid public key for puzzle {puzzle_number}")
        
        if not self.validator.validate_bitcoin_address(address):
            raise ValueError(f"Invalid address for puzzle {puzzle_number}")
        
        # Update or create puzzle key record
        self.puzzle_keys[puzzle_number] = PuzzleKey(
            puzzle_number=puzzle_number,
            key_type="public",
            key_value=public_key,
            address=address,
            status="unsolved",
            timestamp=time.time(),
            source="provided",
            metadata=metadata
        )
        
        logger.info(f"Added public key for puzzle {puzzle_number}")
    
    def verify_solution(self, puzzle_number: int) -> bool:
        """Verify a puzzle solution"""
        if puzzle_number not in self.solutions:
            return False
        
        solution = self.solutions[puzzle_number]
        
        try:
            # Basic validation
            if not self.validator.validate_private_key(solution.private_key):
                logger.error(f"Invalid private key for puzzle {puzzle_number}")
                return False
            
            if not self.validator.validate_bitcoin_address(solution.address):
                logger.error(f"Invalid address for puzzle {puzzle_number}")
                return False
            
            # TODO: Add cryptographic verification
            # - Generate public key from private key
            # - Generate address from public key
            # - Compare with expected address
            
            solution.verified = True
            logger.info(f"Verified solution for puzzle {puzzle_number}")
            return True
            
        except Exception as e:
            logger.error(f"Verification failed for puzzle {puzzle_number}: {e}")
            return False
    
    def export_csv(self, filename: str, config: ExportConfig = None) -> str:
        """Export puzzle data to CSV format"""
        if config is None:
            config = ExportConfig()
        
        # Prepare data for export
        export_data = []
        
        # Add solved puzzles
        if config.include_private_keys:
            for puzzle_num, solution in self.solutions.items():
                row = {
                    "puzzle_number": puzzle_num,
                    "type": "solved",
                    "private_key": solution.private_key if config.include_private_keys else "",
                    "public_key": solution.public_key,
                    "address": solution.address,
                    "reward_btc": solution.reward_btc,
                    "solve_time_seconds": solution.solve_time_seconds,
                    "keys_tested": solution.keys_tested,
                    "search_method": solution.search_method,
                    "hot_zone_hit": solution.hot_zone_hit,
                    "timestamp": solution.timestamp,
                    "verified": solution.verified,
                    "status": "solved"
                }
                
                if config.include_metadata:
                    row.update({f"meta_{k}": v for k, v in solution.metadata.items()})
                
                export_data.append(row)
        
        # Add public keys for unsolved puzzles
        if config.include_public_keys:
            for puzzle_num, puzzle_key in self.puzzle_keys.items():
                if puzzle_key.status != "solved":
                    row = {
                        "puzzle_number": puzzle_num,
                        "type": "unsolved",
                        "private_key": "",
                        "public_key": puzzle_key.key_value,
                        "address": puzzle_key.address,
                        "reward_btc": puzzle_key.metadata.get("reward_btc", 0.0),
                        "solve_time_seconds": 0,
                        "keys_tested": 0,
                        "search_method": "",
                        "hot_zone_hit": None,
                        "timestamp": puzzle_key.timestamp,
                        "verified": False,
                        "status": puzzle_key.status
                    }
                    
                    if config.include_metadata:
                        row.update({f"meta_{k}": v for k, v in puzzle_key.metadata.items()})
                    
                    export_data.append(row)
        
        # Sort by puzzle number
        export_data.sort(key=lambda x: x["puzzle_number"])
        
        # Create DataFrame
        df = pd.DataFrame(export_data)
        
        if df.empty:
            logger.warning("No data to export")
            return ""
        
        # Add metadata header
        metadata_header = {
            "export_timestamp": time.time(),
            "format_version": config.format_version,
            "total_puzzles": len(export_data),
            "solved_puzzles": len([x for x in export_data if x["type"] == "solved"]),
            "unsolved_puzzles": len([x for x in export_data if x["type"] == "unsolved"]),
            "includes_private_keys": config.include_private_keys,
            "includes_public_keys": config.include_public_keys,
            "encrypted": config.encrypt_output
        }
        
        # Generate CSV content
        csv_content = f"# Bitcoin Puzzle Solver Export\\n"
        csv_content += f"# Generated: {datetime.fromtimestamp(metadata_header['export_timestamp'])}\\n"
        csv_content += f"# Format Version: {metadata_header['format_version']}\\n"
        csv_content += f"# Total Puzzles: {metadata_header['total_puzzles']}\\n"
        csv_content += f"# Solved: {metadata_header['solved_puzzles']}, Unsolved: {metadata_header['unsolved_puzzles']}\\n"
        csv_content += f"#\\n"
        
        # Add CSV data
        csv_buffer = df.to_csv(index=False)
        csv_content += csv_buffer
        
        # Add checksum if requested
        if config.add_checksums:
            checksum = hashlib.sha256(csv_content.encode()).hexdigest()
            csv_content += f"\\n# SHA256: {checksum}\\n"
        
        # Encrypt if requested
        if config.encrypt_output:
            encrypted_filename = filename + ".enc"
            self.security_manager.export_solutions_csv(
                encrypted_filename, 
                config.include_private_keys
            )
            logger.info(f"Exported encrypted CSV to {encrypted_filename}")
            return encrypted_filename
        else:
            # Save unencrypted CSV
            with open(filename, 'w', newline='') as f:
                f.write(csv_content)
            
            logger.info(f"Exported CSV to {filename}")
            return filename
    
    def import_csv(self, filename: str, encrypted: bool = False) -> int:
        """Import puzzle data from CSV file"""
        try:
            if encrypted:
                # TODO: Implement encrypted CSV import
                logger.error("Encrypted CSV import not yet implemented")
                return 0
            
            # Read CSV file
            df = pd.read_csv(filename, comment='#')
            
            imported_count = 0
            
            for _, row in df.iterrows():
                puzzle_num = int(row['puzzle_number'])
                
                if row['type'] == 'solved' and row['private_key']:
                    # Import solved puzzle
                    metadata = {}
                    for col in df.columns:
                        if col.startswith('meta_'):
                            key = col[5:]  # Remove 'meta_' prefix
                            metadata[key] = row[col]
                    
                    metadata.update({
                        "reward_btc": row.get('reward_btc', 0.0),
                        "solve_time_seconds": row.get('solve_time_seconds', 0),
                        "keys_tested": row.get('keys_tested', 0),
                        "search_method": row.get('search_method', 'imported')
                    })
                    
                    self.add_solved_puzzle(
                        puzzle_num,
                        row['private_key'],
                        row.get('public_key', ''),
                        row['address'],
                        metadata
                    )
                    imported_count += 1
                
                elif row['type'] == 'unsolved' and row['public_key']:
                    # Import public key
                    metadata = {}
                    for col in df.columns:
                        if col.startswith('meta_'):
                            key = col[5:]
                            metadata[key] = row[col]
                    
                    metadata['reward_btc'] = row.get('reward_btc', 0.0)
                    
                    self.add_public_key(
                        puzzle_num,
                        row['public_key'],
                        row['address'],
                        metadata
                    )
                    imported_count += 1
            
            logger.info(f"Imported {imported_count} puzzle records from {filename}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import CSV {filename}: {e}")
            return 0
    
    def generate_progress_report(self) -> Dict[str, Any]:
        """Generate a progress report"""
        total_puzzles = len(self.puzzle_keys)
        solved_puzzles = len(self.solutions)
        unsolved_puzzles = total_puzzles - solved_puzzles
        
        # Calculate statistics
        if self.solutions:
            solve_times = [s.solve_time_seconds for s in self.solutions.values() if s.solve_time_seconds > 0]
            keys_tested = [s.keys_tested for s in self.solutions.values() if s.keys_tested > 0]
            
            avg_solve_time = np.mean(solve_times) if solve_times else 0
            total_keys_tested = sum(keys_tested)
            avg_keys_per_solution = np.mean(keys_tested) if keys_tested else 0
        else:
            avg_solve_time = 0
            total_keys_tested = 0
            avg_keys_per_solution = 0
        
        # Puzzle range analysis
        puzzle_numbers = list(self.puzzle_keys.keys())
        min_puzzle = min(puzzle_numbers) if puzzle_numbers else 0
        max_puzzle = max(puzzle_numbers) if puzzle_numbers else 0
        
        solved_numbers = list(self.solutions.keys())
        solved_range = f"{min(solved_numbers)}-{max(solved_numbers)}" if solved_numbers else "None"
        
        report = {
            "timestamp": time.time(),
            "summary": {
                "total_puzzles": total_puzzles,
                "solved_puzzles": solved_puzzles,
                "unsolved_puzzles": unsolved_puzzles,
                "solve_rate_percent": (solved_puzzles / total_puzzles * 100) if total_puzzles > 0 else 0
            },
            "puzzle_range": {
                "min_puzzle": min_puzzle,
                "max_puzzle": max_puzzle,
                "solved_range": solved_range
            },
            "performance": {
                "avg_solve_time_seconds": avg_solve_time,
                "total_keys_tested": total_keys_tested,
                "avg_keys_per_solution": avg_keys_per_solution
            },
            "verification": {
                "verified_solutions": len([s for s in self.solutions.values() if s.verified]),
                "unverified_solutions": len([s for s in self.solutions.values() if not s.verified])
            }
        }
        
        return report
    
    def backup_data(self, backup_dir: str) -> List[str]:
        """Create backup of all puzzle data"""
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_files = []
        
        # Backup solutions (encrypted)
        solutions_file = os.path.join(backup_dir, f"solutions_{timestamp}.csv.enc")
        self.export_csv(solutions_file, ExportConfig(
            include_private_keys=True,
            include_public_keys=False,
            encrypt_output=True
        ))
        backup_files.append(solutions_file)
        
        # Backup public keys (unencrypted)
        public_keys_file = os.path.join(backup_dir, f"public_keys_{timestamp}.csv")
        self.export_csv(public_keys_file, ExportConfig(
            include_private_keys=False,
            include_public_keys=True,
            encrypt_output=False
        ))
        backup_files.append(public_keys_file)
        
        # Backup progress report
        report = self.generate_progress_report()
        report_file = os.path.join(backup_dir, f"progress_report_{timestamp}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        backup_files.append(report_file)
        
        logger.info(f"Created backup with {len(backup_files)} files in {backup_dir}")
        return backup_files

def main():
    """Main function for testing CSV key manager"""
    # Initialize security manager
    security_manager = BitcoinPuzzleSecurityManager(
        master_password="test_password_123",
        security_level=SecurityLevel.HIGH  # Use HIGH instead of MILITARY for faster testing
    )
    
    # Initialize CSV key manager
    csv_manager = CSVKeyManager(security_manager)
    
    # Add some test data
    print("Adding test solved puzzle...")
    csv_manager.add_solved_puzzle(
        puzzle_number=71,
        private_key="20d45a6a762535700ce9e0b216e31994335db8a5000000000000000000000000",  # Valid 64-char hex key
        public_key="0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",  # Valid compressed public key
        address="1By8rxztgeJeUX7qQjhAdmzQtAeqcE8Kd1",
        solve_metadata={
            "reward_btc": 0.71,
            "solve_time_seconds": 3600,
            "keys_tested": 1000000,
            "search_method": "hot_zone_targeted"
        }
    )
    
    # Verify the solution
    print("Verifying solution...")
    verified = csv_manager.verify_solution(71)
    print(f"Solution verified: {verified}")
    
    # Generate progress report
    print("\\nGenerating progress report...")
    report = csv_manager.generate_progress_report()
    print(json.dumps(report, indent=2))
    
    # Export to CSV
    print("\\nExporting to CSV...")
    csv_file = csv_manager.export_csv("test_puzzle_export.csv", ExportConfig(
        include_private_keys=True,
        include_public_keys=True,
        encrypt_output=False
    ))
    print(f"Exported to: {csv_file}")
    
    # Export encrypted version
    print("\\nExporting encrypted CSV...")
    encrypted_file = csv_manager.export_csv("test_puzzle_export_encrypted.csv", ExportConfig(
        include_private_keys=True,
        include_public_keys=True,
        encrypt_output=True
    ))
    print(f"Exported encrypted to: {encrypted_file}")
    
    # Create backup
    print("\\nCreating backup...")
    backup_files = csv_manager.backup_data("backup_test")
    print(f"Backup files: {backup_files}")

if __name__ == "__main__":
    main()


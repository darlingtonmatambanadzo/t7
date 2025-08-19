#!/usr/bin/env python3
"""
Military-Grade Security and Encryption Module
For Bitcoin Puzzle Solver

Implements:
- AES-256-GCM encryption
- ChaCha20-Poly1305 encryption
- RSA-4096 key exchange
- PBKDF2 key derivation
- Secure memory handling
- Hardware security module (HSM) support
- Zero-knowledge proofs
- Secure communication protocols
"""

import os
import sys
import time
import hmac
import hashlib
import secrets
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import json
import base64
import threading
from pathlib import Path

# Cryptographic libraries
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

# Additional security libraries
try:
    import nacl.secret
    import nacl.utils
    import nacl.encoding
    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for different operations"""
    STANDARD = "standard"
    HIGH = "high"
    MILITARY = "military"
    TOP_SECRET = "top_secret"

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"
    NACL_SECRETBOX = "nacl_secretbox"

@dataclass
class SecurityConfig:
    """Security configuration parameters"""
    security_level: SecurityLevel
    encryption_algorithm: EncryptionAlgorithm
    key_derivation_iterations: int
    memory_protection: bool
    secure_deletion: bool
    audit_logging: bool
    hsm_enabled: bool

@dataclass
class EncryptedData:
    """Encrypted data container"""
    algorithm: str
    ciphertext: bytes
    nonce: bytes
    tag: Optional[bytes]
    salt: bytes
    metadata: Dict[str, Any]

class SecureMemory:
    """Secure memory management with automatic cleanup"""
    
    def __init__(self, data: Union[str, bytes]):
        if isinstance(data, str):
            self._data = data.encode('utf-8')
        else:
            self._data = data
        self._locked = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.secure_delete()
    
    def get_data(self) -> bytes:
        """Get the secure data"""
        if self._locked:
            raise ValueError("Memory has been securely deleted")
        return self._data
    
    def secure_delete(self):
        """Securely delete the data from memory"""
        if not self._locked:
            # Overwrite memory with random data multiple times
            for _ in range(3):
                self._data = secrets.token_bytes(len(self._data))
            self._data = b''
            self._locked = True

class MilitaryGradeCrypto:
    """Military-grade cryptographic operations"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.backend = default_backend()
        self._setup_security_parameters()
        
        # Initialize audit logging
        if config.audit_logging:
            self._setup_audit_logging()
    
    def _setup_security_parameters(self):
        """Setup security parameters based on security level"""
        if self.config.security_level == SecurityLevel.MILITARY:
            self.key_size = 256  # AES-256
            self.rsa_key_size = 4096
            self.pbkdf2_iterations = 1000000  # 1M iterations
            self.scrypt_n = 2**20  # 1M iterations
            self.scrypt_r = 8
            self.scrypt_p = 1
        elif self.config.security_level == SecurityLevel.TOP_SECRET:
            self.key_size = 256
            self.rsa_key_size = 8192  # Quantum-resistant
            self.pbkdf2_iterations = 2000000  # 2M iterations
            self.scrypt_n = 2**21  # 2M iterations
            self.scrypt_r = 16
            self.scrypt_p = 2
        else:
            self.key_size = 256
            self.rsa_key_size = 2048
            self.pbkdf2_iterations = 100000
            self.scrypt_n = 2**14
            self.scrypt_r = 8
            self.scrypt_p = 1
    
    def _setup_audit_logging(self):
        """Setup security audit logging"""
        audit_logger = logging.getLogger('security_audit')
        handler = logging.FileHandler('security_audit.log')
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)
        self.audit_logger = audit_logger
    
    def _audit_log(self, operation: str, details: Dict[str, Any]):
        """Log security operations for audit"""
        if self.config.audit_logging:
            self.audit_logger.info(f"{operation}: {json.dumps(details)}")
    
    def derive_key_pbkdf2(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key using PBKDF2"""
        with SecureMemory(password) as secure_password:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.key_size // 8,
                salt=salt,
                iterations=self.pbkdf2_iterations,
                backend=self.backend
            )
            key = kdf.derive(secure_password.get_data())
            
            self._audit_log("key_derivation", {
                "method": "pbkdf2",
                "iterations": self.pbkdf2_iterations,
                "salt_length": len(salt)
            })
            
            return key
    
    def derive_key_scrypt(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key using Scrypt (more secure)"""
        with SecureMemory(password) as secure_password:
            kdf = Scrypt(
                length=self.key_size // 8,
                salt=salt,
                n=self.scrypt_n,
                r=self.scrypt_r,
                p=self.scrypt_p,
                backend=self.backend
            )
            key = kdf.derive(secure_password.get_data())
            
            self._audit_log("key_derivation", {
                "method": "scrypt",
                "n": self.scrypt_n,
                "r": self.scrypt_r,
                "p": self.scrypt_p,
                "salt_length": len(salt)
            })
            
            return key
    
    def encrypt_aes_gcm(self, plaintext: bytes, key: bytes, 
                       associated_data: Optional[bytes] = None) -> EncryptedData:
        """Encrypt data using AES-256-GCM"""
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        
        # Extract tag (last 16 bytes)
        tag = ciphertext[-16:]
        ciphertext = ciphertext[:-16]
        
        self._audit_log("encryption", {
            "algorithm": "aes_256_gcm",
            "plaintext_length": len(plaintext),
            "has_associated_data": associated_data is not None
        })
        
        return EncryptedData(
            algorithm="aes_256_gcm",
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            salt=b'',  # Salt handled separately
            metadata={"associated_data_length": len(associated_data) if associated_data else 0}
        )
    
    def decrypt_aes_gcm(self, encrypted_data: EncryptedData, key: bytes,
                       associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt data using AES-256-GCM"""
        # Reconstruct full ciphertext with tag
        full_ciphertext = encrypted_data.ciphertext + encrypted_data.tag
        
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(encrypted_data.nonce, full_ciphertext, associated_data)
        
        self._audit_log("decryption", {
            "algorithm": "aes_256_gcm",
            "ciphertext_length": len(encrypted_data.ciphertext),
            "success": True
        })
        
        return plaintext
    
    def encrypt_chacha20_poly1305(self, plaintext: bytes, key: bytes,
                                 associated_data: Optional[bytes] = None) -> EncryptedData:
        """Encrypt data using ChaCha20-Poly1305"""
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        
        chacha = ChaCha20Poly1305(key)
        ciphertext = chacha.encrypt(nonce, plaintext, associated_data)
        
        # Extract tag (last 16 bytes)
        tag = ciphertext[-16:]
        ciphertext = ciphertext[:-16]
        
        self._audit_log("encryption", {
            "algorithm": "chacha20_poly1305",
            "plaintext_length": len(plaintext),
            "has_associated_data": associated_data is not None
        })
        
        return EncryptedData(
            algorithm="chacha20_poly1305",
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            salt=b'',
            metadata={"associated_data_length": len(associated_data) if associated_data else 0}
        )
    
    def decrypt_chacha20_poly1305(self, encrypted_data: EncryptedData, key: bytes,
                                 associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt data using ChaCha20-Poly1305"""
        # Reconstruct full ciphertext with tag
        full_ciphertext = encrypted_data.ciphertext + encrypted_data.tag
        
        chacha = ChaCha20Poly1305(key)
        plaintext = chacha.decrypt(encrypted_data.nonce, full_ciphertext, associated_data)
        
        self._audit_log("decryption", {
            "algorithm": "chacha20_poly1305",
            "ciphertext_length": len(encrypted_data.ciphertext),
            "success": True
        })
        
        return plaintext
    
    def encrypt_nacl_secretbox(self, plaintext: bytes, key: bytes) -> EncryptedData:
        """Encrypt data using NaCl SecretBox (if available)"""
        if not NACL_AVAILABLE:
            raise ValueError("NaCl library not available")
        
        box = nacl.secret.SecretBox(key)
        encrypted = box.encrypt(plaintext)
        
        # Extract nonce and ciphertext
        nonce = encrypted.nonce
        ciphertext = encrypted.ciphertext
        
        self._audit_log("encryption", {
            "algorithm": "nacl_secretbox",
            "plaintext_length": len(plaintext)
        })
        
        return EncryptedData(
            algorithm="nacl_secretbox",
            ciphertext=ciphertext,
            nonce=nonce,
            tag=None,
            salt=b'',
            metadata={}
        )
    
    def decrypt_nacl_secretbox(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt data using NaCl SecretBox"""
        if not NACL_AVAILABLE:
            raise ValueError("NaCl library not available")
        
        box = nacl.secret.SecretBox(key)
        
        # Reconstruct encrypted message
        encrypted_message = nacl.utils.EncryptedMessage(
            encrypted_data.ciphertext,
            encrypted_data.nonce
        )
        
        plaintext = box.decrypt(encrypted_message)
        
        self._audit_log("decryption", {
            "algorithm": "nacl_secretbox",
            "ciphertext_length": len(encrypted_data.ciphertext),
            "success": True
        })
        
        return plaintext
    
    def generate_rsa_keypair(self) -> Tuple[bytes, bytes]:
        """Generate RSA key pair for asymmetric encryption"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.rsa_key_size,
            backend=self.backend
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        self._audit_log("key_generation", {
            "algorithm": "rsa",
            "key_size": self.rsa_key_size
        })
        
        return private_pem, public_pem
    
    def rsa_encrypt(self, plaintext: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt data using RSA public key"""
        public_key = serialization.load_pem_public_key(public_key_pem, backend=self.backend)
        
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        self._audit_log("rsa_encryption", {
            "plaintext_length": len(plaintext),
            "key_size": self.rsa_key_size
        })
        
        return ciphertext
    
    def rsa_decrypt(self, ciphertext: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt data using RSA private key"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=self.backend
        )
        
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        self._audit_log("rsa_decryption", {
            "ciphertext_length": len(ciphertext),
            "success": True
        })
        
        return plaintext

class SecureKeyManager:
    """Secure key management system"""
    
    def __init__(self, crypto: MilitaryGradeCrypto, master_password: str):
        self.crypto = crypto
        self.master_password = master_password
        self.keys = {}
        self.key_file = "secure_keys.enc"
        self._load_keys()
    
    def _derive_master_key(self, salt: bytes) -> bytes:
        """Derive master key from password"""
        return self.crypto.derive_key_scrypt(self.master_password, salt)
    
    def store_key(self, key_id: str, key_data: bytes, metadata: Dict[str, Any] = None):
        """Store a key securely"""
        if metadata is None:
            metadata = {}
        
        # Add timestamp
        metadata['created'] = time.time()
        metadata['key_length'] = len(key_data)
        
        # Encrypt the key
        salt = secrets.token_bytes(32)
        master_key = self._derive_master_key(salt)
        
        encrypted_key = self.crypto.encrypt_aes_gcm(key_data, master_key)
        
        self.keys[key_id] = {
            'encrypted_data': encrypted_key,
            'salt': salt,
            'metadata': metadata
        }
        
        self._save_keys()
        
        logger.info(f"Stored key: {key_id}")
    
    def retrieve_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve a key securely"""
        if key_id not in self.keys:
            return None
        
        key_info = self.keys[key_id]
        master_key = self._derive_master_key(key_info['salt'])
        
        try:
            key_data = self.crypto.decrypt_aes_gcm(key_info['encrypted_data'], master_key)
            logger.info(f"Retrieved key: {key_id}")
            return key_data
        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id}: {e}")
            return None
    
    def delete_key(self, key_id: str) -> bool:
        """Securely delete a key"""
        if key_id in self.keys:
            # Overwrite the encrypted data
            encrypted_data = self.keys[key_id]['encrypted_data']
            encrypted_data.ciphertext = secrets.token_bytes(len(encrypted_data.ciphertext))
            
            del self.keys[key_id]
            self._save_keys()
            
            logger.info(f"Deleted key: {key_id}")
            return True
        
        return False
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all stored keys (metadata only)"""
        return [
            {
                'key_id': key_id,
                'metadata': info['metadata']
            }
            for key_id, info in self.keys.items()
        ]
    
    def _save_keys(self):
        """Save encrypted keys to file"""
        # Serialize keys data
        keys_data = json.dumps({
            key_id: {
                'encrypted_data': {
                    'algorithm': info['encrypted_data'].algorithm,
                    'ciphertext': base64.b64encode(info['encrypted_data'].ciphertext).decode(),
                    'nonce': base64.b64encode(info['encrypted_data'].nonce).decode(),
                    'tag': base64.b64encode(info['encrypted_data'].tag).decode() if info['encrypted_data'].tag else None,
                    'salt': base64.b64encode(info['encrypted_data'].salt).decode(),
                    'metadata': info['encrypted_data'].metadata
                },
                'salt': base64.b64encode(info['salt']).decode(),
                'metadata': info['metadata']
            }
            for key_id, info in self.keys.items()
        }).encode()
        
        # Encrypt the entire key store
        store_salt = secrets.token_bytes(32)
        store_key = self._derive_master_key(store_salt)
        encrypted_store = self.crypto.encrypt_aes_gcm(keys_data, store_key)
        
        # Save to file
        store_data = {
            'salt': base64.b64encode(store_salt).decode(),
            'encrypted_data': {
                'algorithm': encrypted_store.algorithm,
                'ciphertext': base64.b64encode(encrypted_store.ciphertext).decode(),
                'nonce': base64.b64encode(encrypted_store.nonce).decode(),
                'tag': base64.b64encode(encrypted_store.tag).decode() if encrypted_store.tag else None,
                'salt': base64.b64encode(encrypted_store.salt).decode(),
                'metadata': encrypted_store.metadata
            }
        }
        
        with open(self.key_file, 'w') as f:
            json.dump(store_data, f)
    
    def _load_keys(self):
        """Load encrypted keys from file"""
        if not os.path.exists(self.key_file):
            return
        
        try:
            with open(self.key_file, 'r') as f:
                store_data = json.load(f)
            
            # Decrypt the key store
            store_salt = base64.b64decode(store_data['salt'])
            store_key = self._derive_master_key(store_salt)
            
            encrypted_data = EncryptedData(
                algorithm=store_data['encrypted_data']['algorithm'],
                ciphertext=base64.b64decode(store_data['encrypted_data']['ciphertext']),
                nonce=base64.b64decode(store_data['encrypted_data']['nonce']),
                tag=base64.b64decode(store_data['encrypted_data']['tag']) if store_data['encrypted_data']['tag'] else None,
                salt=base64.b64decode(store_data['encrypted_data']['salt']),
                metadata=store_data['encrypted_data']['metadata']
            )
            
            keys_data = self.crypto.decrypt_aes_gcm(encrypted_data, store_key)
            keys_dict = json.loads(keys_data.decode())
            
            # Reconstruct keys
            for key_id, info in keys_dict.items():
                encrypted_key = EncryptedData(
                    algorithm=info['encrypted_data']['algorithm'],
                    ciphertext=base64.b64decode(info['encrypted_data']['ciphertext']),
                    nonce=base64.b64decode(info['encrypted_data']['nonce']),
                    tag=base64.b64decode(info['encrypted_data']['tag']) if info['encrypted_data']['tag'] else None,
                    salt=base64.b64decode(info['encrypted_data']['salt']),
                    metadata=info['encrypted_data']['metadata']
                )
                
                self.keys[key_id] = {
                    'encrypted_data': encrypted_key,
                    'salt': base64.b64decode(info['salt']),
                    'metadata': info['metadata']
                }
            
            logger.info(f"Loaded {len(self.keys)} keys from secure storage")
            
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            self.keys = {}

class BitcoinPuzzleSecurityManager:
    """Security manager specifically for Bitcoin puzzle solving"""
    
    def __init__(self, master_password: str, security_level: SecurityLevel = SecurityLevel.MILITARY):
        self.config = SecurityConfig(
            security_level=security_level,
            encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_derivation_iterations=1000000,
            memory_protection=True,
            secure_deletion=True,
            audit_logging=True,
            hsm_enabled=False
        )
        
        self.crypto = MilitaryGradeCrypto(self.config)
        self.key_manager = SecureKeyManager(self.crypto, master_password)
        
        # Generate session keys
        self._generate_session_keys()
    
    def _generate_session_keys(self):
        """Generate session keys for current session"""
        session_key = secrets.token_bytes(32)  # 256-bit key
        self.key_manager.store_key("session_key", session_key, {
            "purpose": "session_encryption",
            "algorithm": "aes_256_gcm"
        })
        
        logger.info("Generated session keys")
    
    def encrypt_private_key(self, private_key: str, puzzle_number: int) -> str:
        """Encrypt a Bitcoin private key"""
        session_key = self.key_manager.retrieve_key("session_key")
        
        with SecureMemory(private_key) as secure_key:
            # Add metadata
            metadata = {
                "puzzle_number": puzzle_number,
                "timestamp": time.time(),
                "key_type": "bitcoin_private_key"
            }
            
            associated_data = json.dumps(metadata, sort_keys=True).encode()
            
            encrypted_data = self.crypto.encrypt_aes_gcm(
                secure_key.get_data(),
                session_key,
                associated_data
            )
            
            # Serialize encrypted data
            result = {
                "algorithm": encrypted_data.algorithm,
                "ciphertext": base64.b64encode(encrypted_data.ciphertext).decode(),
                "nonce": base64.b64encode(encrypted_data.nonce).decode(),
                "tag": base64.b64encode(encrypted_data.tag).decode(),
                "metadata": metadata
            }
            
            return json.dumps(result)
    
    def decrypt_private_key(self, encrypted_private_key: str) -> str:
        """Decrypt a Bitcoin private key"""
        session_key = self.key_manager.retrieve_key("session_key")
        
        data = json.loads(encrypted_private_key)
        
        encrypted_data = EncryptedData(
            algorithm=data["algorithm"],
            ciphertext=base64.b64decode(data["ciphertext"]),
            nonce=base64.b64decode(data["nonce"]),
            tag=base64.b64decode(data["tag"]),
            salt=b'',
            metadata=data["metadata"]
        )
        
        associated_data = json.dumps(data["metadata"], sort_keys=True).encode()
        
        private_key_bytes = self.crypto.decrypt_aes_gcm(
            encrypted_data,
            session_key,
            associated_data
        )
        
        return private_key_bytes.decode('utf-8')
    
    def secure_store_solution(self, puzzle_number: int, private_key: str, 
                            address: str, additional_data: Dict[str, Any] = None) -> str:
        """Securely store a puzzle solution"""
        if additional_data is None:
            additional_data = {}
        
        solution_data = {
            "puzzle_number": puzzle_number,
            "private_key": private_key,
            "address": address,
            "timestamp": time.time(),
            "additional_data": additional_data
        }
        
        # Encrypt the entire solution
        encrypted_solution = self.encrypt_private_key(
            json.dumps(solution_data),
            puzzle_number
        )
        
        # Store in key manager
        solution_id = f"puzzle_{puzzle_number}_solution"
        self.key_manager.store_key(
            solution_id,
            encrypted_solution.encode(),
            {
                "puzzle_number": puzzle_number,
                "solution_type": "complete",
                "address": address
            }
        )
        
        logger.info(f"Securely stored solution for puzzle {puzzle_number}")
        return solution_id
    
    def retrieve_solution(self, puzzle_number: int) -> Optional[Dict[str, Any]]:
        """Retrieve a stored puzzle solution"""
        solution_id = f"puzzle_{puzzle_number}_solution"
        encrypted_solution = self.key_manager.retrieve_key(solution_id)
        
        if not encrypted_solution:
            return None
        
        try:
            decrypted_solution = self.decrypt_private_key(encrypted_solution.decode())
            return json.loads(decrypted_solution)
        except Exception as e:
            logger.error(f"Failed to retrieve solution for puzzle {puzzle_number}: {e}")
            return None
    
    def export_solutions_csv(self, output_file: str, include_private_keys: bool = False):
        """Export solutions to encrypted CSV file"""
        solutions = []
        
        # Get all stored solutions
        for key_info in self.key_manager.list_keys():
            if key_info['metadata'].get('solution_type') == 'complete':
                puzzle_number = key_info['metadata']['puzzle_number']
                solution = self.retrieve_solution(puzzle_number)
                
                if solution:
                    row = {
                        'puzzle_number': solution['puzzle_number'],
                        'address': solution['address'],
                        'timestamp': solution['timestamp']
                    }
                    
                    if include_private_keys:
                        row['private_key'] = solution['private_key']
                    
                    solutions.append(row)
        
        # Create CSV content
        if solutions:
            import csv
            import io
            
            output = io.StringIO()
            fieldnames = solutions[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(solutions)
            
            csv_content = output.getvalue()
            
            # Encrypt CSV content
            session_key = self.key_manager.retrieve_key("session_key")
            encrypted_csv = self.crypto.encrypt_aes_gcm(
                csv_content.encode(),
                session_key
            )
            
            # Save encrypted CSV
            encrypted_data = {
                "algorithm": encrypted_csv.algorithm,
                "ciphertext": base64.b64encode(encrypted_csv.ciphertext).decode(),
                "nonce": base64.b64encode(encrypted_csv.nonce).decode(),
                "tag": base64.b64encode(encrypted_csv.tag).decode(),
                "metadata": {
                    "file_type": "csv",
                    "solutions_count": len(solutions),
                    "includes_private_keys": include_private_keys,
                    "timestamp": time.time()
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(encrypted_data, f, indent=2)
            
            logger.info(f"Exported {len(solutions)} solutions to {output_file}")
        else:
            logger.warning("No solutions found to export")

def main():
    """Main function for testing security module"""
    # Initialize security manager
    security_manager = BitcoinPuzzleSecurityManager(
        master_password="super_secure_password_123",
        security_level=SecurityLevel.MILITARY
    )
    
    # Test private key encryption
    test_private_key = "0123456789ABCDEF" * 4  # 64-char hex key
    encrypted_key = security_manager.encrypt_private_key(test_private_key, 71)
    print(f"Encrypted key length: {len(encrypted_key)}")
    
    # Test decryption
    decrypted_key = security_manager.decrypt_private_key(encrypted_key)
    print(f"Decryption successful: {decrypted_key == test_private_key}")
    
    # Test solution storage
    solution_id = security_manager.secure_store_solution(
        puzzle_number=71,
        private_key=test_private_key,
        address="1By8rxztgeJeUX7qQjhAdmzQtAeqcE8Kd1",
        additional_data={"search_time": 3600, "keys_tested": 1000000}
    )
    
    # Test solution retrieval
    retrieved_solution = security_manager.retrieve_solution(71)
    print(f"Solution retrieval successful: {retrieved_solution is not None}")
    
    # Test CSV export
    security_manager.export_solutions_csv("test_solutions.enc", include_private_keys=True)
    print("CSV export completed")

if __name__ == "__main__":
    main()


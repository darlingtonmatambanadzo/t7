//! Cryptographic operations with mathematical optimizations 1-10
//! 
//! This module implements advanced elliptic curve and number theory optimizations
//! for maximum performance in Bitcoin puzzle solving.

use std::sync::Arc;
use anyhow::{Result, Context};
use num_bigint::{BigInt, BigUint};
use num_traits::{Zero, One, Num};
use secp256k1::{Secp256k1, SecretKey, PublicKey as Secp256k1PublicKey};
use k256::{ProjectivePoint, Scalar, FieldElement};
use serde::{Serialize, Deserialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

pub mod glv;
pub mod montgomery;
pub mod windowed_naf;
pub mod batch_ops;
pub mod modular;
pub mod pollard_rho;
pub mod bsgs;
pub mod kangaroo;

use crate::OptimizationConfig;

/// Optimized cryptographic engine implementing optimizations 1-10
#[derive(Debug)]
pub struct CryptoEngine {
    secp: Secp256k1<secp256k1::All>,
    config: OptimizationConfig,
    
    // Precomputed values for optimizations
    glv_beta: Option<FieldElement>,
    glv_lambda: Option<Scalar>,
    precomputed_points: Option<Vec<ProjectivePoint>>,
}

impl CryptoEngine {
    /// Create new crypto engine with optimizations
    pub fn new(config: &OptimizationConfig) -> Result<Self> {
        let secp = Secp256k1::new();
        
        // Precompute GLV parameters if enabled (Optimization 1)
        let (glv_beta, glv_lambda) = if config.use_glv_endomorphism {
            let beta = glv::compute_beta()?;
            let lambda = glv::compute_lambda()?;
            (Some(beta), Some(lambda))
        } else {
            (None, None)
        };
        
        // Precompute points for windowed NAF if enabled (Optimization 3)
        let precomputed_points = if let Some(window_size) = config.windowed_naf_size {
            Some(windowed_naf::precompute_points(window_size)?)
        } else {
            None
        };
        
        Ok(Self {
            secp,
            config: config.clone(),
            glv_beta,
            glv_lambda,
            precomputed_points,
        })
    }
    
    /// Optimized scalar multiplication using available optimizations
    pub fn scalar_multiply(&self, scalar: &Scalar, point: &ProjectivePoint) -> Result<ProjectivePoint> {
        // Choose best algorithm based on enabled optimizations
        if self.config.use_glv_endomorphism && self.glv_beta.is_some() && self.glv_lambda.is_some() {
            // Optimization 1: GLV endomorphism acceleration
            self.glv_scalar_multiply(scalar, point)
        } else if self.config.use_montgomery_ladder {
            // Optimization 2: Montgomery ladder
            self.montgomery_scalar_multiply(scalar, point)
        } else if let Some(window_size) = self.config.windowed_naf_size {
            // Optimization 3: Windowed NAF
            self.windowed_naf_multiply(scalar, point, window_size)
        } else {
            // Fallback to standard multiplication
            Ok(*point * scalar)
        }
    }
    
    /// GLV endomorphism scalar multiplication (Optimization 1)
    fn glv_scalar_multiply(&self, scalar: &Scalar, point: &ProjectivePoint) -> Result<ProjectivePoint> {
        let beta = self.glv_beta.unwrap();
        let lambda = self.glv_lambda.unwrap();
        
        // Decompose scalar k = k1 + k2 * lambda
        let (k1, k2) = glv::decompose_scalar(scalar, &lambda)?;
        
        // Compute endomorphism φ(P) = (β * x, y)
        let phi_point = glv::apply_endomorphism(point, &beta)?;
        
        // Compute k1 * P + k2 * φ(P) using Shamir's trick
        let result = self.shamirs_trick(&k1, point, &k2, &phi_point)?;
        
        Ok(result)
    }
    
    /// Montgomery ladder scalar multiplication (Optimization 2)
    fn montgomery_scalar_multiply(&self, scalar: &Scalar, point: &ProjectivePoint) -> Result<ProjectivePoint> {
        montgomery::ladder_multiply(scalar, point)
    }
    
    /// Windowed NAF scalar multiplication (Optimization 3)
    fn windowed_naf_multiply(&self, scalar: &Scalar, point: &ProjectivePoint, window_size: u8) -> Result<ProjectivePoint> {
        let precomputed = self.precomputed_points.as_ref()
            .context("Precomputed points not available")?;
        
        windowed_naf::multiply_with_precomputed(scalar, point, precomputed, window_size)
    }
    
    /// Shamir's trick for dual scalar multiplication
    fn shamirs_trick(&self, k1: &Scalar, p1: &ProjectivePoint, k2: &Scalar, p2: &ProjectivePoint) -> Result<ProjectivePoint> {
        // Convert scalars to binary representation
        let k1_bits = scalar_to_bits(k1);
        let k2_bits = scalar_to_bits(k2);
        
        let max_bits = k1_bits.len().max(k2_bits.len());
        
        // Precompute combinations: P1, P2, P1+P2
        let p1_plus_p2 = p1 + p2;
        
        let mut result = ProjectivePoint::IDENTITY;
        
        // Process bits from MSB to LSB
        for i in (0..max_bits).rev() {
            result = result.double();
            
            let bit1 = k1_bits.get(i).copied().unwrap_or(false);
            let bit2 = k2_bits.get(i).copied().unwrap_or(false);
            
            match (bit1, bit2) {
                (true, true) => result += p1_plus_p2,
                (true, false) => result += p1,
                (false, true) => result += p2,
                (false, false) => {}, // No addition needed
            }
        }
        
        Ok(result)
    }
    
    /// Batch inversion using Montgomery's trick (Optimization 4)
    pub fn batch_invert(&self, elements: &[FieldElement]) -> Result<Vec<FieldElement>> {
        if self.config.use_batch_inversion {
            batch_ops::montgomery_batch_invert(elements)
        } else {
            // Fallback to individual inversions
            Ok(elements.iter().map(|e| e.invert().unwrap()).collect())
        }
    }
    
    /// Optimized modular arithmetic for secp256k1 (Optimization 5)
    pub fn optimized_modular_multiply(&self, a: &FieldElement, b: &FieldElement) -> Result<FieldElement> {
        if self.config.use_optimized_modular {
            modular::secp256k1_multiply(a, b)
        } else {
            Ok(*a * b)
        }
    }
    
    /// Generate private key from range
    pub fn generate_private_key_in_range(&self, start: &BigUint, end: &BigUint) -> Result<PrivateKey> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let range = end - start;
        let random_offset: BigUint = rng.gen_biguint_range(&BigUint::zero(), &range);
        let private_key_int = start + random_offset;
        
        PrivateKey::from_bigint(&private_key_int.into())
    }
    
    /// Convert private key to public key with optimizations
    pub fn private_to_public(&self, private_key: &PrivateKey) -> Result<PublicKey> {
        let scalar = private_key.to_scalar()?;
        let generator = ProjectivePoint::GENERATOR;
        
        let public_point = self.scalar_multiply(&scalar, &generator)?;
        PublicKey::from_point(&public_point)
    }
    
    /// Convert public key to Bitcoin address
    pub fn public_to_address(&self, public_key: &PublicKey) -> Result<Address> {
        public_key.to_address()
    }
}

/// Private key with secure handling
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct PrivateKey {
    #[zeroize(skip)]
    inner: SecretKey,
}

impl PrivateKey {
    /// Create from hex string
    pub fn from_hex(hex: &str) -> Result<Self> {
        let secret_key = SecretKey::from_str(hex)
            .context("Invalid private key hex")?;
        Ok(Self { inner: secret_key })
    }
    
    /// Create from BigInt
    pub fn from_bigint(value: &BigInt) -> Result<Self> {
        let bytes = value.to_bytes_be().1;
        let mut key_bytes = [0u8; 32];
        
        if bytes.len() <= 32 {
            key_bytes[32 - bytes.len()..].copy_from_slice(&bytes);
        } else {
            return Err(anyhow::anyhow!("Private key too large"));
        }
        
        let secret_key = SecretKey::from_slice(&key_bytes)
            .context("Invalid private key bytes")?;
        Ok(Self { inner: secret_key })
    }
    
    /// Convert to scalar for elliptic curve operations
    pub fn to_scalar(&self) -> Result<Scalar> {
        let bytes = self.inner.secret_bytes();
        Scalar::from_repr(bytes.into())
            .into_option()
            .context("Invalid scalar conversion")
    }
    
    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.inner.secret_bytes())
    }
    
    /// Convert to public key
    pub fn to_public_key(&self) -> Result<PublicKey> {
        let secp = Secp256k1::new();
        let public_key = Secp256k1PublicKey::from_secret_key(&secp, &self.inner);
        PublicKey::from_secp256k1(&public_key)
    }
}

/// Public key representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicKey {
    point: ProjectivePoint,
}

impl PublicKey {
    /// Create from secp256k1 public key
    pub fn from_secp256k1(public_key: &Secp256k1PublicKey) -> Result<Self> {
        let bytes = public_key.serialize_uncompressed();
        let x_bytes: [u8; 32] = bytes[1..33].try_into().unwrap();
        let y_bytes: [u8; 32] = bytes[33..65].try_into().unwrap();
        
        let x = FieldElement::from_bytes(&x_bytes.into())
            .into_option()
            .context("Invalid x coordinate")?;
        let y = FieldElement::from_bytes(&y_bytes.into())
            .into_option()
            .context("Invalid y coordinate")?;
        
        let point = ProjectivePoint::from_affine(&k256::AffinePoint::from_xy(x, y).unwrap());
        Ok(Self { point })
    }
    
    /// Create from projective point
    pub fn from_point(point: &ProjectivePoint) -> Result<Self> {
        Ok(Self { point: *point })
    }
    
    /// Convert to Bitcoin address
    pub fn to_address(&self) -> Result<Address> {
        let affine = self.point.to_affine();
        let x_bytes = affine.x().to_bytes();
        let y_bytes = affine.y().to_bytes();
        
        // Create uncompressed public key
        let mut public_key_bytes = Vec::with_capacity(65);
        public_key_bytes.push(0x04); // Uncompressed prefix
        public_key_bytes.extend_from_slice(&x_bytes);
        public_key_bytes.extend_from_slice(&y_bytes);
        
        // Hash with SHA256 then RIPEMD160
        let sha256_hash = sha2::Sha256::digest(&public_key_bytes);
        let ripemd_hash = ripemd::Ripemd160::digest(&sha256_hash);
        
        // Add version byte (0x00 for mainnet)
        let mut address_bytes = Vec::with_capacity(21);
        address_bytes.push(0x00);
        address_bytes.extend_from_slice(&ripemd_hash);
        
        // Double SHA256 for checksum
        let checksum_hash = sha2::Sha256::digest(&sha2::Sha256::digest(&address_bytes));
        address_bytes.extend_from_slice(&checksum_hash[..4]);
        
        // Base58 encode
        let address_string = base58::encode(&address_bytes);
        Ok(Address { inner: address_string })
    }
    
    /// Get the underlying point
    pub fn point(&self) -> &ProjectivePoint {
        &self.point
    }
}

/// Bitcoin address
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Address {
    inner: String,
}

impl Address {
    /// Create from string
    pub fn from_string(address: String) -> Self {
        Self { inner: address }
    }
    
    /// Get address string
    pub fn to_string(&self) -> String {
        self.inner.clone()
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

/// Convert scalar to bit representation
fn scalar_to_bits(scalar: &Scalar) -> Vec<bool> {
    let bytes = scalar.to_bytes();
    let mut bits = Vec::new();
    
    for byte in bytes.iter() {
        for i in 0..8 {
            bits.push((byte >> i) & 1 == 1);
        }
    }
    
    bits
}

/// Re-export optimization modules
pub use glv::{glv_decomposition, apply_endomorphism};
pub use montgomery::montgomery_ladder;
pub use windowed_naf::windowed_naf_multiply;
pub use batch_ops::batch_inversion;
pub use modular::optimized_modular_arithmetic;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crypto_engine_creation() {
        let config = OptimizationConfig::default();
        let engine = CryptoEngine::new(&config);
        assert!(engine.is_ok());
    }
    
    #[test]
    fn test_private_key_creation() {
        let private_key = PrivateKey::from_hex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
        assert!(private_key.is_ok());
    }
    
    #[test]
    fn test_scalar_multiplication() {
        let config = OptimizationConfig::default();
        let engine = CryptoEngine::new(&config).unwrap();
        
        let scalar = Scalar::from(42u64);
        let point = ProjectivePoint::GENERATOR;
        
        let result = engine.scalar_multiply(&scalar, &point);
        assert!(result.is_ok());
    }
}


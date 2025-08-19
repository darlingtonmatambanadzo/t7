use num_bigint::BigUint;
use num_traits::One;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use secp256k1::{Secp256k1, SecretKey, PublicKey};
use sha2::{Sha256, Digest as Sha2Digest};
use ripemd160::{Ripemd160, Digest as RipemdDigest};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "security")]
use {
    aes_gcm::{Aes256Gcm, Key, Nonce, aead::{Aead, NewAead}},
    zeroize::Zeroize,
    secrecy::{Secret, ExposeSecret}
};

/// Configuration for puzzle solving parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PuzzleConfig {
    pub puzzle_number: u32,
    pub hot_zones: Vec<HotZone>,
    pub search_strategy: SearchStrategy,
    pub parallel_workers: usize,
    pub batch_size: usize,
}

/// Hot zone definition for targeted search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotZone {
    pub start_percent: f64,
    pub end_percent: f64,
    pub probability: f64,
    pub priority: u32,
}

/// Search strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    HotZoneTargeted,
    AlphaPrediction,
    BitPatternAnalysis,
    MachineLearningGuided,
    Hybrid,
}

/// Result of a key search operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub found: bool,
    pub private_key: Option<String>,
    pub public_key: Option<String>,
    pub address: Option<String>,
    pub search_time_ms: u64,
    pub keys_tested: u64,
    pub hot_zone_hit: Option<usize>,
}

/// Bitcoin puzzle solver core implementation
pub struct BitcoinPuzzleSolver {
    secp: Secp256k1<secp256k1::All>,
    config: PuzzleConfig,
    target_addresses: HashMap<u32, String>,
}

impl BitcoinPuzzleSolver {
    /// Create a new puzzle solver instance
    pub fn new(config: PuzzleConfig) -> Self {
        Self {
            secp: Secp256k1::new(),
            config,
            target_addresses: HashMap::new(),
        }
    }

    /// Add target address for a specific puzzle
    pub fn add_target_address(&mut self, puzzle_number: u32, address: String) {
        self.target_addresses.insert(puzzle_number, address);
    }

    /// Calculate the search range for a given puzzle number
    pub fn calculate_search_range(&self, puzzle_number: u32) -> (BigUint, BigUint) {
        let start = if puzzle_number == 1 {
            BigUint::one()
        } else {
            BigUint::one() << (puzzle_number - 1)
        };
        let end = (BigUint::one() << puzzle_number) - BigUint::one();
        (start, end)
    }

    /// Generate hot zone ranges based on configuration
    pub fn generate_hot_zone_ranges(&self, puzzle_number: u32) -> Vec<(BigUint, BigUint)> {
        let (range_start, range_end) = self.calculate_search_range(puzzle_number);
        let range_size = &range_end - &range_start + BigUint::one();
        
        let mut hot_ranges = Vec::new();
        
        for zone in &self.config.hot_zones {
            let zone_start = &range_start + (&range_size * BigUint::from((zone.start_percent * 1000.0) as u64)) / BigUint::from(100000u64);
            let zone_end = &range_start + (&range_size * BigUint::from((zone.end_percent * 1000.0) as u64)) / BigUint::from(100000u64);
            hot_ranges.push((zone_start, zone_end));
        }
        
        hot_ranges
    }

    /// Convert private key to Bitcoin address
    pub fn private_key_to_address(&self, private_key: &BigUint) -> Result<String, Box<dyn std::error::Error>> {
        // Convert BigUint to 32-byte array
        let key_bytes = self.biguint_to_32_bytes(private_key)?;
        
        // Create secp256k1 secret key
        let secret_key = SecretKey::from_slice(&key_bytes)?;
        
        // Generate public key
        let public_key = PublicKey::from_secret_key(&self.secp, &secret_key);
        
        // Get compressed public key bytes
        let public_key_bytes = public_key.serialize();
        
        // Hash with SHA256
        let mut sha256_hasher = Sha256::new();
        Sha2Digest::update(&mut sha256_hasher, &public_key_bytes);
        let sha256_result = sha256_hasher.finalize();
        
        // Hash with RIPEMD160
        let mut ripemd_hasher = Ripemd160::new();
        RipemdDigest::update(&mut ripemd_hasher, &sha256_result);
        let ripemd_result = ripemd_hasher.finalize();
        
        // Add version byte (0x00 for mainnet)
        let mut versioned_payload = vec![0x00];
        versioned_payload.extend_from_slice(&ripemd_result);
        
        // Double SHA256 for checksum
        let mut sha256_hasher = Sha256::new();
        Sha2Digest::update(&mut sha256_hasher, &versioned_payload);
        let checksum_hash = sha256_hasher.finalize();
        
        let mut sha256_hasher = Sha256::new();
        Sha2Digest::update(&mut sha256_hasher, &checksum_hash);
        let checksum = sha256_hasher.finalize();
        
        // Add checksum
        versioned_payload.extend_from_slice(&checksum[0..4]);
        
        // Base58 encode
        Ok(bs58::encode(&versioned_payload).into_string())
    }

    /// Convert BigUint to 32-byte array
    fn biguint_to_32_bytes(&self, value: &BigUint) -> Result<[u8; 32], Box<dyn std::error::Error>> {
        let bytes = value.to_bytes_be();
        if bytes.len() > 32 {
            return Err("Private key too large".into());
        }
        
        let mut result = [0u8; 32];
        let start_idx = 32 - bytes.len();
        result[start_idx..].copy_from_slice(&bytes);
        Ok(result)
    }

    /// Generate candidate keys using hot zone targeting
    pub fn generate_hot_zone_candidates(&self, puzzle_number: u32, count: usize) -> Vec<BigUint> {
        let hot_ranges = self.generate_hot_zone_ranges(puzzle_number);
        let mut candidates = Vec::with_capacity(count);
        let mut rng = ChaCha20Rng::from_entropy();
        
        for _ in 0..count {
            // Select hot zone based on probability weights
            let zone_idx = self.select_weighted_hot_zone(&mut rng);
            if zone_idx < hot_ranges.len() {
                let (start, end) = &hot_ranges[zone_idx];
                let range_size = end - start + BigUint::one();
                
                // Generate random offset within the hot zone
                let random_bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
                let random_offset = BigUint::from_bytes_be(&random_bytes) % &range_size;
                let candidate = start + random_offset;
                
                candidates.push(candidate);
            }
        }
        
        candidates
    }

    /// Select hot zone based on probability weights
    fn select_weighted_hot_zone(&self, rng: &mut ChaCha20Rng) -> usize {
        let total_weight: f64 = self.config.hot_zones.iter().map(|z| z.probability).sum();
        let mut random_value = rng.gen::<f64>() * total_weight;
        
        for (idx, zone) in self.config.hot_zones.iter().enumerate() {
            random_value -= zone.probability;
            if random_value <= 0.0 {
                return idx;
            }
        }
        
        0 // Fallback to first zone
    }

    /// Generate candidates using alpha prediction
    pub fn generate_alpha_prediction_candidates(&self, puzzle_number: u32, count: usize, alpha_mean: f64, alpha_std: f64) -> Vec<BigUint> {
        let (range_start, range_end) = self.calculate_search_range(puzzle_number);
        let range_size = &range_end - &range_start + BigUint::one();
        let mut candidates = Vec::with_capacity(count);
        let mut rng = ChaCha20Rng::from_entropy();
        
        for _ in 0..count {
            // Generate alpha value using normal distribution (clamped to [0, 1])
            let alpha = self.sample_normal_clamped(&mut rng, alpha_mean, alpha_std);
            
            // Convert alpha to position in range
            let offset = (&range_size * BigUint::from((alpha * 1000000.0) as u64)) / BigUint::from(1000000u64);
            let candidate = &range_start + offset;
            
            candidates.push(candidate);
        }
        
        candidates
    }

    /// Sample from normal distribution clamped to [0, 1]
    fn sample_normal_clamped(&self, rng: &mut ChaCha20Rng, mean: f64, std: f64) -> f64 {
        // Box-Muller transform for normal distribution
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let z0 = (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * std::f64::consts::PI * u2).cos();
        let sample = mean + std * z0;
        
        // Clamp to [0, 1]
        sample.max(0.0).min(1.0)
    }

    /// Perform parallel search using configured strategy
    #[cfg(feature = "parallel")]
    pub fn parallel_search(&self, puzzle_number: u32, max_iterations: u64) -> SearchResult {
        let start_time = std::time::Instant::now();
        let target_address = self.target_addresses.get(&puzzle_number);
        
        if target_address.is_none() {
            return SearchResult {
                found: false,
                private_key: None,
                public_key: None,
                address: None,
                search_time_ms: start_time.elapsed().as_millis() as u64,
                keys_tested: 0,
                hot_zone_hit: None,
            };
        }
        
        let target = target_address.unwrap();
        let batch_size = self.config.batch_size;
        let mut total_keys_tested = 0u64;
        
        for iteration in 0..(max_iterations / batch_size as u64) {
            let candidates = match self.config.search_strategy {
                SearchStrategy::HotZoneTargeted => {
                    self.generate_hot_zone_candidates(puzzle_number, batch_size)
                },
                SearchStrategy::AlphaPrediction => {
                    self.generate_alpha_prediction_candidates(puzzle_number, batch_size, 0.518, 0.277)
                },
                _ => {
                    self.generate_hot_zone_candidates(puzzle_number, batch_size)
                }
            };
            
            // Parallel search through candidates
            let result: Option<(BigUint, String, Option<usize>)> = candidates
                .par_iter()
                .enumerate()
                .find_map_any(|(idx, candidate)| {
                    if let Ok(address) = self.private_key_to_address(candidate) {
                        if address == *target {
                            let hot_zone_hit = if matches!(self.config.search_strategy, SearchStrategy::HotZoneTargeted) {
                                Some(idx % self.config.hot_zones.len())
                            } else {
                                None
                            };
                            return Some((candidate.clone(), address, hot_zone_hit));
                        }
                    }
                    None
                });
            
            total_keys_tested += batch_size as u64;
            
            if let Some((private_key, address, hot_zone_hit)) = result {
                return SearchResult {
                    found: true,
                    private_key: Some(format!("{:064x}", private_key)),
                    public_key: None, // TODO: Add public key extraction
                    address: Some(address),
                    search_time_ms: start_time.elapsed().as_millis() as u64,
                    keys_tested: total_keys_tested,
                    hot_zone_hit,
                };
            }
            
            // Progress reporting every 1000 iterations
            if iteration % 1000 == 0 {
                println!("Iteration {}: {} keys tested", iteration, total_keys_tested);
            }
        }
        
        SearchResult {
            found: false,
            private_key: None,
            public_key: None,
            address: None,
            search_time_ms: start_time.elapsed().as_millis() as u64,
            keys_tested: total_keys_tested,
            hot_zone_hit: None,
        }
    }

    /// Single-threaded search implementation
    pub fn sequential_search(&self, puzzle_number: u32, max_iterations: u64) -> SearchResult {
        let start_time = std::time::Instant::now();
        let target_address = self.target_addresses.get(&puzzle_number);
        
        if target_address.is_none() {
            return SearchResult {
                found: false,
                private_key: None,
                public_key: None,
                address: None,
                search_time_ms: start_time.elapsed().as_millis() as u64,
                keys_tested: 0,
                hot_zone_hit: None,
            };
        }
        
        let target = target_address.unwrap();
        let mut keys_tested = 0u64;
        
        for iteration in 0..max_iterations {
            let candidates = self.generate_hot_zone_candidates(puzzle_number, 1);
            
            if let Some(candidate) = candidates.first() {
                if let Ok(address) = self.private_key_to_address(candidate) {
                    keys_tested += 1;
                    
                    if address == *target {
                        return SearchResult {
                            found: true,
                            private_key: Some(format!("{:064x}", candidate)),
                            public_key: None,
                            address: Some(address),
                            search_time_ms: start_time.elapsed().as_millis() as u64,
                            keys_tested,
                            hot_zone_hit: Some(0),
                        };
                    }
                }
            }
            
            if iteration % 10000 == 0 {
                println!("Iteration {}: {} keys tested", iteration, keys_tested);
            }
        }
        
        SearchResult {
            found: false,
            private_key: None,
            public_key: None,
            address: None,
            search_time_ms: start_time.elapsed().as_millis() as u64,
            keys_tested,
            hot_zone_hit: None,
        }
    }
}

/// Create default hot zones based on analysis
pub fn create_default_hot_zones() -> Vec<HotZone> {
    vec![
        HotZone {
            start_percent: 30.0,
            end_percent: 40.0,
            probability: 0.12,
            priority: 2,
        },
        HotZone {
            start_percent: 40.0,
            end_percent: 50.0,
            probability: 0.12,
            priority: 2,
        },
        HotZone {
            start_percent: 60.0,
            end_percent: 70.0,
            probability: 0.173,
            priority: 1,
        },
        HotZone {
            start_percent: 90.0,
            end_percent: 100.0,
            probability: 0.12,
            priority: 2,
        },
    ]
}

/// Create default puzzle configuration
pub fn create_default_config(puzzle_number: u32) -> PuzzleConfig {
    PuzzleConfig {
        puzzle_number,
        hot_zones: create_default_hot_zones(),
        search_strategy: SearchStrategy::HotZoneTargeted,
        parallel_workers: num_cpus::get(),
        batch_size: 10000,
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn bitcoin_puzzle_solver_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBitcoinPuzzleSolver>()?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyclass]
struct PyBitcoinPuzzleSolver {
    solver: BitcoinPuzzleSolver,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBitcoinPuzzleSolver {
    #[new]
    fn new(puzzle_number: u32) -> Self {
        let config = create_default_config(puzzle_number);
        Self {
            solver: BitcoinPuzzleSolver::new(config),
        }
    }
    
    fn add_target_address(&mut self, puzzle_number: u32, address: String) {
        self.solver.add_target_address(puzzle_number, address);
    }
    
    fn search(&self, puzzle_number: u32, max_iterations: u64) -> String {
        #[cfg(feature = "parallel")]
        let result = self.solver.parallel_search(puzzle_number, max_iterations);
        
        #[cfg(not(feature = "parallel"))]
        let result = self.solver.sequential_search(puzzle_number, max_iterations);
        
        serde_json::to_string(&result).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_range_calculation() {
        let config = create_default_config(10);
        let solver = BitcoinPuzzleSolver::new(config);
        
        let (start, end) = solver.calculate_search_range(10);
        assert_eq!(start, BigUint::from(512u32));
        assert_eq!(end, BigUint::from(1023u32));
    }

    #[test]
    fn test_hot_zone_generation() {
        let config = create_default_config(10);
        let solver = BitcoinPuzzleSolver::new(config);
        
        let hot_ranges = solver.generate_hot_zone_ranges(10);
        assert_eq!(hot_ranges.len(), 4);
    }

    #[test]
    fn test_candidate_generation() {
        let config = create_default_config(10);
        let solver = BitcoinPuzzleSolver::new(config);
        
        let candidates = solver.generate_hot_zone_candidates(10, 100);
        assert_eq!(candidates.len(), 100);
        
        // Verify candidates are within valid range
        let (start, end) = solver.calculate_search_range(10);
        for candidate in candidates {
            assert!(candidate >= start && candidate <= end);
        }
    }
}


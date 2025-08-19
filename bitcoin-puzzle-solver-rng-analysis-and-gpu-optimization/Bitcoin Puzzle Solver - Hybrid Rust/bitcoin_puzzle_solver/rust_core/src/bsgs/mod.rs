//! Baby-step Giant-step (BSGS) algorithm implementation
//! 
//! High-performance BSGS algorithm with GLV endomorphism optimization and ML-guided
//! search space reduction for Bitcoin puzzle solving.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use num_bigint::BigUint;
use num_traits::{Zero, One};
use rayon::prelude::*;

use crate::crypto::{CryptoError, PrivateKey, PublicKey, BitcoinAddress, Secp256k1Engine};
use crate::puzzle::{SearchResult, PuzzleRange};

pub mod glv_endomorphism;
pub mod search_window;
pub mod cache_manager;

pub use glv_endomorphism::*;
pub use search_window::*;
pub use cache_manager::*;

/// BSGS algorithm error types
#[derive(thiserror::Error, Debug)]
pub enum BSGSError {
    #[error("Invalid search window: {0}")]
    InvalidSearchWindow(String),
    
    #[error("Cache operation failed: {0}")]
    CacheError(String),
    
    #[error("GLV endomorphism error: {0}")]
    GLVError(String),
    
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    #[error("Crypto error: {0}")]
    Crypto(#[from] CryptoError),
}

/// BSGS algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BSGSConfig {
    /// Baby step table size (2^baby_step_bits)
    pub baby_step_bits: u32,
    
    /// Enable GLV endomorphism optimization
    pub use_glv_endomorphism: bool,
    
    /// Enable cache-aligned memory operations
    pub use_cache_alignment: bool,
    
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    
    /// Number of parallel threads
    pub thread_count: usize,
    
    /// Progress reporting interval (number of giant steps)
    pub progress_interval: u64,
    
    /// Enable detailed statistics
    pub collect_statistics: bool,
}

impl Default for BSGSConfig {
    fn default() -> Self {
        Self {
            baby_step_bits: 28, // 268M entries as mentioned in SPR
            use_glv_endomorphism: true,
            use_cache_alignment: true,
            max_memory_mb: 16384, // 16GB default
            thread_count: num_cpus::get(),
            progress_interval: 1000000, // Report every 1M giant steps
            collect_statistics: true,
        }
    }
}

/// Search window for BSGS algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchWindow {
    /// Start of the search range
    pub start: BigUint,
    
    /// End of the search range (exclusive)
    pub end: BigUint,
    
    /// Predicted center point from ML model
    pub predicted_center: Option<BigUint>,
    
    /// Confidence score from ML prediction (0.0 to 1.0)
    pub confidence: f64,
}

impl SearchWindow {
    /// Create a new search window
    pub fn new(start: BigUint, end: BigUint) -> Self {
        Self {
            start,
            end,
            predicted_center: None,
            confidence: 0.0,
        }
    }
    
    /// Create a search window from a puzzle range
    pub fn from_range(range: &PuzzleRange) -> Self {
        Self::new(range.start.clone(), range.end.clone())
    }
    
    /// Create a focused search window around a predicted center
    pub fn focused_window(center: BigUint, radius: u64, confidence: f64) -> Self {
        let radius_big = BigUint::from(radius);
        let start = if center > radius_big {
            center.clone() - &radius_big
        } else {
            BigUint::zero()
        };
        let end = center.clone() + &radius_big;
        
        Self {
            start,
            end,
            predicted_center: Some(center),
            confidence,
        }
    }
    
    /// Get the size of the search window
    pub fn size(&self) -> BigUint {
        &self.end - &self.start
    }
    
    /// Check if the window is valid
    pub fn is_valid(&self) -> bool {
        self.start < self.end
    }
    
    /// Create a subrange of this window
    pub fn subrange(&self, start_offset: u128, end_offset: u128) -> Result<SearchWindow, BSGSError> {
        let start_big = BigUint::from(start_offset);
        let end_big = BigUint::from(end_offset);
        
        if start_big >= end_big {
            return Err(BSGSError::InvalidSearchWindow("Invalid subrange offsets".to_string()));
        }
        
        let new_start = &self.start + start_big;
        let new_end = &self.start + end_big;
        
        if new_end > self.end {
            return Err(BSGSError::InvalidSearchWindow("Subrange exceeds window bounds".to_string()));
        }
        
        Ok(SearchWindow::new(new_start, new_end))
    }
}

/// BSGS algorithm statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BSGSStatistics {
    /// Total baby steps computed
    pub baby_steps_computed: u64,
    
    /// Total giant steps computed
    pub giant_steps_computed: u64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Memory usage in MB
    pub memory_usage_mb: usize,
    
    /// Computation time in seconds
    pub computation_time_secs: f64,
    
    /// Keys per second rate
    pub keys_per_second: f64,
    
    /// GLV endomorphism usage
    pub glv_operations: u64,
}

/// Baby step table entry
#[derive(Debug, Clone)]
struct BabyStepEntry {
    /// The private key value
    key: BigUint,
    
    /// The corresponding public key point (compressed)
    public_key: [u8; 33],
}

/// BSGS algorithm engine
pub struct BSGSEngine {
    config: BSGSConfig,
    secp_engine: Arc<Secp256k1Engine>,
    glv_engine: Option<GLVEndomorphism>,
    cache_manager: CacheManager,
    statistics: Arc<Mutex<BSGSStatistics>>,
    current_window: Option<SearchWindow>,
}

impl BSGSEngine {
    /// Create a new BSGS engine
    pub fn new(config: BSGSConfig, secp_engine: Arc<Secp256k1Engine>) -> Result<Self, BSGSError> {
        log::info!("Initializing BSGS engine with {} baby step bits", config.baby_step_bits);
        
        // Initialize GLV endomorphism if enabled
        let glv_engine = if config.use_glv_endomorphism {
            Some(GLVEndomorphism::new()?)
        } else {
            None
        };
        
        // Initialize cache manager
        let cache_manager = CacheManager::new(
            config.max_memory_mb,
            config.use_cache_alignment,
        )?;
        
        let statistics = Arc::new(Mutex::new(BSGSStatistics::default()));
        
        Ok(Self {
            config,
            secp_engine,
            glv_engine,
            cache_manager,
            statistics,
            current_window: None,
        })
    }
    
    /// Set the current search window
    pub fn set_search_window(&mut self, window: SearchWindow) -> Result<(), BSGSError> {
        if !window.is_valid() {
            return Err(BSGSError::InvalidSearchWindow("Invalid search window".to_string()));
        }
        
        log::info!("Setting search window: {} to {} (size: {})", 
                  window.start, window.end, window.size());
        
        self.current_window = Some(window);
        Ok(())
    }
    
    /// Execute BSGS search for the target address
    pub async fn search(&mut self, window: &SearchWindow, target_address: &BitcoinAddress) -> Result<SearchResult, BSGSError> {
        log::info!("Starting BSGS search for address: {}", target_address);
        
        let start_time = std::time::Instant::now();
        
        // Set the search window
        self.set_search_window(window.clone())?;
        
        // Calculate optimal baby step size
        let baby_step_size = self.calculate_optimal_baby_step_size(window)?;
        log::info!("Using baby step size: 2^{} = {}", self.config.baby_step_bits, baby_step_size);
        
        // Build baby step table
        let baby_table = self.build_baby_step_table(window, baby_step_size).await?;
        log::info!("Baby step table built with {} entries", baby_table.len());
        
        // Execute giant steps
        let result = self.execute_giant_steps(window, &baby_table, target_address).await?;
        
        // Update statistics
        if self.config.collect_statistics {
            let mut stats = self.statistics.lock().unwrap();
            stats.computation_time_secs = start_time.elapsed().as_secs_f64();
            stats.keys_per_second = (stats.baby_steps_computed + stats.giant_steps_computed) as f64 / stats.computation_time_secs;
        }
        
        if result.found {
            log::info!("BSGS search completed successfully in {:.2}s", start_time.elapsed().as_secs_f64());
        } else {
            log::info!("BSGS search completed without finding solution in {:.2}s", start_time.elapsed().as_secs_f64());
        }
        
        Ok(result)
    }
    
    /// Calculate optimal baby step size based on search window
    fn calculate_optimal_baby_step_size(&self, window: &SearchWindow) -> Result<BigUint, BSGSError> {
        let window_size = window.size();
        let baby_step_size = BigUint::one() << self.config.baby_step_bits;
        
        // Ensure baby step size doesn't exceed window size
        if baby_step_size > window_size {
            let optimal_bits = window_size.bits() / 2;
            Ok(BigUint::one() << optimal_bits)
        } else {
            Ok(baby_step_size)
        }
    }
    
    /// Build the baby step table
    async fn build_baby_step_table(&mut self, window: &SearchWindow, baby_step_size: BigUint) -> Result<HashMap<[u8; 33], BigUint>, BSGSError> {
        log::info!("Building baby step table...");
        
        let mut baby_table = HashMap::new();
        let mut current_key = window.start.clone();
        let mut steps = 0u64;
        
        // Use parallel processing for baby steps
        let chunk_size = 10000; // Process in chunks for memory efficiency
        
        while &current_key < &window.end && steps < baby_step_size.to_u64_digits()[0] {
            let chunk_end = std::cmp::min(
                &current_key + BigUint::from(chunk_size),
                window.end.clone()
            );
            
            // Process chunk in parallel
            let chunk_results: Result<Vec<_>, _> = (0..chunk_size)
                .into_par_iter()
                .map(|i| {
                    let key_value = &current_key + BigUint::from(i);
                    if key_value >= chunk_end {
                        return Ok(None);
                    }
                    
                    // Create private key and derive public key
                    let private_key = PrivateKey::from_bigint(&key_value)
                        .map_err(BSGSError::Crypto)?;
                    let public_key = self.secp_engine.derive_public_key(&private_key)
                        .map_err(BSGSError::Crypto)?;
                    
                    Ok(Some((public_key.compressed_bytes(), key_value)))
                })
                .collect();
            
            // Add results to baby table
            for result in chunk_results? {
                if let Some((pubkey_bytes, key_value)) = result {
                    baby_table.insert(pubkey_bytes, key_value);
                    steps += 1;
                }
            }
            
            current_key = chunk_end;
            
            // Report progress
            if steps % 100000 == 0 {
                log::debug!("Baby steps progress: {} / {}", steps, baby_step_size);
            }
        }
        
        // Update statistics
        if self.config.collect_statistics {
            let mut stats = self.statistics.lock().unwrap();
            stats.baby_steps_computed = steps;
        }
        
        log::info!("Baby step table completed with {} entries", baby_table.len());
        Ok(baby_table)
    }
    
    /// Execute giant steps to find the solution
    async fn execute_giant_steps(
        &mut self,
        window: &SearchWindow,
        baby_table: &HashMap<[u8; 33], BigUint>,
        target_address: &BitcoinAddress,
    ) -> Result<SearchResult, BSGSError> {
        log::info!("Executing giant steps...");
        
        let baby_step_size = BigUint::one() << self.config.baby_step_bits;
        let mut giant_steps = 0u64;
        
        // Start from the beginning of the window
        let mut current_giant = window.start.clone();
        
        while &current_giant < &window.end {
            // Calculate the giant step point
            let giant_private_key = PrivateKey::from_bigint(&current_giant)
                .map_err(BSGSError::Crypto)?;
            let giant_public_key = self.secp_engine.derive_public_key(&giant_private_key)
                .map_err(BSGSError::Crypto)?;
            
            // Apply GLV endomorphism if enabled
            let search_points = if let Some(ref glv) = self.glv_engine {
                glv.generate_search_points(&giant_public_key)?
            } else {
                vec![giant_public_key.compressed_bytes()]
            };
            
            // Check each search point against baby table
            for point_bytes in search_points {
                if let Some(baby_key) = baby_table.get(&point_bytes) {
                    // Found a collision! Calculate the actual private key
                    let solution_key = &current_giant + baby_key;
                    
                    // Verify the solution
                    let solution_private_key = PrivateKey::from_bigint(&solution_key)
                        .map_err(BSGSError::Crypto)?;
                    
                    if self.secp_engine.verify_solution(&solution_private_key, target_address)? {
                        log::info!("Solution found: {}", hex::encode(&solution_private_key.to_bytes()));
                        return Ok(SearchResult::found(solution_private_key));
                    }
                }
            }
            
            // Move to next giant step
            current_giant += &baby_step_size;
            giant_steps += 1;
            
            // Report progress
            if giant_steps % self.config.progress_interval == 0 {
                log::info!("Giant steps progress: {} (current key: {})", giant_steps, current_giant);
            }
            
            // Update statistics
            if self.config.collect_statistics {
                let mut stats = self.statistics.lock().unwrap();
                stats.giant_steps_computed = giant_steps;
                if let Some(ref glv) = self.glv_engine {
                    stats.glv_operations += search_points.len() as u64;
                }
            }
        }
        
        log::info!("Giant steps completed: {} steps executed", giant_steps);
        Ok(SearchResult::not_found())
    }
    
    /// Clear internal caches
    pub fn clear_cache(&mut self) -> Result<(), BSGSError> {
        self.cache_manager.clear_all()?;
        Ok(())
    }
    
    /// Get current statistics
    pub fn get_statistics(&self) -> BSGSStatistics {
        self.statistics.lock().unwrap().clone()
    }
    
    /// Estimate memory usage for given configuration
    pub fn estimate_memory_usage(&self, window: &SearchWindow) -> usize {
        let baby_step_size = BigUint::one() << self.config.baby_step_bits;
        let entry_size = 33 + 32; // Public key + private key
        let table_size = baby_step_size.to_u64_digits()[0] as usize * entry_size;
        
        // Add overhead for HashMap and other structures
        table_size + (table_size / 4) // 25% overhead estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::Secp256k1Engine;
    
    #[tokio::test]
    async fn test_bsgs_engine_creation() {
        let config = BSGSConfig::default();
        let secp_engine = Arc::new(Secp256k1Engine::new().unwrap());
        let engine = BSGSEngine::new(config, secp_engine);
        assert!(engine.is_ok());
    }
    
    #[test]
    fn test_search_window() {
        let start = BigUint::from(100u32);
        let end = BigUint::from(200u32);
        let window = SearchWindow::new(start.clone(), end.clone());
        
        assert_eq!(window.start, start);
        assert_eq!(window.end, end);
        assert_eq!(window.size(), BigUint::from(100u32));
        assert!(window.is_valid());
    }
    
    #[test]
    fn test_focused_window() {
        let center = BigUint::from(1000u32);
        let radius = 100u64;
        let window = SearchWindow::focused_window(center.clone(), radius, 0.8);
        
        assert_eq!(window.predicted_center, Some(center));
        assert_eq!(window.confidence, 0.8);
        assert_eq!(window.size(), BigUint::from(200u32));
    }
    
    #[test]
    fn test_memory_estimation() {
        let config = BSGSConfig {
            baby_step_bits: 20, // Smaller for testing
            ..Default::default()
        };
        let secp_engine = Arc::new(Secp256k1Engine::new().unwrap());
        let engine = BSGSEngine::new(config, secp_engine).unwrap();
        
        let window = SearchWindow::new(BigUint::from(1u32), BigUint::from(1000000u32));
        let memory_usage = engine.estimate_memory_usage(&window);
        assert!(memory_usage > 0);
    }
}


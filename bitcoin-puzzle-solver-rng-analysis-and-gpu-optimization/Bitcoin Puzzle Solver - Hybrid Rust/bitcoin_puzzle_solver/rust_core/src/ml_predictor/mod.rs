//! Machine Learning predictor module
//! 
//! Implements ML-guided hot zone prediction using Random Forest regression
//! to identify high-probability search areas for Bitcoin puzzle solving.

use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use num_bigint::BigUint;
use num_traits::{Zero, One};

use crate::bsgs::SearchWindow;

pub mod training_data;
pub mod feature_extraction;
pub mod prediction_model;
pub mod python_interface;

pub use training_data::*;
pub use feature_extraction::*;
pub use prediction_model::*;
pub use python_interface::*;

/// ML prediction error types
#[derive(thiserror::Error, Debug)]
pub enum MLError {
    #[error("Model not trained: {0}")]
    ModelNotTrained(String),
    
    #[error("Invalid training data: {0}")]
    InvalidTrainingData(String),
    
    #[error("Prediction failed: {0}")]
    PredictionFailed(String),
    
    #[error("Python interface error: {0}")]
    PythonInterface(String),
    
    #[error("Feature extraction error: {0}")]
    FeatureExtraction(String),
    
    #[error("Model serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("I/O error: {0}")]
    IO(#[from] std::io::Error),
}

/// Training data point for a solved puzzle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PuzzleDataPoint {
    /// Puzzle number (1-160)
    pub puzzle_number: u32,
    
    /// The actual private key that was found
    pub private_key: BigUint,
    
    /// The range start for this puzzle
    pub range_start: BigUint,
    
    /// The range end for this puzzle
    pub range_end: BigUint,
    
    /// Position within the range (0.0 to 1.0)
    pub position_percent: f64,
    
    /// Bitcoin address
    pub address: String,
    
    /// Date solved (if known)
    pub solved_date: Option<String>,
    
    /// Additional features extracted from the key
    pub features: HashMap<String, f64>,
}

impl PuzzleDataPoint {
    /// Create a new puzzle data point
    pub fn new(
        puzzle_number: u32,
        private_key: BigUint,
        range_start: BigUint,
        range_end: BigUint,
        address: String,
    ) -> Self {
        let range_size = &range_end - &range_start;
        let position = &private_key - &range_start;
        let position_percent = if range_size > BigUint::zero() {
            position.to_f64().unwrap_or(0.0) / range_size.to_f64().unwrap_or(1.0)
        } else {
            0.0
        };
        
        Self {
            puzzle_number,
            private_key,
            range_start,
            range_end,
            position_percent,
            address,
            solved_date: None,
            features: HashMap::new(),
        }
    }
    
    /// Extract features from the private key
    pub fn extract_features(&mut self) -> Result<(), MLError> {
        let feature_extractor = FeatureExtractor::new();
        self.features = feature_extractor.extract_key_features(&self.private_key)?;
        Ok(())
    }
}

/// Training dataset for the ML model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    /// All solved puzzle data points
    pub data_points: Vec<PuzzleDataPoint>,
    
    /// Feature names used in training
    pub feature_names: Vec<String>,
    
    /// Statistics about the training data
    pub statistics: TrainingStatistics,
}

impl TrainingData {
    /// Create new empty training data
    pub fn new() -> Self {
        Self {
            data_points: Vec::new(),
            feature_names: Vec::new(),
            statistics: TrainingStatistics::default(),
        }
    }
    
    /// Load training data from CSV file
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, MLError> {
        log::info!("Loading training data from CSV: {:?}", path.as_ref());
        
        let mut reader = csv::Reader::from_path(path)?;
        let mut data_points = Vec::new();
        
        for result in reader.deserialize() {
            let record: CSVPuzzleRecord = result
                .map_err(|e| MLError::InvalidTrainingData(e.to_string()))?;
            
            // Skip unsolved puzzles
            if record.status != "SOLVED" {
                continue;
            }
            
            // Parse private key
            let private_key = if record.private_key.starts_with("0x") {
                BigUint::parse_bytes(&record.private_key[2..].as_bytes(), 16)
            } else {
                BigUint::parse_bytes(record.private_key.as_bytes(), 16)
            }.ok_or_else(|| MLError::InvalidTrainingData(
                format!("Invalid private key: {}", record.private_key)
            ))?;
            
            // Calculate range bounds
            let range_start = BigUint::one() << (record.puzzle_number - 1);
            let range_end = BigUint::one() << record.puzzle_number;
            
            let mut data_point = PuzzleDataPoint::new(
                record.puzzle_number,
                private_key,
                range_start,
                range_end,
                record.bitcoin_address,
            );
            
            data_point.solved_date = Some(record.solved_date);
            data_point.extract_features()?;
            
            data_points.push(data_point);
        }
        
        log::info!("Loaded {} solved puzzles from CSV", data_points.len());
        
        let mut training_data = Self {
            data_points,
            feature_names: Vec::new(),
            statistics: TrainingStatistics::default(),
        };
        
        training_data.compute_statistics();
        Ok(training_data)
    }
    
    /// Compute statistics about the training data
    fn compute_statistics(&mut self) {
        if self.data_points.is_empty() {
            return;
        }
        
        // Extract feature names from first data point
        if let Some(first_point) = self.data_points.first() {
            self.feature_names = first_point.features.keys().cloned().collect();
            self.feature_names.sort();
        }
        
        // Compute position statistics
        let positions: Vec<f64> = self.data_points.iter()
            .map(|dp| dp.position_percent)
            .collect();
        
        self.statistics.mean_position = positions.iter().sum::<f64>() / positions.len() as f64;
        
        let variance = positions.iter()
            .map(|pos| (pos - self.statistics.mean_position).powi(2))
            .sum::<f64>() / positions.len() as f64;
        self.statistics.std_position = variance.sqrt();
        
        self.statistics.min_position = positions.iter().cloned().fold(f64::INFINITY, f64::min);
        self.statistics.max_position = positions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Compute puzzle number statistics
        let puzzle_numbers: Vec<u32> = self.data_points.iter()
            .map(|dp| dp.puzzle_number)
            .collect();
        
        self.statistics.min_puzzle = *puzzle_numbers.iter().min().unwrap_or(&0);
        self.statistics.max_puzzle = *puzzle_numbers.iter().max().unwrap_or(&0);
        self.statistics.total_puzzles = self.data_points.len();
        
        log::info!("Training data statistics: {} puzzles, position mean: {:.4}, std: {:.4}", 
                  self.statistics.total_puzzles, 
                  self.statistics.mean_position, 
                  self.statistics.std_position);
    }
    
    /// Get feature matrix for ML training
    pub fn get_feature_matrix(&self) -> Vec<Vec<f64>> {
        self.data_points.iter()
            .map(|dp| {
                let mut features = vec![dp.puzzle_number as f64];
                for feature_name in &self.feature_names {
                    features.push(dp.features.get(feature_name).copied().unwrap_or(0.0));
                }
                features
            })
            .collect()
    }
    
    /// Get target values (position percentages)
    pub fn get_targets(&self) -> Vec<f64> {
        self.data_points.iter()
            .map(|dp| dp.position_percent)
            .collect()
    }
}

/// Statistics about the training data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingStatistics {
    pub total_puzzles: usize,
    pub min_puzzle: u32,
    pub max_puzzle: u32,
    pub mean_position: f64,
    pub std_position: f64,
    pub min_position: f64,
    pub max_position: f64,
}

/// CSV record structure for puzzle data
#[derive(Debug, Deserialize)]
struct CSVPuzzleRecord {
    #[serde(rename = "#")]
    puzzle_number: u32,
    #[serde(rename = "Private Key")]
    private_key: String,
    #[serde(rename = "Bitcoin Address")]
    bitcoin_address: String,
    #[serde(rename = "Status")]
    status: String,
    #[serde(rename = "Solved")]
    solved_date: String,
}

/// Hot zone prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotZonePrediction {
    /// Predicted center position (0.0 to 1.0)
    pub predicted_position: f64,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    
    /// Search window around the prediction
    pub search_window: SearchWindow,
    
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

/// Main ML predictor for hot zone identification
pub struct HotZonePredictor {
    /// Trained prediction model
    model: Option<PredictionModel>,
    
    /// Training data used to train the model
    training_data: Option<TrainingData>,
    
    /// Python interface for ML operations
    python_interface: PythonMLInterface,
    
    /// Feature extractor
    feature_extractor: FeatureExtractor,
}

impl HotZonePredictor {
    /// Create a new hot zone predictor
    pub async fn new() -> Result<Self, MLError> {
        log::info!("Initializing hot zone predictor");
        
        let python_interface = PythonMLInterface::new().await?;
        let feature_extractor = FeatureExtractor::new();
        
        Ok(Self {
            model: None,
            training_data: None,
            python_interface,
            feature_extractor,
        })
    }
    
    /// Load and train the model with puzzle data
    pub async fn train_model<P: AsRef<Path>>(&mut self, csv_path: P) -> Result<(), MLError> {
        log::info!("Training ML model for hot zone prediction");
        
        // Load training data
        let training_data = TrainingData::from_csv(csv_path)?;
        log::info!("Loaded training data with {} solved puzzles", training_data.data_points.len());
        
        // Train the model using Python interface
        let model = self.python_interface.train_random_forest(&training_data).await?;
        
        self.model = Some(model);
        self.training_data = Some(training_data);
        
        log::info!("ML model training completed successfully");
        Ok(())
    }
    
    /// Predict hot zones for a given puzzle number
    pub async fn predict_hot_zones(&self, puzzle_number: u32) -> Result<Vec<SearchWindow>, MLError> {
        let model = self.model.as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Model must be trained before prediction".to_string()))?;
        
        log::info!("Predicting hot zones for puzzle #{}", puzzle_number);
        
        // Create feature vector for prediction
        let features = vec![puzzle_number as f64]; // Basic feature: just puzzle number
        
        // Get prediction from model
        let prediction = self.python_interface.predict(&model, &features).await?;
        
        // Calculate search range for this puzzle
        let range_start = BigUint::one() << (puzzle_number - 1);
        let range_end = BigUint::one() << puzzle_number;
        let range_size = &range_end - &range_start;
        
        // Calculate predicted center
        let predicted_offset = range_size.to_f64().unwrap_or(0.0) * prediction.predicted_position;
        let predicted_center = &range_start + BigUint::from(predicted_offset as u64);
        
        // Create focused search window (2^40 keys around center as per SPR)
        let search_radius = 1u64 << 40; // 1 trillion keys
        let search_window = SearchWindow::focused_window(
            predicted_center,
            search_radius,
            prediction.confidence,
        );
        
        log::info!("Predicted hot zone: center at {:.2}% of range, confidence: {:.2}", 
                  prediction.predicted_position * 100.0, prediction.confidence);
        
        Ok(vec![search_window])
    }
    
    /// Predict multiple hot zones with different strategies
    pub async fn predict_multiple_zones(&self, puzzle_number: u32, num_zones: usize) -> Result<Vec<SearchWindow>, MLError> {
        let model = self.model.as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Model must be trained before prediction".to_string()))?;
        
        log::info!("Predicting {} hot zones for puzzle #{}", num_zones, puzzle_number);
        
        let mut zones = Vec::new();
        
        // Primary prediction
        let primary_zones = self.predict_hot_zones(puzzle_number).await?;
        zones.extend(primary_zones);
        
        // Add additional zones based on statistical analysis
        if num_zones > 1 {
            let training_data = self.training_data.as_ref().unwrap();
            
            // Find similar puzzles for additional predictions
            let similar_puzzles: Vec<&PuzzleDataPoint> = training_data.data_points.iter()
                .filter(|dp| {
                    let diff = (dp.puzzle_number as i32 - puzzle_number as i32).abs();
                    diff <= 5 // Consider puzzles within 5 numbers as similar
                })
                .collect();
            
            if !similar_puzzles.is_empty() {
                // Create zones based on similar puzzle positions
                let range_start = BigUint::one() << (puzzle_number - 1);
                let range_end = BigUint::one() << puzzle_number;
                let range_size = &range_end - &range_start;
                
                for (i, similar_puzzle) in similar_puzzles.iter().take(num_zones - 1).enumerate() {
                    let center_offset = range_size.to_f64().unwrap_or(0.0) * similar_puzzle.position_percent;
                    let center = &range_start + BigUint::from(center_offset as u64);
                    
                    let confidence = 0.5 - (i as f64 * 0.1); // Decreasing confidence
                    let search_radius = 1u64 << 39; // Smaller radius for secondary zones
                    
                    let zone = SearchWindow::focused_window(center, search_radius, confidence);
                    zones.push(zone);
                }
            }
        }
        
        log::info!("Generated {} hot zones for puzzle #{}", zones.len(), puzzle_number);
        Ok(zones)
    }
    
    /// Evaluate model performance on test data
    pub async fn evaluate_model(&self, test_data: &TrainingData) -> Result<ModelEvaluation, MLError> {
        let model = self.model.as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Model must be trained before evaluation".to_string()))?;
        
        log::info!("Evaluating model performance on {} test samples", test_data.data_points.len());
        
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();
        
        for data_point in &test_data.data_points {
            let features = vec![data_point.puzzle_number as f64];
            let prediction = self.python_interface.predict(&model, &features).await?;
            
            predictions.push(prediction.predicted_position);
            actuals.push(data_point.position_percent);
        }
        
        // Calculate evaluation metrics
        let evaluation = ModelEvaluation::calculate(&predictions, &actuals);
        
        log::info!("Model evaluation: MAE: {:.4}, RMSE: {:.4}, R²: {:.4}", 
                  evaluation.mean_absolute_error, 
                  evaluation.root_mean_square_error, 
                  evaluation.r_squared);
        
        Ok(evaluation)
    }
    
    /// Save the trained model to disk
    pub async fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), MLError> {
        let model = self.model.as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("No model to save".to_string()))?;
        
        self.python_interface.save_model(model, path).await?;
        log::info!("Model saved to: {:?}", path.as_ref());
        Ok(())
    }
    
    /// Load a trained model from disk
    pub async fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<(), MLError> {
        let model = self.python_interface.load_model(path).await?;
        self.model = Some(model);
        log::info!("Model loaded from: {:?}", path.as_ref());
        Ok(())
    }
}

/// Model evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvaluation {
    pub mean_absolute_error: f64,
    pub root_mean_square_error: f64,
    pub r_squared: f64,
    pub max_error: f64,
    pub predictions_within_10_percent: f64,
}

impl ModelEvaluation {
    /// Calculate evaluation metrics from predictions and actual values
    pub fn calculate(predictions: &[f64], actuals: &[f64]) -> Self {
        assert_eq!(predictions.len(), actuals.len());
        let n = predictions.len() as f64;
        
        // Mean Absolute Error
        let mae = predictions.iter()
            .zip(actuals.iter())
            .map(|(pred, actual)| (pred - actual).abs())
            .sum::<f64>() / n;
        
        // Root Mean Square Error
        let mse = predictions.iter()
            .zip(actuals.iter())
            .map(|(pred, actual)| (pred - actual).powi(2))
            .sum::<f64>() / n;
        let rmse = mse.sqrt();
        
        // R-squared
        let actual_mean = actuals.iter().sum::<f64>() / n;
        let ss_tot = actuals.iter()
            .map(|actual| (actual - actual_mean).powi(2))
            .sum::<f64>();
        let ss_res = predictions.iter()
            .zip(actuals.iter())
            .map(|(pred, actual)| (actual - pred).powi(2))
            .sum::<f64>();
        let r_squared = 1.0 - (ss_res / ss_tot);
        
        // Maximum error
        let max_error = predictions.iter()
            .zip(actuals.iter())
            .map(|(pred, actual)| (pred - actual).abs())
            .fold(0.0, f64::max);
        
        // Predictions within 10% accuracy
        let within_10_percent = predictions.iter()
            .zip(actuals.iter())
            .filter(|(pred, actual)| (pred - actual).abs() < 0.1)
            .count() as f64 / n;
        
        Self {
            mean_absolute_error: mae,
            root_mean_square_error: rmse,
            r_squared,
            max_error,
            predictions_within_10_percent: within_10_percent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_puzzle_data_point() {
        let private_key = BigUint::from(12345u32);
        let range_start = BigUint::from(10000u32);
        let range_end = BigUint::from(20000u32);
        
        let data_point = PuzzleDataPoint::new(
            71,
            private_key,
            range_start,
            range_end,
            "1Address123".to_string(),
        );
        
        assert_eq!(data_point.puzzle_number, 71);
        assert!(data_point.position_percent > 0.0 && data_point.position_percent < 1.0);
    }
    
    #[test]
    fn test_training_statistics() {
        let mut training_data = TrainingData::new();
        
        // Add some test data points
        for i in 1..=5 {
            let private_key = BigUint::from(i * 1000u32);
            let range_start = BigUint::from(0u32);
            let range_end = BigUint::from(10000u32);
            
            let data_point = PuzzleDataPoint::new(
                i,
                private_key,
                range_start,
                range_end,
                format!("1Address{}", i),
            );
            
            training_data.data_points.push(data_point);
        }
        
        training_data.compute_statistics();
        
        assert_eq!(training_data.statistics.total_puzzles, 5);
        assert_eq!(training_data.statistics.min_puzzle, 1);
        assert_eq!(training_data.statistics.max_puzzle, 5);
    }
    
    #[test]
    fn test_model_evaluation() {
        let predictions = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let actuals = vec![0.12, 0.18, 0.32, 0.38, 0.52];
        
        let evaluation = ModelEvaluation::calculate(&predictions, &actuals);
        
        assert!(evaluation.mean_absolute_error > 0.0);
        assert!(evaluation.root_mean_square_error > 0.0);
        assert!(evaluation.r_squared > 0.0);
    }
}


"""
Statistical and Probabilistic Optimizations (31-35)

This module implements advanced statistical analysis and probabilistic modeling
techniques for optimizing Bitcoin puzzle solving strategies.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize, differential_evolution
from scipy.special import gamma, digamma
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import networkx as nx
import pymc3 as pm
import arviz as az
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianInferenceEngine:
    """
    Optimization 31: Bayesian Inference for Key Prediction
    
    Implements Bayesian networks and MCMC sampling for probabilistic
    key prediction with uncertainty quantification.
    """
    
    def __init__(self, puzzle_data: pd.DataFrame):
        self.puzzle_data = puzzle_data
        self.bayesian_network = None
        self.mcmc_trace = None
        self.model = None
        
    def build_bayesian_network(self) -> nx.DiGraph:
        """Build Bayesian network representing key dependencies"""
        logger.info("Building Bayesian network for key dependencies...")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes for different key features
        nodes = [
            'puzzle_number',
            'key_position_pct',
            'address_entropy',
            'bit_pattern_complexity',
            'generation_timestamp',
            'key_strength'
        ]
        
        G.add_nodes_from(nodes)
        
        # Add edges based on conditional dependencies
        edges = [
            ('puzzle_number', 'key_position_pct'),
            ('puzzle_number', 'key_strength'),
            ('key_position_pct', 'address_entropy'),
            ('bit_pattern_complexity', 'key_strength'),
            ('generation_timestamp', 'bit_pattern_complexity')
        ]
        
        G.add_edges_from(edges)
        
        self.bayesian_network = G
        logger.info(f"Bayesian network built with {len(nodes)} nodes and {len(edges)} edges")
        
        return G
    
    def fit_bayesian_model(self, target_puzzle: int):
        """Fit Bayesian model using PyMC3"""
        logger.info(f"Fitting Bayesian model for puzzle {target_puzzle}...")
        
        # Prepare data
        solved_puzzles = self.puzzle_data[self.puzzle_data['private_key'].notna()]
        
        # Extract features
        puzzle_numbers = solved_puzzles['puzzle'].values
        key_positions = self._calculate_key_positions(solved_puzzles)
        
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            beta = pm.Normal('beta', mu=0, sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Linear relationship
            mu = alpha + beta * puzzle_numbers
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=key_positions)
            
            # Posterior sampling
            trace = pm.sample(2000, tune=1000, return_inferencedata=True)
            
        self.model = model
        self.mcmc_trace = trace
        
        # Posterior predictive check
        with model:
            posterior_pred = pm.sample_posterior_predictive(trace, samples=1000)
        
        logger.info("Bayesian model fitting completed")
        return trace
    
    def predict_key_position(self, puzzle_number: int, num_samples: int = 1000) -> Dict[str, float]:
        """Predict key position with uncertainty quantification"""
        if self.model is None or self.mcmc_trace is None:
            raise ValueError("Model must be fitted before prediction")
        
        with self.model:
            # Posterior predictive sampling
            pm.set_data({'puzzle_numbers': [puzzle_number]})
            posterior_pred = pm.sample_posterior_predictive(
                self.mcmc_trace, 
                samples=num_samples,
                var_names=['y_obs']
            )
        
        predictions = posterior_pred['y_obs'].flatten()
        
        return {
            'mean_position_pct': float(np.mean(predictions)),
            'std_position_pct': float(np.std(predictions)),
            'credible_interval_95': [
                float(np.percentile(predictions, 2.5)),
                float(np.percentile(predictions, 97.5))
            ],
            'credible_interval_68': [
                float(np.percentile(predictions, 16)),
                float(np.percentile(predictions, 84))
            ]
        }
    
    def _calculate_key_positions(self, puzzle_data: pd.DataFrame) -> np.ndarray:
        """Calculate key position percentages"""
        positions = []
        
        for _, row in puzzle_data.iterrows():
            puzzle_num = row['puzzle']
            private_key = row['private_key']
            
            if pd.notna(private_key):
                start_range = 2**(puzzle_num - 1)
                end_range = 2**puzzle_num - 1
                key_int = int(private_key, 16)
                position_pct = (key_int - start_range) / (end_range - start_range) * 100
                positions.append(position_pct)
        
        return np.array(positions)
    
    def compute_mutual_information(self, features: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute mutual information between features and target"""
        mi_scores = {}
        
        for i, feature_name in enumerate(['puzzle_num', 'entropy', 'complexity']):
            if i < features.shape[1]:
                mi_score = mutual_info_score(features[:, i], target)
                mi_scores[feature_name] = mi_score
        
        return mi_scores

class ExtremeValueAnalyzer:
    """
    Optimization 32: Extreme Value Theory for Tail Event Analysis
    
    Applies extreme value theory to model rare solution discovery events
    and optimize search strategies for tail regions.
    """
    
    def __init__(self):
        self.gev_params = None
        self.pot_params = None
        self.threshold = None
        
    def fit_generalized_extreme_value(self, data: np.ndarray) -> Dict[str, float]:
        """Fit Generalized Extreme Value distribution"""
        logger.info("Fitting Generalized Extreme Value distribution...")
        
        # Block maxima approach
        block_size = max(10, len(data) // 20)
        blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]
        block_maxima = [np.max(block) for block in blocks if len(block) == block_size]
        
        # Fit GEV distribution
        shape, loc, scale = stats.genextreme.fit(block_maxima)
        
        self.gev_params = {
            'shape': shape,
            'location': loc,
            'scale': scale
        }
        
        logger.info(f"GEV parameters: shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")
        
        return self.gev_params
    
    def fit_peaks_over_threshold(self, data: np.ndarray, threshold_percentile: float = 95) -> Dict[str, float]:
        """Fit Peaks Over Threshold model using Generalized Pareto Distribution"""
        logger.info("Fitting Peaks Over Threshold model...")
        
        # Determine threshold
        self.threshold = np.percentile(data, threshold_percentile)
        
        # Extract exceedances
        exceedances = data[data > self.threshold] - self.threshold
        
        if len(exceedances) < 10:
            logger.warning("Insufficient exceedances for reliable POT fitting")
            return {}
        
        # Fit Generalized Pareto Distribution
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
        
        self.pot_params = {
            'shape': shape,
            'scale': scale,
            'threshold': self.threshold,
            'num_exceedances': len(exceedances),
            'exceedance_rate': len(exceedances) / len(data)
        }
        
        logger.info(f"POT parameters: shape={shape:.4f}, scale={scale:.4f}, threshold={self.threshold:.4f}")
        
        return self.pot_params
    
    def estimate_return_levels(self, return_periods: List[int]) -> Dict[int, float]:
        """Estimate return levels for given return periods"""
        if self.gev_params is None:
            raise ValueError("GEV model must be fitted first")
        
        return_levels = {}
        shape, loc, scale = self.gev_params['shape'], self.gev_params['location'], self.gev_params['scale']
        
        for T in return_periods:
            if abs(shape) < 1e-6:  # Gumbel case
                level = loc - scale * np.log(-np.log(1 - 1/T))
            else:  # Frechet or Weibull case
                level = loc + (scale / shape) * ((-np.log(1 - 1/T))**(-shape) - 1)
            
            return_levels[T] = level
        
        return return_levels
    
    def estimate_exceedance_probabilities(self, levels: List[float]) -> Dict[float, float]:
        """Estimate exceedance probabilities using POT model"""
        if self.pot_params is None:
            raise ValueError("POT model must be fitted first")
        
        exceedance_probs = {}
        shape, scale = self.pot_params['shape'], self.pot_params['scale']
        threshold = self.pot_params['threshold']
        exceedance_rate = self.pot_params['exceedance_rate']
        
        for level in levels:
            if level <= threshold:
                # Use empirical probability for levels below threshold
                prob = 1.0
            else:
                # Use GPD for levels above threshold
                excess = level - threshold
                if abs(shape) < 1e-6:  # Exponential case
                    gpd_prob = np.exp(-excess / scale)
                else:
                    gpd_prob = (1 + shape * excess / scale)**(-1/shape)
                
                prob = exceedance_rate * gpd_prob
            
            exceedance_probs[level] = prob
        
        return exceedance_probs
    
    def optimize_search_thresholds(self, computational_budget: float) -> Dict[str, float]:
        """Optimize search thresholds based on extreme value analysis"""
        if self.pot_params is None:
            raise ValueError("POT model must be fitted first")
        
        # Define optimization objective
        def objective(threshold_pct):
            threshold = np.percentile(np.linspace(0, 100, 1000), threshold_pct[0])
            prob = self.estimate_exceedance_probabilities([threshold])[threshold]
            
            # Balance between probability and computational cost
            cost = computational_budget * (1 - threshold_pct[0] / 100)
            benefit = prob * 1000  # Scale factor for benefit
            
            return -(benefit / cost)  # Minimize negative utility
        
        # Optimize threshold
        result = minimize(objective, x0=[95], bounds=[(80, 99.9)], method='L-BFGS-B')
        
        optimal_threshold_pct = result.x[0]
        optimal_threshold = np.percentile(np.linspace(0, 100, 1000), optimal_threshold_pct)
        
        return {
            'optimal_threshold_percentile': optimal_threshold_pct,
            'optimal_threshold_value': optimal_threshold,
            'expected_success_rate': self.estimate_exceedance_probabilities([optimal_threshold])[optimal_threshold]
        }

class InformationTheoryAnalyzer:
    """
    Optimization 33: Information Theory for Feature Selection
    
    Uses mutual information, entropy analysis, and information-theoretic
    measures to identify the most informative features for puzzle solving.
    """
    
    def __init__(self):
        self.feature_importance = {}
        self.entropy_analysis = {}
        
    def compute_shannon_entropy(self, data: np.ndarray, bins: int = 50) -> float:
        """Compute Shannon entropy of data"""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def compute_conditional_entropy(self, X: np.ndarray, Y: np.ndarray, bins: int = 50) -> float:
        """Compute conditional entropy H(Y|X)"""
        # Discretize continuous variables
        X_discrete = pd.cut(X, bins=bins, labels=False)
        Y_discrete = pd.cut(Y, bins=bins, labels=False)
        
        # Compute joint and marginal probabilities
        joint_counts = pd.crosstab(X_discrete, Y_discrete)
        joint_probs = joint_counts / joint_counts.sum().sum()
        
        marginal_x = joint_probs.sum(axis=1)
        
        conditional_entropy = 0
        for x_val in joint_probs.index:
            if marginal_x[x_val] > 0:
                conditional_probs = joint_probs.loc[x_val] / marginal_x[x_val]
                conditional_probs = conditional_probs[conditional_probs > 0]
                h_y_given_x = -np.sum(conditional_probs * np.log2(conditional_probs))
                conditional_entropy += marginal_x[x_val] * h_y_given_x
        
        return conditional_entropy
    
    def compute_mutual_information_matrix(self, features: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Compute mutual information matrix between all features"""
        n_features = features.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    mi_matrix[i, j] = self.compute_shannon_entropy(features[:, i])
                else:
                    mi_matrix[i, j] = mutual_info_score(
                        pd.cut(features[:, i], bins=20, labels=False),
                        pd.cut(features[:, j], bins=20, labels=False)
                    )
        
        return pd.DataFrame(mi_matrix, index=feature_names, columns=feature_names)
    
    def analyze_key_randomness(self, private_keys: List[str]) -> Dict[str, float]:
        """Analyze randomness of private keys using information theory"""
        randomness_metrics = {}
        
        # Convert keys to bit sequences
        bit_sequences = []
        for key in private_keys:
            if pd.notna(key):
                key_int = int(key, 16)
                bit_seq = format(key_int, '0256b')
                bit_sequences.append([int(b) for b in bit_seq])
        
        if not bit_sequences:
            return randomness_metrics
        
        bit_array = np.array(bit_sequences)
        
        # Bit-wise entropy
        bit_entropies = []
        for i in range(bit_array.shape[1]):
            bit_column = bit_array[:, i]
            p1 = np.mean(bit_column)
            p0 = 1 - p1
            
            if p0 > 0 and p1 > 0:
                entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
            else:
                entropy = 0
            
            bit_entropies.append(entropy)
        
        randomness_metrics['mean_bit_entropy'] = np.mean(bit_entropies)
        randomness_metrics['min_bit_entropy'] = np.min(bit_entropies)
        randomness_metrics['entropy_variance'] = np.var(bit_entropies)
        
        # Sequential correlation
        correlations = []
        for lag in range(1, min(32, bit_array.shape[1])):
            for i in range(bit_array.shape[1] - lag):
                corr = np.corrcoef(bit_array[:, i], bit_array[:, i + lag])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        randomness_metrics['mean_sequential_correlation'] = np.mean(correlations) if correlations else 0
        randomness_metrics['max_sequential_correlation'] = np.max(correlations) if correlations else 0
        
        # Kolmogorov complexity approximation (compression ratio)
        import zlib
        complexity_ratios = []
        for bit_seq in bit_sequences:
            bit_string = ''.join(map(str, bit_seq))
            compressed = zlib.compress(bit_string.encode())
            ratio = len(compressed) / len(bit_string)
            complexity_ratios.append(ratio)
        
        randomness_metrics['mean_compression_ratio'] = np.mean(complexity_ratios)
        randomness_metrics['min_compression_ratio'] = np.min(complexity_ratios)
        
        return randomness_metrics
    
    def select_informative_features(self, X: np.ndarray, y: np.ndarray, 
                                  feature_names: List[str], top_k: int = 10) -> List[str]:
        """Select most informative features using mutual information"""
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, mi_scores))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        self.feature_importance = dict(sorted_features)
        
        # Return top k features
        return [feature for feature, _ in sorted_features[:top_k]]
    
    def compute_information_gain(self, X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
        """Compute information gain for a specific feature"""
        # Entropy of target
        h_y = self.compute_shannon_entropy(y)
        
        # Conditional entropy H(Y|X)
        h_y_given_x = self.compute_conditional_entropy(X[:, feature_idx], y)
        
        # Information gain
        information_gain = h_y - h_y_given_x
        
        return information_gain

class SurvivalAnalyzer:
    """
    Optimization 34: Survival Analysis for Search Time Modeling
    
    Models the time required to find puzzle solutions using survival analysis
    techniques, accounting for censoring and competing risks.
    """
    
    def __init__(self):
        self.km_fitter = KaplanMeierFitter()
        self.cox_fitter = CoxPHFitter()
        self.survival_data = None
        
    def prepare_survival_data(self, search_logs: pd.DataFrame) -> pd.DataFrame:
        """Prepare survival analysis data from search logs"""
        survival_data = []
        
        for search_id in search_logs['search_id'].unique():
            search_data = search_logs[search_logs['search_id'] == search_id]
            
            # Calculate search duration
            start_time = search_data['timestamp'].min()
            end_time = search_data['timestamp'].max()
            duration = (end_time - start_time).total_seconds()
            
            # Determine if solution was found (event occurred)
            solution_found = search_data['solution_found'].any()
            
            # Extract covariates
            puzzle_number = search_data['puzzle_number'].iloc[0]
            algorithm = search_data['algorithm'].iloc[0]
            gpu_count = search_data['gpu_count'].iloc[0]
            search_range_size = search_data['search_range_size'].iloc[0]
            
            survival_data.append({
                'search_id': search_id,
                'duration': duration,
                'event': solution_found,
                'puzzle_number': puzzle_number,
                'algorithm': algorithm,
                'gpu_count': gpu_count,
                'search_range_size': search_range_size
            })
        
        self.survival_data = pd.DataFrame(survival_data)
        return self.survival_data
    
    def fit_kaplan_meier(self, duration_col: str = 'duration', event_col: str = 'event') -> Dict[str, Any]:
        """Fit Kaplan-Meier survival curve"""
        if self.survival_data is None:
            raise ValueError("Survival data must be prepared first")
        
        logger.info("Fitting Kaplan-Meier survival curve...")
        
        durations = self.survival_data[duration_col]
        events = self.survival_data[event_col]
        
        self.km_fitter.fit(durations, events)
        
        # Extract key statistics
        median_survival = self.km_fitter.median_survival_time_
        survival_at_times = {
            '1_hour': self.km_fitter.survival_function_at_times(3600).iloc[0],
            '1_day': self.km_fitter.survival_function_at_times(86400).iloc[0],
            '1_week': self.km_fitter.survival_function_at_times(604800).iloc[0]
        }
        
        return {
            'median_survival_time': median_survival,
            'survival_probabilities': survival_at_times,
            'confidence_interval': self.km_fitter.confidence_interval_
        }
    
    def fit_cox_regression(self, covariates: List[str]) -> Dict[str, Any]:
        """Fit Cox proportional hazards model"""
        if self.survival_data is None:
            raise ValueError("Survival data must be prepared first")
        
        logger.info("Fitting Cox proportional hazards model...")
        
        # Prepare data for Cox regression
        cox_data = self.survival_data[['duration', 'event'] + covariates].copy()
        
        self.cox_fitter.fit(cox_data, duration_col='duration', event_col='event')
        
        # Extract results
        hazard_ratios = self.cox_fitter.hazard_ratios_
        p_values = self.cox_fitter.summary['p']
        confidence_intervals = self.cox_fitter.confidence_intervals_
        
        return {
            'hazard_ratios': hazard_ratios.to_dict(),
            'p_values': p_values.to_dict(),
            'confidence_intervals': confidence_intervals.to_dict(),
            'concordance_index': self.cox_fitter.concordance_index_
        }
    
    def predict_solution_time(self, covariates: Dict[str, float], 
                            confidence_level: float = 0.95) -> Dict[str, float]:
        """Predict solution discovery time for given covariates"""
        if self.cox_fitter.summary is None:
            raise ValueError("Cox model must be fitted first")
        
        # Create prediction dataframe
        pred_data = pd.DataFrame([covariates])
        
        # Predict survival function
        survival_func = self.cox_fitter.predict_survival_function(pred_data)
        
        # Find median survival time
        median_time = None
        for time, prob in survival_func.iloc[:, 0].items():
            if prob <= 0.5:
                median_time = time
                break
        
        # Confidence intervals (approximate)
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        times = survival_func.index
        probs = survival_func.iloc[:, 0].values
        
        lower_time = times[np.argmin(np.abs(probs - upper_percentile / 100))]
        upper_time = times[np.argmin(np.abs(probs - lower_percentile / 100))]
        
        return {
            'median_solution_time': median_time,
            'confidence_interval': [lower_time, upper_time],
            'probability_solved_1_day': float(survival_func.loc[86400].iloc[0]) if 86400 in survival_func.index else None
        }
    
    def compare_algorithms(self, algorithm_col: str = 'algorithm') -> Dict[str, Any]:
        """Compare survival curves between different algorithms"""
        if self.survival_data is None:
            raise ValueError("Survival data must be prepared first")
        
        algorithms = self.survival_data[algorithm_col].unique()
        comparison_results = {}
        
        # Fit separate KM curves for each algorithm
        km_curves = {}
        for algorithm in algorithms:
            algo_data = self.survival_data[self.survival_data[algorithm_col] == algorithm]
            km_fitter = KaplanMeierFitter()
            km_fitter.fit(algo_data['duration'], algo_data['event'], label=algorithm)
            km_curves[algorithm] = km_fitter
        
        # Pairwise log-rank tests
        logrank_results = {}
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                data1 = self.survival_data[self.survival_data[algorithm_col] == algo1]
                data2 = self.survival_data[self.survival_data[algorithm_col] == algo2]
                
                test_result = logrank_test(
                    data1['duration'], data2['duration'],
                    data1['event'], data2['event']
                )
                
                logrank_results[f"{algo1}_vs_{algo2}"] = {
                    'test_statistic': test_result.test_statistic,
                    'p_value': test_result.p_value,
                    'significant': test_result.p_value < 0.05
                }
        
        comparison_results['km_curves'] = km_curves
        comparison_results['logrank_tests'] = logrank_results
        
        return comparison_results

class MultiObjectiveBayesianOptimizer:
    """
    Optimization 35: Multi-Objective Bayesian Optimization
    
    Optimizes multiple competing objectives (accuracy, speed, resource usage)
    using Gaussian processes and Pareto frontier analysis.
    """
    
    def __init__(self, objective_names: List[str]):
        self.objective_names = objective_names
        self.observations = []
        self.pareto_frontier = None
        self.gp_models = {}
        
    def add_observation(self, parameters: Dict[str, float], objectives: Dict[str, float]):
        """Add new observation to the optimization history"""
        observation = {
            'parameters': parameters.copy(),
            'objectives': objectives.copy()
        }
        self.observations.append(observation)
        
    def fit_gaussian_processes(self):
        """Fit Gaussian process models for each objective"""
        if len(self.observations) < 5:
            logger.warning("Insufficient observations for GP fitting")
            return
        
        # Extract parameter and objective arrays
        param_names = list(self.observations[0]['parameters'].keys())
        X = np.array([[obs['parameters'][name] for name in param_names] 
                     for obs in self.observations])
        
        for obj_name in self.objective_names:
            y = np.array([obs['objectives'][obj_name] for obs in self.observations])
            
            # Fit GP using scikit-learn
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
            
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, 
                                        normalize_y=True, random_state=42)
            gp.fit(X, y)
            
            self.gp_models[obj_name] = gp
        
        logger.info(f"Fitted GP models for {len(self.objective_names)} objectives")
    
    def find_pareto_frontier(self) -> List[Dict]:
        """Find Pareto-optimal solutions"""
        if not self.observations:
            return []
        
        # Extract objective values
        objectives_matrix = np.array([
            [obs['objectives'][name] for name in self.objective_names]
            for obs in self.observations
        ])
        
        # Find Pareto frontier (assuming minimization for all objectives)
        pareto_indices = []
        n_points = len(objectives_matrix)
        
        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if np.all(objectives_matrix[j] <= objectives_matrix[i]) and \
                       np.any(objectives_matrix[j] < objectives_matrix[i]):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        self.pareto_frontier = [self.observations[i] for i in pareto_indices]
        
        logger.info(f"Found {len(self.pareto_frontier)} Pareto-optimal solutions")
        return self.pareto_frontier
    
    def acquisition_function(self, parameters: Dict[str, float], 
                           weights: Dict[str, float] = None) -> float:
        """Compute acquisition function value for given parameters"""
        if not self.gp_models:
            return np.random.random()  # Random exploration if no models
        
        # Default equal weights
        if weights is None:
            weights = {name: 1.0 / len(self.objective_names) for name in self.objective_names}
        
        param_names = list(parameters.keys())
        X_test = np.array([[parameters[name] for name in param_names]])
        
        acquisition_value = 0
        
        for obj_name in self.objective_names:
            if obj_name in self.gp_models:
                gp = self.gp_models[obj_name]
                
                # Predict mean and variance
                mean, std = gp.predict(X_test, return_std=True)
                
                # Expected Improvement acquisition
                if len(self.observations) > 0:
                    best_observed = min(obs['objectives'][obj_name] for obs in self.observations)
                    improvement = best_observed - mean[0]
                    z = improvement / (std[0] + 1e-9)
                    ei = improvement * stats.norm.cdf(z) + std[0] * stats.norm.pdf(z)
                else:
                    ei = std[0]  # Pure exploration
                
                acquisition_value += weights[obj_name] * ei
        
        return acquisition_value
    
    def suggest_next_parameters(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                              weights: Dict[str, float] = None) -> Dict[str, float]:
        """Suggest next parameters to evaluate using acquisition function"""
        param_names = list(parameter_bounds.keys())
        bounds = [parameter_bounds[name] for name in param_names]
        
        def objective(x):
            params = dict(zip(param_names, x))
            return -self.acquisition_function(params, weights)  # Minimize negative acquisition
        
        # Optimize acquisition function
        result = differential_evolution(objective, bounds, seed=42, maxiter=100)
        
        suggested_params = dict(zip(param_names, result.x))
        
        return suggested_params
    
    def analyze_trade_offs(self) -> Dict[str, Any]:
        """Analyze trade-offs between objectives"""
        if not self.pareto_frontier:
            self.find_pareto_frontier()
        
        if len(self.pareto_frontier) < 2:
            return {"message": "Insufficient Pareto solutions for trade-off analysis"}
        
        # Compute trade-off statistics
        trade_offs = {}
        
        for i, obj1 in enumerate(self.objective_names):
            for j, obj2 in enumerate(self.objective_names[i+1:], i+1):
                obj1_values = [sol['objectives'][obj1] for sol in self.pareto_frontier]
                obj2_values = [sol['objectives'][obj2] for sol in self.pareto_frontier]
                
                # Correlation between objectives
                correlation = np.corrcoef(obj1_values, obj2_values)[0, 1]
                
                # Trade-off slope (approximate)
                if len(obj1_values) > 1:
                    slope = np.polyfit(obj1_values, obj2_values, 1)[0]
                else:
                    slope = 0
                
                trade_offs[f"{obj1}_vs_{obj2}"] = {
                    'correlation': correlation,
                    'trade_off_slope': slope
                }
        
        return {
            'trade_offs': trade_offs,
            'pareto_solutions': self.pareto_frontier,
            'num_pareto_solutions': len(self.pareto_frontier)
        }
    
    def recommend_solution(self, preference_weights: Dict[str, float]) -> Dict[str, Any]:
        """Recommend best solution based on user preferences"""
        if not self.pareto_frontier:
            self.find_pareto_frontier()
        
        if not self.pareto_frontier:
            return {"message": "No solutions available"}
        
        # Normalize weights
        total_weight = sum(preference_weights.values())
        normalized_weights = {k: v/total_weight for k, v in preference_weights.items()}
        
        # Score each Pareto solution
        best_score = float('-inf')
        best_solution = None
        
        for solution in self.pareto_frontier:
            score = 0
            for obj_name, weight in normalized_weights.items():
                if obj_name in solution['objectives']:
                    # Assuming lower is better, use negative value
                    score += weight * (-solution['objectives'][obj_name])
            
            if score > best_score:
                best_score = score
                best_solution = solution
        
        return {
            'recommended_solution': best_solution,
            'score': best_score,
            'preference_weights': normalized_weights
        }

class StatisticalOptimizer:
    """
    Main statistical optimizer integrating all statistical optimizations (31-35)
    """
    
    def __init__(self, puzzle_data: pd.DataFrame, config: Dict):
        self.puzzle_data = puzzle_data
        self.config = config
        
        # Initialize optimizers
        self.bayesian_engine = BayesianInferenceEngine(puzzle_data) if config.get('use_bayesian', True) else None
        self.extreme_value_analyzer = ExtremeValueAnalyzer() if config.get('use_extreme_value', True) else None
        self.info_theory_analyzer = InformationTheoryAnalyzer() if config.get('use_info_theory', True) else None
        self.survival_analyzer = SurvivalAnalyzer() if config.get('use_survival', True) else None
        self.multi_objective_optimizer = MultiObjectiveBayesianOptimizer(
            ['solution_time', 'computational_cost', 'success_probability']
        ) if config.get('use_multi_objective', True) else None
        
        logger.info(f"Initialized StatisticalOptimizer with {sum(1 for x in [self.bayesian_engine, self.extreme_value_analyzer, self.info_theory_analyzer, self.survival_analyzer, self.multi_objective_optimizer] if x is not None)} optimizers")
    
    def optimize_search_strategy(self, target_puzzle: int) -> Dict[str, Any]:
        """Optimize search strategy using all statistical methods"""
        optimization_results = {}
        
        # Bayesian inference for key prediction
        if self.bayesian_engine:
            logger.info("Running Bayesian inference optimization...")
            self.bayesian_engine.build_bayesian_network()
            self.bayesian_engine.fit_bayesian_model(target_puzzle)
            bayesian_prediction = self.bayesian_engine.predict_key_position(target_puzzle)
            optimization_results['bayesian_prediction'] = bayesian_prediction
        
        # Extreme value analysis for tail events
        if self.extreme_value_analyzer:
            logger.info("Running extreme value analysis...")
            # Use historical solution times or key positions
            if 'solution_time' in self.puzzle_data.columns:
                solution_times = self.puzzle_data['solution_time'].dropna().values
                self.extreme_value_analyzer.fit_generalized_extreme_value(solution_times)
                self.extreme_value_analyzer.fit_peaks_over_threshold(solution_times)
                
                return_levels = self.extreme_value_analyzer.estimate_return_levels([10, 100, 1000])
                optimization_results['extreme_value_analysis'] = {
                    'return_levels': return_levels,
                    'gev_params': self.extreme_value_analyzer.gev_params,
                    'pot_params': self.extreme_value_analyzer.pot_params
                }
        
        # Information theory analysis
        if self.info_theory_analyzer:
            logger.info("Running information theory analysis...")
            if 'private_key' in self.puzzle_data.columns:
                private_keys = self.puzzle_data['private_key'].dropna().tolist()
                randomness_analysis = self.info_theory_analyzer.analyze_key_randomness(private_keys)
                optimization_results['randomness_analysis'] = randomness_analysis
        
        # Multi-objective optimization
        if self.multi_objective_optimizer:
            logger.info("Running multi-objective optimization...")
            # Add some sample observations for demonstration
            for i in range(10):
                params = {
                    'search_range_size': np.random.uniform(1e12, 1e15),
                    'gpu_count': np.random.randint(1, 9),
                    'algorithm_type': np.random.randint(0, 3)
                }
                objectives = {
                    'solution_time': np.random.exponential(3600),  # seconds
                    'computational_cost': np.random.uniform(10, 1000),  # dollars
                    'success_probability': np.random.uniform(0.1, 0.9)
                }
                self.multi_objective_optimizer.add_observation(params, objectives)
            
            self.multi_objective_optimizer.fit_gaussian_processes()
            pareto_solutions = self.multi_objective_optimizer.find_pareto_frontier()
            trade_off_analysis = self.multi_objective_optimizer.analyze_trade_offs()
            
            optimization_results['multi_objective'] = {
                'pareto_solutions': pareto_solutions,
                'trade_offs': trade_off_analysis
            }
        
        return optimization_results
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive optimization report"""
        report = []
        report.append("# Statistical Optimization Report")
        report.append("=" * 50)
        
        if 'bayesian_prediction' in results:
            bp = results['bayesian_prediction']
            report.append("\n## Bayesian Key Position Prediction")
            report.append(f"- Predicted position: {bp['mean_position_pct']:.2f}%")
            report.append(f"- Uncertainty (std): {bp['std_position_pct']:.2f}%")
            report.append(f"- 95% credible interval: [{bp['credible_interval_95'][0]:.2f}%, {bp['credible_interval_95'][1]:.2f}%]")
        
        if 'extreme_value_analysis' in results:
            eva = results['extreme_value_analysis']
            report.append("\n## Extreme Value Analysis")
            if 'return_levels' in eva:
                report.append("Return levels:")
                for period, level in eva['return_levels'].items():
                    report.append(f"- {period} period: {level:.2f}")
        
        if 'randomness_analysis' in results:
            ra = results['randomness_analysis']
            report.append("\n## Key Randomness Analysis")
            report.append(f"- Mean bit entropy: {ra.get('mean_bit_entropy', 0):.4f}")
            report.append(f"- Sequential correlation: {ra.get('mean_sequential_correlation', 0):.4f}")
            report.append(f"- Compression ratio: {ra.get('mean_compression_ratio', 0):.4f}")
        
        if 'multi_objective' in results:
            mo = results['multi_objective']
            report.append("\n## Multi-Objective Optimization")
            report.append(f"- Pareto solutions found: {mo['trade_offs']['num_pareto_solutions']}")
            if 'trade_offs' in mo['trade_offs']:
                report.append("Trade-off analysis:")
                for trade_off, stats in mo['trade_offs']['trade_offs'].items():
                    report.append(f"- {trade_off}: correlation = {stats['correlation']:.3f}")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Create sample puzzle data
    sample_data = pd.DataFrame({
        'puzzle': range(1, 21),
        'private_key': [hex(i)[2:] for i in range(1, 21)],
        'solution_time': np.random.exponential(1800, 20),  # seconds
        'address': ['1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH'] * 20
    })
    
    # Configuration
    config = {
        'use_bayesian': True,
        'use_extreme_value': True,
        'use_info_theory': True,
        'use_survival': True,
        'use_multi_objective': True
    }
    
    # Initialize optimizer
    optimizer = StatisticalOptimizer(sample_data, config)
    
    # Run optimization
    results = optimizer.optimize_search_strategy(target_puzzle=71)
    
    # Generate report
    report = optimizer.generate_optimization_report(results)
    print(report)
    
    logger.info("Statistical optimizations (31-35) implementation complete!")


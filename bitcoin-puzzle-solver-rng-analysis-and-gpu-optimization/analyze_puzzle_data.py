#!/usr/bin/env python3
"""
Bitcoin Puzzle Data Analysis Script
Analyzes patterns in solved Bitcoin puzzle private keys
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

def load_puzzle_data(filename):
    """Load and parse the Bitcoin puzzle CSV data"""
    df = pd.read_csv(filename, names=['puzzle_num', 'address', 'private_key_hex'])
    
    # Convert hex private keys to integers
    df['private_key_int'] = df['private_key_hex'].apply(lambda x: int(x, 16))
    
    # Calculate the range for each puzzle (2^(n-1) to 2^n - 1)
    df['range_start'] = 2**(df['puzzle_num'] - 1)
    df['range_end'] = 2**df['puzzle_num'] - 1
    df['range_size'] = df['range_end'] - df['range_start'] + 1
    
    # Calculate relative position within range (alpha value)
    # Handle special case for puzzle 1 where range size is 1
    range_diff = df['range_end'] - df['range_start']
    df['alpha'] = np.where(
        range_diff > 0,
        (df['private_key_int'] - df['range_start']) / range_diff,
        0.0  # For puzzle 1, set alpha to 0
    )
    
    # Calculate percentage position within range
    df['position_percent'] = df['alpha'] * 100
    
    return df

def analyze_patterns(df):
    """Analyze patterns in the private keys"""
    analysis = {}
    
    # Basic statistics
    analysis['total_puzzles'] = len(df)
    analysis['alpha_stats'] = {
        'mean': df['alpha'].mean(),
        'median': df['alpha'].median(),
        'std': df['alpha'].std(),
        'min': df['alpha'].min(),
        'max': df['alpha'].max()
    }
    
    # Position distribution analysis
    analysis['position_distribution'] = {
        'first_quarter': len(df[df['alpha'] < 0.25]),
        'second_quarter': len(df[(df['alpha'] >= 0.25) & (df['alpha'] < 0.5)]),
        'third_quarter': len(df[(df['alpha'] >= 0.5) & (df['alpha'] < 0.75)]),
        'fourth_quarter': len(df[df['alpha'] >= 0.75])
    }
    
    # Bit pattern analysis
    analysis['bit_patterns'] = analyze_bit_patterns(df)
    
    # Hot zones analysis (areas with higher concentration)
    analysis['hot_zones'] = identify_hot_zones(df)
    
    return analysis

def analyze_bit_patterns(df):
    """Analyze bit patterns in private keys"""
    patterns = {}
    
    for _, row in df.iterrows():
        key_bin = bin(row['private_key_int'])[2:].zfill(row['puzzle_num'])
        
        # Count leading zeros
        leading_zeros = len(key_bin) - len(key_bin.lstrip('0'))
        
        # Count trailing zeros
        trailing_zeros = len(key_bin) - len(key_bin.rstrip('0'))
        
        # Count total ones and zeros
        ones_count = key_bin.count('1')
        zeros_count = key_bin.count('0')
        
        patterns[row['puzzle_num']] = {
            'binary': key_bin,
            'leading_zeros': leading_zeros,
            'trailing_zeros': trailing_zeros,
            'ones_count': ones_count,
            'zeros_count': zeros_count,
            'bit_density': ones_count / len(key_bin)
        }
    
    return patterns

def identify_hot_zones(df, num_bins=20):
    """Identify hot zones where keys are more likely to be found"""
    # Create histogram of alpha values
    hist, bin_edges = np.histogram(df['alpha'], bins=num_bins)
    
    # Find bins with above-average frequency
    avg_frequency = len(df) / num_bins
    hot_zones = []
    
    for i, count in enumerate(hist):
        if count > avg_frequency:
            hot_zones.append({
                'start': bin_edges[i],
                'end': bin_edges[i+1],
                'count': count,
                'probability': count / len(df)
            })
    
    return hot_zones

def create_visualizations(df, analysis):
    """Create visualizations of the data"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Alpha distribution histogram
    axes[0, 0].hist(df['alpha'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Alpha Values (Relative Position)')
    axes[0, 0].set_xlabel('Alpha (0 = start of range, 1 = end of range)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['alpha'].mean(), color='red', linestyle='--', label=f'Mean: {df["alpha"].mean():.3f}')
    axes[0, 0].legend()
    
    # Position percentage scatter plot
    axes[0, 1].scatter(df['puzzle_num'], df['position_percent'], alpha=0.7, color='green')
    axes[0, 1].set_title('Key Position vs Puzzle Number')
    axes[0, 1].set_xlabel('Puzzle Number')
    axes[0, 1].set_ylabel('Position Percentage in Range')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bit density analysis
    bit_densities = [analysis['bit_patterns'][num]['bit_density'] for num in df['puzzle_num']]
    axes[1, 0].scatter(df['puzzle_num'], bit_densities, alpha=0.7, color='orange')
    axes[1, 0].set_title('Bit Density vs Puzzle Number')
    axes[1, 0].set_xlabel('Puzzle Number')
    axes[1, 0].set_ylabel('Bit Density (1s / total bits)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quarter distribution pie chart
    quarters = list(analysis['position_distribution'].values())
    labels = ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']
    axes[1, 1].pie(quarters, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Distribution Across Range Quarters')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/puzzle_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_predictions(df, analysis):
    """Generate predictions for unsolved puzzles based on patterns"""
    predictions = {}
    
    # Use the hot zones to predict likely ranges for unsolved puzzles
    hot_zones = analysis['hot_zones']
    
    # Predict for puzzles 71-80 (next targets)
    for puzzle_num in range(71, 81):
        range_start = 2**(puzzle_num - 1)
        range_end = 2**puzzle_num - 1
        range_size = range_end - range_start + 1
        
        predicted_ranges = []
        for zone in hot_zones:
            zone_start = range_start + int(zone['start'] * range_size)
            zone_end = range_start + int(zone['end'] * range_size)
            predicted_ranges.append({
                'start': hex(zone_start),
                'end': hex(zone_end),
                'probability': zone['probability'],
                'size': zone_end - zone_start + 1
            })
        
        predictions[puzzle_num] = {
            'full_range': {'start': hex(range_start), 'end': hex(range_end)},
            'predicted_hot_zones': predicted_ranges,
            'total_reduction': sum(zone['probability'] for zone in hot_zones)
        }
    
    return predictions

def main():
    print("Loading Bitcoin puzzle data...")
    df = load_puzzle_data('/home/ubuntu/bitcoin_puzzle_data.csv')
    
    print(f"Loaded {len(df)} solved puzzles")
    print(f"Puzzle range: {df['puzzle_num'].min()} to {df['puzzle_num'].max()}")
    
    print("\nAnalyzing patterns...")
    analysis = analyze_patterns(df)
    
    print("\nCreating visualizations...")
    create_visualizations(df, analysis)
    
    print("\nGenerating predictions...")
    predictions = generate_predictions(df, analysis)
    
    # Save analysis results
    results = {
        'analysis': analysis,
        'predictions': predictions,
        'data_summary': {
            'total_puzzles': len(df),
            'min_puzzle': int(df['puzzle_num'].min()),
            'max_puzzle': int(df['puzzle_num'].max()),
            'alpha_mean': float(df['alpha'].mean()),
            'alpha_std': float(df['alpha'].std())
        }
    }
    
    with open('/home/ubuntu/puzzle_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*50)
    print("BITCOIN PUZZLE ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total puzzles analyzed: {analysis['total_puzzles']}")
    print(f"Alpha mean: {analysis['alpha_stats']['mean']:.4f}")
    print(f"Alpha std: {analysis['alpha_stats']['std']:.4f}")
    print(f"Alpha median: {analysis['alpha_stats']['median']:.4f}")
    
    print("\nPosition Distribution:")
    for quarter, count in analysis['position_distribution'].items():
        percentage = (count / analysis['total_puzzles']) * 100
        print(f"  {quarter}: {count} puzzles ({percentage:.1f}%)")
    
    print(f"\nIdentified {len(analysis['hot_zones'])} hot zones")
    print("Hot zones (high probability areas):")
    for i, zone in enumerate(analysis['hot_zones']):
        print(f"  Zone {i+1}: {zone['start']:.3f} - {zone['end']:.3f} "
              f"(probability: {zone['probability']:.3f})")
    
    print(f"\nGenerated predictions for puzzles 71-80")
    print("Analysis complete! Check puzzle_analysis.png and puzzle_analysis_results.json")

if __name__ == "__main__":
    main()


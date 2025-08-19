#!/usr/bin/env python3
"""
Simplified Bitcoin Puzzle Data Analysis Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def analyze_puzzles():
    # Read the CSV data
    df = pd.read_csv('/home/ubuntu/bitcoin_puzzle_data.csv', names=['puzzle_num', 'address', 'private_key_hex'])
    
    print(f"Loaded {len(df)} puzzles")
    print(f"Puzzle range: {df['puzzle_num'].min()} to {df['puzzle_num'].max()}")
    
    # Convert hex to integers
    df['private_key_int'] = df['private_key_hex'].apply(lambda x: int(x, 16))
    
    # Calculate ranges and alpha values
    results = []
    for _, row in df.iterrows():
        puzzle_num = row['puzzle_num']
        private_key = row['private_key_int']
        
        # Calculate the theoretical range for this puzzle
        if puzzle_num == 1:
            range_start = 1
            range_end = 1
            alpha = 0.0  # Special case
        else:
            range_start = 2**(puzzle_num - 1)
            range_end = 2**puzzle_num - 1
            alpha = (private_key - range_start) / (range_end - range_start)
        
        results.append({
            'puzzle_num': puzzle_num,
            'private_key_hex': row['private_key_hex'],
            'private_key_int': private_key,
            'range_start': range_start,
            'range_end': range_end,
            'alpha': alpha,
            'position_percent': alpha * 100,
            'address': row['address']
        })
    
    results_df = pd.DataFrame(results)
    
    # Basic statistics (excluding puzzle 1 for alpha calculations)
    alpha_data = results_df[results_df['puzzle_num'] > 1]['alpha']
    
    print("\n" + "="*50)
    print("BITCOIN PUZZLE ANALYSIS RESULTS")
    print("="*50)
    
    print(f"\nAlpha Statistics (excluding puzzle 1):")
    print(f"  Mean: {alpha_data.mean():.4f}")
    print(f"  Median: {alpha_data.median():.4f}")
    print(f"  Std Dev: {alpha_data.std():.4f}")
    print(f"  Min: {alpha_data.min():.4f}")
    print(f"  Max: {alpha_data.max():.4f}")
    
    # Position distribution
    quarters = {
        'Q1 (0-25%)': len(alpha_data[alpha_data < 0.25]),
        'Q2 (25-50%)': len(alpha_data[(alpha_data >= 0.25) & (alpha_data < 0.5)]),
        'Q3 (50-75%)': len(alpha_data[(alpha_data >= 0.5) & (alpha_data < 0.75)]),
        'Q4 (75-100%)': len(alpha_data[alpha_data >= 0.75])
    }
    
    print(f"\nPosition Distribution:")
    for quarter, count in quarters.items():
        percentage = (count / len(alpha_data)) * 100
        print(f"  {quarter}: {count} puzzles ({percentage:.1f}%)")
    
    # Identify patterns
    print(f"\nPattern Analysis:")
    
    # Hot zones analysis
    hist, bin_edges = np.histogram(alpha_data, bins=10)
    avg_frequency = len(alpha_data) / 10
    
    print(f"Hot zones (above average frequency of {avg_frequency:.1f}):")
    for i, count in enumerate(hist):
        if count > avg_frequency:
            start_pct = bin_edges[i] * 100
            end_pct = bin_edges[i+1] * 100
            probability = count / len(alpha_data)
            print(f"  {start_pct:.1f}% - {end_pct:.1f}%: {count} keys (probability: {probability:.3f})")
    
    # Bit pattern analysis
    print(f"\nBit Pattern Analysis:")
    for _, row in results_df.head(10).iterrows():
        key_bin = bin(row['private_key_int'])[2:].zfill(row['puzzle_num'])
        ones_count = key_bin.count('1')
        bit_density = ones_count / len(key_bin) if len(key_bin) > 0 else 0
        print(f"  Puzzle {row['puzzle_num']:2d}: {key_bin} (density: {bit_density:.3f})")
    
    # Generate predictions for next puzzles
    print(f"\nPredictions for Puzzles 71-75:")
    
    # Use the hot zones to predict likely ranges
    hot_zones = []
    for i, count in enumerate(hist):
        if count > avg_frequency:
            hot_zones.append({
                'start': bin_edges[i],
                'end': bin_edges[i+1],
                'probability': count / len(alpha_data)
            })
    
    for puzzle_num in range(71, 76):
        range_start = 2**(puzzle_num - 1)
        range_end = 2**puzzle_num - 1
        range_size = range_end - range_start + 1
        
        print(f"\n  Puzzle {puzzle_num}:")
        print(f"    Full range: {hex(range_start)} - {hex(range_end)}")
        print(f"    Range size: {range_size:,} keys")
        
        total_reduction = 0
        for zone in hot_zones:
            zone_start = range_start + int(zone['start'] * range_size)
            zone_end = range_start + int(zone['end'] * range_size)
            zone_size = zone_end - zone_start + 1
            reduction = zone_size / range_size
            total_reduction += reduction
            
            print(f"    Hot zone: {hex(zone_start)} - {hex(zone_end)} "
                  f"({zone_size:,} keys, {reduction:.1%} of range)")
        
        print(f"    Total search reduction: {total_reduction:.1%}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(alpha_data, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Alpha Distribution (Relative Position in Range)')
    plt.xlabel('Alpha (0 = start, 1 = end)')
    plt.ylabel('Frequency')
    plt.axvline(alpha_data.mean(), color='red', linestyle='--', label=f'Mean: {alpha_data.mean():.3f}')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.scatter(results_df['puzzle_num'], results_df['position_percent'], alpha=0.7)
    plt.title('Key Position vs Puzzle Number')
    plt.xlabel('Puzzle Number')
    plt.ylabel('Position Percentage in Range')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    bit_densities = []
    for _, row in results_df.iterrows():
        key_bin = bin(row['private_key_int'])[2:].zfill(row['puzzle_num'])
        density = key_bin.count('1') / len(key_bin) if len(key_bin) > 0 else 0
        bit_densities.append(density)
    
    plt.scatter(results_df['puzzle_num'], bit_densities, alpha=0.7, color='green')
    plt.title('Bit Density vs Puzzle Number')
    plt.xlabel('Puzzle Number')
    plt.ylabel('Bit Density (1s / total bits)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    quarter_values = list(quarters.values())
    quarter_labels = list(quarters.keys())
    plt.pie(quarter_values, labels=quarter_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution Across Range Quarters')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/puzzle_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    analysis_results = {
        'total_puzzles': len(results_df),
        'alpha_stats': {
            'mean': float(alpha_data.mean()),
            'median': float(alpha_data.median()),
            'std': float(alpha_data.std()),
            'min': float(alpha_data.min()),
            'max': float(alpha_data.max())
        },
        'position_distribution': quarters,
        'hot_zones': hot_zones,
        'puzzle_data': results_df.to_dict('records')
    }
    
    with open('/home/ubuntu/puzzle_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\nAnalysis complete!")
    print(f"Visualization saved to: puzzle_analysis.png")
    print(f"Detailed results saved to: puzzle_analysis_results.json")
    
    return results_df, analysis_results

if __name__ == "__main__":
    analyze_puzzles()


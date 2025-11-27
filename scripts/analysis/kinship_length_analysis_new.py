"""
Kinship LENGTH_CM distribution analysis and visualization for new filtered datasets.

Creates:
1. Individual line plots for kinship 1-6 and UN per LENGTH_CM threshold
2. Combined line plots showing all kinship categories per threshold
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_percentile_file(kinship_level, threshold):
    """Load individual percentile TSV file."""
    kinship_str = str(kinship_level)
    file_path = f'data/raw/new/cm_over_{threshold}/percentile_{kinship_str}.tsv'
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
    
    return pd.read_csv(file_path, sep='\t')


def filter_un_data(df):
    """
    Filter UN kinship data to remove X-1_vs_X-2 pairs (same sample duplicates).
    """
    duplicate_patterns = []
    for i in range(1, 7):
        pattern = f"[{i}-1_vs_{i}-2]"
        duplicate_patterns.append(pattern)
    
    mask = ~df['PAIR_ID'].isin(duplicate_patterns)
    original_count = len(df)
    filtered_df = df[mask].copy()
    removed_count = original_count - len(filtered_df)
    
    print(f"  UN file: Removed {removed_count} records (filtered from {original_count} to {len(filtered_df)})")
    return filtered_df


def create_length_distribution(df):
    """
    Create LENGTH_CM distribution from percentile columns (0%, 10%, 20%... 100%).
    
    Returns dict with:
    - 'lengths': actual LENGTH_CM values (from 0%-100% columns)
    - 'counts': count of pairs in each bin
    - 'percentages': percentage per bin
    """
    # Extract percentile columns - these represent actual LENGTH_CM values at different percentiles
    percentile_cols = [col for col in df.columns if col.endswith('%')]
    percentile_cols = sorted(percentile_cols, key=lambda x: float(x.rstrip('%')))
    
    # Get all LENGTH_CM values from all percentile columns across all pairs
    all_values = []
    for col in percentile_cols:
        all_values.extend(df[col].dropna().values)
    
    if len(all_values) == 0:
        print("  Warning: No valid LENGTH_CM values found")
        return {'lengths': [], 'counts': [], 'percentages': [], 'total': 0}
    
    # Create bins with 0.05 cM granularity
    bin_size = 0.05
    min_val = 0
    max_val = max(all_values)
    
    bin_edges = np.arange(0, max_val + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2
    
    # Count values in bins
    counts = np.histogram(all_values, bins=bin_edges)[0]
    total = len(all_values)
    percentages = (counts / total * 100) if total > 0 else counts
    
    return {
        'lengths': bin_centers.tolist(),
        'counts': counts.tolist(),
        'percentages': percentages.tolist(),
        'total': total,
        'max_length': max_val
    }


def plot_individual_kinship(kinship_num, df, threshold, output_dir, global_max_length=None):
    """
    Create individual line plot for a kinship level.
    Shows both count and percentage on dual Y-axis.
    If global_max_length is provided, use it for consistent x-axis scaling.
    """
    dist = create_length_distribution(df)
    
    if not dist['lengths']:
        print(f"  Skipping plot: no data")
        return dist
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot count on primary Y-axis
    color1 = '#4C78A8'
    ax1.set_xlabel('LENGTH_CM (centiMorgans)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', color=color1, fontsize=14, fontweight='bold')
    line1 = ax1.plot(dist['lengths'], dist['counts'], marker='o', color=color1,
                     linewidth=1, markersize=3, label='Count')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis limits for consistency across all plots in this threshold set
    if global_max_length:
        ax1.set_xlim(-global_max_length * 0.02, global_max_length * 1.05)
    
    # Set x-axis ticks at 25 cM intervals (major) and 5 cM (minor)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    
    # Plot percentage on secondary Y-axis
    ax2 = ax1.twinx()
    color2 = '#E45756'
    ax2.set_ylabel('Percentage (%)', color=color2, fontsize=14, fontweight='bold')
    line2 = ax2.plot(dist['lengths'], dist['percentages'], marker='s', color=color2,
                     linewidth=1, markersize=3, label='Percentage', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and legend
    kinship_label = f'Kinship {kinship_num}' if kinship_num != 'UN' else 'Kinship UN'
    title = f'{kinship_label}: LENGTH_CM Distribution (cM > {threshold}, n={dist["total"]})'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'kinship_{kinship_num}_length_distribution_cM_{threshold}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()
    
    return dist


def plot_combined_kinship(all_distributions, threshold, output_dir):
    """
    Create combined line plot overlaying all kinship categories (percentage only).
    X-axis shows full range of values from each distribution.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    colors = ['#4C78A8', '#E45756', '#54A24B', '#F58518', '#B279A2', '#FF9DA6', '#9D755D']
    kinship_labels = ['Kinship 1', 'Kinship 2', 'Kinship 3', 'Kinship 4',
                      'Kinship 5', 'Kinship 6', 'Kinship UN']
    kinship_nums = [1, 2, 3, 4, 5, 6, 'UN']
    
    # Find max length across all distributions
    max_length = max([dist['max_length'] for dist in all_distributions.values() if dist['lengths']])
    
    # Plot each kinship as percentage
    for kinship_num, color, label in zip(kinship_nums, colors, kinship_labels):
        key = f'kinship_{kinship_num}'
        if key in all_distributions:
            dist = all_distributions[key]
            if dist['lengths']:
                ax.plot(dist['lengths'], dist['percentages'], marker='o', color=color,
                       linewidth=1.2, markersize=4, label=label, alpha=0.8)
    
    ax.set_xlabel('LENGTH_CM (centiMorgans)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Kinship LENGTH_CM Distribution Comparison (cM > {threshold})',
                fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis to show full range of data (no unnecessary zero padding)
    ax.set_xlim(-max_length * 0.02, max_length * 1.05)
    
    # Set x-axis ticks at 25 cM intervals (major) and 5 cM (minor)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(True, alpha=0.3, which='major')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'kinship_combined_length_distribution_cM_{threshold}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()


def main():
    thresholds = [1, 3, 6, 10]
    
    print("\n" + "=" * 70)
    print("New Dataset: Kinship LENGTH_CM Analysis")
    print("=" * 70)
    
    for threshold in thresholds:
        print(f"\n{'='*70}")
        print(f"Processing Threshold: cM > {threshold}")
        print(f"{'='*70}")
        
        output_dir = os.path.join('reports', f'kinship_length_analysis_cM_{threshold}')
        os.makedirs(output_dir, exist_ok=True)
        
        all_distributions = {}
        global_max = 0
        
        # First pass: load all data to determine global max
        kinship_data = {}
        for kinship_num in range(1, 7):
            df = load_percentile_file(kinship_num, threshold)
            if df is not None:
                kinship_data[kinship_num] = df
        
        # UN file
        df_un = load_percentile_file('UN', threshold)
        if df_un is not None:
            kinship_data['UN'] = df_un
        
        # Calculate global max across all kinship levels
        for kinship_num, df in kinship_data.items():
            percentile_cols = [col for col in df.columns if col.endswith('%')]
            if percentile_cols:
                max_val = df[percentile_cols].max().max()
                global_max = max(global_max, max_val)
        
        # Second pass: plot with consistent x-axis
        # Process kinship 1-6
        for kinship_num in range(1, 7):
            print(f"\n  Kinship {kinship_num}:")
            df = kinship_data.get(kinship_num)
            if df is not None:
                dist = plot_individual_kinship(kinship_num, df, threshold, output_dir, global_max)
                all_distributions[f'kinship_{kinship_num}'] = dist
                print(f"    Records: {len(df)}, LENGTH_CM range: {dist['lengths'][0]:.2f} - {dist['max_length']:.2f} cM")
            else:
                all_distributions[f'kinship_{kinship_num}'] = {'lengths': [], 'counts': [], 'percentages': [], 'total': 0}
        
        # Process kinship UN
        print(f"\n  Kinship UN:")
        df_un = kinship_data.get('UN')
        if df_un is not None:
            print(f"    Records before filtering: {len(df_un)}")
            df_un_filtered = filter_un_data(df_un)
            print(f"    Records after filtering: {len(df_un_filtered)}")
            dist_un = plot_individual_kinship('UN', df_un_filtered, threshold, output_dir, global_max)
            all_distributions['kinship_UN'] = dist_un
            if dist_un['lengths']:
                print(f"    LENGTH_CM range: {dist_un['lengths'][0]:.2f} - {dist_un['max_length']:.2f} cM")
        else:
            all_distributions['kinship_UN'] = {'lengths': [], 'counts': [], 'percentages': [], 'total': 0}
        
        # Combined plot
        print(f"\n  Creating combined plot...")
        plot_combined_kinship(all_distributions, threshold, output_dir)
        
        # Summary
        print(f"\n  Summary for cM > {threshold}:")
        for kinship_name, dist in all_distributions.items():
            if dist['total'] > 0:
                print(f"    {kinship_name}: {dist['total']} samples, max percentile: {max(dist['percentages']):.2f}%")
        
        print(f"  All plots saved to: {output_dir}/")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

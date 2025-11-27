"""
Kinship LENGTH_CM distribution analysis and visualization.

Creates:
1. Individual line plots for kinship_1-6 and kinship_UN (with count & percentage on Y-axis)
2. Combined line plot overlaying all 7 kinship files (percentage only)
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_kinship_data(kinship_file):
    """Load kinship TSV file."""
    return pd.read_csv(kinship_file, sep='\t')


def filter_un_data(df):
    """
    Filter UN kinship data to remove X-1_vs_X-2 pairs (same sample, different experimental runs).
    
    Removes patterns: 1-1_vs_1-2, 2-1_vs_2-2, 3-1_vs_3-2, ..., 6-1_vs_6-2
    """
    duplicate_patterns = []
    for i in range(1, 7):
        # Pattern: [X-1_vs_X-2] where X = 1-6
        pattern = f"[{i}-1_vs_{i}-2]"
        duplicate_patterns.append(pattern)
    
    # Filter out rows matching these patterns
    mask = ~df['PAIR_ID'].isin(duplicate_patterns)
    original_count = len(df)
    filtered_df = df[mask].copy()
    removed_count = original_count - len(filtered_df)
    
    print(f"UN file: Removed {removed_count} records (filtered from {original_count} to {len(filtered_df)})")
    return filtered_df


def create_length_distribution(df):
    """
    Create LENGTH_CM distribution data with 0.05 cM bin size.
    
    Returns dict with:
    - 'bins': length bins (0, 0.05, 0.10, 0.15, ..., MAX)
    - 'counts': count per bin
    - 'percentages': percentage per bin
    """
    # Use 0.05 cM bin size for finer granularity
    bin_size = 0.05
    lengths = df['LENGTH_CM']
    
    # Get range from 0 to max
    min_val = 0
    max_val = lengths.max()
    
    # Create bins at 0.2 cM intervals
    bin_edges = np.arange(0, max_val + bin_size, bin_size)
    bin_centers = np.arange(0, max_val + bin_size, bin_size)[:-1] + bin_size/2
    
    # Use np.digitize to assign values to bins
    bin_assignments = np.digitize(lengths, bin_edges) - 1
    
    # Count occurrences
    counts = {}
    for i in range(len(bin_edges) - 1):
        counts[i] = (bin_assignments == i).sum()
    
    total = len(df)
    percentages = {k: (v / total * 100) for k, v in counts.items()}
    
    # Use actual bin centers for X-axis
    bin_labels = [bin_edges[i] for i in range(len(bin_edges) - 1)]
    
    return {
        'bins': bin_labels,
        'counts': [counts[i] for i in range(len(bin_edges) - 1)],
        'percentages': [percentages[i] for i in range(len(bin_edges) - 1)],
        'total': total
    }


def plot_individual_kinship(kinship_num, df, output_dir):
    """
    Create individual line plot for a kinship file.
    
    Shows both count and percentage on dual Y-axis.
    """
    dist = create_length_distribution(df)
    
    # Setup plot with dual Y-axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot count on primary Y-axis
    color1 = '#4C78A8'
    ax1.set_xlabel('LENGTH_CM (centiMorgans)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', color=color1, fontsize=14, fontweight='bold')
    line1 = ax1.plot(dist['bins'], dist['counts'], marker='o', color=color1, 
                     linewidth=2, markersize=6, label='Count')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot percentage on secondary Y-axis
    ax2 = ax1.twinx()
    color2 = '#E45756'
    ax2.set_ylabel('Percentage (%)', color=color2, fontsize=14, fontweight='bold')
    line2 = ax2.plot(dist['bins'], dist['percentages'], marker='s', color=color2, 
                     linewidth=2, markersize=6, label='Percentage', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and legend
    title = f'Kinship {kinship_num}: LENGTH_CM Distribution (n={dist["total"]})'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Increase x-axis tick frequency
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=30))
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'kinship_{kinship_num}_length_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return dist


def plot_combined_kinship(all_distributions, output_dir):
    """
    Create combined line plot overlaying all 7 kinship files (percentage only).
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    colors = ['#4C78A8', '#E45756', '#54A24B', '#F58518', '#B279A2', '#FF9DA6', '#9D755D']
    kinship_labels = ['Kinship 1', 'Kinship 2', 'Kinship 3', 'Kinship 4', 
                      'Kinship 5', 'Kinship 6', 'Kinship UN']
    
    # Plot each kinship as percentage
    for (kinship_name, dist), color, label in zip(all_distributions.items(), colors, kinship_labels):
        ax.plot(dist['bins'], dist['percentages'], marker='o', color=color, 
               linewidth=2.5, markersize=7, label=label, alpha=0.8)
    
    ax.set_xlabel('LENGTH_CM (centiMorgans)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Kinship LENGTH_CM Distribution Comparison (All Kinship Categories)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 10)
    
    # Increase x-axis tick frequency
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=30))
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'kinship_combined_length_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Paths
    data_dir = 'data/raw'
    output_dir = os.path.join('reports', 'kinship_length_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process kinship files 1-6
    print("\n" + "="*60)
    print("Processing Kinship Files 1-6")
    print("="*60)
    
    all_distributions = {}
    
    for kinship_num in range(1, 7):
        kinship_file = os.path.join(data_dir, f'kinship_{kinship_num}.tsv')
        print(f"\nLoading kinship_{kinship_num}.tsv...")
        
        df = load_kinship_data(kinship_file)
        print(f"  Records: {len(df)}")
        print(f"  LENGTH_CM range: {df['LENGTH_CM'].min():.2f} - {df['LENGTH_CM'].max():.2f}")
        
        dist = plot_individual_kinship(kinship_num, df, output_dir)
        all_distributions[f'kinship_{kinship_num}'] = dist
    
    # Process kinship UN
    print(f"\nLoading kinship_UN.tsv...")
    kinship_file = os.path.join(data_dir, 'kinship_UN.tsv')
    df_un = load_kinship_data(kinship_file)
    print(f"  Records before filtering: {len(df_un)}")
    print(f"  LENGTH_CM range: {df_un['LENGTH_CM'].min():.2f} - {df_un['LENGTH_CM'].max():.2f}")
    
    # Filter UN data
    df_un_filtered = filter_un_data(df_un)
    print(f"  LENGTH_CM range after filtering: {df_un_filtered['LENGTH_CM'].min():.2f} - {df_un_filtered['LENGTH_CM'].max():.2f}")
    
    dist_un = plot_individual_kinship('UN', df_un_filtered, output_dir)
    all_distributions['kinship_UN'] = dist_un
    
    # Combined plot
    print("\n" + "="*60)
    print("Creating Combined Plot")
    print("="*60)
    plot_combined_kinship(all_distributions, output_dir)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir}/")
    
    # Summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    for kinship_name, dist in all_distributions.items():
        print(f"\n{kinship_name}:")
        print(f"  Total records: {dist['total']}")
        print(f"  LENGTH_CM range: {dist['bins'][0]} - {dist['bins'][-1]} cM")
        print(f"  Max count: {max(dist['counts'])} (at {dist['bins'][dist['counts'].index(max(dist['counts']))]} cM)")
        print(f"  Max percentage: {max(dist['percentages']):.2f}%")


if __name__ == '__main__':
    main()

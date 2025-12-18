#!/usr/bin/env python3
"""
Convert new merged_cm_over_*.tsv files to CSV format for training pipeline.
Maps new percentile-based features to model input format.
"""

import os
import pandas as pd
import argparse


def filter_duplicate_samples(df):
    """
    Filter UN kinship data to remove X-1_vs_X-2 pairs (same sample duplicates).
    """
    duplicate_patterns = []
    for i in range(1, 7):
        pattern = f"[{i}-1_vs_{i}-2]"
        duplicate_patterns.append(pattern)
    
    mask = ~df['pair'].isin(duplicate_patterns)
    original_count = len(df)
    filtered_df = df[mask].copy()
    removed_count = original_count - len(filtered_df)
    print(f"  Removed {removed_count} duplicate sample pairs (X-1 vs X-2)")
    return filtered_df


def convert_merged_tsv_to_csv(threshold, data_dir='new'):
    """
    Convert merged_cm_over_X.tsv to model_input_with_kinship_filtered_cM_X_percentile.csv
    Uses percentile and statistical columns as features.
    """
    threshold = str(threshold)
    base_path = f'data/raw/{data_dir}'
    filename = f'merged_cm_over_{threshold}.tsv'
    input_tsv = os.path.join(base_path, filename)
    output_csv = f'data/raw/model_input_with_kinship_filtered_cM_{threshold}_percentile.csv'
    
    # Robust path discovery: if not in root, check subdirectories
    if not os.path.exists(input_tsv):
        found = False
        for root, dirs, files in os.walk(base_path):
            if filename in files:
                input_tsv = os.path.join(root, filename)
                found = True
                break
        if not found:
            print(f"CRITICAL ERROR: {filename} not found in {base_path} or its subdirectories.")
            return False
    
    # Read new format
    df = pd.read_csv(input_tsv, sep='\t')
    print(f"\nLoading {input_tsv}")
    print(f"  Shape: {df.shape}")
    
    # Rename PAIR_ID to pair for compatibility
    df.rename(columns={'PAIR_ID': 'pair'}, inplace=True)
    
    # Keep essential columns
    feature_cols = ['pair', 'kinship', 'n_segment', 'mean', 'median', 'max', 'total_length']
    percentile_cols = [col for col in df.columns if col.endswith('%')]
    
    keep_cols = feature_cols + percentile_cols
    df_out = df[keep_cols].copy()
    
    # Ensure kinship is string
    df_out['kinship'] = df_out['kinship'].astype(str)
    
    # Filter duplicate samples (X-1 vs X-2 pairs)
    df_out = filter_duplicate_samples(df_out)
    
    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False, sep='\t')
    
    print(f"Saved: {output_csv}")
    print(f"  Shape: {df_out.shape}")
    print(f"  Kinship distribution:\n{df_out['kinship'].value_counts().sort_index()}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare new dataset TSVs for training')
    parser.add_argument('--thresholds', type=str, nargs='+', default=['1', '3', '6', '10'],
                       help='cM thresholds to process')
    parser.add_argument('--data-dir', type=str, default='new',
                       help='Data directory name (e.g., "new", "251128")')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Converting TSV datasets to CSV format (source: data/raw/{args.data_dir})")
    print("=" * 70)
    
    success = True
    for threshold in args.thresholds:
        if not convert_merged_tsv_to_csv(threshold, args.data_dir):
            success = False
    
    if not success:
        print("\n" + "!" * 70)
        print("CRITICAL FAILURE: Some datasets could not be converted.")
        print("Stopping pipeline to prevent training on stale data.")
        print("!" * 70)
        import sys
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("All datasets converted successfully")
    print("=" * 70)


if __name__ == '__main__':
    main()

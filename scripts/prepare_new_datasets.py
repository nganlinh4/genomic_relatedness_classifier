#!/usr/bin/env python3
"""
Convert new merged_cm_over_*.tsv files to CSV format for training pipeline.
Maps new percentile-based features to model input format.
"""

import os
import pandas as pd
import argparse


def convert_merged_tsv_to_csv(threshold):
    """
    Convert merged_cm_over_X.tsv to model_input_with_kinship_filtered_cM_X_percentile.csv
    Uses percentile and statistical columns as features.
    """
    threshold = str(threshold)
    input_tsv = f'data/raw/new/merged_cm_over_{threshold}.tsv'
    output_csv = f'data/raw/model_input_with_kinship_filtered_cM_{threshold}_percentile.csv'
    
    if not os.path.exists(input_tsv):
        print(f"Error: {input_tsv} not found")
        return False
    
    # Read new format
    df = pd.read_csv(input_tsv, sep='\t')
    print(f"\nLoading {input_tsv}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Rename PAIR_ID to pair for compatibility
    df.rename(columns={'PAIR_ID': 'pair'}, inplace=True)
    
    # Keep essential columns: pair, kinship, and all statistical/percentile features
    # The percentile columns (0%, 10%, 20%... 100%) + stats (n_segment, mean, median, max, total_length)
    # will serve as features for the model
    feature_cols = ['pair', 'kinship', 'n_segment', 'mean', 'median', 'max', 'total_length']
    percentile_cols = [col for col in df.columns if col.endswith('%')]
    
    keep_cols = feature_cols + percentile_cols
    df_out = df[keep_cols].copy()
    
    # Ensure kinship is string
    df_out['kinship'] = df_out['kinship'].astype(str)
    
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
    args = parser.parse_args()
    
    print("=" * 70)
    print("Converting new TSV datasets to CSV format")
    print("=" * 70)
    
    for threshold in args.thresholds:
        convert_merged_tsv_to_csv(threshold)
    
    print("\n" + "=" * 70)
    print("All datasets converted successfully")
    print("=" * 70)


if __name__ == '__main__':
    main()

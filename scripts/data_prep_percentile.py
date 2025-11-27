import sys
import os
import argparse
import pandas as pd
import re


def main():
    parser = argparse.ArgumentParser(description='Prepare merged percentile-based dataset with optional UN filtering.')
    parser.add_argument('dataset', type=str, help='cM_1, cM_3, cM_6, cM_10')
    parser.add_argument('--drop-un', action='store_true', help="Drop rows where kinship == 'UN' (UN-removed scenario)")
    args = parser.parse_args()

    dataset = args.dataset
    raw_csv = os.path.join('data', 'raw', f'model_input_with_kinship_filtered_{dataset}_percentile.csv')
    scenario_suffix = '_noUN' if args.drop_un else ''
    out_csv = os.path.join('data', 'processed', f'merged_{dataset}{scenario_suffix}.csv')

    # Load percentile-based dataset
    sep = '\t' if raw_csv.endswith('.tsv') or raw_csv.endswith('.txt') else None
    df_csv = pd.read_csv(raw_csv, sep=sep)
    
    # Keep pair + kinship + all features (percentiles and statistics)
    keep_cols = ['pair', 'kinship', 'n_segment', 'mean', 'median', 'max', 'total_length']
    percentile_cols = [col for col in df_csv.columns if col.endswith('%')]
    keep_cols.extend(percentile_cols)
    
    df_csv = df_csv[keep_cols]
    df_csv['pair'] = df_csv['pair'].astype(str).str.strip('[]')

    if args.drop_un:
        before = len(df_csv)
        df_csv = df_csv[df_csv['kinship'] != 'UN'].reset_index(drop=True)
        after = len(df_csv)
        print(f"Dropped UN rows: {before - after} (from {before} to {after})")

    # Save (no merging needed for percentile-based data)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_csv.to_csv(out_csv, index=False)
    scenario = 'UN-removed' if args.drop_un else 'UN-included'
    print(f"Percentile dataset saved to {out_csv} (scenario: {scenario})")
    print(f"  Shape: {df_csv.shape}")
    print(f"  Features: {len(keep_cols)} columns")
    print(f"  Kinship distribution:\n{df_csv['kinship'].value_counts().sort_index()}")


if __name__ == '__main__':
    main()

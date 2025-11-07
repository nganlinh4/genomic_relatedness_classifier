import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict

"""
Feature Engineering Hook

Purpose:
    Experiment with aggregate features derived from the top-importance features
    already selected by the RandomForest process. For each dataset & scenario we:
      1. Load merged_<dataset>[ _noUN].csv
      2. Load top_features_<dataset>[ _noUN].pkl (the 50 selected features)
      3. Compute engineered aggregates over those top features:
         - mean, std, min, max, median
         - skew, kurtosis
         - quantiles (q10, q25, q75, q90)
      4. Append these engineered columns to a new DataFrame alongside original selected features.
      5. Persist to data/processed/engineered_<dataset>_<scenario>.csv (ignored by git).

    This file is a placeholder for iterative experimentation (e.g., grouped aggregates,
    interaction terms, polynomial expansions) and is intentionally lightweight.

Usage:
    python scripts/feature_engineering.py cM_1 --scenario included
    python scripts/feature_engineering.py cM_1 --scenario noUN

    Integrate into pipeline later by replacing scaler/top feature loading in
    training with these engineered columns (optional).
"""

import pickle
from scipy.stats import skew, kurtosis

AGG_PREFIX = "agg_"

def compute_aggregates(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    values = df[feature_cols].values.astype(float)
    # Ensure 2D
    if values.ndim != 2:
        raise ValueError("Expected 2D array for feature aggregate computation")
    row_stats = {}
    # Row-wise aggregates
    row_stats['mean'] = np.mean(values, axis=1)
    row_stats['std'] = np.std(values, axis=1)
    row_stats['min'] = np.min(values, axis=1)
    row_stats['max'] = np.max(values, axis=1)
    row_stats['median'] = np.median(values, axis=1)
    row_stats['skew'] = skew(values, axis=1, bias=False)
    row_stats['kurtosis'] = kurtosis(values, axis=1, bias=False)
    for q in [0.10, 0.25, 0.75, 0.90]:
        row_stats[f'q{int(q*100)}'] = np.quantile(values, q, axis=1)
    return row_stats


def main():
    parser = argparse.ArgumentParser(description='Engineer aggregate features over top-importance feature subset')
    parser.add_argument('dataset', type=str, help='cM_1, cM_3, cM_6')
    parser.add_argument('--scenario', type=str, choices=['included','noUN'], default='included', help='Scenario: included or noUN')
    args = parser.parse_args()

    dataset = args.dataset
    scenario = args.scenario
    suffix = '' if scenario == 'included' else '_noUN'

    merged_path = os.path.join('data', 'processed', f'merged_{dataset}{suffix}.csv')
    top_features_path = os.path.join('data', 'processed', f'top_features_{dataset}{suffix}.pkl')

    if not os.path.exists(merged_path):
        print(f"Merged CSV not found: {merged_path}. Run data_prep first.")
        sys.exit(1)
    if not os.path.exists(top_features_path):
        print(f"Top features pickle not found: {top_features_path}. Run feature_selection first.")
        sys.exit(1)

    with open(top_features_path, 'rb') as f:
        top_features = pickle.load(f)

    df = pd.read_csv(merged_path)

    # Compute aggregates row-wise across selected top features
    aggregates = compute_aggregates(df, top_features)

    # Append engineered columns
    for key, arr in aggregates.items():
        df[f'{AGG_PREFIX}{key}'] = arr

    # Persist engineered dataset
    out_path = os.path.join('data', 'processed', f'engineered_{dataset}_{scenario}.csv')
    df.to_csv(out_path, index=False)
    print(f"Engineered feature dataset saved to {out_path}")
    print("Added columns:")
    for key in aggregates.keys():
        print(f"  - {AGG_PREFIX}{key}")

    # Simple summary of correlations (optional quick insight)
    try:
        corr_subset = df[[f'{AGG_PREFIX}mean', f'{AGG_PREFIX}std', f'{AGG_PREFIX}min', f'{AGG_PREFIX}max']].corr()
        print("\nCorrelation (subset of engineered features):")
        print(corr_subset)
    except Exception:
        pass

if __name__ == '__main__':
    main()

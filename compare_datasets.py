#!/usr/bin/env python3
"""Compare old data/raw/new with new external data."""

import pandas as pd
import os
from pathlib import Path

OLD_DIR = Path("/home/moagen/linh/genomic_relatedness_classifier/data/raw/new")
NEW_DIR = Path("/mg6-18t/prj/2506_LHC_SPO_Kinship/model_improvement/cbj_process/Analysis/Proto/ibis_test/genetic_distance/percentile_outputs_by_threshold_over_filtered")

def compare_merged_files():
    """Compare merged_cm_over_*.tsv files."""
    print("=" * 80)
    print("COMPARING MERGED_CM_OVER FILES")
    print("=" * 80)
    
    for threshold in [1, 3, 6, 10]:
        old_file = OLD_DIR / f"merged_cm_over_{threshold}.tsv"
        new_file = NEW_DIR / f"merged_cm_over_{threshold}.tsv"
        
        print(f"\n--- merged_cm_over_{threshold}.tsv ---")
        
        old_df = pd.read_csv(old_file, sep="\t")
        new_df = pd.read_csv(new_file, sep="\t")
        
        print(f"Old shape: {old_df.shape}, New shape: {new_df.shape}")
        print(f"Old columns: {list(old_df.columns)}")
        print(f"New columns: {list(new_df.columns)}")
        
        # Check for column differences
        old_cols = set(old_df.columns)
        new_cols = set(new_df.columns)
        if old_cols != new_cols:
            print(f"  Columns removed: {old_cols - new_cols}")
            print(f"  Columns added: {new_cols - old_cols}")
        
        # Check data types
        print(f"  Old dtypes:\n{old_df.dtypes.to_string()}")
        print(f"  New dtypes:\n{new_df.dtypes.to_string()}")
        
        # Check for row differences
        if 'sample_id' in old_df.columns:
            old_samples = set(old_df['sample_id'].unique())
            new_samples = set(new_df['sample_id'].unique())
            print(f"  Old unique samples: {len(old_samples)}, New unique samples: {len(new_samples)}")
            if old_samples != new_samples:
                print(f"  Samples removed: {len(old_samples - new_samples)}")
                print(f"  Samples added: {len(new_samples - old_samples)}")
        
        # Check for value differences
        common_cols = old_cols & new_cols
        for col in common_cols:
            if old_df[col].dtype in ['float64', 'int64']:
                old_stats = old_df[col].describe()
                new_stats = new_df[col].describe()
                if not old_stats.equals(new_stats):
                    print(f"  {col} stats differ:")
                    print(f"    Old: min={old_stats['min']:.4f}, max={old_stats['max']:.4f}, mean={old_stats['mean']:.4f}")
                    print(f"    New: min={new_stats['min']:.4f}, max={new_stats['max']:.4f}, mean={new_stats['mean']:.4f}")

def compare_percentile_files():
    """Compare percentile files in subdirectories."""
    print("\n" + "=" * 80)
    print("COMPARING PERCENTILE FILES")
    print("=" * 80)
    
    for threshold in [1, 3, 6, 10]:
        old_subdir = OLD_DIR / f"cm_over_{threshold}"
        new_subdir = NEW_DIR / f"cm_over_{threshold}"
        
        print(f"\n--- cm_over_{threshold}/ ---")
        
        old_files = sorted([f for f in os.listdir(old_subdir) if f.endswith('.tsv')])
        new_files = sorted([f for f in os.listdir(new_subdir) if f.endswith('.tsv')])
        
        print(f"Old files: {len(old_files)}, New files: {len(new_files)}")
        if old_files != new_files:
            print(f"  Files removed: {set(old_files) - set(new_files)}")
            print(f"  Files added: {set(new_files) - set(old_files)}")
        
        # Check first file as sample
        if old_files and new_files:
            sample_file = old_files[0]
            old_df = pd.read_csv(old_subdir / sample_file, sep="\t")
            new_df = pd.read_csv(new_subdir / sample_file, sep="\t")
            
            print(f"  Sample file ({sample_file}): Old shape: {old_df.shape}, New shape: {new_df.shape}")
            print(f"    Old columns: {list(old_df.columns)}")
            print(f"    New columns: {list(new_df.columns)}")

def compare_percentile_all():
    """Compare percentile_all_thresholds.tsv."""
    print("\n" + "=" * 80)
    print("COMPARING PERCENTILE_ALL_THRESHOLDS.TSV")
    print("=" * 80)
    
    old_file = OLD_DIR / "percentile_all_thresholds.tsv"
    new_file = NEW_DIR / "percentile_all_thresholds.tsv"
    
    old_df = pd.read_csv(old_file, sep="\t")
    new_df = pd.read_csv(new_file, sep="\t")
    
    print(f"Old shape: {old_df.shape}, New shape: {new_df.shape}")
    print(f"Old columns: {list(old_df.columns)}")
    print(f"New columns: {list(new_df.columns)}")
    
    if old_df.shape[0] != new_df.shape[0]:
        print(f"  Row count difference: {new_df.shape[0] - old_df.shape[0]}")

if __name__ == "__main__":
    compare_merged_files()
    compare_percentile_files()
    compare_percentile_all()
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

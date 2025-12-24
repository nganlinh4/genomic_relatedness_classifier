import sys
import os
import argparse
import pandas as pd
import re

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
    print(f"  Filtered out {removed_count} duplicate sample pairs (X-1 vs X-2)")
    return filtered_df

def parse_stats_file(path):
    rows = []
    if not os.path.exists(path):
        print(f"Warning: stats file missing: {path}")
        return pd.DataFrame(columns=['pair'])
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip() or 'allChr' not in line:
                continue
            m = re.search(r"\[(.*?)\]", line)
            if not m:
                continue
            pair_raw = m.group(1)
            pair = pair_raw.replace(' ', '')
            tail = line.split('allChr', 1)[1]
            stats = {'pair': pair}
            for mm in re.finditer(r'([^:]+?):(\S+)', tail):
                key = mm.group(1).strip()
                val = mm.group(2).strip()
                try:
                    stats[key] = float(val)
                except ValueError:
                    stats[key] = val
            rows.append(stats)
    df = pd.DataFrame(rows)
    if not df.empty:
        for c in df.columns:
            if c == 'pair': continue
            df[c] = pd.to_numeric(df[c], errors='ignore')
    return df

def process_percentile_mode(dataset, drop_un, data_dir='new'):
    """Logic derived from data_prep_percentile.py"""
    
    # Check for 251211 structure
    if '251211' in data_dir:
        # Map cM_1 -> merged_cm_over_1.tsv
        map_name = dataset.replace('cM', 'cm_over')
        raw_csv = os.path.join('data', 'raw', '251211', 'percentile_outputs_only_autosomal', f'merged_{map_name}.tsv')
        sep = '\t'
    else:
        raw_csv = os.path.join('data', 'raw', f'model_input_with_kinship_filtered_{dataset}_percentile.csv')
        sep = None # Auto-detect

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Percentile input file not found: {raw_csv}")

    df_csv = pd.read_csv(raw_csv, sep=sep, engine='python')
    
    # Rename PAIR_ID to pair if needed
    if 'PAIR_ID' in df_csv.columns:
        df_csv.rename(columns={'PAIR_ID': 'pair'}, inplace=True)

    keep_cols = ['pair', 'kinship', 'n_segment', 'mean', 'median', 'max', 'total_length']
    percentile_cols = [col for col in df_csv.columns if col.endswith('%')]
    keep_cols.extend(percentile_cols)
    
    # Filter columns that actually exist
    existing_cols = [c for c in keep_cols if c in df_csv.columns]
    df_csv = df_csv[existing_cols]
    
    if 'pair' in df_csv.columns:
        df_csv['pair'] = df_csv['pair'].astype(str).str.strip('[]')
    
    # Filter duplicate samples
    df_csv = filter_duplicate_samples(df_csv)
    
    return df_csv

def process_standard_mode(dataset):
    """Logic derived from original data_prep.py"""
    raw_csv = os.path.join('data', 'raw', f'model_input_with_kinship_filtered_{dataset}.csv')
    merged_info_path = os.path.join('data', 'raw', 'merged_info.out')
    merged_added_path = os.path.join('data', 'raw', 'merged_added_info.out')

    df_primary = parse_stats_file(merged_info_path)
    df_added = parse_stats_file(merged_added_path)

    if not df_primary.empty and not df_added.empty:
        df_merged = pd.merge(df_primary, df_added, on='pair', how='outer', suffixes=('', '_added'))
        primary_cols = set(df_primary.columns) - {'pair'}
        added_cols = set(df_added.columns) - {'pair'}
        common_cols = sorted(primary_cols & added_cols)
        
        for col in common_cols:
            added_col = f"{col}_added"
            if added_col in df_merged.columns:
                df_merged[col] = df_merged[col].combine_first(df_merged[added_col])
                df_merged.drop(columns=[added_col], inplace=True)
    elif not df_primary.empty:
        df_merged = df_primary
    else:
        df_merged = df_added

    sep = '\t' if raw_csv.endswith('.tsv') or raw_csv.endswith('.txt') else None
    df_csv = pd.read_csv(raw_csv, sep=sep)
    keep_cols = ['pair', 'IBD1_len', 'IBD2_len', 'R1', 'R2', 'Num_Segs', 'Total_len', 'kinship']
    df_csv = df_csv[keep_cols]
    df_csv['pair'] = df_csv['pair'].astype(str).str.strip('[]')

    # Filter duplicate samples
    df_csv = filter_duplicate_samples(df_csv)

    df_final = pd.merge(df_csv, df_merged, on='pair', how='inner')
    return df_final

def main():
    parser = argparse.ArgumentParser(description='Prepare merged feature dataset.')
    parser.add_argument('dataset', type=str, help='cM_1, cM_3, cM_6, cM_10')
    parser.add_argument('--drop-un', action='store_true', help="Drop rows where kinship == 'UN'")
    parser.add_argument('--type', type=str, choices=['standard', 'percentile'], default='standard', 
                        help="Data source type. 'standard' uses .out files, 'percentile' uses pre-calculated percentile CSVs.")
    parser.add_argument('--data-dir', type=str, default='new',
                        help="Data directory name (e.g., 'new', '251128')")
    args = parser.parse_args()

    dataset = args.dataset
    scenario_suffix = '_noUN' if args.drop_un else ''
    out_csv = os.path.join('data', 'processed', f'merged_{dataset}{scenario_suffix}.csv')

    print(f"Processing {dataset} (Type: {args.type}, Drop UN: {args.drop_un}, Data Dir: {args.data_dir})...")

    if args.type == 'percentile':
        df_final = process_percentile_mode(dataset, args.drop_un, args.data_dir)
    else:
        df_final = process_standard_mode(dataset)

    # Common filtering logic
    if args.drop_un:
        before = len(df_final)
        df_final = df_final[df_final['kinship'] != 'UN'].reset_index(drop=True)
        after = len(df_final)
        print(f"Dropped UN rows: {before - after} (from {before} to {after})")

    # Create and save artifacts for benchmark script
    import pickle
    from sklearn.preprocessing import StandardScaler
    
    # Identify features
    exclude = ['pair', 'kinship']
    features = [c for c in df_final.columns if c not in exclude]
    
    # Save features list
    feat_path = os.path.join('data', 'processed', f'top_features_{dataset}{scenario_suffix}.pkl')
    with open(feat_path, 'wb') as f:
        pickle.dump(features, f)
    
    # Fit and save scaler
    scaler = StandardScaler()
    X = df_final[features]
    scaler.fit(X)
    
    scaler_path = os.path.join('data', 'processed', f'scaler_{dataset}{scenario_suffix}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_final.to_csv(out_csv, index=False)
    print(f"Saved data to {out_csv}")
    print(f"Saved artifacts to {feat_path} and {scaler_path}")

if __name__ == '__main__':
    main()

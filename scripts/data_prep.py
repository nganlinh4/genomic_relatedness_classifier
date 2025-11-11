import sys
import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Prepare merged feature dataset with optional UN filtering.')
    parser.add_argument('dataset', type=str, help='cM_1, cM_3, cM_6')
    parser.add_argument('--drop-un', action='store_true', help="Drop rows where kinship == 'UN' (UN-removed scenario)")
    args = parser.parse_args()

    dataset = args.dataset
    raw_csv = os.path.join('data', 'raw', f'model_input_with_kinship_filtered_{dataset}.csv')
    merged_info_path = os.path.join('data', 'raw', 'merged_info.out')
    merged_added_path = os.path.join('data', 'raw', 'merged_added_info.out')
    scenario_suffix = '_noUN' if args.drop_un else ''
    out_csv = os.path.join('data', 'processed', f'merged_{dataset}{scenario_suffix}.csv')

    def parse_stats_file(path):
        rows = []
        if not os.path.exists(path):
            print(f"Warning: stats file missing: {path}")
            return pd.DataFrame(columns=['pair'])
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                pair = parts[0].strip('[]')
                stats = {'pair': pair}
                for part in parts[2:]:  # skip pair and allChr token
                    if ':' not in part:
                        continue
                    key, val = part.split(':', 1)
                    try:
                        stats[key] = float(val)
                    except ValueError:
                        continue
                rows.append(stats)
        return pd.DataFrame(rows)

    df_primary = parse_stats_file(merged_info_path)
    df_added = parse_stats_file(merged_added_path)

    # Merge the two stats sets on 'pair'; suffix collisions distinctly
    if not df_primary.empty and not df_added.empty:
        df_merged = pd.merge(df_primary, df_added, on='pair', how='outer', suffixes=('', '_added'))
    elif not df_primary.empty:
        df_merged = df_primary
    else:
        df_merged = df_added
    print(f"Primary stats cols: {len(df_primary.columns) - 1}, Added stats cols: {len(df_added.columns) - 1}, Final stats cols: {len(df_merged.columns) - 1}")

    # Load raw labeled pairs for this dataset
    sep = '\t' if raw_csv.endswith('.tsv') or raw_csv.endswith('.txt') else None
    df_csv = pd.read_csv(raw_csv, sep=sep)
    # Keep essential columns
    keep_cols = ['pair', 'IBD1_len', 'IBD2_len', 'R1', 'R2', 'Num_Segs', 'Total_len', 'kinship']
    df_csv = df_csv[keep_cols]
    df_csv['pair'] = df_csv['pair'].astype(str).str.strip('[]')

    if args.drop_un:
        before = len(df_csv)
        df_csv = df_csv[df_csv['kinship'] != 'UN'].reset_index(drop=True)
        after = len(df_csv)
        print(f"Dropped UN rows: {before - after} (from {before} to {after})")

    # Merge
    df_final = pd.merge(df_csv, df_merged, on='pair', how='inner')

    # Save
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_final.to_csv(out_csv, index=False)
    scenario = 'UN-removed' if args.drop_un else 'UN-included'
    print(f"Merged dataset saved to {out_csv} (scenario: {scenario})")


if __name__ == '__main__':
    main()

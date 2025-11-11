import sys
import os
import argparse
import pandas as pd
import re


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
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip() or 'allChr' not in line:
                    continue
                m = re.search(r"\[(.*?)\]", line)
                if not m:
                    continue
                pair_raw = m.group(1)
                # normalize pair id to match CSV format: remove spaces inside brackets
                pair = pair_raw.replace(' ', '')
                tail = line.split('allChr', 1)[1]
                stats = {'pair': pair}
                # Robust key:value extraction; keys may contain spaces and symbols (e.g., '<1cM (%)')
                for mm in re.finditer(r'([^:]+?):(\S+)', tail):
                    key = mm.group(1).strip()
                    val = mm.group(2).strip()
                    # Keep the last occurrence if duplicate percentile keys appear
                    try:
                        stats[key] = float(val)
                    except ValueError:
                        stats[key] = val
                rows.append(stats)
        df = pd.DataFrame(rows)
        # Attempt numeric conversion for all non-pair columns
        if not df.empty:
            for c in df.columns:
                if c == 'pair':
                    continue
                df[c] = pd.to_numeric(df[c], errors='ignore')
        return df

    df_primary = parse_stats_file(merged_info_path)
    df_added = parse_stats_file(merged_added_path)

    # Merge the two stats sets on 'pair'; use suffix for collisions temporarily
    if not df_primary.empty and not df_added.empty:
        df_merged = pd.merge(df_primary, df_added, on='pair', how='outer', suffixes=('', '_added'))

        # Identify column groups
        primary_cols = set(df_primary.columns) - {'pair'}
        added_cols = set(df_added.columns) - {'pair'}
        common_cols = sorted(primary_cols & added_cols)
        added_only_cols = sorted(added_cols - primary_cols)

        # For common columns, prefer the base (primary) values; if missing, backfill from _added, then drop _added
        dropped = []
        for col in common_cols:
            added_col = f"{col}_added"
            if added_col in df_merged.columns:
                # backfill NaNs in primary with values from added
                df_merged[col] = df_merged[col].combine_first(df_merged[added_col])
                df_merged.drop(columns=[added_col], inplace=True)
                dropped.append(added_col)

        print(
            "Stats merge: primary cols = {}, added cols = {}, common = {}, added-only kept = {}, duplicate cols dropped = {}".format(
                len(primary_cols), len(added_cols), len(common_cols), len(added_only_cols), len(dropped)
            )
        )
    elif not df_primary.empty:
        df_merged = df_primary
        print("Stats merge: only primary stats available; cols = {}".format(len(df_merged.columns) - 1))
    else:
        df_merged = df_added
        print("Stats merge: only added stats available; cols = {}".format(len(df_merged.columns) - 1))

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

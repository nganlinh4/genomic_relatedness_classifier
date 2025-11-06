import sys
import os
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/data_prep.py <dataset>")
        print("dataset: cM_1, cM_3, cM_6")
        sys.exit(1)

    dataset = sys.argv[1]
    raw_csv = os.path.join('data', 'raw', f'model_input_with_kinship_filtered_{dataset}.csv')
    merged_info_path = os.path.join('data', 'raw', 'merged_info.out')
    out_csv = os.path.join('data', 'processed', f'merged_{dataset}.csv')

    # Parse merged_info.out
    data = []
    with open(merged_info_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            pair = parts[0].strip('[]')
            stats = {'pair': pair}
            for part in parts[2:]:  # skip pair and allChr
                if ':' not in part:
                    continue
                key, val = part.split(':', 1)
                try:
                    stats[key] = float(val)
                except ValueError:
                    # Skip if value not float-parsable
                    continue
            data.append(stats)

    df_merged = pd.DataFrame(data)

    # Load raw labeled pairs for this dataset
    sep = '\t' if raw_csv.endswith('.tsv') or raw_csv.endswith('.txt') else None
    df_csv = pd.read_csv(raw_csv, sep=sep)
    # Keep essential columns
    keep_cols = ['pair', 'IBD1_len', 'IBD2_len', 'R1', 'R2', 'Num_Segs', 'Total_len', 'kinship']
    df_csv = df_csv[keep_cols]
    df_csv['pair'] = df_csv['pair'].astype(str).str.strip('[]')

    # Merge
    df_final = pd.merge(df_csv, df_merged, on='pair', how='inner')

    # Save
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_final.to_csv(out_csv, index=False)
    print(f"Merged dataset saved to {out_csv}")


if __name__ == '__main__':
    main()

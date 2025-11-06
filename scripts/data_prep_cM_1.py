import pandas as pd

# Process merged_info.out
data = []
with open('data/raw/merged_info.out', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        pair = parts[0].strip('[]')
        stats = {'pair': pair}
        for part in parts[2:]:  # skip pair and allChr
            key, val = part.split(':', 1)
            stats[key] = float(val)
        data.append(stats)

df_merged = pd.DataFrame(data)

# Process model_input_with_kinship_filtered_cM_1.csv
df_csv = pd.read_csv('data/raw/model_input_with_kinship_filtered_cM_1.csv', sep='\t')
df_csv = df_csv[['pair', 'IBD1_len', 'IBD2_len', 'R1', 'R2', 'Num_Segs', 'Total_len', 'kinship']]
df_csv['pair'] = df_csv['pair'].str.strip('[]')

# Merge
df_final = pd.merge(df_csv, df_merged, on='pair')

# Save
df_final.to_csv('data/processed/merged_cM_1.csv', index=False)

print("Merged dataset saved to data/processed/merged_cM_1.csv")